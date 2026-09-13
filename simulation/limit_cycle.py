"""Pacing a cell model to its limit cycle.

Paced at a fixed period, these models take thousands of beats to settle: the
action potential stabilises within a few hundred, but the intracellular sodium
and potassium concentrations drift for far longer.  Both the baseline subject
generator and the drug-block simulator need the same procedure — integrate in
blocks, compare features between consecutive blocks, stop when nothing moves by
more than its tolerance — so it lives here once.

Convergence is judged on the features returned by
:meth:`~cell_models.base.CardiacCellModel.calculate_limit_state_features`: the
AP features *and* the min/max of every ionic concentration.  Including the
concentrations is what makes the test meaningful; a model whose APD has
flattened may still be far from steady state.

Tolerances are per species and per feature, and are read from
``data/ap_features/tolerances/``.  See
``simulation/compute_limit_cycle_tolerances.py`` for how they were derived.
"""

import gc
from dataclasses import dataclass, field

import numpy as np

#: Beats to integrate between convergence checks.
CHECK_EVERY_N_CYCLES = 100

#: Give up after this many checks, i.e. 25,000 beats at the default block size.
MAX_CHECKS = 250

#: Pacing period in milliseconds (1 Hz).
STIM_PERIOD = 1000


@dataclass
class LimitCycleResult:
    """Outcome of a :func:`run_to_limit_cycle` call.

    Attributes:
        converged: True if every feature moved less than its tolerance between
            the last two blocks.
        solver_failed: True if the integrator failed, produced a non-finite
            value, or hit an infeasible AP with ``stop_on_bad_AP``. When set,
            the trajectory fields are None and ``converged`` is False.
        num_cycles: Beats actually simulated.
        feature_history: Feature name to the list of its values, one entry per
            block. Used for the convergence diagnostic plots.
        feasibility: Feature name to whether it sat inside its physiological
            bounds on the final block.
        voi, states, algebraic, rates: The final block's trajectory.
    """

    converged: bool
    solver_failed: bool
    num_cycles: int
    feature_history: dict = field(default_factory=dict)
    feasibility: dict = field(default_factory=dict)
    voi: np.ndarray = None
    states: np.ndarray = None
    algebraic: np.ndarray = None
    rates: np.ndarray = None

    @property
    def final_features(self):
        """Feature values from the last completed block."""
        return {k: v[-1] for k, v in self.feature_history.items() if v}

    @property
    def has_missing_features(self):
        """True if any feature was unmeasurable on the final block.

        Usually means repolarisation never reached the required threshold, so
        the "limit cycle" is degenerate rather than genuinely converged.
        """
        return any(v[-1] is None for v in self.feature_history.values() if v)


def _final_cycle_range(voi, model):
    """Index range covering the last complete beat of a block."""
    return [len(voi) - model.min_steps_needed - 1, len(voi) - 1]


def run_to_limit_cycle(
    model,
    t_0=0.0,
    stim_period=STIM_PERIOD,
    check_every_n_cycles=CHECK_EVERY_N_CYCLES,
    max_checks=MAX_CHECKS,
    stop_on_bad_AP=False,
    check_feasibility=False,
    verbose=False,
):
    """Pace ``model`` in blocks until its features stop changing.

    Integration resumes from ``model.init_states``, so callers control the
    starting point: the baseline generator perturbs the constants first, and the
    drug-block simulator loads a saved baseline limit state.

    Args:
        model: A :class:`~cell_models.base.CardiacCellModel` subclass instance.
        t_0: Simulation time to start from, in milliseconds. Non-zero when
            continuing from a saved state, so the stimulus stays in phase.
        stim_period: Pacing period in milliseconds.
        check_every_n_cycles: Beats per block.
        max_checks: Maximum number of blocks before giving up.
        stop_on_bad_AP: Abort as soon as an AP falls outside its feasibility
            bounds. Used when simulating drug block, where an infeasible AP
            means that concentration is not viable.
        check_feasibility: Also record which features were physiological on each
            block. Used when generating subjects, which must be both converged
            and feasible.
        verbose: Print the feature blocking convergence at each check.

    Returns:
        A :class:`LimitCycleResult`.
    """
    hyperparameters = model.ode_hyperparameters
    num_steps = model.min_steps_needed * check_every_n_cycles + 1

    voi = states = algebraic = rates = None
    feature_history = {}
    feasibility = {}
    converged = False
    check_count = 0

    while not converged and check_count < max_checks:
        t_start = t_0 + check_count * check_every_n_cycles * stim_period
        t_max = t_start + check_every_n_cycles * stim_period

        if check_count > 0:
            # Continue from the end of the previous block.
            model.set_states(states[:, -1])
            del voi, states, algebraic, rates
            gc.collect()

        voi, states, algebraic, rates = model.solve(
            stim_period=stim_period,
            t_start=t_start,
            t_max=t_max,
            num_steps=num_steps,
            num_ap_checks=0,
            plot_apd=False,
            stop_on_bad_AP=stop_on_bad_AP,
            save_states=False,
            **hyperparameters,
        )

        if voi is None:
            return LimitCycleResult(
                converged=False,
                solver_failed=True,
                num_cycles=check_count * check_every_n_cycles,
                feature_history=feature_history,
            )

        idx_range = _final_cycle_range(voi, model)
        features = model.calculate_limit_state_features(
            voi, states, rates, algebraic, time_idx_range=idx_range
        )
        for key, value in features.items():
            feature_history.setdefault(key, []).append(value)

        if check_feasibility:
            feasibility = model.check_ap_feasibility(
                voi, states, rates, algebraic,
                time_idx_range=idx_range, return_dict=True,
            )

        check_count += 1
        if check_count > 1:
            converged = _all_features_settled(
                model, feature_history, feasibility,
                check_count * check_every_n_cycles, verbose,
            )

    return LimitCycleResult(
        converged=converged,
        solver_failed=False,
        num_cycles=check_count * check_every_n_cycles,
        feature_history=feature_history,
        feasibility=feasibility,
        voi=voi,
        states=states,
        algebraic=algebraic,
        rates=rates,
    )


def _all_features_settled(model, feature_history, feasibility, cycles, verbose):
    """True if every feature moved less than its tolerance in the last block.

    A feature that has become unmeasurable, or that has settled at a value
    outside its physiological bounds, also terminates the search — the model is
    not going to improve by pacing it longer — but the caller can tell those
    cases apart via ``LimitCycleResult.has_missing_features`` and
    ``.feasibility``.
    """
    settled = True
    for key, values in feature_history.items():
        if values[-1] is None or values[-2] is None:
            if verbose:
                print(f"  {cycles} cycles: {key} unmeasurable")
            return True  # degenerate; no point pacing further

        diff = abs(values[-1] - values[-2])
        within_tol = np.isclose(0.0, diff, atol=model.limit_cycle_tol_dict[key])

        if within_tol and feasibility.get(key) is False:
            if verbose:
                print(f"  {cycles} cycles: {key} converged but outside AP limits")
            return True  # settled on an infeasible value

        if not within_tol:
            if verbose:
                print(
                    f"  {cycles} cycles: {key} still moving, "
                    f"diff {diff:.3g} > tol {model.limit_cycle_tol_dict[key]:.3g}"
                )
            settled = False

    return settled
