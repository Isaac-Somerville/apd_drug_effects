"""Derive each species' per-feature convergence tolerances.

:func:`~simulation.limit_cycle.run_to_limit_cycle` decides a model has settled
when every feature moves less than its tolerance between consecutive blocks.
Those tolerances have to come from somewhere, and setting them by hand would
make "converged" an arbitrary judgement.

Instead they are measured. Starting from a state already at the limit cycle,
the model is paced for many further blocks and every feature recorded. Whatever
residual movement remains is, by construction, the movement of a *converged*
model: the numerical noise floor of the integrator at this step size, not real
drift. The tolerance for a feature is the largest consecutive change observed
over that run, so the convergence test reads as "has stopped moving by more
than a converged model moves anyway".

Two floors are applied:

* Time-valued features (APD, CTD, triangulation) are floored at the sampling
  step, ``stim_period / min_steps_needed``. APD is read off the integration
  grid, so it is quantised to that step and a tolerance below it could never be
  satisfied — the feature would appear to jump by a whole sample.
* Everything else is floored at 1e-16, guarding against a tolerance of exactly
  zero when a feature happens not to move at all.

**Guinea pig.** That model integrates in seconds, so its APD is ~0.16 rather
than ~160, while the sampling-step floor is computed from a stimulus period
expressed in milliseconds. The floor therefore lands orders of magnitude above
the feature values and the time-valued tolerances never bind: guinea-pig
convergence is decided entirely by its ionic concentrations. That is a real
property of the shipped tolerances and the published dataset was produced with
them, so it is preserved here rather than corrected.

Writes ``data/ap_features/tolerances/``. The shipped files are the published
ones; regenerating them changes which runs count as converged.

Example:
    python -m simulation.compute_limit_cycle_tolerances --species Dog
"""

import argparse
import gc
import json

from cell_models import NUM_CYCLES_LIMIT_STATE, SPECIES, build_model
from paths import LIMIT_CYCLE_TOLERANCES_DIR, ensure
from simulation.limit_cycle import CHECK_EVERY_N_CYCLES, STIM_PERIOD
from simulation.simulate_drug_block import baseline_state_path

#: Blocks to pace past the limit cycle when sampling residual movement.
NUM_BLOCKS = 100

#: Features measured in the model's time unit, floored at the sampling step.
TIME_FEATURES = ["APD_40", "APD_50", "APD_90", "tri_90_40", "CTD_50", "CTD_90"]

#: Floor for everything else: only guards against a tolerance of exactly zero.
ABSOLUTE_FLOOR = 1e-16

#: Subject whose limit state defines the species baseline.
BASELINE_SUBJECT = 0


def tolerances_filename(species, num_cycles):
    return f"{species}_{num_cycles}_cycles_limit_cycle_tolerances.json"


def sample_residual_movement(species, num_blocks=NUM_BLOCKS,
                             block_cycles=CHECK_EVERY_N_CYCLES, verbose=True):
    """Pace past the limit cycle and return each feature's value per block.

    Returns:
        Feature name to a list of ``num_blocks`` values.
    """
    model = build_model(species)
    state_path = baseline_state_path(species, BASELINE_SUBJECT)
    if not state_path.exists():
        raise FileNotFoundError(
            f"No baseline limit state for {species} at {state_path}. "
            f"Run simulation.generate_subjects first."
        )
    with open(state_path) as f:
        limit_state = json.load(f)

    model.set_states(limit_state["states"])
    model.set_constants(limit_state["constants"])
    t_0 = limit_state["voi"]
    num_steps = model.min_steps_needed * block_cycles + 1

    history = {}
    voi = states = algebraic = rates = None
    for block in range(num_blocks):
        t_start = t_0 + block * block_cycles * STIM_PERIOD
        if block > 0:
            model.set_states(states[:, -1])
            del voi, states, algebraic, rates
            gc.collect()

        voi, states, algebraic, rates = model.solve(
            stim_period=STIM_PERIOD,
            t_start=t_start,
            t_max=t_start + block_cycles * STIM_PERIOD,
            num_steps=num_steps,
            num_ap_checks=0,
            plot_apd=False,
            stop_on_bad_AP=False,
            save_states=False,
            **model.ode_hyperparameters,
        )
        if voi is None:
            raise RuntimeError(
                f"{species}: solver failed at block {block} while sampling "
                f"residual movement."
            )

        features = model.calculate_limit_state_features(
            voi, states, rates, algebraic,
            time_idx_range=[len(voi) - model.min_steps_needed - 1, len(voi) - 1],
        )
        for key, value in features.items():
            history.setdefault(key, []).append(value)

        if verbose and (block + 1) % 20 == 0:
            print(f"  {species}: {(block + 1) * block_cycles} cycles")

    return history, model


def derive_tolerances(history, model):
    """Largest consecutive change per feature, floored as described above."""
    time_step = STIM_PERIOD / model.min_steps_needed
    tolerances = {}
    for key, values in history.items():
        floor = max(time_step, ABSOLUTE_FLOOR) if key in TIME_FEATURES else ABSOLUTE_FLOOR
        usable = [v for v in values if v is not None]
        diffs = [
            abs(usable[i] - usable[i - 1]) for i in range(1, len(usable))
        ]
        tolerances[key] = max(max(diffs, default=0.0), floor)
    return tolerances


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ensure(LIMIT_CYCLE_TOLERANCES_DIR)

    for species in args.species:
        history, model = sample_residual_movement(
            species, num_blocks=args.num_blocks, verbose=not args.quiet
        )
        tolerances = derive_tolerances(history, model)
        path = LIMIT_CYCLE_TOLERANCES_DIR / tolerances_filename(
            species, NUM_CYCLES_LIMIT_STATE[species]
        )
        with open(path, "w") as f:
            json.dump(tolerances, f, indent=4)
        print(f"{species}: {len(tolerances)} tolerances  ->  {path.name}")


if __name__ == "__main__":
    main()
