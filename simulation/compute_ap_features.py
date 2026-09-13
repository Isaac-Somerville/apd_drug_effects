"""Measure each species' baseline AP features and derive its feasibility bounds.

Two stages, both writing per-species JSON that the cell models load on
construction.

Baseline features
-----------------
Pace one beat of the unperturbed model and record the nine AP features. Written
to ``data/ap_features/defaults/``. These describe the species' reference action
potential: the drug-block dataset is expressed relative to each subject's own
baseline, and these are the population-level equivalent.

Feasibility bounds
------------------
:meth:`~cell_models.base.CardiacCellModel.check_ap_feasibility` needs an upper
and lower bound on every feature, per species. Fixing them by hand for six
species would be arbitrary, so instead one set of *human* bounds is taken as
given (:data:`HUMAN_AP_LIMITS`, from the physiological ranges reported for human
ventricular myocytes) and scaled onto each species by the ratio of that
species' baseline feature to the human baseline:

    bound_species = bound_human * (feature_species / feature_human)

So a mouse, whose APD90 is roughly a fifteenth of a human's, inherits bounds a
fifteenth as wide. The assumption is that "how far from baseline is still
physiological" is a property of the *shape* of the AP rather than its
timescale — a species-invariant fractional tolerance. That is an approximation,
but it makes the bounds reproducible from one hand-set reference rather than
six, and it is only ever used to reject grossly non-physiological simulations,
not to make fine distinctions. Written to ``data/ap_features/limits/``.

Guinea pig is a special case throughout: it integrates in seconds, so its
features come back in seconds and its bounds are therefore in seconds too. The
ratio is dimensionless, so the scaling handles this correctly on its own.

Relationship to the shipped files
---------------------------------
**The shipped files are authoritative and this script does not reproduce them
bit-for-bit.** They were measured from an intermediate series of saved states,
one per cycle count, which was not retained; only the final limit state per
subject survives, and that is what this script starts from. Where the two
coincide the agreement is exact — dog matches to 1e-12 — and where they do not,
it does not: guinea pig APDs differ by ~3%, its calcium-transient durations by
much more, and pig's ``dvdt_max`` by ~34%.

This does not affect the dataset. The drug-block simulations read the *shipped*
bounds and tolerances, and the dataset built from them reproduces bit-for-bit
(see ``verification/``). This script documents how those files were derived and
lets them be regenerated for a new species or a changed model, but rerunning it
over the published species would move the feasibility bounds and so change
which simulations are accepted. Write elsewhere with ``--out-dir`` unless that
is what you intend.

Example:
    python -m simulation.compute_ap_features --species Dog Mouse
    python -m simulation.compute_ap_features --from-initial-conditions
"""

import argparse
import json
from pathlib import Path

from cell_models import NUM_CYCLES_LIMIT_STATE, SPECIES, build_model
from paths import AP_LIMITS_DIR, DEFAULT_AP_FEATURES_DIR, ensure
from simulation.limit_cycle import STIM_PERIOD
from simulation.simulate_drug_block import baseline_state_path

#: Physiological bounds on human ventricular AP features, in ms / mV /
#: mV per ms. Every other species' bounds are derived from these by ratio.
HUMAN_AP_LIMITS = {
    "APD_40": [85, 320],
    "APD_50": [110, 350],
    "APD_90": [180, 440],
    "tri_90_40": [50, 150],
    "dvdt_max": [100, 1000],
    "v_peak": [10, 55],
    "RMP": [-95, -80],
    "CTD_50": [120, 420],
    "CTD_90": [220, 785],
}

#: The species the bounds are anchored on.
REFERENCE_SPECIES = "Human"

#: Subject whose limit state defines the species baseline. Subject 0 is the
#: unperturbed model, so this is the published parameterisation rather than a
#: sampled one.
BASELINE_SUBJECT = 0


def features_filename(species, num_cycles):
    return f"{species}_{num_cycles}_cycles_default_AP_features.json"


def limits_filename(species, num_cycles):
    return f"{species}_{num_cycles}_cycles_AP_limits.json"


def measure_features(species, from_initial_conditions=False):
    """Pace one beat and return that species' AP features.

    Args:
        species: One of :data:`~cell_models.SPECIES`.
        from_initial_conditions: Start from the model's published initial
            conditions instead of its converged limit state. Those are the
            values the source paper reports, so this measures the AP *before*
            the long pacing drift — recorded under a cycle count of 0.

    Returns:
        ``(features, num_cycles)``, or ``(None, num_cycles)`` if the solver
        failed.
    """
    num_cycles = 0 if from_initial_conditions else NUM_CYCLES_LIMIT_STATE[species]
    model = build_model(species, num_cycles_feasible_ap=0)

    t_0 = 0.0
    if not from_initial_conditions:
        state_path = baseline_state_path(species, BASELINE_SUBJECT)
        if not state_path.exists():
            raise FileNotFoundError(
                f"No baseline limit state for {species} at {state_path}. "
                f"Run simulation.generate_subjects, or pass "
                f"--from-initial-conditions."
            )
        with open(state_path) as f:
            limit_state = json.load(f)
        model.set_states(limit_state["states"])
        model.set_constants(limit_state["constants"])
        t_0 = limit_state["voi"]

    # One beat, sampled exactly as the drug-block stage samples it, so the
    # features are quantised the same way as every APD in the dataset.
    voi, states, algebraic, rates = model.solve(
        stim_period=STIM_PERIOD,
        t_start=t_0,
        t_max=t_0 + STIM_PERIOD,
        num_steps=int(model.min_steps_needed),
        num_ap_checks=0,
        plot_apd=False,
        stop_on_bad_AP=False,
        save_states=False,
        **model.ode_hyperparameters,
    )
    if voi is None:
        return None, num_cycles
    return model.calculate_ap_features(voi, states, rates, algebraic), num_cycles


def derive_limits(species_features, reference_features):
    """Scale the human bounds onto a species by its baseline feature ratios.

    Features the species does not report (no calcium transient, say) are simply
    absent from the result, and are then never checked for that species.
    """
    limits = {}
    for feature, bounds in HUMAN_AP_LIMITS.items():
        value = species_features.get(feature)
        reference = reference_features.get(feature)
        if value is None or reference is None or reference == 0:
            continue
        ratio = value / reference
        limits[feature] = [bounds[0] * ratio, bounds[1] * ratio]
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument(
        "--from-initial-conditions", action="store_true",
        help="Measure from the published initial conditions rather than the "
             "converged limit state.",
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Skip the feasibility-bound stage.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Write under this directory instead of overwriting the shipped "
             "files, which the published dataset depends on. Creates "
             "'defaults/' and 'limits/' subdirectories.",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        features_dir, limits_dir = DEFAULT_AP_FEATURES_DIR, AP_LIMITS_DIR
    else:
        features_dir = Path(args.out_dir) / "defaults"
        limits_dir = Path(args.out_dir) / "limits"
    ensure(features_dir)
    ensure(limits_dir)

    # The reference species is always needed, since the bounds are ratios
    # against it, even when it was not asked for.
    species_list = list(args.species)
    if not args.features_only and REFERENCE_SPECIES not in species_list:
        species_list.append(REFERENCE_SPECIES)

    measured = {}
    for species in species_list:
        features, num_cycles = measure_features(
            species, from_initial_conditions=args.from_initial_conditions
        )
        if features is None:
            print(f"{species}: solver failed, skipping")
            continue
        measured[species] = (features, num_cycles)

        path = features_dir / features_filename(species, num_cycles)
        with open(path, "w") as f:
            json.dump(features, f, indent=4)
        apd90 = features.get("APD_90")
        unit = "s" if species == "Guinea Pig" else "ms"
        print(f"{species}: APD90 = {apd90:.6g} {unit}  ->  {path.name}")

    if args.features_only:
        return

    if REFERENCE_SPECIES not in measured:
        raise RuntimeError(
            f"{REFERENCE_SPECIES} features are required to derive bounds for "
            f"any species, and they could not be measured."
        )
    reference_features, _ = measured[REFERENCE_SPECIES]

    for species in args.species:
        if species not in measured:
            continue
        features, num_cycles = measured[species]
        limits = derive_limits(features, reference_features)
        path = limits_dir / limits_filename(species, num_cycles)
        with open(path, "w") as f:
            json.dump(limits, f, indent=4)
        print(f"{species}: {len(limits)} bounds  ->  {path.name}")


if __name__ == "__main__":
    main()
