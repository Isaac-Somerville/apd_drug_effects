"""Find the coarsest integration grid that still resolves the action potential.

Every downstream stage measures AP features on a grid of ``min_steps_needed``
points per beat, so this number sets both the accuracy of every APD in the
dataset and the cost of producing it — the drug-block stage integrates tens of
thousands of beats per subject, so a grid twice as fine costs twice as much.

The search is a downward sweep: pace one beat at 5000 steps, drop 100 at a
time, and stop when the AP first fails :meth:`check_ap_feasibility`. Too coarse
a grid does not fail gracefully — the upstroke is undersampled, ``dvdt_max``
collapses and the repolarisation thresholds land on the wrong samples — so
feasibility is a usable proxy for "the grid still resolves the AP". The last
grid that passed is kept.

The result is quantisation, not just cost: APD is read off this grid, so it is
a multiple of ``stim_period / (min_steps_needed - 1)``. That is why published
baseline APDs are values like 188.188... rather than round numbers, and why
changing this number changes every APD in the dataset.

Writes ``data/ap_features/min_steps_needed_rounded.json``, keyed by species.
The shipped file is the published one; regenerating it invalidates the dataset.

Example:
    python -m simulation.compute_min_steps --species Dog Mouse
"""

import argparse
import json
import time

from cell_models import SPECIES, build_model
from paths import MIN_STEPS_NEEDED
from simulation.limit_cycle import STIM_PERIOD

#: Finest grid tried, in integration steps per beat.
MAX_STEPS = 5000

#: Step-count decrement per iteration of the downward sweep.
STEP_DECREMENT = 100

#: Never go below this, regardless of feasibility.
MIN_STEPS = 100


def min_steps_for_species(species, max_steps=MAX_STEPS, decrement=STEP_DECREMENT,
                          min_steps=MIN_STEPS, verbose=True):
    """Coarsest grid at which ``species`` still produces a feasible AP.

    Returns:
        Step count per beat, or ``max_steps`` if even that fails — which means
        the model is misconfigured rather than the grid being too coarse.
    """
    model = build_model(species, num_cycles_feasible_ap=0)
    num_steps = max_steps
    last_good = None

    while num_steps >= min_steps:
        start = time.time()
        voi, states, algebraic, rates = model.solve(
            stim_period=STIM_PERIOD,
            t_max=STIM_PERIOD,
            num_steps=num_steps,
            num_ap_checks=0,
            plot_apd=False,
            stop_on_bad_AP=True,
            save_states=False,
            **model.ode_hyperparameters,
        )
        feasible = voi is not None and model.check_ap_feasibility(
            voi, states, rates, algebraic
        )
        if verbose:
            print(
                f"  {species}: {num_steps:5d} steps -> "
                f"{'feasible' if feasible else 'INFEASIBLE'} "
                f"({time.time() - start:.1f}s)"
            )
        if not feasible:
            break
        last_good = num_steps
        num_steps -= decrement

    return last_good if last_good is not None else max_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument(
        "--out", default=None,
        help="Override the output path. Defaults to the shipped file, which "
             "the published dataset depends on.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_path = MIN_STEPS_NEEDED if args.out is None else args.out

    existing = {}
    if MIN_STEPS_NEEDED.exists():
        with open(MIN_STEPS_NEEDED) as f:
            existing = json.load(f)

    results = dict(existing)
    for species in args.species:
        steps = min_steps_for_species(species, verbose=not args.quiet)
        was = existing.get(species)
        note = "" if was is None else f"  (published value: {was})"
        print(f"{species}: {steps} steps per beat{note}")
        results[species] = steps

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
