"""Generate virtual subjects: perturbed models paced to a feasible limit cycle.

Between-subject variability is introduced by perturbing the model's maximal
conductances.  Each constant listed in ``data/model_constants/base_values/`` is
resampled lognormally about its published value, with the coefficient of
variation for that constant taken from ``data/ap_features/parameter_CVs.json``.
Using a lognormal keeps conductances positive and preserves their sign.

A perturbed model is only accepted if it both

1. reaches a limit cycle within :data:`~simulation.limit_cycle.MAX_CHECKS`
   blocks, and
2. produces a physiologically feasible action potential there — every AP
   feature inside the bounds in ``data/ap_features/limits/``.

Perturbations that fail either test are discarded and the seed is advanced, so
the subject population is a *rejection sample* from the perturbation
distribution, conditioned on physiological viability. Rejection rates differ
sharply between species, which is why subject counts vary.

To keep the search cheap, each attempt warm-starts from the limit state of the
already-accepted subject whose constants are closest (see
:meth:`~cell_models.base.CardiacCellModel.calculate_model_distance`), rather
than from the published initial conditions.

Output, per species x subject:
    ``data/saved_states/baseline/<species>/v<n>_limit_state.json``

Examples:
    python -m simulation.generate_subjects --species Dog --subjects 0-19
    python -m simulation.generate_subjects --species Mouse --subjects 7 --verbose
"""

import argparse
import json
import os
import re

from cell_models import SPECIES, build_model
from paths import BASELINE_STATES, ensure
from simulation.limit_cycle import STIM_PERIOD, run_to_limit_cycle
from simulation.simulate_drug_block import MAX_SUBJECT_IDX, parse_subject_range

#: Seeds for subject ``n`` start here and increment on each rejected attempt.
#: Deterministic in the subject index, so a given subject is reproducible
#: independently of which others have been generated.
SEED_STRIDE = 10_000

#: Abandon a subject after this many rejected perturbations.
MAX_ATTEMPTS = 200

_STATE_FILE_RE = re.compile(r"^v(\d+)_limit_state\.json$")


def existing_subjects(species):
    """Indices of subjects already generated for ``species``, sorted."""
    folder = BASELINE_STATES / species
    if not folder.exists():
        return []
    found = []
    for file in folder.iterdir():
        match = _STATE_FILE_RE.match(file.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def load_existing_states(species, idxs):
    """Load the saved limit states for ``idxs``, keyed by index."""
    states = {}
    for idx in idxs:
        path = BASELINE_STATES / species / f"v{idx}_limit_state.json"
        with open(path) as f:
            states[idx] = json.load(f)
    return states


def _warm_start(model, saved_states):
    """Seed the model's states from the most similar accepted subject.

    Similarity is measured on the constants, not the states: two subjects with
    near-identical conductances settle into near-identical limit cycles, so
    starting from that neighbour's state saves most of the pacing.
    """
    if not saved_states:
        return
    closest = min(
        saved_states,
        key=lambda i: model.calculate_model_distance(
            model2_constants=saved_states[i]["constants"]
        ),
    )
    model.set_states(saved_states[closest]["states"])


def generate_subject(species, subject_idx, overwrite=False, verbose=False):
    """Find and save one feasible, converged subject.

    Returns:
        The seed that produced the accepted subject, or None if every attempt
        was rejected.
    """
    out_dir = ensure(BASELINE_STATES / species)
    out_path = out_dir / f"v{subject_idx}_limit_state.json"
    if out_path.exists() and not overwrite:
        print(f"[{species}/v{subject_idx}] already exists, skipping")
        return None

    others = [i for i in existing_subjects(species) if i != subject_idx]
    saved_states = load_existing_states(species, others)

    model = build_model(species)
    base_seed = SEED_STRIDE * abs(subject_idx - 1)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        seed = base_seed + attempt
        model.initConsts(perturbation="lognormal", seed=seed)
        _warm_start(model, saved_states)

        result = run_to_limit_cycle(
            model,
            stim_period=STIM_PERIOD,
            stop_on_bad_AP=False,
            check_feasibility=True,
            verbose=verbose,
        )

        if result.solver_failed:
            if verbose:
                print(f"[{species}/v{subject_idx}] attempt {attempt}: solver failed")
            continue
        if not result.converged or result.has_missing_features:
            if verbose:
                print(f"[{species}/v{subject_idx}] attempt {attempt}: no limit cycle")
            continue
        if not all(result.feasibility.values()):
            failed = [k for k, ok in result.feasibility.items() if not ok]
            if verbose:
                print(
                    f"[{species}/v{subject_idx}] attempt {attempt}: "
                    f"AP infeasible ({', '.join(failed)})"
                )
            continue

        model.save_model(
            result.voi[-1],
            result.states[:, -1],
            result.algebraic[:, -1],
            model.constants,
            f"{out_dir}/",
            f"v{subject_idx}_limit_state.json",
            seed=seed,
        )
        print(
            f"[{species}/v{subject_idx}] accepted on attempt {attempt} "
            f"(seed {seed}) after {result.num_cycles} cycles"
        )
        return seed

    print(f"[{species}/v{subject_idx}] no feasible subject in {MAX_ATTEMPTS} attempts")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument(
        "--subjects", default=f"0-{MAX_SUBJECT_IDX}",
        help='Subject indices, e.g. "0-19" or "1,4,7". Default: all.',
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    subject_idxs = parse_subject_range(args.subjects)
    combos = [(sp, idx) for idx in subject_idxs for sp in args.species]

    # Under PBS, each task generates exactly one subject.
    array_index = os.environ.get("PBS_ARRAY_INDEX")
    if array_index is not None:
        combos = [combos[int(array_index) % len(combos)]]
        print(f"PBS task {array_index}: {combos[0][0]} / v{combos[0][1]}")

    for species, subject_idx in combos:
        generate_subject(
            species, subject_idx, overwrite=args.overwrite, verbose=args.verbose
        )


if __name__ == "__main__":
    main()
