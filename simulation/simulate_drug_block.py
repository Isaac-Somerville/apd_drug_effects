"""Simulate APD under drug block, for one species x drug x subject.

For each concentration in :data:`CONCENTRATIONS`, the subject's drug-free limit
state is loaded, the drug is applied at that fixed concentration, and the model
is paced until it settles into a *new* limit cycle.  The APD90 of that new cycle
is the measurement; the state it settled into is saved so the run can be
resumed or inspected.

This is the expensive step of the pipeline: 11 concentrations x up to 25,000
beats each, per subject.  On a cluster, run it as a PBS array job (see
``simulation/shell/run_simulate_drug_block.sh``) — the task index is unpacked
into a (species, drug, subject) triple by :func:`unpack_array_index`.

Outputs, per species x drug x subject:
    ``outputs/simulation/drug_block/<drug>/<species>/v<n>/apd.csv``
        APD90 at each concentration.
    ``data/saved_states/<drug>/<species>/v<n>/<conc>_limit_state.json``
        The limit state reached at that concentration.

Examples:
    python -m simulation.simulate_drug_block --species Dog --drug quinidine \\
        --subjects 0-19
    python -m simulation.simulate_drug_block           # every combination
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cell_models import SPECIES, build_model
from drug import Drug, load_drug_table
from paths import BASELINE_STATES, SAVED_STATES, SIMULATION_OUT, ensure
from simulation.limit_cycle import STIM_PERIOD, run_to_limit_cycle

#: Number of non-zero concentrations simulated per drug.
NUM_CONCENTRATIONS = 9

#: Highest concentration simulated, as a multiple of the drug's free
#: therapeutic concentration. Three times therapeutic is the exposure the CiPA
#: proarrhythmia assessment is built around.
MAX_THERAPEUTIC_MULTIPLE = 3

#: Highest subject index generated per species.
MAX_SUBJECT_IDX = 41


def drug_concentrations(drug):
    """The nine non-zero concentrations simulated for ``drug``, in nM.

    Evenly spaced from ``3 * c_therapeutic / 9`` up to ``3 * c_therapeutic``, so
    the grid is *per drug*: each compound is sampled over its own clinically
    relevant range rather than over one grid shared across drugs. Experiments
    therefore index concentrations by position (0-8), not by absolute value.

    Concentration zero is not simulated — it is the subject's drug-free limit
    cycle, which is already saved under ``data/saved_states/baseline/``.
    """
    c_end = MAX_THERAPEUTIC_MULTIPLE * drug.c_therapeutic
    return [
        c_end * (i + 1) / NUM_CONCENTRATIONS for i in range(NUM_CONCENTRATIONS)
    ]


def concentration_label(conc):
    """Filename-safe label for a concentration, e.g. ``"3237.00"``."""
    return f"{conc:.2f}"


def baseline_state_path(species, subject_idx):
    """Path to a subject's drug-free limit state."""
    return BASELINE_STATES / species / f"v{subject_idx}_limit_state.json"


def unpack_array_index(index, species_list, drug_list, subject_idxs):
    """Map a flat PBS array index onto a (species, drug, subject) triple.

    Drug varies fastest, then species, then subject, so a partial array run
    still covers whole species x drug blocks.
    """
    n_drugs = len(drug_list)
    n_species = len(species_list)
    drug = drug_list[index % n_drugs]
    species = species_list[(index // n_drugs) % n_species]
    subject = subject_idxs[(index // (n_drugs * n_species)) % len(subject_idxs)]
    return species, drug, subject


def baseline_apd(model, baseline):
    """APD90 of the drug-free limit cycle, i.e. the concentration-zero value.

    The saved state is already converged, so this paces a *single* beat purely
    to measure the AP rather than searching for convergence.

    That single beat is sampled at exactly ``min_steps_needed`` points, giving a
    step of ``stim_period / (min_steps_needed - 1)``, and APD is read off that
    grid. The measurement is therefore quantised to the step size — for the dog,
    1000/999 ms, which is why baseline APDs come out as values like
    188.188... rather than round numbers. Changing the step count changes the
    quantisation, so it is fixed here to keep the dataset reproducible.
    """
    model.set_states(baseline["states"])
    model.set_constants(baseline["constants"])
    model.drug = None

    voi, states, algebraic, rates = model.solve(
        stim_period=STIM_PERIOD,
        t_start=baseline["voi"],
        t_max=baseline["voi"] + STIM_PERIOD,
        num_steps=int(model.min_steps_needed),
        num_ap_checks=0,
        plot_apd=False,
        stop_on_bad_AP=False,
        save_states=False,
        **model.ode_hyperparameters,
    )
    if voi is None:
        return None
    return model.calculate_ap_features(voi, states, rates, algebraic)["APD_90"]


def simulate_subject(species, drug_name, subject_idx, overwrite=False, verbose=False):
    """Run every concentration for one subject and write its APD table.

    Returns:
        A Series of APD90 indexed by concentration label, with the drug-free
        value first. Entries are NaN where the model failed to solve or never
        produced a measurable APD90 — a real outcome meaning the cell is not
        viable at that concentration, which ``build_apd_dataset.py`` then
        decides how to handle.
    """
    out_dir = ensure(SIMULATION_OUT / "drug_block" / drug_name / species / f"v{subject_idx}")
    states_dir = ensure(SAVED_STATES / drug_name / species / f"v{subject_idx}")
    apd_path = out_dir / "apd.csv"

    if apd_path.exists() and not overwrite:
        print(f"[{species}/{drug_name}/v{subject_idx}] already done, skipping")
        return pd.read_csv(apd_path, index_col=0).iloc[:, 0]

    state_path = baseline_state_path(species, subject_idx)
    if not state_path.exists():
        raise FileNotFoundError(
            f"No baseline limit state for {species} v{subject_idx} at {state_path}. "
            f"Run simulation.generate_subjects first."
        )
    with open(state_path) as f:
        baseline = json.load(f)

    drug_data = load_drug_table()
    model = build_model(species)
    concentrations = drug_concentrations(Drug(drug_name, drug_data=drug_data))
    labels = [concentration_label(c) for c in concentrations]

    apd = pd.Series(
        index=[concentration_label(0.0)] + labels, dtype=float, name=f"v{subject_idx}"
    )
    apd[concentration_label(0.0)] = baseline_apd(model, baseline)

    for conc, label in zip(concentrations, labels):
        # Every concentration restarts from the same drug-free limit state, so
        # the concentrations are independent rather than a titration sequence.
        model.set_states(baseline["states"])
        model.set_constants(baseline["constants"])
        model.init_drug(drug_name, drug_amount=conc, drug_data=drug_data)

        result = run_to_limit_cycle(
            model,
            t_0=baseline["voi"],
            stim_period=STIM_PERIOD,
            stop_on_bad_AP=True,
            verbose=verbose,
        )

        if result.solver_failed:
            print(f"[{species}/{drug_name}/v{subject_idx}] {label} nM: failed to solve")
            continue

        if not result.converged:
            print(
                f"[{species}/{drug_name}/v{subject_idx}] {label} nM: "
                f"no limit cycle after {result.num_cycles} cycles"
            )

        apd90 = result.final_features.get("APD_90")
        if apd90 is not None:
            apd[label] = apd90

        model.save_model(
            result.voi[-1],
            result.states[:, -1],
            result.algebraic[:, -1],
            model.constants,
            file_path=f"{states_dir}/",
            file_name=f"{label}_limit_state.json",
            seed=baseline["seed"],
        )

    apd.to_frame().to_csv(apd_path)
    print(f"[{species}/{drug_name}/v{subject_idx}] wrote {apd_path}")
    return apd


def parse_subject_range(spec):
    """Parse ``"0-19"``, ``"3"`` or ``"1,4,7"`` into a list of indices."""
    idxs = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            idxs.extend(range(int(lo), int(hi) + 1))
        else:
            idxs.append(int(part))
    return idxs


def main():
    drug_list = load_drug_table().index.tolist()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument("--drug", nargs="+", default=drug_list, choices=drug_list)
    parser.add_argument(
        "--subjects", default=f"0-{MAX_SUBJECT_IDX}",
        help='Subject indices, e.g. "0-19" or "1,4,7". Default: all.',
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run combinations whose apd.csv already exists.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    subject_idxs = parse_subject_range(args.subjects)

    # Under PBS, each task handles exactly one triple.
    array_index = os.environ.get("PBS_ARRAY_INDEX")
    if array_index is not None:
        species, drug, subject = unpack_array_index(
            int(array_index), args.species, args.drug, subject_idxs
        )
        print(f"PBS task {array_index}: {species} / {drug} / v{subject}")
        combos = [(species, drug, subject)]
    else:
        combos = [
            (sp, dr, sub)
            for sub in subject_idxs
            for sp in args.species
            for dr in args.drug
        ]

    for species, drug, subject in combos:
        if not baseline_state_path(species, subject).exists():
            continue  # subject was never generated for this species
        simulate_subject(
            species, drug, subject,
            overwrite=args.overwrite, verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
