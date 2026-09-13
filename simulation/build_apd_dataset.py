"""Merge per-subject drug-block runs into the analysis dataset, and clean it.

Takes the per-subject APD tables written by ``simulate_drug_block`` and produces
the single long-format table the experiments consume,
``data/simulations/apd_drug_block.csv``, with one row per
(species, drug, subject, concentration).

Derived columns
---------------
``APD_90``
    Simulated APD90 in ms (seconds for guinea pig; see
    :mod:`cell_models`).
``APD_90_relative``
    Fractional change from that subject's own drug-free APD,
    ``(APD - APD_0) / APD_0``. Normalising within subject removes the large
    between-subject and between-species differences in absolute APD, leaving
    the drug effect — which is what transfers across species.
``APD_90_relative_noisy``
    ``APD_90_relative`` plus Gaussian observation noise, sd 0.03, drawn under
    :data:`NOISE_SEED`. Noise is forced to exactly zero at zero concentration,
    since the relative change there is zero by construction and a noisy
    baseline would corrupt every other point for that subject. This is the
    column the GP experiments treat as the observation.

    The draw is positional, so it is defined by the seed *and* the row set:
    the k-th value belongs to whatever row sat at position k. It was drawn over
    a frame that also held two further cardiac models, which are not shipped
    here because their simulated drug effects on APD differed significantly
    from the literature. A subset of a positional draw cannot be reproduced by
    redrawing over the smaller frame — removing rows shifts every later value,
    and the very first row of the original frame belonged to one of the
    excluded models, so the two sequences diverge immediately.

    The original sequence is therefore replayed at full length and each row
    takes the value at the position it originally occupied, recorded in
    ``data/simulations/published_noise_index.csv``. This reproduces the
    published observations exactly, which is what lets this repository
    regenerate the numbers in the paper rather than merely resemble them.
    See :func:`add_observation_noise`.

Cleaning
--------
The per-run checks are real but local. :func:`~simulation.limit_cycle.run_to_limit_cycle`
requires every limit-state feature to move less than its tolerance between
consecutive blocks, and drug-block runs pace with ``stop_on_bad_AP``, so an AP
that leaves its feasibility bounds aborts the run. Neither test can see a run
that is self-consistent and individually plausible at every check and still
non-physiological. Two failure modes get through:

* **Alternans.** Features are measured on the last beat of each 100-beat block,
  so a period-2 rhythm is always sampled at the same point in its cycle.
  Successive checks agree to within tolerance and the run is marked converged,
  even though APD is varying from beat to beat, and the value recorded is one of
  the two alternating APDs rather than a settled one. Both of them can sit well
  inside the feasibility bounds, so nothing aborts either.

* **Erratic concentration response.** Every concentration restarts from the same
  drug-free limit state and is solved independently, so nothing ever compares
  one concentration with the next. A curve that varies erratically, or steps
  discontinuously between neighbouring concentrations, passes every per-run
  test. The response is not required to be monotone in concentration; it is
  required not to behave like that.

A run that fails to converge is warned about but its APD is still recorded, so
all of these reach this point looking valid. They were identified by inspecting
the concentration-response curves, and the tables below are the authoritative
record of what was excluded from the published analysis:

* :data:`EXCLUDED_SUBJECTS` — the subject is unusable for that drug across the
  board; its rows are dropped.
* :data:`EXCLUDED_POINTS` — the curve is sound lower down but not at these
  concentrations. Those rows are kept with the APD columns set to NaN, which
  records that the concentration was simulated while keeping the value out of
  every fit.

Example:
    python -m simulation.build_apd_dataset
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

from cell_models import SPECIES, build_model
from drug import load_drug_table
from paths import APD_DATASET, SIMULATION_OUT
from simulation.simulate_drug_block import MAX_SUBJECT_IDX

#: Seed for the observation-noise draw, fixed so the dataset is reproducible.
NOISE_SEED = 43

#: Standard deviation of the observation noise on the relative APD change.
NOISE_SD = 0.03

#: Length of the original noise sequence. The published draw ran over a frame
#: holding two further cardiac models, so replaying it needs that frame's row
#: count even though this release does not ship the rows themselves.
PUBLISHED_DRAW_LENGTH = 27349

#: Each retained row's position in that original sequence.
PUBLISHED_NOISE_INDEX = APD_DATASET.parent / "published_noise_index.csv"

# ---------------------------------------------------------------------------
# Manually identified non-physiological points (see module docstring)
# ---------------------------------------------------------------------------

#: Subjects whose response to a drug is unusable at every concentration, keyed
#: by (species, drug). Every row for these subject/drug pairs is dropped.
EXCLUDED_SUBJECTS = {
    ("Dog", "bepridil"): [18],
    ("Dog", "chlorpromazine"): [12],
    ("Dog", "cisapride"): [1, 12],
    ("Dog", "dofetilide"): [1, 12, 13, 17, 18, 21, 26, 31, 35],
    ("Dog", "mexiletine"): [1, 12, 13, 18, 31, 35],
    ("Dog", "ondansetron"): [12, 18, 35],
    ("Dog", "quinidine"): [1, 12],
    ("Dog", "ranolazine"): [1, 12, 13, 18, 31, 35],
    ("Dog", "sotalol"): [1, 12, 13, 31],
    ("Dog", "terfenadine"): [1, 12],
    ("Guinea Pig", "diltiazem"): [9],
    ("Guinea Pig", "quinidine"): [2, 12],
    ("Human", "cisapride"): [2, 15, 30],
    ("Human", "diltiazem"): [6],
    ("Human", "dofetilide"): [2, 15, 22],
    ("Human", "mexiletine"): [2, 7, 9, 13, 24, 34],
    ("Human", "ranolazine"): [13, 24],
    ("Human", "sotalol"): [2, 22],
    ("Mouse", "cisapride"): [31],
    ("Mouse", "quinidine"): [2, 25],
    ("Rabbit", "quinidine"): [31],
}

#: Individual concentrations to exclude, keyed by (species, drug, subject). The
#: rows are kept and their APD columns set to NaN, so the dataset still records
#: that the concentration was simulated.
#:
#: Every exclusion in this release is a NaN rather than a missing row, so the
#: row set is a property of which simulations were run, never of which results
#: were judged usable.
EXCLUDED_POINTS = {
    ("Guinea Pig", "bepridil", 2): [66.0, 77.0, 88.0, 99.0],
    ("Guinea Pig", "bepridil", 15): [77.0, 99.0],
    ("Guinea Pig", "bepridil", 25): [99.0],
    ("Guinea Pig", "chlorpromazine", 29): [38.0, 50.67, 63.33, 76.0, 88.67, 101.33],
    ("Guinea Pig", "ondansetron", 29): [46.33, 324.33],
    ("Guinea Pig", "quinidine", 5): [6474.0, 7553.0, 8632.0, 9711.0],
    ("Guinea Pig", "quinidine", 8): [6474.0],
    ("Guinea Pig", "quinidine", 9): [4316.0, 5395.0, 6474.0, 7553.0, 8632.0, 9711.0],
    ("Guinea Pig", "quinidine", 16): [8632.0, 9711.0],
    ("Guinea Pig", "quinidine", 18): [4316.0, 5395.0, 6474.0, 8632.0, 9711.0],
    ("Guinea Pig", "quinidine", 20): [7553.0, 8632.0, 9711.0],
    ("Guinea Pig", "quinidine", 21): [3237.0, 4316.0],
    ("Guinea Pig", "ranolazine", 15): [3896.4],
    ("Guinea Pig", "terfenadine", 15): [9.33],
    ("Guinea Pig", "verapamil", 9): [189.0, 216.0, 243.0],
    ("Human", "bepridil", 1): [44.0, 55.0, 66.0, 77.0, 88.0, 99.0],
    ("Human", "diltiazem", 16): [244.0, 284.67, 325.33, 366.0],
    ("Human", "dofetilide", 1): [2.67, 3.33, 4.0],
    ("Human", "mexiletine", 1): [
        2752.67, 4129.0, 5505.33, 6881.67, 8258.0, 9634.33, 11010.67, 12387.0,
    ],
    ("Human", "quinidine", 15): [7553.0, 8632.0, 9711.0],
    ("Human", "quinidine", 18): [4316.0],
    ("Human", "ranolazine", 1): [2597.6, 3247.0, 3896.4, 4545.8, 5195.2, 5844.6],
    ("Human", "ranolazine", 15): [4545.8, 5195.2, 5844.6],
    ("Human", "sotalol", 15): [19586.67, 24483.33, 34276.67, 39173.33, 44070.0],
    ("Mouse", "chlorpromazine", 13): [50.67],
    ("Mouse", "diltiazem", 16): [40.67],
    ("Mouse", "ondansetron", 13): [46.33, 92.67],
    ("Mouse", "quinidine", 6): [6474.0, 7553.0, 8632.0, 9711.0],
    ("Mouse", "quinidine", 16): [5395.0, 6474.0],
    ("Mouse", "quinidine", 28): [8632.0, 9711.0],
    ("Mouse", "ranolazine", 2): [5844.6],
    ("Mouse", "sotalol", 13): [34276.67],
    ("Mouse", "terfenadine", 13): [5.33],
    ("Pig", "chlorpromazine", 7): [114.0],
    ("Rabbit", "diltiazem", 7): [162.67, 203.33, 244.0, 284.67, 325.33, 366.0],
    ("Rabbit", "diltiazem", 14): [203.33, 244.0, 284.67, 325.33, 366.0],
    ("Rabbit", "quinidine", 7): [9711.0],
    ("Rabbit", "quinidine", 8): [5395.0, 6474.0],
    ("Rabbit", "quinidine", 9): [6474.0, 7553.0],
    ("Rabbit", "quinidine", 10): [7553.0, 8632.0, 9711.0],
    ("Rabbit", "quinidine", 11): [7553.0, 8632.0, 9711.0],
    ("Rabbit", "quinidine", 21): [5395.0, 6474.0],
    ("Rabbit", "quinidine", 24): [8632.0, 9711.0],
    ("Rabbit", "quinidine", 26): [7553.0, 8632.0, 9711.0],
    ("Rabbit", "verapamil", 2): [216.0, 243.0],
    ("Rabbit", "verapamil", 7): [243.0],
    ("Rabbit", "verapamil", 13): [108.0, 135.0, 162.0, 189.0, 216.0, 243.0],
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def load_runs(species_list, drug_list, max_subject_idx=MAX_SUBJECT_IDX):
    """Collect every per-subject ``apd.csv`` into one long-format frame.

    APD is converted to milliseconds here. The guinea pig model integrates in
    seconds, so its values are scaled by 1000; every other species is already
    in ms. Doing it at merge time means the per-subject files always hold the
    model's native units, while the dataset is uniformly in ms.
    """
    rows = []
    for species in species_list:
        # The conversion factor is a property of the model, not the subject, so
        # one instance per species suffices.
        model = build_model(species)
        conversion = model.time_unit_conversion[model.time_unit]
        for drug in drug_list:
            for subject_idx in range(max_subject_idx + 1):
                path = (
                    SIMULATION_OUT / "drug_block" / drug / species
                    / f"v{subject_idx}" / "apd.csv"
                )
                if not path.exists():
                    continue
                # round_trip: pandas' default float parser is accurate only to
                # within one ulp, which would make the rebuilt dataset differ
                # from the simulation output in the last bit of every value.
                apd = pd.read_csv(path, index_col=0, float_precision="round_trip")
                if apd.isna().all().all():
                    continue
                frame = apd.reset_index()
                frame.columns = ["Drug_Concentration", "APD_90"]
                frame["Species"] = species
                frame["Drug"] = drug
                frame["Subject_ID"] = subject_idx
                frame["Drug_Concentration"] = frame["Drug_Concentration"].astype(float)
                frame["APD_90"] = (
                    pd.to_numeric(frame["APD_90"], errors="coerce") * conversion
                )
                rows.append(frame)

    if not rows:
        raise FileNotFoundError(
            f"No per-subject apd.csv files under {SIMULATION_OUT / 'drug_block'}. "
            f"Run simulation.simulate_drug_block first."
        )

    return pd.concat(rows, ignore_index=True)[
        ["Species", "Drug", "Subject_ID", "Drug_Concentration", "APD_90"]
    ]


def add_relative_column(df):
    """Add ``APD_90_relative``: fractional change from each subject's own zero dose.

    Subjects whose drug-free APD is missing or zero yield NaN throughout, since
    there is nothing to normalise against.
    """
    df = df.sort_values(
        ["Species", "Drug", "Subject_ID", "Drug_Concentration"]
    ).reset_index(drop=True)

    # The baseline is specifically the zero-concentration APD, not merely the
    # first available one. If a subject's drug-free simulation failed, that
    # subject has no reference point and every one of its relative values is
    # undefined — taking the next concentration instead would silently rebase
    # the whole curve on a drugged measurement.
    group_keys = ["Species", "Drug", "Subject_ID"]
    zero_dose = (
        df[df["Drug_Concentration"] == 0.0]
        .set_index(group_keys)["APD_90"]
        .rename("baseline")
    )
    baseline = df.set_index(group_keys).index.map(zero_dose)
    baseline = pd.Series(baseline, index=df.index, dtype=float)
    baseline = baseline.where(baseline != 0.0)

    df["APD_90_relative"] = _replay_csv_roundtrip((df["APD_90"] - baseline) / baseline)
    return df


def _replay_csv_roundtrip(values):
    """Round-trip a column through a CSV read with pandas' *default* parser.

    This deliberately reproduces a defect, and removing it would be the bug.

    The research pipeline computed this column, wrote it to an intermediate CSV,
    and read it back with pandas' default float parser — which is accurate only
    to within one ulp. So the published ``APD_90_relative`` sits one ulp below
    the exact expression on about 78% of its rows, and the observation column
    (``APD_90_relative_noisy``) was drawn on top of *that*, not on top of the
    exact value.

    The artefact is therefore part of the data every published number was
    computed from. Recomputing the column cleanly shifts every observation by
    that ulp, which the optimiser amplifies: the research code records it moving
    CRPS by about 1.6% of its own scale — enough that the experiments stop
    reproducing. Verified here: replaying the round-trip reproduces the
    published column on all 19,669 non-missing rows, bit for bit, where the
    clean expression matches only 4,320 of them.
    """
    buffer = io.StringIO()
    pd.Series(np.asarray(values, dtype=float)).to_csv(buffer, index=False)
    buffer.seek(0)
    restored = pd.read_csv(buffer).iloc[:, 0].to_numpy()
    return pd.Series(restored, index=getattr(values, "index", None))


def apply_cleaning(df):
    """Drop and mask the manually identified non-physiological points."""
    apd_cols = [c for c in df.columns if c.startswith("APD_90")]

    def key_mask(species, drug, subject_id):
        return (
            (df["Species"] == species)
            & (df["Drug"] == drug)
            & (df["Subject_ID"] == subject_id)
        )

    removed_rows = 0
    for (species, drug), subject_ids in EXCLUDED_SUBJECTS.items():
        mask = (
            (df["Species"] == species)
            & (df["Drug"] == drug)
            & (df["Subject_ID"].isin(subject_ids))
        )
        removed_rows += int(mask.sum())
        df = df[~mask]

    masked_rows = 0
    for (species, drug, subject_id), concs in EXCLUDED_POINTS.items():
        mask = key_mask(species, drug, subject_id) & df["Drug_Concentration"].isin(concs)
        masked_rows += int(mask.sum())
        df.loc[mask, apd_cols] = np.nan

    print(f"Cleaning: removed {removed_rows} rows, masked {masked_rows} rows")
    return df.reset_index(drop=True)


KEY_COLUMNS = ["Species", "Drug", "Subject_ID", "Drug_Concentration"]


def add_observation_noise(df, seed=NOISE_SEED, sd=NOISE_SD,
                          index_path=None):
    """Add ``APD_90_relative_noisy``, the column the experiments observe.

    The published noise was drawn as **one positional sequence** over a frame
    that also held two cardiac models this release does not ship. A positional
    draw cannot be subsetted: dropping rows shifts every later value, and the
    first row of that frame belonged to an excluded model, so a fresh draw over
    these six species diverges from the published one immediately.

    The sequence is therefore *replayed* at full length, and each row takes the
    value at the position it originally occupied.
    ``data/simulations/published_noise_index.csv`` records those positions,
    since this release does not hold the frame they refer to. The result is the
    published observations themselves, regenerated from a seed rather than
    copied across.

    Noise is zero at zero concentration: the relative change is exactly zero
    there by definition, and perturbing it would shift that subject's whole
    curve rather than adding measurement error to one point.

    A row with no recorded position was excluded before the original draw
    happened, so it has no published value; its relative change is NaN anyway.
    """
    if index_path is None:
        index_path = PUBLISHED_NOISE_INDEX

    positions = pd.read_csv(index_path, float_precision="round_trip")
    rng = np.random.RandomState(seed)
    full_draw = rng.normal(loc=0.0, scale=sd, size=PUBLISHED_DRAW_LENGTH)

    merged = df.merge(positions, on=KEY_COLUMNS, how="left", validate="one_to_one")
    located = merged["draw_index"].notna()

    noise = np.full(len(merged), np.nan)
    noise[located.values] = full_draw[
        merged.loc[located, "draw_index"].astype(int).values
    ]
    noise[merged["Drug_Concentration"].values == 0.0] = 0.0

    merged["APD_90_relative_noisy"] = merged["APD_90_relative"] + noise

    # Restore the original row order as well as the original values. The index
    # carries it for free -- positions ascend in the order of the frame the draw
    # ran over -- and it matters: the GP fits are assembled in frame order, and
    # summing identical values in a different order changes the last bits, which
    # the optimiser amplifies. Building this file sorted rather than in the
    # original order moved CRPS by ~1.4e-4, roughly 1.4% of its own scale, on
    # data that was otherwise bit-for-bit identical. Rows with no position were
    # not in that frame at all, so they go last.
    merged = merged.sort_values("draw_index", na_position="last", kind="stable")
    return merged.drop(columns=["draw_index"]).reset_index(drop=True)


def build(species_list=None, drug_list=None, out_path=APD_DATASET):
    """Run the whole merge-clean-derive pipeline and write the dataset.

    Every column is computed here rather than copied from a previous build of
    the dataset. The one external input is
    ``data/simulations/published_noise_index.csv``, which supplies row
    positions, not values: the observations themselves are still generated from
    :data:`NOISE_SEED`. See :func:`add_observation_noise` for why those
    positions cannot be derived here.

    Args:
        species_list: Species to include. Defaults to all six.
        drug_list: Drugs to include. Defaults to all twelve.
        out_path: Where to write the dataset.
    """
    species_list = species_list or SPECIES
    drug_list = drug_list or load_drug_table().index.tolist()

    df = load_runs(species_list, drug_list)
    print(f"Merged {len(df)} rows from per-subject runs")
    df = add_relative_column(df)
    df = apply_cleaning(df)
    df = add_observation_noise(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument("--out", default=None, help="Override the output path.")
    args = parser.parse_args()

    build(
        species_list=args.species,
        out_path=APD_DATASET if args.out is None else Path(args.out),
    )


if __name__ == "__main__":
    main()
