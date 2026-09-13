"""Compute the ground-truth tables the GP experiments are scored against.

The quantity is the **maximum mean relative APD change** for a drug in a
species, over concentrations up to a multiple of the drug's therapeutic
concentration:

    ground_truth[drug, species] = max over concentrations c <= c_end of
                                  mean over subjects of APD_90_relative

Averaging across subjects first and maximising second matters: it asks "how far
does this drug move the *typical* cell in this species", not "what is the worst
thing that happened to any cell". The maximum is taken because the drug effect
is not monotone in concentration for every drug — some curves turn over — so
the endpoint is not always the largest response.

This is what the experiments predict. A GP trained on human and animal subjects
produces a posterior over that same quantity for a held-out drug, and is scored
against this table by CRPS.

Two tables ship, at ``c_end = 1x`` and ``3x`` the therapeutic concentration,
matching the two single-drug configs. Three times therapeutic is the exposure
the CiPA proarrhythmia assessment is built around; 1x is the conservative case.

Ground truth is computed per species independently, so a species' column
depends only on that species' rows. Dropping a species from the dataset leaves
every surviving column unchanged.

Rows with a missing ``APD_90_relative`` are excluded before averaging, so
NaN-masked points neither contribute nor drag the mean.

Example:
    python -m simulation.calculate_ground_truth
"""

import argparse

import numpy as np
import pandas as pd

from cell_models import SPECIES
from drug import Drug, load_drug_table
from paths import APD_DATASET, GROUND_TRUTH_DIR, ensure

#: Concentration endpoints, as multiples of each drug's therapeutic
#: concentration. One table is written per entry.
CONC_MULTIPLIERS = [1, 3]


def ground_truth_table(df, drug_list, species_list, conc_multiplier):
    """Maximum mean relative APD change per (drug, species).

    Args:
        df: The APD dataset, with ``APD_90_relative`` already non-null.
        drug_list: Drugs to compute, one row each.
        species_list: Species to compute, one column each.
        conc_multiplier: Concentration ceiling, as a multiple of each drug's
            therapeutic concentration.

    Returns:
        DataFrame indexed by drug, with one column per species. Entries are NaN
        where a (drug, species) pair has no usable rows.
    """
    table = pd.DataFrame(index=drug_list, columns=species_list, dtype=float)

    for drug in drug_list:
        c_end = conc_multiplier * Drug(drug).c_therapeutic
        for species in species_list:
            rows = df[
                (df["Species"] == species)
                & (df["Drug"] == drug)
                & (df["Drug_Concentration"] <= c_end)
            ]
            if rows.empty:
                continue
            # Mean across subjects at each concentration, then the largest.
            per_conc = rows.groupby("Drug_Concentration")["APD_90_relative"].mean()
            table.loc[drug, species] = per_conc.max()

    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument(
        "--conc-multipliers", nargs="+", type=int, default=CONC_MULTIPLIERS
    )
    parser.add_argument(
        "--dataset", default=None,
        help="APD dataset to read. Defaults to the shipped one.",
    )
    args = parser.parse_args()

    dataset = APD_DATASET if args.dataset is None else args.dataset
    df = pd.read_csv(dataset, float_precision="round_trip")
    df = df.dropna(subset=["APD_90_relative"])
    drug_list = load_drug_table().index.tolist()

    ensure(GROUND_TRUTH_DIR)
    for multiplier in args.conc_multipliers:
        table = ground_truth_table(df, drug_list, args.species, multiplier)
        out_path = GROUND_TRUTH_DIR / f"ground_truth_relative_{multiplier}x.csv"
        table.to_csv(out_path)
        print(f"Wrote {out_path.name}  ({table.notna().sum().sum()} entries)")


if __name__ == "__main__":
    main()
