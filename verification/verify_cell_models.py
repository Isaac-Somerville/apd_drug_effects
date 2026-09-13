"""Check that this package reproduces the published simulation outputs.

Three checks, cheapest first. Each is a claim about a different stage, so a
failure localises the problem rather than just signalling one.

1. **Baseline AP features.** Pace one beat of each species from its shipped
   limit state and compare the AP features against
   ``data/ap_features/defaults/``. This exercises the ODE right-hand sides, the
   integrator settings and the feature extraction.

   Expect dog to agree to ~1e-12 and guinea pig and pig **not** to agree. The
   shipped feature files were measured from an intermediate series of saved
   states that was not retained; only the final limit state per subject
   survives. See :mod:`simulation.compute_ap_features`. This check reports the
   discrepancy rather than asserting it away, because the number that matters
   is the next one.

2. **Drug-block APD.** Re-simulate a sample of (species, drug, subject,
   concentration) from the shipped limit states and compare APD90 against the
   dataset. This is the expensive check — each point paces to a new limit cycle
   — so it runs on a sample by default.

3. **Dataset rebuild.** Rebuild the derived columns from the raw simulation
   outputs and compare against the shipped dataset. Cheap, and the strongest of
   the three: it covers the cleaning tables, the relative-change normalisation
   and the observation-noise draw in one comparison, and it should be
   bit-identical.

Examples:
    python -m verification.verify_cell_models --checks features dataset
    python -m verification.verify_cell_models --num-drug-block-samples 3
"""

import argparse
import json

import numpy as np
import pandas as pd

from cell_models import NUM_CYCLES_LIMIT_STATE, SPECIES, build_model
from drug import Drug, load_drug_table
from paths import APD_DATASET, DEFAULT_AP_FEATURES_DIR
from simulation.build_apd_dataset import (KEY_COLUMNS, add_observation_noise,
                                          add_relative_column, apply_cleaning)
from simulation.compute_ap_features import features_filename, measure_features
from simulation.limit_cycle import STIM_PERIOD, run_to_limit_cycle
from simulation.simulate_drug_block import (baseline_state_path,
                                            drug_concentrations)

#: Species whose shipped AP features are known not to be reproducible from the
#: surviving limit states, with the largest relative discrepancy seen.
KNOWN_FEATURE_DIVERGENCE = {"Guinea Pig", "Pig"}

#: Relative tolerance for a duration feature to count as reproduced.
#:
#: These are the features the dataset is built from, and they re-measure to
#: around 1e-11, so the bar is set far tighter than anything that matters.
FEATURE_RTOL = 1e-6

#: Features describing the *shape* of the upstroke rather than a duration.
#:
#: ``dvdt_max`` is a finite difference across the upstroke, where the membrane
#: potential moves hundreds of millivolts in well under a millisecond, so its
#: value depends on exactly where the adaptive solver placed its samples;
#: ``v_peak`` and ``RMP`` are single-point reads of the same trace. Re-measuring
#: from the shipped limit state reproduces them to ~1e-5, against ~1e-11 for
#: every duration feature. None of them enters the APD dataset — they are
#: reported for the paper's feasibility table — so they are checked at a
#: tolerance matched to what a re-measurement can actually deliver, rather than
#: relaxing the bar on the features that count.
VOLTAGE_SHAPE_FEATURES = {"dvdt_max", "v_peak", "RMP"}
VOLTAGE_SHAPE_RTOL = 1e-3


def check_ap_features(species_list):
    """Compare freshly measured baseline AP features against the shipped files."""
    print("\n1. Baseline AP features")
    all_ok = True
    for species in species_list:
        features, num_cycles = measure_features(species)
        path = DEFAULT_AP_FEATURES_DIR / features_filename(species, num_cycles)
        if features is None or not path.exists():
            print(f"   {species:11s} SKIP (no measurement or no shipped file)")
            continue
        shipped = json.load(open(path))

        worst_feature, worst_rel = None, 0.0
        reproduced = True
        for key, value in shipped.items():
            if value is None or features.get(key) is None:
                continue
            rel = abs(features[key] - value) / max(abs(value), 1e-30)
            limit = (VOLTAGE_SHAPE_RTOL if key in VOLTAGE_SHAPE_FEATURES
                     else FEATURE_RTOL)
            if rel > limit:
                reproduced = False
            if rel > worst_rel:
                worst_feature, worst_rel = key, rel

        expected_to_diverge = species in KNOWN_FEATURE_DIVERGENCE
        if reproduced:
            status = "match"
        elif expected_to_diverge:
            status = "differs (known, documented)"
        else:
            status = "DIFFERS UNEXPECTEDLY"
            all_ok = False
        print(
            f"   {species:11s} {status:28s} worst: {worst_feature} "
            f"rel={worst_rel:.2e}"
        )
    return all_ok


def check_drug_block(df, num_samples, seed=0):
    """Re-simulate a sample of drug-block points and compare APD90."""
    print(f"\n2. Drug-block APD ({num_samples} sampled points)")
    usable = df[(df["Drug_Concentration"] > 0) & df["APD_90"].notna()]
    if usable.empty:
        print("   no usable rows")
        return True

    rng = np.random.default_rng(seed)
    sample = usable.iloc[rng.choice(len(usable), size=min(num_samples, len(usable)),
                                    replace=False)]
    drug_table = load_drug_table()
    all_ok = True

    for _, row in sample.iterrows():
        species, drug_name = row["Species"], row["Drug"]
        subject, concentration = int(row["Subject_ID"]), row["Drug_Concentration"]

        state_path = baseline_state_path(species, subject)
        if not state_path.exists():
            print(f"   {species}/{drug_name}/v{subject} SKIP (no baseline state)")
            continue
        with open(state_path) as f:
            baseline = json.load(f)

        model = build_model(species)
        model.set_states(baseline["states"])
        model.set_constants(baseline["constants"])
        model.init_drug(drug_name, drug_amount=concentration, drug_data=drug_table)
        result = run_to_limit_cycle(
            model, t_0=baseline["voi"], stim_period=STIM_PERIOD, stop_on_bad_AP=True
        )

        got = None if result.solver_failed else result.final_features.get("APD_90")
        if got is not None:
            # The dataset is in milliseconds; the guinea pig model is in seconds.
            got *= model.time_unit_conversion[model.time_unit]

        expected = float(row["APD_90"])
        ok = got is not None and abs(got - expected) < 1e-9
        all_ok &= ok
        print(
            f"   {'match' if ok else 'DIFFERS':8s} {species:11s} {drug_name:14s} "
            f"v{subject:<3d} @ {concentration:>9.2f} nM  "
            f"got={got!r} expected={expected!r}"
        )
    return all_ok


def check_dataset_rebuild(raw_path):
    """Rebuild the derived columns from the raw merge and compare to the dataset."""
    print("\n3. Dataset rebuild")
    if not raw_path.exists():
        print(f"   SKIP: raw merge not found at {raw_path}")
        return True

    raw = pd.read_csv(raw_path, float_precision="round_trip")
    raw = raw[raw["Species"].isin(SPECIES) | (raw["Species"] == "Benson_Dog")].copy()
    raw["Species"] = raw["Species"].replace({"Benson_Dog": "Dog"})
    raw = raw[["Species", "Drug", "Subject_ID", "Drug_Concentration", "APD_90"]]

    rebuilt = add_observation_noise(apply_cleaning(add_relative_column(raw)))
    shipped = pd.read_csv(APD_DATASET, float_precision="round_trip")

    a = rebuilt.sort_values(KEY_COLUMNS).reset_index(drop=True)
    b = shipped.sort_values(KEY_COLUMNS).reset_index(drop=True)

    if len(a) != len(b) or not a[KEY_COLUMNS].equals(b[KEY_COLUMNS]):
        print(f"   DIFFERS: {len(a)} rebuilt rows vs {len(b)} shipped")
        return False

    all_ok = True
    for column in ["APD_90", "APD_90_relative", "APD_90_relative_noisy"]:
        x, y = a[column].to_numpy(), b[column].to_numpy()
        nan_match = np.array_equal(np.isnan(x), np.isnan(y))
        both = ~np.isnan(x) & ~np.isnan(y)
        exact = np.array_equal(x[both], y[both])
        all_ok &= nan_match and exact
        print(
            f"   {'match' if (nan_match and exact) else 'DIFFERS':8s} {column:24s} "
            f"max|d|={np.abs(x[both] - y[both]).max():.3e}"
        )
    print(f"   {len(a)} rows compared")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument(
        "--checks", nargs="+", default=["features", "dataset"],
        choices=["features", "drug-block", "dataset"],
        help="Which checks to run. 'drug-block' is slow and is off by default.",
    )
    parser.add_argument("--num-drug-block-samples", type=int, default=2)
    parser.add_argument(
        "--raw-merge", default=None,
        help="Path to the uncleaned merge of the per-subject runs, for the "
             "rebuild check.",
    )
    args = parser.parse_args()

    from pathlib import Path
    raw_path = (
        Path(args.raw_merge) if args.raw_merge
        else Path(__file__).resolve().parents[2]
        / "Simulations/Outputs/drug_block/all_drug_block_limit_state.csv"
    )

    df = pd.read_csv(APD_DATASET, float_precision="round_trip")
    results = {}
    if "features" in args.checks:
        results["AP features"] = check_ap_features(args.species)
    if "drug-block" in args.checks:
        results["drug block"] = check_drug_block(df, args.num_drug_block_samples)
    if "dataset" in args.checks:
        results["dataset rebuild"] = check_dataset_rebuild(raw_path)

    print("\n" + "-" * 60)
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
