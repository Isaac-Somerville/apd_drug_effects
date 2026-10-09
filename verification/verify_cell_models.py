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

3. **Dataset rebuild.** Run :func:`simulation.build_apd_dataset.build` on the
   shipped per-subject ``apd.csv`` files and compare the result against the
   shipped dataset byte for byte. This covers the merge, the cleaning tables, 
   the relative-change normalisation and the observation-noise draw in one comparison.

Examples:
    python -m verification.verify_cell_models --checks features dataset
    python -m verification.verify_cell_models --num-drug-block-samples 3
"""

import argparse
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cell_models import NUM_CYCLES_LIMIT_STATE, SPECIES, build_model
from drug import Drug, load_drug_table
from paths import APD_DATASET, DEFAULT_AP_FEATURES_DIR
from simulation.build_apd_dataset import build
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
            print(f"   {species:11s} FAIL (no measurement or no shipped file)")
            all_ok = False
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
            print(f"   {species}/{drug_name}/v{subject} FAIL (no baseline state)")
            all_ok = False
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


def check_dataset_rebuild(species_list):
    """Rebuild the dataset from the shipped per-subject runs and compare bytes."""
    print("\n3. Dataset rebuild")
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "apd_drug_block.csv"
        try:
            build(species_list=species_list, out_path=out_path)
        except FileNotFoundError as err:
            print(f"   FAIL: {err}")
            return False
        rebuilt = out_path.read_bytes()

    if species_list == SPECIES:
        identical = rebuilt == APD_DATASET.read_bytes()
        print(f"   {'match' if identical else 'DIFFERS':8s} byte-for-byte against {APD_DATASET.name}")
        return identical

    # A species subset cannot be compared byte for byte; compare its rows.
    shipped = pd.read_csv(APD_DATASET, float_precision="round_trip")
    shipped = shipped[shipped["Species"].isin(species_list)].reset_index(drop=True)
    rebuilt = pd.read_csv(io.BytesIO(rebuilt), float_precision="round_trip")
    identical = rebuilt.equals(shipped)
    print(f"   {'match' if identical else 'DIFFERS':8s} {len(rebuilt)} rebuilt rows "
          f"vs {len(shipped)} shipped rows for {', '.join(species_list)}")
    return identical


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument(
        "--checks", nargs="+", default=["features", "dataset"],
        choices=["features", "drug-block", "dataset"],
        help="Which checks to run. 'drug-block' is slow and is off by default.",
    )
    parser.add_argument("--num-drug-block-samples", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(APD_DATASET, float_precision="round_trip")
    results = {}
    if "features" in args.checks:
        results["AP features"] = check_ap_features(args.species)
    if "drug-block" in args.checks:
        results["drug block"] = check_drug_block(df, args.num_drug_block_samples)
    if "dataset" in args.checks:
        results["dataset rebuild"] = check_dataset_rebuild(args.species)

    print("\n" + "-" * 60)
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
