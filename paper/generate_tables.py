"""Generate the supplementary LaTeX tables from the shipped data files.

Every number is read from the JSON and CSV files this repository ships, so no
value in the paper is transcribed by hand and the tables cannot drift from the
data they describe.

Five tables, one ``.tex`` file each, written to ``outputs/tables/``:

=========================  =================================================
``tolerances.tex``         Convergence tolerances per species and feature
``baseline_features.tex``  Drug-free AP features of the unperturbed model
``feasible_ranges.tex``    Feasibility bounds per species and feature
``ic50.tex``               IC50 per drug and channel, from the CiPA table
``hill_coefficients.tex``  Hill coefficients, likewise
=========================  =================================================

Each file is a bare ``table`` environment, to be ``\\input`` into the manuscript.
No prose, section headings or formulae are emitted.

Units
-----
The guinea pig model integrates in seconds while every other model uses
milliseconds, so its stored values are in different units. They are converted
on the way into the tables — durations scaled to ms, rates to mV/ms — so a
column comparison is meaningful. Concentrations are left in each model's own
internal units, which the caption states.

Required LaTeX packages: ``booktabs``, ``siunitx``.

Example:
    python -m paper.generate_tables
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from cell_models import NUM_CYCLES_LIMIT_STATE, SPECIES
from drug import ALL_CHANNELS, load_drug_table
from paths import (AP_LIMITS_DIR, DEFAULT_AP_FEATURES_DIR,
                   LIMIT_CYCLE_TOLERANCES_DIR, MIN_STEPS_NEEDED, TABLES_OUT,
                   ensure)

#: Species that integrate in seconds rather than milliseconds.
SECONDS_SPECIES = {"Guinea Pig"}

#: Features measured as durations, so scaled by 1000 for a seconds-based model.
TIME_FEATURES = {"APD_40", "APD_50", "APD_90", "tri_90_40", "CTD_50", "CTD_90"}

AP_FEATURES = ["APD_40", "APD_50", "APD_90", "tri_90_40", "dvdt_max",
               "v_peak", "RMP", "CTD_50", "CTD_90"]

IONIC_FEATURES = ["cai_min", "cai_max", "cai2_min", "cai2_max",
                  "nai_min", "nai_max", "ki_min", "ki_max",
                  "cli_min", "cli_max"]

LABEL = {
    "APD_40": r"$\mathrm{APD}_{40}$", "APD_50": r"$\mathrm{APD}_{50}$",
    "APD_90": r"$\mathrm{APD}_{90}$", "tri_90_40": r"$\mathrm{Tri}_{90-40}$",
    "dvdt_max": r"$(\mathrm{d}V/\mathrm{d}t)_{\max}$",
    "v_peak": r"$V_{\mathrm{peak}}$", "RMP": r"$\mathrm{RMP}$",
    "CTD_50": r"$\mathrm{CTD}_{50}$", "CTD_90": r"$\mathrm{CTD}_{90}$",
    "cai_min": r"$[\mathrm{Ca}^{2+}]_{i}^{\min}$",
    "cai_max": r"$[\mathrm{Ca}^{2+}]_{i}^{\max}$",
    "cai2_min": r"$[\mathrm{Ca}^{2+}]_{i,2}^{\min}$",
    "cai2_max": r"$[\mathrm{Ca}^{2+}]_{i,2}^{\max}$",
    "nai_min": r"$[\mathrm{Na}^{+}]_{i}^{\min}$",
    "nai_max": r"$[\mathrm{Na}^{+}]_{i}^{\max}$",
    "ki_min": r"$[\mathrm{K}^{+}]_{i}^{\min}$",
    "ki_max": r"$[\mathrm{K}^{+}]_{i}^{\max}$",
    "cli_min": r"$[\mathrm{Cl}^{-}]_{i}^{\min}$",
    "cli_max": r"$[\mathrm{Cl}^{-}]_{i}^{\max}$",
}

UNIT = {
    "APD_40": r"\si{\milli\second}", "APD_50": r"\si{\milli\second}",
    "APD_90": r"\si{\milli\second}", "tri_90_40": r"\si{\milli\second}",
    "dvdt_max": r"\si{\milli\volt\per\milli\second}",
    "v_peak": r"\si{\milli\volt}", "RMP": r"\si{\milli\volt}",
    "CTD_50": r"\si{\milli\second}", "CTD_90": r"\si{\milli\second}",
}

CHANNEL_LABEL = {
    "ICaL": r"$I_{\mathrm{CaL}}$", "IK1": r"$I_{\mathrm{K1}}$",
    "IKs": r"$I_{\mathrm{Ks}}$", "INa": r"$I_{\mathrm{Na}}$",
    "INaL": r"$I_{\mathrm{NaL}}$", "Ito": r"$I_{\mathrm{to}}$",
    "hERG": r"$I_{\mathrm{Kr}}$",
}


def convert(feature, value, species, kind="value"):
    """Put a stored value into milliseconds / mV per ms.

    ``kind="value"`` scales durations from seconds to ms for a seconds-based
    model. ``kind="tolerance"`` does not: a stored duration tolerance is already
    the intended millisecond sampling step, so only the rate is rescaled.
    """
    if value is None or species not in SECONDS_SPECIES:
        return value
    if feature == "dvdt_max":
        return value / 1000.0
    if feature in TIME_FEATURES and kind == "value":
        return value * 1000.0
    return value


def sig(value, digits=3):
    return "--" if value is None else f"{value:.{digits}g}"


def load_json(directory, species, suffix):
    """Load one per-species metadata file at its own cycle count."""
    path = directory / f"{species}_{NUM_CYCLES_LIMIT_STATE[species]}_cycles_{suffix}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def table(caption, colspec, header, body_rows, label, extra=None):
    """Assemble one ``table`` environment."""
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\footnotesize",
        r"\begin{tabular}{" + colspec + "}", r"\toprule",
        header + r" \\", r"\midrule",
        *body_rows,
        r"\bottomrule", r"\end{tabular}",
    ]
    if extra:
        lines.append(extra)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def tolerances_table(species_list):
    tolerances = {s: load_json(LIMIT_CYCLE_TOLERANCES_DIR, s,
                               "limit_cycle_tolerances") for s in species_list}
    with open(MIN_STEPS_NEEDED) as f:
        min_steps = json.load(f)

    header = (r"\textbf{Feature} & \textbf{Unit} & "
              + " & ".join(r"{\textbf{" + s + "}}" for s in species_list))
    rows = [
        r"$N_{\mathrm{steps}}$ (beat$^{-1}$) & & "
        + " & ".join("{" + str(min_steps[s]) + "}" for s in species_list) + r" \\",
        r"$N_{\mathrm{ref}}$ (beats) & & "
        + " & ".join("{" + str(NUM_CYCLES_LIMIT_STATE[s]) + "}"
                     for s in species_list) + r" \\",
        r"\midrule",
    ]
    for feature in AP_FEATURES:
        cells = [
            f"{convert(feature, tolerances[s].get(feature), s, 'tolerance'):.2e}"
            if tolerances[s].get(feature) is not None else "{--}"
            for s in species_list
        ]
        rows.append(f"{LABEL[feature]} & {UNIT[feature]} & " + " & ".join(cells) + r" \\")

    rows.append(r"\midrule")
    for feature in IONIC_FEATURES:
        if not any(feature in tolerances[s] for s in species_list):
            continue
        cells = [
            f"{tolerances[s][feature]:.2e}" if feature in tolerances[s] else "{--}"
            for s in species_list
        ]
        rows.append(f"{LABEL[feature]} & & " + " & ".join(cells) + r" \\")

    return table(
        "Absolute tolerances used in the limit-cycle convergence criterion, by "
        "species and feature. $N_{\\mathrm{steps}}$ is the number of samples per "
        "beat and $N_{\\mathrm{ref}}$ the length of the reference run the "
        "tolerances were calibrated from. A dash indicates a quantity that model "
        "does not track. Concentration tolerances are in each model's own "
        "internal units.",
        "l l " + " ".join(["S"] * len(species_list)),
        header, rows, "tab:supp-tolerances",
    )


def baseline_features_table(species_list):
    features = {s: load_json(DEFAULT_AP_FEATURES_DIR, s, "default_AP_features")
                for s in species_list}
    header = (r"\textbf{Feature} & \textbf{Unit} & "
              + " & ".join(r"{\textbf{" + s + "}}" for s in species_list))
    rows = [
        f"{LABEL[f]} & {UNIT[f]} & "
        + " & ".join(sig(convert(f, features[s].get(f), s)) for s in species_list)
        + r" \\"
        for f in AP_FEATURES
    ]
    return table(
        "Drug-free baseline action potential features of the unperturbed model "
        "for each species, measured on the converged limit cycle.",
        "l l " + " ".join(["S"] * len(species_list)),
        header, rows, "tab:supp-baseline-features",
    )


def feasible_ranges_table(species_list):
    limits = {s: load_json(AP_LIMITS_DIR, s, "AP_limits") for s in species_list}
    header = (r"\textbf{Feature} & \textbf{Unit} & "
              + " & ".join(r"\textbf{" + s + "}" for s in species_list))
    rows = []
    for feature in AP_FEATURES:
        cells = []
        for species in species_list:
            bounds = limits[species].get(feature)
            if bounds is None:
                cells.append("--")
                continue
            low = convert(feature, bounds[0], species)
            high = convert(feature, bounds[1], species)
            cells.append(f"{sig(low)}--{sig(high)}")
        rows.append(f"{LABEL[feature]} & {UNIT[feature]} & " + " & ".join(cells) + r" \\")
    return table(
        "Feasible ranges for each action potential feature, by species. A "
        "simulated beat falling outside any of these is rejected as "
        "non-physiological. Ranges are derived by scaling one set of human "
        "bounds onto each species by the ratio of its baseline feature to the "
        "human value.",
        "l l " + " ".join(["c"] * len(species_list)),
        header, rows, "tab:supp-feasible-ranges",
    )


def _channel_table(column_suffix, caption, label, digits):
    drug_data = load_drug_table()
    drugs = drug_data.index.tolist()
    header = (r"\textbf{Drug} & "
              + " & ".join("{" + CHANNEL_LABEL[c] + "}" for c in ALL_CHANNELS))
    rows = []
    for drug in drugs:
        cells = []
        for channel in ALL_CHANNELS:
            value = drug_data.loc[drug, f"{channel}_{column_suffix}"]
            cells.append("{--}" if pd.isna(value) else sig(value, digits))
        rows.append(f"{drug.capitalize()} & " + " & ".join(cells) + r" \\")
    return table(caption, "l " + " ".join(["S"] * len(ALL_CHANNELS)),
                 header, rows, label)


def ic50_table():
    return _channel_table(
        "IC50",
        "$\\mathrm{IC}_{50}$ values (\\si{\\nano\\molar}) for each drug and ion "
        "channel, from the CiPA dataset. A dash indicates a channel with no "
        "reported value for that drug, which is modelled as unblocked.",
        "tab:supp-ic50", 4,
    )


def hill_table():
    return _channel_table(
        "h",
        "Hill coefficients $h$ (dimensionless) for each drug and ion channel, "
        "from the CiPA dataset.",
        "tab:supp-hill", 3,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = ensure(TABLES_OUT if args.out_dir is None else Path(args.out_dir))
    species_list = args.species

    for name, content in [
        ("tolerances", tolerances_table(species_list)),
        ("baseline_features", baseline_features_table(species_list)),
        ("feasible_ranges", feasible_ranges_table(species_list)),
        ("ic50", ic50_table()),
        ("hill_coefficients", hill_table()),
    ]:
        path = out_dir / f"{name}.tex"
        path.write_text(content)
        print(f"Wrote {path}  ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
