# Cardiac APD simulation under drug block

Single-cell ventricular action potential simulations across six species, under
block by twelve drugs from the CiPA reference set. This repository produces the
simulated APD dataset and the ground-truth exceedance probabilities used in the
accompanying research article "Multi-Fidelity Gaussian Processes for Translational Modelling of Clinical Outcomes" (https://arxiv.org/abs/2609.05007). The statistical analysis of that dataset lives in the
companion repository, https://github.com/Isaac-Somerville/translational_mfgps.

## Contents

| Path | Contents |
|---|---|
| `cell_models/` | Six species models, sharing one `CardiacCellModel` interface |
| `simulation/` | Limit-cycle solving, virtual-subject generation, drug-block runs |
| `data/` | Model constants, AP-feature metadata, CiPA parameters, saved states |
| `data/simulations/apd_drug_block.csv` | The merged APD dataset — the analysis input |
| `data/simulations/published_noise_index.csv` | Observation-noise draw positions; required to rebuild the dataset (see below) |
| `paths.py` | Canonical filesystem locations, so scripts run from any directory |

Species: Human, Dog, Guinea Pig, Mouse, Pig, Rabbit. See
`cell_models/__init__.py` for the source model behind each, its time unit, and
its limit-cycle length.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Solve to the limit cycle for one species
python -m simulation.limit_cycle --species Dog

# Simulate drug block
python -m simulation.simulate_drug_block --species Dog --drug quinidine

# Merge per-subject runs into the analysis dataset
python -m simulation.build_apd_dataset
```

Outputs are written under `outputs/`. The merged dataset lands at
`data/simulations/apd_drug_block.csv`, which is the input the companion analysis
repository consumes.

## Reproducing the dataset

In dependency order. Stages 1-3 are the expensive ones and are written to run
either standalone or as a PBS array job (`simulation/shell/`); stage 4 is
seconds.

```bash
# 1. Generate the virtual subject population: perturb each model's maximal
#    conductances, pace each subject to its limit cycle, and keep the ones whose
#    action potential is physiologically feasible.   (hours per species)
python -m simulation.generate_subjects --species Human

# 2. Simulate drug block for every subject and concentration (the bulk of the cost)
python -m simulation.simulate_drug_block --species Human --drug quinidine

# 3. Merge the per-subject runs, clean, and add observation noise
python -m simulation.build_apd_dataset

# 4. Ground-truth exceedance curves at 1x and 3x therapeutic
python -m simulation.calculate_ground_truth
```

Omit `--species` / `--drug` to run the full sweep. Stage 2 is 6 species x 12
drugs x 42 subjects x 10 concentrations, which is why it is the stage the PBS
wrappers in `simulation/shell/` exist for. The shipped `data/saved_states/` hold
the converged limit-cycle states, so stage 1 need only be rerun if a model or
protocol changes.

`simulation/limit_cycle.py` is the shared pacing-to-convergence routine used by
stages 1 and 2 rather than a script in its own right. The three `compute_*.py`
modules regenerate the AP-feature bounds, convergence tolerances and step counts
that those stages read; they are not part of the normal path.

The published dataset is committed at `data/simulations/apd_drug_block.csv`;
stage 4 regenerates it **bit-identically**, including the noise column.

## Verifying a reproduction

```bash
python -m verification.verify_cell_models
```

Checks the baseline AP features per species against the shipped bounds, re-runs
drug block for a sample of (species, drug, subject, concentration) against the
dataset, and rebuilds the whole dataset to assert row-for-row identity.

One documented exception: `simulation/compute_ap_features.py` cannot reproduce
the shipped AP-feature bounds, because those were measured from an intermediate
per-cycle state series that was not retained — only each subject's final limit
state survives. This does not affect the dataset, which is built from the
shipped bounds and does reproduce exactly. The module docstring records it, and
the script writes to `--out-dir` so a rerun cannot silently overwrite the
published bounds.

## Citation

If you use this code or the datasets it ships, please cite:

> Hayden, I. S., D'Souza, A., Niederer, S. and Filippi, S. (2026).
> *Multi-Fidelity Gaussian Processes for Translational Modelling of Clinical
> Outcomes.* arXiv:2609.05007. <https://arxiv.org/abs/2609.05007>

```bibtex
@misc{hayden2026multifidelity,
  title         = {Multi-Fidelity Gaussian Processes for Translational
                   Modelling of Clinical Outcomes},
  author        = {Hayden, Isaac S. and D'Souza, Alicia and
                   Niederer, Steven and Filippi, Sarah},
  year          = {2026},
  eprint        = {2609.05007},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.05007}
}
```

## Licence

**GNU General Public License v3.0 or later** — see `LICENSE`.

The six cardiac cell models are third-party published works. Five are available
under Creative Commons Attribution terms; the human ToR-ORd model derives from
GPL-3.0 source.

Full citations, source URLs and licence terms for every third-party component —
including the CiPA ion-channel data — are in **`THIRD_PARTY_NOTICES.md`**. If you
use this code, please cite the underlying model papers listed there as well as
this repository.