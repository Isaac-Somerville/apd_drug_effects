# Cardiac APD simulation under drug block

Single-cell ventricular action potential simulations across six species, under
block by twelve drugs from the CiPA reference set. This repository produces the
simulated APD dataset and the ground-truth drug effects (maximum mean relative APD change) used in the
accompanying research article "Multi-Fidelity Gaussian Processes for Translational Modelling of Clinical Outcomes" (https://arxiv.org/abs/2609.05007). The statistical analysis of that dataset lives in the
companion repository, https://github.com/Isaac-Somerville/translational_mfgps.

## Contents

| Path | Contents |
|---|---|
| `cell_models/` | Six species models, sharing one `CardiacCellModel` interface |
| `simulation/` | Limit-cycle solving, virtual-subject generation, drug-block runs |
| `data/` | Model constants, AP-feature metadata, CiPA parameters, saved states |
| `data/simulations/apd_drug_block.csv` | The merged APD dataset — the analysis input |
| `outputs/simulation/drug_block/` | Per-subject drug-block APDs (`apd.csv`), the input to the merge; provenance in its README |
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
# Pace the unperturbed model to its reference limit state (subject 0)
python -m simulation.limit_cycle --species Dog

# Generate virtual subjects (perturbed models paced to their limit cycle)
python -m simulation.generate_subjects --species Dog --subjects 0-3

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

# 4. Ground truth: maximum mean relative APD change up to 1x and 3x therapeutic
python -m simulation.calculate_ground_truth
```

Omit `--species` / `--drug` to run the full sweep. Stage 2 is 6 species x 12
drugs x up to 42 subjects x 9 non-zero concentrations (the zero-concentration
APD is measured from the drug-free limit state), each paced for up to 25,000
beats, which necessitates the PBS wrappers in `simulation/shell/`. The shipped `data/saved_states/` hold
the converged limit-cycle states, so stage 1 need only be rerun if a model or
protocol changes.

`simulation/limit_cycle.py` is the shared pacing-to-convergence routine used by
stages 1 and 2; run as a script it produces each species' unperturbed reference
subject 0, which `generate_subjects` does not. The two `compute_*.py`
modules regenerate the AP-feature bounds and convergence tolerances that those
stages read; they are not part of the normal path. The integration steps per beat
in `data/ap_features/min_steps_needed_rounded.json` were chosen by hand for each
species (see `cell_models/base.py`).

The per-subject output of stage 2 ships in `outputs/simulation/drug_block/`
(one `apd.csv` per species x drug x subject, carried over from the published
runs), so stages 3 and 4 run without repeating stage 2. From those files stage 3
regenerates the published `data/simulations/apd_drug_block.csv`
bit-identically, including the noise column, and stage 4 regenerates both
`ground_truth_relative_*.csv` tables bit-identically.

## Verifying a reproduction

```bash
python -m verification.verify_cell_models
```

Checks the baseline AP features per species against the shipped files, and
rebuilds the whole dataset from the shipped per-subject runs to assert
byte-for-byte identity. `--checks drug-block` additionally re-simulates a sample
of (species, drug, subject, concentration) points against the dataset; it is
slow, so it is off by default.

One documented exception: `simulation/compute_ap_features.py` cannot reproduce
the shipped AP-feature bounds, because those were measured from an intermediate
per-cycle state series that was not retained — only each subject's final limit
state survives. This does not affect the dataset, which is built from the
shipped bounds and does reproduce exactly. The module docstring records it.
Both `compute_*.py` scripts write under
`outputs/ap_features/` by default, so a rerun cannot silently overwrite the
published files in `data/ap_features/`.

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