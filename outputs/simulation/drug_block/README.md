# Per-subject drug-block runs

One `apd.csv` per species x drug x subject, at `<drug>/<species>/v<n>/apd.csv`, in
the format `simulation/simulate_drug_block.py` writes: APD90 in the model's native
time unit (seconds for guinea pig, milliseconds otherwise), indexed by concentration
label in nM, with the drug-free value on the `0.00` row. These are the inputs
`simulation/build_apd_dataset.py` merges into `data/simulations/apd_drug_block.csv`,
and they regenerate that file bit-for-bit.

## Provenance

The files carry the published research runs over unchanged rather than re-simulating
them:

- **Non-zero concentrations** are the original runs' per-subject outputs, value for
  value (the dog model was named `Benson_Dog` in the research code). The row set is
  exactly the set of runs in the published merge, 2,073 files.
- **The `0.00` row** is the subject's drug-free APD, recomputed with
  `simulate_drug_block.baseline_apd()` from the shipped limit state in
  `data/saved_states/baseline/`. It reproduces the published value for every subject
  (up to the parsing defect described in `build_apd_dataset.load_runs`).

## Excluded from the dataset

`build_apd_dataset.py` drops the following explicitly, so the dataset comes out the
same whether it is built from these files or from a fresh `simulate_drug_block` sweep.

- **Subjects with no drug-free APD:** Human, Pig and Rabbit subject 31 and guinea pig
  subject 23 (`EXCLUDED_SUBJECTS_ALL_DRUGS`). With no zero-concentration reference
  their relative change is undefined, so they carried no observations. Their files
  are kept here as the record of the runs, with an empty `0.00` row. Guinea pig 23
  has no saved baseline state. The others do, and it yields a drug-free APD (Human
  218.07 ms, Pig 286.29 ms, Rabbit 369.75 ms), but the published runs had none.
- **Guinea pig x quinidine** (`EXCLUDED_PAIRS`): its concentration responses are
  not physiologically plausible. The 30 original runs are not carried over here.

Re-simulating a sample on macOS arm64 with SciPy 1.16 (one full subject per non-human
species plus one human concentration) reproduced six of the seven runs exactly. The
seventh differed in 2 of its 10 values, by about 1e-10 ms. The saved limit states are
not reproduced byte for byte: they agree to within about 5e-4 relative and often
converge on a different 100-beat block. APD is read off the integration grid, so it
survives those differences, apart from the 1e-10 case above.
