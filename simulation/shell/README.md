# Cluster job scripts

PBS array-job wrappers for the two expensive simulation stages. Both Python
modules also run standalone — the array support activates only when
`PBS_ARRAY_INDEX` is set — so these are a convenience for cluster use, not a
requirement.

| Script | Stage | Array size |
|---|---|---|
| `run_generate_subjects.sh` | Sample and converge virtual subjects | one task per (species, subject) |
| `run_simulate_drug_block.sh` | APD at 9 concentrations per subject | one task per (species, drug, subject) |

Set `PUBLIC_REPO` and `VENV` at the top of each script, or export them before
submitting. Array ranges below cover the full published grid: 6 species x 42
subjects, and that times 12 drugs. Narrow them with `--species` / `--drug` /
`--subjects` if you only need part of it.

    qsub simulation/shell/run_generate_subjects.sh
    qsub simulation/shell/run_simulate_drug_block.sh

Walltimes are sized for the slowest species. Human needs 20,000 beats to reach
its limit cycle and the guinea pig model integrates with a much smaller step,
so both take far longer per subject than dog or rabbit.
