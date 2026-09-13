"""Canonical filesystem locations for the package.

Every path is derived from the location of this file, so scripts can be run from
any working directory:

    python -m simulation.simulate_drug_block --species Dog --drug quinidine

Directory map
-------------
``data/``    inputs that ship with the repository (read-only in normal use)
``outputs/`` everything the scripts write: PDFs, CSVs, JSON

The one exception is ``SAVED_STATES``, which is both an input (limit-cycle
states used to start drug-block simulations) and an output (where newly
computed states are written).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Hill-equation parameters (IC50, h), therapeutic and maximum concentrations,
# and CiPA torsade risk category, one row per drug.
CIPA_DIR = DATA_ROOT / "cipa"
CIPA_TABLE = CIPA_DIR / "CiPA_optimal_combined.csv"

# Which of the seven ion channels each species model represents.
CHANNELS_TABLE = DATA_ROOT / "channels" / "channels_per_model.csv"

# Per-species model metadata, keyed by species name.
CONSTANT_IDX_DIR = DATA_ROOT / "model_constants" / "idx"
CONSTANT_BASE_VALUE_DIR = DATA_ROOT / "model_constants" / "base_values"

# Coefficients of variation used when sampling virtual subjects.
PARAMETER_CVS = DATA_ROOT / "ap_features" / "parameter_CVs.json"
# Integration steps per cycle needed for a converged APD, per species.
MIN_STEPS_NEEDED = DATA_ROOT / "ap_features" / "min_steps_needed_rounded.json"
# Physiological feasibility bounds on AP features, per species and cycle count.
AP_LIMITS_DIR = DATA_ROOT / "ap_features" / "limits"
# AP features of the unperturbed (baseline) model, per species and cycle count.
DEFAULT_AP_FEATURES_DIR = DATA_ROOT / "ap_features" / "defaults"
# Convergence tolerances defining "the model has reached its limit cycle".
LIMIT_CYCLE_TOLERANCES_DIR = DATA_ROOT / "ap_features" / "tolerances"

# Simulated APD dataset and the ground truth derived from it. These are the
# outputs this repository exists to produce, and the inputs the companion
# analysis repository consumes.
APD_DATASET = DATA_ROOT / "simulations" / "apd_drug_block.csv"
GROUND_TRUTH_DIR = DATA_ROOT / "simulations"

# ---------------------------------------------------------------------------
# Inputs that are also outputs
# ---------------------------------------------------------------------------

# Limit-cycle states. ``baseline/<species>/`` holds the drug-free states that
# seed every drug-block run; ``<drug>/<species>/`` holds the states reached
# under drug.
SAVED_STATES = DATA_ROOT / "saved_states"
BASELINE_STATES = SAVED_STATES / "baseline"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

SIMULATION_OUT = OUTPUT_ROOT / "simulation"

#: LaTeX tables for the manuscript supplement.
TABLES_OUT = OUTPUT_ROOT / "tables"

#: Verification reports and comparison figures.
VERIFICATION_OUT = OUTPUT_ROOT / "verification"


def ensure(path: Path) -> Path:
    """Create ``path`` as a directory if absent, then return it.

    Convenience for output paths: ``ensure(SIMULATION_OUT) / "quinidine.pdf"``.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
