"""Cardiac simulation pipeline: subjects, drug block, and the derived dataset.

Run the stages in this order; each depends on the one above it.

======================================  =====================================
Module                                  Produces
======================================  =====================================
``compute_min_steps``                   integration steps per beat, per species
``compute_ap_features``                 baseline AP features and feasibility bounds
``compute_limit_cycle_tolerances``      per-feature convergence tolerances
``generate_subjects``                   ``data/saved_states/baseline/``
``simulate_drug_block``                 APD at 11 concentrations, per subject
``build_apd_dataset``                   ``data/simulations/apd_drug_block.csv``
``calculate_ground_truth``              ``data/simulations/ground_truth_*.csv``
======================================  =====================================

The outputs of every stage ship with the repository, so any stage can be run on
its own to reproduce one artefact without recomputing the ones before it.

The two expensive stages are ``generate_subjects`` and ``simulate_drug_block``;
both accept ``PBS_ARRAY_INDEX`` and have wrapper scripts in
``simulation/shell/``.
"""
