# Verification state — pipeline reproducibility (2026-10-07)

Request (Isaac): every .py script whose outputs feed another script runs, needs only
requirements.txt, and reproduces the data in this repo. Ask before any run > 30 min.

Environment: Python 3.13.1 venv, requirements.txt with ONE deviation: scipy 1.16.2
(pinned 1.15.2 and 1.15.3 wheels fail to import on macOS/Darwin 27:
`_spropack...so: section '__DATA/__thread_bss' has a zero-fill section type`).
Platform: Apple M3 arm64. Published runs: HPC Linux x86_64.
Scratch: /private/tmp/claude-502/-Users-ish124-Documents-GitHub-apd-drug-effects/f3bbd9da-6115-4680-89c0-876331be8d56/scratchpad
(venv116/, repo*/ copies, tools/, out/).

Status key: SETTLED (a blinded agent independently agrees + I accept its derivation) /
CHECKED-BY-ME (my run only, agent pending) / PENDING (not yet run).

## Agent log
- V1 (general-purpose, opus, medium): claims 1-5, blinded. Reported ~17:31. Agrees on all
  five (same counts: HEAD build differs 3,018/19,874 APD_90; 2,073 files = merge set;
  188/188 baselines; GT HEAD differs 51/71 & 47/71; tables identical; min_steps none match).
  Derivation accepted -> claims 1-5 SETTLED.
  Novel (V1): Guinea Pig x quinidine absent from published dataset and merge (30 raw files
  exist); EXCLUDED_* entries for it are dead. Reproduced by me (0 rows). Needs independent agent.
- V2 (general-purpose, opus, medium): claim 8 + GP-quinidine finding + Mouse seed question.
  SPAWNED ~17:35. Reported ~17:45. Claim 8: agrees exactly (5/6 byte-identical, Rabbit v13
  2 entries +-1.16e-10, states 0/54 byte-identical); its own reruns byte-identical to mine
  -> environment difference, not run noise. Claim 8 SETTLED.
  GP x quinidine gap: independently found -> SETTLED (V1 + V2). Rerun+rebuild would append
  200 GP-quinidine rows (no noise positions -> noisy column NaN).
  CORRECTION of my Mouse interpretation: foreign-seed / 73-constant drug states are orphan
  sets with no apd.csv and no dataset rows; Mouse v22 sotalol rerun reproduces published.
  I re-checked: 44 suspect Mouse sets, 0 contribute; GP only v23 diltiazem (known).
  Correction -> needs fresh blinded agent (V3).
  Other V2 notes: outputs/simulation/ untracked (new files); 106 stray
  saved_states/<drug>/Guinea Pig/v1_<conc>_limit_state.json files; published_noise_index
  has 20,159 rows (missing row = GP ondansetron v29 46.33, all-NaN).
- V3 (general-purpose, opus, medium): orphan-state question + claims 9, 10 (Dog), 11, 12.
  SPAWNED ~17:50, reported ~18:02. Agrees: only inconsistent state set feeding data is GP
  diltiazem v23 (known) -> Mouse correction SETTLED. Claim 9 numbers identical to mine
  (GP v22 APD 200.880 vs 199.680; Rabbit/Dog exact) -> SETTLED; 34/221 committed subjects
  unreachable by script (6 v0 seed 0, 28 > 200 attempts). Claim 10 Dog byte-identical ->
  SETTLED (Rabbit part: mine only, UNREPRODUCED). Claims 11, 12 SETTLED.
  V3 corrected me: seed range shared by subjects 0 and 2 (not 0 and 1) -> comment fixed;
  outputs README state-diff magnitude understated -> fixed. Further gaps it noted (report
  only): features check skips None features; drug-block check passes with no usable rows.
  VERIFICATION COMPLETE except Rabbit tolerance detail (unreproduced, not re-sent: cost).

## Claims

1. `simulation/build_apd_dataset.py` (with load_runs fix) rebuilds
   `data/simulations/apd_drug_block.csv` byte-identically from the shipped
   `outputs/simulation/drug_block/**/apd.csv`; original code (round_trip read) does not.
   — SETTLED (V1).
2. Shipped `apd.csv` files faithfully carry the original research runs
   (StatML-Mini-Project-2/Simulations/Outputs/drug_block/<drug>/<Benson_Dog|species>/v<n>/raw_apd_limit_state.csv,
   row set = published merge all_drug_block_limit_state.csv; 0.00 row = baseline_apd()).
   — SETTLED (V1).
3. `simulation/calculate_ground_truth.py` (with numpy-mean + trunc16 fix) reproduces
   `ground_truth_relative_{1,3}x.csv` byte-identically; original code does not
   (max |d| 1.1e-16). — SETTLED (V1).
4. `paper/generate_tables.py` reproduces all 5 `outputs/tables/*.tex` byte-identically.
   — SETTLED (V1).
5. `simulation/compute_min_steps.py` does not reproduce `min_steps_needed_rounded.json`
   (got H2700 D5000 GP100 M1800 P100 R1300 vs 3000/1000/2500/7000/1000/2500;
   Mouse 7000 > MAX_STEPS 5000 is unreachable). — SETTLED (V1).
6. `simulation/compute_ap_features.py` does not byte-reproduce defaults/limits
   (documented); Human limits identical. — CHECKED-BY-ME (no agent; already documented as non-reproducible).
7. `baseline_apd()` reproduces every published 0 nM APD (exact or via default-parser
   artefact), except v31 Human/Pig/Rabbit (published NaN, code gives a value) and
   GP v23 (no baseline state shipped). — SETTLED (V1 Q2: 188/188).
8. `simulation/simulate_drug_block.py` reproduces shipped apd.csv for a sample:
   byte-identical for Dog mexiletine v29, GP chlorpromazine v1, Mouse sotalol v22
   (seed-mismatched subject), Mouse ondansetron v41, Pig cisapride v15; Rabbit
   chlorpromazine v13 8/10 exact, 2 differ ~1e-10; Human chlorpromazine v17 @12.67
   exact. Drug states: 0/54 byte-identical, max rel diff 1.7e-5, voi equal 26/54.
   Logs: scratchpad/out/batchA/. — SETTLED (V2).
9. `simulation/generate_subjects.py`: full script, attempt-1 subjects, others present.
   Dog v4, GP v22, Mouse v41, Pig v14, Rabbit v40: same seed accepted, constants ~1e-16,
   states not bitwise (max rel 1.9e-4; Pig 0.47 on some variable), converged one block
   off; baseline APD exact for Dog/Mouse/Pig/Rabbit, GP 200.880 vs 199.680 ms.
   Human v27: attempt 1 rejected (APD_40 unmeasurable at 200 cycles) where original
   accepted it; stopped at ~26 min. Structural blockers: v0 (seed 0) no script;
   28 subjects needed > MAX_ATTEMPTS=200 (max 1168). Logs scratchpad/out/batchB/.
   — SETTLED (V3).
10. `simulation/compute_limit_cycle_tolerances.py`: Dog byte-identical (17/17);
    Rabbit 6/13 exact, nai_min/max 9x smaller than shipped, cai/v_peak/dvdt within 17%.
    Human/GP/Mouse/Pig not run (cost). — Dog SETTLED (V3); Rabbit UNREPRODUCED.
11. requirements.txt is complete: imports are stdlib + numpy/scipy/pandas/numba/
    matplotlib only; no sys.path hacks; `pip check` clean. Not installable as pinned
    on macOS 27 (scipy). — SETTLED (V3).
12. verifier fix: dataset check now byte-compares a real rebuild; missing inputs FAIL.
    Full default run PASS on fresh copy. — SETTLED (V3).

## Report-only observations (no fix: would change published data)
- Baseline APD grid step = period/(min_steps-1); drug-block grid = period/min_steps.
- GP: t_0 = baseline voi in seconds passed as ms -> runs start at t=1.2 s, 0.2-beat
  stimulus phase offset vs saved state.
- Mouse v2,12,13,14,15,23,32: drug states end before baseline start time; v1,9,21,23,
  24,28 exceed 25,000-beat cap; GP v12 off-block times -> those drug runs did not
  start from the shipped baseline states.
- All private raw_apd files were rewritten once through pandas default parser.

## Session 2 (2026-10-08) — changes requested by Isaac
- compute_min_steps.py deleted; references removed; hand-chosen note in cell_models/base.py.
- GP x quinidine: not an explicit exclusion originally. Per-pair merge file deleted in
  private commit b9dae6d872 (2026-05-15, "Rerun guinea pig quinidine block"), never re-merged;
  final merge (8afaa83c07, 2026-05-21) skipped it silently. Now explicit: EXCLUDED_PAIRS.
- Subjects with no drug-free APD (Human/Pig/Rabbit v31, GP v23) now dropped via
  EXCLUDED_SUBJECTS_ALL_DRUGS: 230 rows, all NaN relative/noisy; 230 noise-index rows removed.
  New dataset (19,930 rows) == published minus those lines, line for line. Ground truth
  byte-unchanged. Fresh-run scenario (GP-quinidine runs present, v31 baselines filled) builds
  the same bytes. Verifier PASS. Companion translational_mfgps: load_apd_dataset drops NaN
  truth rows -> identical loaded frame (19,669 rows, order, subject counts); only pandas row
  labels differ, which nothing persists. Its data copy not updated (other repo).
- Not agent-verified (Isaac did not request /verify for session 2).

## Session 3 (2026-10-09)
- MAX_ATTEMPTS removed (added in 07c0e8a3d7 "Make first versions of public repos"; cluster
  simulate_new_subjects.py had no cap). generate_subjects now skips subject 0.
- v0 origin: private 5c98e7430c (unperturbed, initial conditions, convergence checks from
  block 10, max 300) for Human/GP/Pig/Rabbit; Dog and Mouse shipped v0 = fixed 10,000 / 3,000
  beats. New `python -m simulation.limit_cycle` encodes both. Reproduced (my runs only):
  GP/Pig/Rabbit same beat count, constants equal, APD exact, states <=1.7e-4; Dog/Mouse fixed
  length: same voi, states <=8.9e-6, APD exact. Human v0 NOT run (~30-60 min; ask).
- Mouse leftover drug states: for each affected subject, leftover drugs == drugs missing from
  dataset for that subject (gaps, not contamination).

## Session 4 (2026-10-09)
- Isaac: no Human v0 run, no warm_start_from record. Deleted 76 leftover drug-state folders
  (683 files; Mouse 44, GP 32), none with apd.csv or dataset rows. Remaining drug states all
  match their baseline seed/model. Verifier PASS.
- Deleted 106 stray data/saved_states/<drug>/Guinea Pig/v1_<conc>_limit_state.json files
  (outside v1/ folders, unread by code). Verifier PASS.
