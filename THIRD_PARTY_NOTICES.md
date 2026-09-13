# Third-party notices

This repository is distributed under the GNU General Public License v3.0 (see
`LICENSE`). It incorporates third-party material as set out below.

Original work by the author, not derived from any third-party source:
`cell_models/base.py`, `cell_models/drug.py`, `paths.py` and everything under
`simulation/`.

---

## Why this repository is GPL-3.0

Of the six cardiac cell models, five are available under Creative Commons
Attribution terms, which are attribution-only and permit relicensing of an
adaptation. One — the human ToR-ORd model — derives from GPL-3.0 source. 
This repository is therefore GPL-3.0.

The companion analysis repository (https://github.com/Isaac-Somerville/translational_mfgps) contains no cell model code
and does not link to any. The two communicate through CSV files on disk.

---

## 1. Human — ToR-ORd-dynCl (epicardial), Tomek et al. 2020

- **Used by:** `cell_models/human_torord.py`
- **Source:** `ToRORd_dynCl_epi.cellml` from <https://github.com/jtmff/torord>
- **Licence:** **GPL-3.0**
- **Citation:** Tomek J, Bueno-Orovio A, Passini E, Zhou X, Minchole A,
  Britton O, Bartolucci C, Severi S, Shrier A, Virag L, Varro A, Rodriguez B
  (2019). Development, calibration, and validation of a novel human ventricular
  myocyte model in health, disease, and drug block. *eLife* 8:e48890.
  doi:[10.7554/eLife.48890](https://doi.org/10.7554/eLife.48890)
- **Dynamic-chloride variant:** Tomek J, Bueno-Orovio A, Rodriguez B (2020).
  ToR-ORd-dynCl: an update of the ToR-ORd model of human ventricular
  cardiomyocyte with dynamic intracellular chloride. *bioRxiv* 2020.06.01.127043.
  doi:[10.1101/2020.06.01.127043](https://doi.org/10.1101/2020.06.01.127043)

Note: the Physiome Model Repository hosts a Tomek 2019 exposure under CC BY 3.0,
but that is the base ToR-ORd, not the dynamic-chloride variant used here. 
The CC BY grant does not extend to this file.

## 2. Dog — Benson et al. 2008

- **Used by:** `cell_models/dog_benson.py`
- **Source:** Physiome Model Repository, <https://models.physiomeproject.org/e/412>
- **Licence:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)
- **Citation:** Benson AP, Aslanidi OV, Zhang H, Holden AV (2008). The canine
  virtual ventricular wall: a platform for dissecting pharmacological effects on
  propagation and arrhythmogenesis. *Progress in Biophysics and Molecular
  Biology* 96(1–3):187–208. PMID: 17915298

## 3. Guinea Pig — Pasek et al. 2008

- **Used by:** `cell_models/guinea_pig.py`
- **Source:** Physiome Model Repository, model [pasek_model_2008](https://models.physiomeproject.org/exposure/dc0a1bb790d7e2d70974977174a869ec/pasek_simurda_orchard_christe_2008.cellml/view)
- **Licence:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)
- **Citation:** Pasek M, Simurda J, Orchard CH, Christe G (2008). A model of the
  guinea-pig ventricular cardiac myocyte incorporating a transverse-axial tubular
  system. *Progress in Biophysics and Molecular Biology* 96(1–3):258–280.
  PMID: 17888503

This model integrates in **seconds** rather than milliseconds; see the note in
`cell_models/__init__.py`.

## 4. Mouse — Li et al. 2009 (C57BL/6 wild-type)

- **Used by:** `cell_models/mouse.py`
- **Source:** Physiome Model Repository,
  <https://models.physiomeproject.org/e/38> (`Li_Smith_2009_C57BL7_WT.cellml`)
- **Licence:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)
- **Citation:** Li L, Niederer SA, Idigo W, Zhang YH, Swietach P, Casadei B, and Smith NP (2010). 
  A mathematical model of the murine ventricular myocyte: a data-driven biophysically based 
  approach applied to mice overexpressing the canine NCX isoform. 
  American Journal of Physiology-Heart and Circulatory Physiology, 299(4):H1045–H1063.

## 5. Pig — Gaur et al. 2021

- **Used by:** `cell_models/pig.py`
- **Source:** supporting information (S1 Model) of the article below
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0), per PLOS policy
- **Citation:** Gaur N, Qi XY, Benoist D, Bernus O, Coronel R, Nattel S,
  Vigmond EJ (2021). A computational model of pig ventricular cardiomyocyte
  electrophysiology and calcium handling: Translation from pig to human
  electrophysiology. *PLOS Computational Biology* 17(6):e1009137.
  doi:[10.1371/journal.pcbi.1009137](https://doi.org/10.1371/journal.pcbi.1009137)
  PMID: 34191797

## 6. Rabbit — Mahajan et al. 2008

- **Used by:** `cell_models/rabbit.py`
- **Source:** Physiome Model Repository, model [mahajan_2008](https://models.physiomeproject.org/exposure/a5586b72d07ce03fc40fc98ee846d7a5/mahajan_shiferaw_sato_baher_olcese_xie_yang_chen_restrepo_karma_garfinkel_qu_weiss_2008.cellml/view)
- **Licence:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)
- **Citation:** Mahajan A, Shiferaw Y, Sato D, Baher A, Olcese R, Xie LH,
  Yang MJ, Chen PS, Restrepo JG, Karma A, Garfinkel A, Qu Z, Weiss JN (2008).
  A rabbit ventricular action potential model replicating cardiac dynamics at
  rapid heart rates. *Biophysical Journal* 94(2):392–410. PMID: 18160660

---

## 7. CiPA ion-channel data

- **Used by:** `data/cipa/CiPA_optimal_combined.csv`,
  `data/cipa/CiPA_training_drugs.csv`
- **Source:** <https://github.com/FDA/CiPA>
- **Upstream licence:** GPL-3.0
- **Authors:** Kelly Chang and Zhihua Li, who developed the upstream code as Oak
  Ridge Institute for Science and Education (ORISE) research fellows at the U.S.
  Food and Drug Administration.

`CiPA_optimal_combined.csv` is not a verbatim upstream file: it is a compilation
assembled by the author from the per-drug `IC50_optimal.csv` files, `drug_block.csv`
and `CiPA_training_drugs.csv`. No upstream source code is reproduced here.

**FDA disclaimer (from the upstream repository):** This software and documentation were developed by the authors in their capacities as Oak Ridge Institute for Science and Education (ORISE) research fellows at the U.S. Food and Drug Administration (FDA). FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, FDA makes no representations that the use of the Software will not infringe any patent or proprietary rights of third parties. The use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions.

---

## Dependencies

Runtime dependencies (`numpy`, `scipy`, `pandas`, `numba`, `llvmlite`,
`matplotlib`) are distributed under their own permissive BSD-family licences and
are not redistributed as part of this repository.
