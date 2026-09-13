"""Human ventricular myocyte model (ToR-ORd-dynCl, epicardial).

Derived from ``ToRORd_dynCl_epi.cellml`` in the ToR-ORd reference implementation
at https://github.com/jtmff/torord, which is licensed GPL-3.0.  This file is a
translation of that model into Python and is therefore a modified version of it
in the sense of GPLv3 section 5.

Copyright (C) 2026 Isaac S. Hayden
Portions copyright (C) the ToR-ORd authors (Tomek, Bueno-Orovio, Rodriguez et al.)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License (``LICENSE`` in the repository root) for more details.

Reference:
    Tomek J, Bueno-Orovio A, Passini E, Zhou X, Minchole A, Britton O,
    Bartolucci C, Severi S, Shrier A, Virag L, Varro A, Rodriguez B (2019).
    Development, calibration, and validation of a novel human ventricular
    myocyte model in health, disease, and drug block. eLife 8:e48890.
    doi:10.7554/eLife.48890

    Dynamic-chloride update: Tomek J, Bueno-Orovio A, Rodriguez B (2020).
    ToR-ORd-dynCl: an update of the ToR-ORd model of human ventricular
    cardiomyocyte with dynamic intracellular chloride.
    bioRxiv 2020.06.01.127043. doi:10.1101/2020.06.01.127043
"""

import numpy as np
import time
import math
from cell_models.base import CardiacCellModel
from numba import njit


class HumanModel(CardiacCellModel):
    def __init__(
        self,
        drug=None,
        drug_amount=None,
        num_cycles_feasible_ap=None
    ):
        super().__init__(
            sizeStates=45 + 1,
            sizeAlgebraic=226 + 7,
            sizeConstants=162,
            species="Human",
            num_cycles_feasible_ap=num_cycles_feasible_ap,
            num_cycles_limit_state=20_000,
            state_idx_dict = {"cai": 9, "nai": 3, "ki" : 5, "cli" : 10}
        )
        self.initConsts()
        self.createLegends()
        if drug is not None:
            super().init_drug(drug, drug_amount)
        else:
            self.drug = None
            self.drug_amount = None

    def initConsts(self, perturbation=None, seed=42):
        """Initialize model-specific constants and initial conditions."""
        np.random.seed(seed)

        # Initialise arrays
        constants = np.zeros(self.sizeConstants)
        states = np.zeros(self.sizeStates)

        # Initialise constants that are never perturbed
        constants[0] = 1.0  # cell type (1=epicardial)
        constants[14] = 0  # stim start
        constants[15] = 1e15  # stim end
        constants[17] = 1000  # stim period
        constants[8] = 1 # ion charges
        constants[9] = 2
        constants[10] = 1
        constants[11] = -1
        constants[103] = 1 # this zeros out an expression
        constants[6] = 310 # physical constants
        constants[7] = 96485
        constants[5] = 8314

        # Initialise states
        states[0] = -9.074563e01
        states[1] = 1.273541e-02
        states[2] = 5.749921e-05
        states[3] = 1.340062e01
        states[4] = 1.340094e01
        states[5] = 1.523639e02
        states[6] = 1.523638e02
        states[7] = 1.806794e00
        states[8] = 1.805047e00
        states[9] = 6.621816e-05
        states[10] = 3.431721e01
        states[11] = 3.431719e01
        states[12] = 5.253231e-04
        states[13] = 8.645148e-01
        states[14] = 8.644571e-01
        states[15] = 7.313656e-01
        states[16] = 8.643527e-01
        states[17] = 1.117969e-04
        states[18] = 5.916536e-01
        states[19] = 3.476812e-01
        states[20] = 8.320408e-04
        states[21] = 9.997242e-01
        states[22] = 9.997235e-01
        states[23] = 4.239121e-04
        states[24] = 9.997242e-01
        states[25] = 9.997241e-01
        states[26] = -2.486527e-36
        states[27] = 1.000000e00
        states[28] = 9.510602e-01
        states[29] = 1.000000e00
        states[30] = 9.999377e-01
        states[31] = 9.999886e-01
        states[32] = 1.000000e00
        states[33] = 1.000000e00
        states[34] = 3.049523e-04
        states[35] = 5.272668e-04
        states[36] = 9.984733e-01
        states[37] = 7.393045e-04
        states[38] = 6.029079e-04
        states[39] = 5.678255e-06
        states[40] = 1.787783e-04
        states[41] = 2.233584e-01
        states[42] = 1.418247e-04
        states[43] = 6.778827e-25
        states[44] = -1.581941e-23

        # Perturb constants
        base_value_idxs = list(self.constant_base_value_dict.keys())
        different_cv_idxs = set()
        for param in self.constant_idx_dict:
            # set those with different CVs first
            if param not in self.parameter_CVs:
                continue
            idx = self.constant_idx_dict[param]
            # track which idxs have had their CVs set
            different_cv_idxs.add(int(idx))
            cv = self.parameter_CVs[param]
            reference_value = self.constant_base_value_dict[str(idx)]
            constants[idx] = self.perturb_constant(
                reference_value=reference_value, perturbation=perturbation, cv=cv
            )
        for idx in base_value_idxs:
            if int(idx) in different_cv_idxs:
                # skip those already set
                continue
            # 95% CI = [0.5, 2] fold change with lognormal distribution
            cv = np.sqrt(np.exp((np.log(2) / 1.96) ** 2) - 1)
            constants[int(idx)] = self.perturb_constant(
                reference_value=self.constant_base_value_dict[str(idx)],
                perturbation=perturbation,
                cv=cv,
            )

        constants[112] = (
            1000.00 * 3.14000 * constants[13] * constants[13] * constants[12]
        )
        constants[113] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[24] * 1.30000,
                True,
                constants[24],
            ]
        )
        constants[114] = np.power(constants[3] / constants[37], 0.240000)
        constants[115] = 1.00000 / (
            1.00000 + np.power(constants[38] / constants[39], 2.00000)
        )
        constants[116] = 3.00000 * constants[41]
        constants[117] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[42] * 0.600000,
                True,
                constants[42],
            ]
        )
        constants[118] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[43] * 2.00000,
                np.equal(constants[0], 2.00000),
                constants[43] * 2.00000,
                True,
                constants[43],
            ]
        )
        constants[119] = 1.00000 - constants[48]
        constants[120] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[47] * 1.20000,
                np.equal(constants[0], 2.00000),
                constants[47] * 2.00000,
                True,
                constants[47],
            ]
        )
        constants[121] = (
            0.500000
            * (constants[1] + constants[3] + constants[4] + 4.00000 * constants[2])
        ) / 1000.00
        constants[122] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[54] * 1.30000,
                np.equal(constants[0], 2.00000),
                constants[54] * 0.800000,
                True,
                constants[54],
            ]
        )
        constants[123] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[57] * 1.40000,
                True,
                constants[57],
            ]
        )
        constants[124] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[58] * 1.20000,
                np.equal(constants[0], 2.00000),
                constants[58] * 1.30000,
                True,
                constants[58],
            ]
        )
        constants[125] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[95] * 0.600000,
                True,
                constants[95],
            ]
        )
        constants[126] = (0.500000 * constants[108]) / 1.00000
        constants[127] = 1.25000 * constants[108]
        constants[128] = super().custom_piecewise(
            [np.equal(constants[0], 1.00000), 1.30000, True, 1.00000]
        )
        constants[129] = (
            2.00000 * 3.14000 * constants[13] * constants[13]
            + 2.00000 * 3.14000 * constants[13] * constants[12]
        )
        constants[130] = 1.10000 * constants[120]
        constants[131] = 0.00125000 * constants[120]
        constants[132] = 0.000357400 * constants[120]
        constants[133] = 1.82000e06 * (np.power(constants[52] * constants[6], -1.50000))
        constants[134] = (0.500000 * constants[127]) / 1.00000
        constants[135] = 2.00000 * constants[129]
        constants[136] = 0.00125000 * constants[130]
        constants[137] = 0.000357400 * constants[130]
        constants[138] = np.exp(
            -constants[133]
            * 4.00000
            * (
                (np.power(constants[121], 1.0 / 2))
                / (1.00000 + np.power(constants[121], 1.0 / 2))
                - 0.300000 * constants[121]
            )
        )
        constants[139] = np.exp(
            -constants[133]
            * 1.00000
            * (
                (np.power(constants[121], 1.0 / 2))
                / (1.00000 + np.power(constants[121], 1.0 / 2))
                - 0.300000 * constants[121]
            )
        )
        constants[140] = np.exp(
            -constants[133]
            * 1.00000
            * (
                (np.power(constants[121], 1.0 / 2))
                / (1.00000 + np.power(constants[121], 1.0 / 2))
                - 0.300000 * constants[121]
            )
        )
        constants[141] = 0.680000 * constants[112]
        constants[142] = 0.0552000 * constants[112]
        constants[143] = 0.00480000 * constants[112]
        constants[144] = 0.0200000 * constants[112]
        constants[145] = (
            constants[63]
            + 1.00000
            + (constants[1] / constants[60]) * (1.00000 + constants[1] / constants[61])
        )
        constants[146] = (constants[1] * constants[1]) / (
            constants[145] * constants[60] * constants[61]
        )
        constants[147] = 1.00000 / constants[145]
        constants[148] = constants[147] * constants[2] * constants[67]
        constants[149] = constants[68]
        constants[150] = constants[68]
        constants[151] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[72] * 1.10000,
                np.equal(constants[0], 2.00000),
                constants[72] * 1.40000,
                True,
                constants[72],
            ]
        )
        constants[152] = (
            constants[63]
            + 1.00000
            + (constants[1] / constants[60]) * (1.00000 + constants[1] / constants[61])
        )
        constants[153] = (constants[1] * constants[1]) / (
            constants[152] * constants[60] * constants[61]
        )
        constants[154] = 1.00000 / constants[152]
        constants[155] = constants[154] * constants[2] * constants[67]
        constants[156] = constants[68]
        constants[157] = constants[68]
        constants[158] = constants[74] * constants[86]
        constants[159] = constants[75]
        constants[160] = ((constants[79] * constants[87]) / constants[88]) / (
            1.00000 + constants[87] / constants[88]
        )
        constants[161] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                constants[94] * 0.900000,
                np.equal(constants[0], 2.00000),
                constants[94] * 0.700000,
                True,
                constants[94],
            ]
        )
        self.init_states = states
        self.constants = constants
        return states, constants

    def createLegends(self):
        legend_states = [""] * self.sizeStates
        legend_rates = [""] * self.sizeStates
        legend_algebraic = [""] * self.sizeAlgebraic
        legend_voi = ""
        legend_constants = [""] * self.sizeConstants
        legend_states[-1] = "drug concentration (nM)"
        legend_rates[-1] = "d/dt drug concentration (nM/ms)"
        for i in range(len(self.channel_list)):
            legend_algebraic[i - len(self.channel_list)] = (
                "proportion of "
                + self.channel_list[i]
                + " channel uninhibited (dimensionless)"
            )
        legend_voi = "time in component environment (millisecond)"
        legend_constants[0] = "celltype in component environment (dimensionless)"
        legend_constants[1] = "nao in component extracellular (millimolar)"
        legend_constants[2] = "cao in component extracellular (millimolar)"
        legend_constants[3] = "ko in component extracellular (millimolar)"
        legend_constants[4] = "clo in component extracellular (millimolar)"
        legend_constants[5] = (
            "R in component physical_constants (joule_per_kilomole_kelvin)"
        )
        legend_constants[6] = "T in component physical_constants (kelvin)"
        legend_constants[7] = "F in component physical_constants (coulomb_per_mole)"
        legend_constants[8] = "zna in component physical_constants (dimensionless)"
        legend_constants[9] = "zca in component physical_constants (dimensionless)"
        legend_constants[10] = "zk in component physical_constants (dimensionless)"
        legend_constants[11] = "zcl in component physical_constants (dimensionless)"
        legend_constants[12] = "L in component cell_geometry (centimeter)"
        legend_constants[13] = "rad in component cell_geometry (centimeter)"
        legend_constants[112] = "vcell in component cell_geometry (microliter)"
        legend_constants[129] = "Ageo in component cell_geometry (centimeter_squared)"
        legend_constants[135] = "Acap in component cell_geometry (centimeter_squared)"
        legend_constants[141] = "vmyo in component cell_geometry (microliter)"
        legend_constants[142] = "vnsr in component cell_geometry (microliter)"
        legend_constants[143] = "vjsr in component cell_geometry (microliter)"
        legend_constants[144] = "vss in component cell_geometry (microliter)"
        legend_states[0] = "v in component membrane (millivolt)"
        legend_algebraic[25] = "vffrt in component membrane (coulomb_per_mole)"
        legend_algebraic[28] = "vfrt in component membrane (dimensionless)"
        legend_algebraic[70] = "INa in component INa (microA_per_microF)"
        legend_algebraic[72] = "INaL in component INaL (microA_per_microF)"
        legend_algebraic[78] = "Ito in component Ito (microA_per_microF)"
        legend_algebraic[114] = "ICaL in component ICaL (microA_per_microF)"
        legend_algebraic[115] = "ICaNa in component ICaL (microA_per_microF)"
        legend_algebraic[116] = "ICaK in component ICaL (microA_per_microF)"
        legend_algebraic[117] = "IKr in component IKr (microA_per_microF)"
        legend_algebraic[119] = "IKs in component IKs (microA_per_microF)"
        legend_algebraic[123] = "IK1 in component IK1 (microA_per_microF)"
        legend_algebraic[155] = "INaCa_i in component INaCa (microA_per_microF)"
        legend_algebraic[185] = "INaCa_ss in component INaCa (microA_per_microF)"
        legend_algebraic[204] = "INaK in component INaK (microA_per_microF)"
        legend_algebraic[207] = "INab in component INab (microA_per_microF)"
        legend_algebraic[206] = "IKb in component IKb (microA_per_microF)"
        legend_algebraic[211] = "IpCa in component IpCa (microA_per_microF)"
        legend_algebraic[209] = "ICab in component ICab (microA_per_microF)"
        legend_algebraic[216] = "IClCa in component ICl (microA_per_microF)"
        legend_algebraic[218] = "IClb in component ICl (microA_per_microF)"
        legend_algebraic[68] = "I_katp in component I_katp (microA_per_microF)"
        legend_algebraic[11] = "Istim in component membrane (microA_per_microF)"
        legend_constants[14] = "i_Stim_Start in component membrane (millisecond)"
        legend_constants[15] = "i_Stim_End in component membrane (millisecond)"
        legend_constants[16] = (
            "i_Stim_Amplitude in component membrane (microA_per_microF)"
        )
        legend_constants[17] = "i_Stim_Period in component membrane (millisecond)"
        legend_constants[18] = (
            "i_Stim_PulseDuration in component membrane (millisecond)"
        )
        legend_constants[19] = "KmCaMK in component CaMK (millimolar)"
        legend_constants[20] = (
            "aCaMK in component CaMK (per_millimolar_per_millisecond)"
        )
        legend_constants[21] = "bCaMK in component CaMK (per_millisecond)"
        legend_constants[22] = "CaMKo in component CaMK (dimensionless)"
        legend_constants[23] = "KmCaM in component CaMK (millimolar)"
        legend_algebraic[43] = "CaMKb in component CaMK (millimolar)"
        legend_algebraic[49] = "CaMKa in component CaMK (millimolar)"
        legend_states[1] = "CaMKt in component CaMK (millimolar)"
        legend_states[2] = "cass in component intracellular_ions (millimolar)"
        legend_constants[24] = "cmdnmax_b in component intracellular_ions (millimolar)"
        legend_constants[113] = "cmdnmax in component intracellular_ions (millimolar)"
        legend_constants[25] = "kmcmdn in component intracellular_ions (millimolar)"
        legend_constants[26] = "trpnmax in component intracellular_ions (millimolar)"
        legend_constants[27] = "kmtrpn in component intracellular_ions (millimolar)"
        legend_constants[28] = "BSRmax in component intracellular_ions (millimolar)"
        legend_constants[29] = "KmBSR in component intracellular_ions (millimolar)"
        legend_constants[30] = "BSLmax in component intracellular_ions (millimolar)"
        legend_constants[31] = "KmBSL in component intracellular_ions (millimolar)"
        legend_constants[32] = "csqnmax in component intracellular_ions (millimolar)"
        legend_constants[33] = "kmcsqn in component intracellular_ions (millimolar)"
        legend_states[3] = "nai in component intracellular_ions (millimolar)"
        legend_states[4] = "nass in component intracellular_ions (millimolar)"
        legend_states[5] = "ki in component intracellular_ions (millimolar)"
        legend_states[6] = "kss in component intracellular_ions (millimolar)"
        legend_states[7] = "cansr in component intracellular_ions (millimolar)"
        legend_states[8] = "cajsr in component intracellular_ions (millimolar)"
        legend_states[9] = "cai in component intracellular_ions (millimolar)"
        legend_states[10] = "cli in component intracellular_ions (millimolar)"
        legend_states[11] = "clss in component intracellular_ions (millimolar)"
        legend_algebraic[93] = "ICaL_ss in component ICaL (microA_per_microF)"
        legend_algebraic[94] = "ICaNa_ss in component ICaL (microA_per_microF)"
        legend_algebraic[97] = "ICaK_ss in component ICaL (microA_per_microF)"
        legend_algebraic[111] = "ICaL_i in component ICaL (microA_per_microF)"
        legend_algebraic[112] = "ICaNa_i in component ICaL (microA_per_microF)"
        legend_algebraic[113] = "ICaK_i in component ICaL (microA_per_microF)"
        legend_algebraic[210] = "JdiffNa in component diff (millimolar_per_millisecond)"
        legend_algebraic[213] = "Jdiff in component diff (millimolar_per_millisecond)"
        legend_algebraic[224] = "Jup in component SERCA (millimolar_per_millisecond)"
        legend_algebraic[208] = "JdiffK in component diff (millimolar_per_millisecond)"
        legend_algebraic[221] = "JdiffCl in component diff (millimolar_per_millisecond)"
        legend_algebraic[217] = "Jrel in component ryr (millimolar_per_millisecond)"
        legend_algebraic[225] = (
            "Jtr in component trans_flux (millimolar_per_millisecond)"
        )
        legend_algebraic[53] = "Bcai in component intracellular_ions (dimensionless)"
        legend_algebraic[59] = "Bcajsr in component intracellular_ions (dimensionless)"
        legend_algebraic[56] = "Bcass in component intracellular_ions (dimensionless)"
        legend_algebraic[214] = "IClCa_sl in component ICl (microA_per_microF)"
        legend_algebraic[212] = "IClCa_junc in component ICl (microA_per_microF)"
        legend_constants[34] = "PKNa in component reversal_potentials (dimensionless)"
        legend_algebraic[63] = "ENa in component reversal_potentials (millivolt)"
        legend_algebraic[64] = "EK in component reversal_potentials (millivolt)"
        legend_algebraic[65] = "EKs in component reversal_potentials (millivolt)"
        legend_algebraic[66] = "ECl in component reversal_potentials (millivolt)"
        legend_algebraic[67] = "EClss in component reversal_potentials (millivolt)"
        legend_constants[35] = "gkatp in component I_katp (milliS_per_microF)"
        legend_constants[36] = "fkatp in component I_katp (dimensionless)"
        legend_constants[37] = "K_o_n in component I_katp (millimolar)"
        legend_constants[38] = "A_atp in component I_katp (millimolar)"
        legend_constants[39] = "K_atp in component I_katp (millimolar)"
        legend_constants[114] = "akik in component I_katp (dimensionless)"
        legend_constants[115] = "bkik in component I_katp (dimensionless)"
        legend_algebraic[0] = "mss in component INa (dimensionless)"
        legend_algebraic[13] = "tm in component INa (millisecond)"
        legend_states[12] = "m in component INa (dimensionless)"
        legend_algebraic[1] = "hss in component INa (dimensionless)"
        legend_algebraic[14] = "ah in component INa (dimensionless)"
        legend_algebraic[29] = "bh in component INa (dimensionless)"
        legend_algebraic[37] = "th in component INa (millisecond)"
        legend_states[13] = "h in component INa (dimensionless)"
        legend_algebraic[38] = "jss in component INa (dimensionless)"
        legend_algebraic[15] = "aj in component INa (dimensionless)"
        legend_algebraic[30] = "bj in component INa (dimensionless)"
        legend_algebraic[44] = "tj in component INa (millisecond)"
        legend_states[14] = "j in component INa (dimensionless)"
        legend_algebraic[45] = "hssp in component INa (dimensionless)"
        legend_states[15] = "hp in component INa (dimensionless)"
        legend_algebraic[50] = "tjp in component INa (millisecond)"
        legend_states[16] = "jp in component INa (dimensionless)"
        legend_algebraic[69] = "fINap in component INa (dimensionless)"
        legend_constants[40] = "GNa in component INa (milliS_per_microF)"
        legend_algebraic[2] = "mLss in component INaL (dimensionless)"
        legend_algebraic[16] = "tmL in component INaL (millisecond)"
        legend_states[17] = "mL in component INaL (dimensionless)"
        legend_constants[41] = "thL in component INaL (millisecond)"
        legend_algebraic[3] = "hLss in component INaL (dimensionless)"
        legend_states[18] = "hL in component INaL (dimensionless)"
        legend_algebraic[4] = "hLssp in component INaL (dimensionless)"
        legend_constants[116] = "thLp in component INaL (millisecond)"
        legend_states[19] = "hLp in component INaL (dimensionless)"
        legend_constants[42] = "GNaL_b in component INaL (milliS_per_microF)"
        legend_constants[117] = "GNaL in component INaL (milliS_per_microF)"
        legend_algebraic[71] = "fINaLp in component INaL (dimensionless)"
        legend_constants[43] = "Gto_b in component Ito (milliS_per_microF)"
        legend_algebraic[5] = "ass in component Ito (dimensionless)"
        legend_algebraic[17] = "ta in component Ito (millisecond)"
        legend_states[20] = "a in component Ito (dimensionless)"
        legend_constants[44] = "EKshift in component Ito (millivolt)"
        legend_algebraic[6] = "iss in component Ito (dimensionless)"
        legend_algebraic[18] = "delta_epi in component Ito (dimensionless)"
        legend_algebraic[31] = "tiF_b in component Ito (millisecond)"
        legend_algebraic[39] = "tiS_b in component Ito (millisecond)"
        legend_algebraic[46] = "tiF in component Ito (millisecond)"
        legend_algebraic[51] = "tiS in component Ito (millisecond)"
        legend_algebraic[73] = "AiF in component Ito (dimensionless)"
        legend_algebraic[74] = "AiS in component Ito (dimensionless)"
        legend_states[21] = "iF in component Ito (dimensionless)"
        legend_states[22] = "iS in component Ito (dimensionless)"
        legend_algebraic[75] = "i in component Ito (dimensionless)"
        legend_algebraic[32] = "assp in component Ito (dimensionless)"
        legend_states[23] = "ap in component Ito (dimensionless)"
        legend_algebraic[54] = "dti_develop in component Ito (dimensionless)"
        legend_algebraic[57] = "dti_recover in component Ito (dimensionless)"
        legend_algebraic[60] = "tiFp in component Ito (millisecond)"
        legend_algebraic[61] = "tiSp in component Ito (millisecond)"
        legend_states[24] = "iFp in component Ito (dimensionless)"
        legend_states[25] = "iSp in component Ito (dimensionless)"
        legend_algebraic[76] = "ip in component Ito (dimensionless)"
        legend_constants[118] = "Gto in component Ito (milliS_per_microF)"
        legend_algebraic[77] = "fItop in component Ito (dimensionless)"
        legend_constants[45] = "Kmn in component ICaL (millimolar)"
        legend_constants[46] = "k2n in component ICaL (per_millisecond)"
        legend_constants[47] = "PCa_b in component ICaL (dimensionless)"
        legend_algebraic[7] = "dss in component ICaL (dimensionless)"
        legend_states[26] = "d in component ICaL (dimensionless)"
        legend_algebraic[8] = "fss in component ICaL (dimensionless)"
        legend_constants[48] = "Aff in component ICaL (dimensionless)"
        legend_constants[119] = "Afs in component ICaL (dimensionless)"
        legend_states[27] = "ff in component ICaL (dimensionless)"
        legend_states[28] = "fs in component ICaL (dimensionless)"
        legend_algebraic[79] = "f in component ICaL (dimensionless)"
        legend_algebraic[19] = "fcass in component ICaL (dimensionless)"
        legend_algebraic[9] = "jcass in component ICaL (dimensionless)"
        legend_algebraic[80] = "Afcaf in component ICaL (dimensionless)"
        legend_algebraic[81] = "Afcas in component ICaL (dimensionless)"
        legend_states[29] = "fcaf in component ICaL (dimensionless)"
        legend_states[30] = "fcas in component ICaL (dimensionless)"
        legend_algebraic[82] = "fca in component ICaL (dimensionless)"
        legend_states[31] = "jca in component ICaL (dimensionless)"
        legend_states[32] = "ffp in component ICaL (dimensionless)"
        legend_algebraic[83] = "fp in component ICaL (dimensionless)"
        legend_states[33] = "fcafp in component ICaL (dimensionless)"
        legend_algebraic[84] = "fcap in component ICaL (dimensionless)"
        legend_algebraic[10] = "km2n in component ICaL (per_millisecond)"
        legend_algebraic[20] = "anca_ss in component ICaL (dimensionless)"
        legend_states[34] = "nca_ss in component ICaL (dimensionless)"
        legend_algebraic[21] = "anca_i in component ICaL (dimensionless)"
        legend_states[35] = "nca_i in component ICaL (dimensionless)"
        legend_algebraic[89] = "PhiCaL_ss in component ICaL (dimensionless)"
        legend_algebraic[90] = "PhiCaNa_ss in component ICaL (dimensionless)"
        legend_algebraic[91] = "PhiCaK_ss in component ICaL (dimensionless)"
        legend_algebraic[108] = "PhiCaL_i in component ICaL (dimensionless)"
        legend_algebraic[109] = "PhiCaNa_i in component ICaL (dimensionless)"
        legend_algebraic[110] = "PhiCaK_i in component ICaL (dimensionless)"
        legend_constants[120] = "PCa in component ICaL (dimensionless)"
        legend_constants[130] = "PCap in component ICaL (dimensionless)"
        legend_constants[131] = "PCaNa in component ICaL (dimensionless)"
        legend_constants[132] = "PCaK in component ICaL (dimensionless)"
        legend_constants[136] = "PCaNap in component ICaL (dimensionless)"
        legend_constants[137] = "PCaKp in component ICaL (dimensionless)"
        legend_algebraic[92] = "fICaLp in component ICaL (dimensionless)"
        legend_algebraic[22] = "td in component ICaL (millisecond)"
        legend_algebraic[23] = "tff in component ICaL (millisecond)"
        legend_algebraic[24] = "tfs in component ICaL (millisecond)"
        legend_algebraic[33] = "tfcaf in component ICaL (millisecond)"
        legend_algebraic[34] = "tfcas in component ICaL (millisecond)"
        legend_constants[49] = "tjca in component ICaL (millisecond)"
        legend_algebraic[35] = "tffp in component ICaL (millisecond)"
        legend_algebraic[40] = "tfcafp in component ICaL (millisecond)"
        legend_constants[50] = "vShift in component ICaL (millivolt)"
        legend_constants[51] = "offset in component ICaL (millisecond)"
        legend_constants[121] = "Io in component ICaL (dimensionless)"
        legend_algebraic[85] = "Iss in component ICaL (dimensionless)"
        legend_algebraic[100] = "Ii in component ICaL (dimensionless)"
        legend_constants[52] = "dielConstant in component ICaL (per_kelvin)"
        legend_constants[133] = "constA in component ICaL (dimensionless)"
        legend_constants[138] = "gamma_cao in component ICaL (dimensionless)"
        legend_algebraic[86] = "gamma_cass in component ICaL (dimensionless)"
        legend_algebraic[103] = "gamma_cai in component ICaL (dimensionless)"
        legend_constants[139] = "gamma_nao in component ICaL (dimensionless)"
        legend_algebraic[87] = "gamma_nass in component ICaL (dimensionless)"
        legend_algebraic[106] = "gamma_nai in component ICaL (dimensionless)"
        legend_constants[140] = "gamma_ko in component ICaL (dimensionless)"
        legend_algebraic[88] = "gamma_kss in component ICaL (dimensionless)"
        legend_algebraic[107] = "gamma_ki in component ICaL (dimensionless)"
        legend_constants[53] = "ICaL_fractionSS in component ICaL (dimensionless)"
        legend_constants[54] = "GKr_b in component IKr (milliS_per_microF)"
        legend_states[36] = "C1 in component IKr (dimensionless)"
        legend_states[37] = "C2 in component IKr (dimensionless)"
        legend_states[38] = "C3 in component IKr (dimensionless)"
        legend_states[39] = "I in component IKr (dimensionless)"
        legend_states[40] = "O in component IKr (dimensionless)"
        legend_algebraic[41] = "alpha in component IKr (per_millisecond)"
        legend_algebraic[47] = "beta in component IKr (per_millisecond)"
        legend_constants[55] = "alpha_1 in component IKr (per_millisecond)"
        legend_constants[56] = "beta_1 in component IKr (per_millisecond)"
        legend_algebraic[42] = "alpha_2 in component IKr (per_millisecond)"
        legend_algebraic[48] = "beta_2 in component IKr (per_millisecond)"
        legend_algebraic[52] = "alpha_i in component IKr (per_millisecond)"
        legend_algebraic[55] = "beta_i in component IKr (per_millisecond)"
        legend_algebraic[58] = "alpha_C2ToI in component IKr (per_millisecond)"
        legend_algebraic[62] = "beta_ItoC2 in component IKr (per_millisecond)"
        legend_constants[122] = "GKr in component IKr (milliS_per_microF)"
        legend_constants[57] = "GKs_b in component IKs (milliS_per_microF)"
        legend_constants[123] = "GKs in component IKs (milliS_per_microF)"
        legend_algebraic[12] = "xs1ss in component IKs (dimensionless)"
        legend_algebraic[26] = "xs2ss in component IKs (dimensionless)"
        legend_algebraic[27] = "txs1 in component IKs (millisecond)"
        legend_states[41] = "xs1 in component IKs (dimensionless)"
        legend_states[42] = "xs2 in component IKs (dimensionless)"
        legend_algebraic[118] = "KsCa in component IKs (dimensionless)"
        legend_algebraic[36] = "txs2 in component IKs (millisecond)"
        legend_constants[124] = "GK1 in component IK1 (milliS_per_microF)"
        legend_constants[58] = "GK1_b in component IK1 (milliS_per_microF)"
        legend_algebraic[120] = "aK1 in component IK1 (dimensionless)"
        legend_algebraic[121] = "bK1 in component IK1 (dimensionless)"
        legend_algebraic[122] = "K1ss in component IK1 (dimensionless)"
        legend_constants[59] = "INaCa_fractionSS in component INaCa (dimensionless)"
        legend_constants[60] = "kna1 in component INaCa (per_millisecond)"
        legend_constants[61] = "kna2 in component INaCa (per_millisecond)"
        legend_constants[62] = "kna3 in component INaCa (per_millisecond)"
        legend_constants[63] = "kasymm in component INaCa (dimensionless)"
        legend_constants[64] = "wna in component INaCa (dimensionless)"
        legend_constants[65] = "wca in component INaCa (dimensionless)"
        legend_constants[66] = "wnaca in component INaCa (dimensionless)"
        legend_constants[67] = "kcaon in component INaCa (per_millisecond)"
        legend_constants[68] = "kcaoff in component INaCa (per_millisecond)"
        legend_constants[69] = "qna in component INaCa (dimensionless)"
        legend_constants[70] = "qca in component INaCa (dimensionless)"
        legend_algebraic[125] = "hna in component INaCa (dimensionless)"
        legend_algebraic[124] = "hca in component INaCa (dimensionless)"
        legend_constants[71] = "KmCaAct in component INaCa (millimolar)"
        legend_constants[72] = "Gncx_b in component INaCa (milliS_per_microF)"
        legend_constants[151] = "Gncx in component INaCa (milliS_per_microF)"
        legend_algebraic[126] = "h1_i in component INaCa (dimensionless)"
        legend_algebraic[127] = "h2_i in component INaCa (dimensionless)"
        legend_algebraic[128] = "h3_i in component INaCa (dimensionless)"
        legend_algebraic[129] = "h4_i in component INaCa (dimensionless)"
        legend_algebraic[130] = "h5_i in component INaCa (dimensionless)"
        legend_algebraic[131] = "h6_i in component INaCa (dimensionless)"
        legend_algebraic[132] = "h7_i in component INaCa (dimensionless)"
        legend_algebraic[133] = "h8_i in component INaCa (dimensionless)"
        legend_algebraic[134] = "h9_i in component INaCa (dimensionless)"
        legend_constants[145] = "h10_i in component INaCa (dimensionless)"
        legend_constants[146] = "h11_i in component INaCa (dimensionless)"
        legend_constants[147] = "h12_i in component INaCa (dimensionless)"
        legend_constants[148] = "k1_i in component INaCa (dimensionless)"
        legend_constants[149] = "k2_i in component INaCa (dimensionless)"
        legend_algebraic[135] = "k3p_i in component INaCa (dimensionless)"
        legend_algebraic[136] = "k3pp_i in component INaCa (dimensionless)"
        legend_algebraic[137] = "k3_i in component INaCa (dimensionless)"
        legend_algebraic[140] = "k4_i in component INaCa (dimensionless)"
        legend_algebraic[138] = "k4p_i in component INaCa (dimensionless)"
        legend_algebraic[139] = "k4pp_i in component INaCa (dimensionless)"
        legend_constants[150] = "k5_i in component INaCa (dimensionless)"
        legend_algebraic[141] = "k6_i in component INaCa (dimensionless)"
        legend_algebraic[142] = "k7_i in component INaCa (dimensionless)"
        legend_algebraic[143] = "k8_i in component INaCa (dimensionless)"
        legend_algebraic[144] = "x1_i in component INaCa (dimensionless)"
        legend_algebraic[145] = "x2_i in component INaCa (dimensionless)"
        legend_algebraic[146] = "x3_i in component INaCa (dimensionless)"
        legend_algebraic[147] = "x4_i in component INaCa (dimensionless)"
        legend_algebraic[148] = "E1_i in component INaCa (dimensionless)"
        legend_algebraic[149] = "E2_i in component INaCa (dimensionless)"
        legend_algebraic[150] = "E3_i in component INaCa (dimensionless)"
        legend_algebraic[151] = "E4_i in component INaCa (dimensionless)"
        legend_algebraic[152] = "allo_i in component INaCa (dimensionless)"
        legend_algebraic[153] = (
            "JncxNa_i in component INaCa (millimolar_per_millisecond)"
        )
        legend_algebraic[154] = (
            "JncxCa_i in component INaCa (millimolar_per_millisecond)"
        )
        legend_algebraic[156] = "h1_ss in component INaCa (dimensionless)"
        legend_algebraic[157] = "h2_ss in component INaCa (dimensionless)"
        legend_algebraic[158] = "h3_ss in component INaCa (dimensionless)"
        legend_algebraic[159] = "h4_ss in component INaCa (dimensionless)"
        legend_algebraic[160] = "h5_ss in component INaCa (dimensionless)"
        legend_algebraic[161] = "h6_ss in component INaCa (dimensionless)"
        legend_algebraic[162] = "h7_ss in component INaCa (dimensionless)"
        legend_algebraic[163] = "h8_ss in component INaCa (dimensionless)"
        legend_algebraic[164] = "h9_ss in component INaCa (dimensionless)"
        legend_constants[152] = "h10_ss in component INaCa (dimensionless)"
        legend_constants[153] = "h11_ss in component INaCa (dimensionless)"
        legend_constants[154] = "h12_ss in component INaCa (dimensionless)"
        legend_constants[155] = "k1_ss in component INaCa (dimensionless)"
        legend_constants[156] = "k2_ss in component INaCa (dimensionless)"
        legend_algebraic[165] = "k3p_ss in component INaCa (dimensionless)"
        legend_algebraic[166] = "k3pp_ss in component INaCa (dimensionless)"
        legend_algebraic[167] = "k3_ss in component INaCa (dimensionless)"
        legend_algebraic[170] = "k4_ss in component INaCa (dimensionless)"
        legend_algebraic[168] = "k4p_ss in component INaCa (dimensionless)"
        legend_algebraic[169] = "k4pp_ss in component INaCa (dimensionless)"
        legend_constants[157] = "k5_ss in component INaCa (dimensionless)"
        legend_algebraic[171] = "k6_ss in component INaCa (dimensionless)"
        legend_algebraic[172] = "k7_ss in component INaCa (dimensionless)"
        legend_algebraic[173] = "k8_ss in component INaCa (dimensionless)"
        legend_algebraic[174] = "x1_ss in component INaCa (dimensionless)"
        legend_algebraic[175] = "x2_ss in component INaCa (dimensionless)"
        legend_algebraic[176] = "x3_ss in component INaCa (dimensionless)"
        legend_algebraic[177] = "x4_ss in component INaCa (dimensionless)"
        legend_algebraic[178] = "E1_ss in component INaCa (dimensionless)"
        legend_algebraic[179] = "E2_ss in component INaCa (dimensionless)"
        legend_algebraic[180] = "E3_ss in component INaCa (dimensionless)"
        legend_algebraic[181] = "E4_ss in component INaCa (dimensionless)"
        legend_algebraic[182] = "allo_ss in component INaCa (dimensionless)"
        legend_algebraic[183] = (
            "JncxNa_ss in component INaCa (millimolar_per_millisecond)"
        )
        legend_algebraic[184] = (
            "JncxCa_ss in component INaCa (millimolar_per_millisecond)"
        )
        legend_constants[73] = "k1p in component INaK (per_millisecond)"
        legend_constants[74] = "k1m in component INaK (per_millisecond)"
        legend_constants[75] = "k2p in component INaK (per_millisecond)"
        legend_constants[76] = "k2m in component INaK (per_millisecond)"
        legend_constants[77] = "k3p in component INaK (per_millisecond)"
        legend_constants[78] = "k3m in component INaK (per_millisecond)"
        legend_constants[79] = "k4p in component INaK (per_millisecond)"
        legend_constants[80] = "k4m in component INaK (per_millisecond)"
        legend_constants[81] = "Knai0 in component INaK (millimolar)"
        legend_constants[82] = "Knao0 in component INaK (millimolar)"
        legend_constants[83] = "delta in component INaK (millivolt)"
        legend_constants[84] = "Kki in component INaK (per_millisecond)"
        legend_constants[85] = "Kko in component INaK (per_millisecond)"
        legend_constants[86] = "MgADP in component INaK (millimolar)"
        legend_constants[87] = "MgATP in component INaK (millimolar)"
        legend_constants[88] = "Kmgatp in component INaK (millimolar)"
        legend_constants[89] = "H in component INaK (millimolar)"
        legend_constants[90] = "eP in component INaK (dimensionless)"
        legend_constants[91] = "Khp in component INaK (millimolar)"
        legend_constants[92] = "Knap in component INaK (millimolar)"
        legend_constants[93] = "Kxkur in component INaK (millimolar)"
        legend_constants[94] = "Pnak_b in component INaK (milliS_per_microF)"
        legend_constants[161] = "Pnak in component INaK (milliS_per_microF)"
        legend_algebraic[186] = "Knai in component INaK (millimolar)"
        legend_algebraic[187] = "Knao in component INaK (millimolar)"
        legend_algebraic[188] = "P in component INaK (dimensionless)"
        legend_algebraic[189] = "a1 in component INaK (dimensionless)"
        legend_constants[158] = "b1 in component INaK (dimensionless)"
        legend_constants[159] = "a2 in component INaK (dimensionless)"
        legend_algebraic[190] = "b2 in component INaK (dimensionless)"
        legend_algebraic[191] = "a3 in component INaK (dimensionless)"
        legend_algebraic[192] = "b3 in component INaK (dimensionless)"
        legend_constants[160] = "a4 in component INaK (dimensionless)"
        legend_algebraic[193] = "b4 in component INaK (dimensionless)"
        legend_algebraic[194] = "x1 in component INaK (dimensionless)"
        legend_algebraic[195] = "x2 in component INaK (dimensionless)"
        legend_algebraic[196] = "x3 in component INaK (dimensionless)"
        legend_algebraic[197] = "x4 in component INaK (dimensionless)"
        legend_algebraic[198] = "E1 in component INaK (dimensionless)"
        legend_algebraic[199] = "E2 in component INaK (dimensionless)"
        legend_algebraic[200] = "E3 in component INaK (dimensionless)"
        legend_algebraic[201] = "E4 in component INaK (dimensionless)"
        legend_algebraic[202] = "JnakNa in component INaK (millimolar_per_millisecond)"
        legend_algebraic[203] = "JnakK in component INaK (millimolar_per_millisecond)"
        legend_algebraic[205] = "xkb in component IKb (dimensionless)"
        legend_constants[95] = "GKb_b in component IKb (milliS_per_microF)"
        legend_constants[125] = "GKb in component IKb (milliS_per_microF)"
        legend_constants[96] = "PNab in component INab (milliS_per_microF)"
        legend_constants[97] = "PCab in component ICab (milliS_per_microF)"
        legend_constants[98] = "GpCa in component IpCa (milliS_per_microF)"
        legend_constants[99] = "KmCap in component IpCa (millimolar)"
        legend_constants[100] = "GClCa in component ICl (milliS_per_microF)"
        legend_constants[101] = "GClb in component ICl (milliS_per_microF)"
        legend_constants[102] = "KdClCa in component ICl (millimolar)"
        legend_constants[103] = "Fjunc in component ICl (dimensionless)"
        legend_constants[104] = "tauNa in component diff (millisecond)"
        legend_constants[105] = "tauK in component diff (millisecond)"
        legend_constants[106] = "tauCa in component diff (millisecond)"
        legend_constants[107] = "tauCl in component diff (millisecond)"
        legend_constants[108] = "bt in component ryr (millisecond)"
        legend_constants[126] = "a_rel in component ryr (millimolar_per_millisecond)"
        legend_algebraic[95] = (
            "Jrel_inf_b in component ryr (millimolar_per_millisecond)"
        )
        legend_algebraic[98] = "Jrel_inf in component ryr (millimolar_per_millisecond)"
        legend_algebraic[101] = "tau_rel_b in component ryr (millisecond)"
        legend_algebraic[104] = "tau_rel in component ryr (millisecond)"
        legend_states[43] = "Jrel_np in component ryr (millimolar_per_millisecond)"
        legend_constants[127] = "btp in component ryr (millisecond)"
        legend_constants[134] = "a_relp in component ryr (millimolar_per_millisecond)"
        legend_algebraic[96] = (
            "Jrel_infp_b in component ryr (millimolar_per_millisecond)"
        )
        legend_algebraic[99] = "Jrel_infp in component ryr (millimolar_per_millisecond)"
        legend_algebraic[102] = "tau_relp_b in component ryr (millisecond)"
        legend_algebraic[105] = "tau_relp in component ryr (millisecond)"
        legend_states[44] = "Jrel_p in component ryr (millimolar_per_millisecond)"
        legend_constants[109] = "cajsr_half in component ryr (millimolar)"
        legend_algebraic[215] = "fJrelp in component ryr (dimensionless)"
        legend_constants[110] = "Jrel_b in component ryr (dimensionless)"
        legend_constants[128] = "upScale in component SERCA (dimensionless)"
        legend_algebraic[219] = "Jupnp in component SERCA (millimolar_per_millisecond)"
        legend_algebraic[220] = "Jupp in component SERCA (millimolar_per_millisecond)"
        legend_algebraic[222] = "fJupp in component SERCA (dimensionless)"
        legend_algebraic[223] = "Jleak in component SERCA (millimolar_per_millisecond)"
        legend_constants[111] = "Jup_b in component SERCA (dimensionless)"
        legend_rates[0] = "d/dt v in component membrane (millivolt)"
        legend_rates[1] = "d/dt CaMKt in component CaMK (millimolar)"
        legend_rates[3] = "d/dt nai in component intracellular_ions (millimolar)"
        legend_rates[4] = "d/dt nass in component intracellular_ions (millimolar)"
        legend_rates[5] = "d/dt ki in component intracellular_ions (millimolar)"
        legend_rates[6] = "d/dt kss in component intracellular_ions (millimolar)"
        legend_rates[10] = "d/dt cli in component intracellular_ions (millimolar)"
        legend_rates[11] = "d/dt clss in component intracellular_ions (millimolar)"
        legend_rates[9] = "d/dt cai in component intracellular_ions (millimolar)"
        legend_rates[2] = "d/dt cass in component intracellular_ions (millimolar)"
        legend_rates[7] = "d/dt cansr in component intracellular_ions (millimolar)"
        legend_rates[8] = "d/dt cajsr in component intracellular_ions (millimolar)"
        legend_rates[12] = "d/dt m in component INa (dimensionless)"
        legend_rates[13] = "d/dt h in component INa (dimensionless)"
        legend_rates[14] = "d/dt j in component INa (dimensionless)"
        legend_rates[15] = "d/dt hp in component INa (dimensionless)"
        legend_rates[16] = "d/dt jp in component INa (dimensionless)"
        legend_rates[17] = "d/dt mL in component INaL (dimensionless)"
        legend_rates[18] = "d/dt hL in component INaL (dimensionless)"
        legend_rates[19] = "d/dt hLp in component INaL (dimensionless)"
        legend_rates[20] = "d/dt a in component Ito (dimensionless)"
        legend_rates[21] = "d/dt iF in component Ito (dimensionless)"
        legend_rates[22] = "d/dt iS in component Ito (dimensionless)"
        legend_rates[23] = "d/dt ap in component Ito (dimensionless)"
        legend_rates[24] = "d/dt iFp in component Ito (dimensionless)"
        legend_rates[25] = "d/dt iSp in component Ito (dimensionless)"
        legend_rates[26] = "d/dt d in component ICaL (dimensionless)"
        legend_rates[27] = "d/dt ff in component ICaL (dimensionless)"
        legend_rates[28] = "d/dt fs in component ICaL (dimensionless)"
        legend_rates[29] = "d/dt fcaf in component ICaL (dimensionless)"
        legend_rates[30] = "d/dt fcas in component ICaL (dimensionless)"
        legend_rates[31] = "d/dt jca in component ICaL (dimensionless)"
        legend_rates[32] = "d/dt ffp in component ICaL (dimensionless)"
        legend_rates[33] = "d/dt fcafp in component ICaL (dimensionless)"
        legend_rates[34] = "d/dt nca_ss in component ICaL (dimensionless)"
        legend_rates[35] = "d/dt nca_i in component ICaL (dimensionless)"
        legend_rates[38] = "d/dt C3 in component IKr (dimensionless)"
        legend_rates[37] = "d/dt C2 in component IKr (dimensionless)"
        legend_rates[36] = "d/dt C1 in component IKr (dimensionless)"
        legend_rates[40] = "d/dt O in component IKr (dimensionless)"
        legend_rates[39] = "d/dt I in component IKr (dimensionless)"
        legend_rates[41] = "d/dt xs1 in component IKs (dimensionless)"
        legend_rates[42] = "d/dt xs2 in component IKs (dimensionless)"
        legend_rates[43] = "d/dt Jrel_np in component ryr (millimolar_per_millisecond)"
        legend_rates[44] = "d/dt Jrel_p in component ryr (millimolar_per_millisecond)"
        self.legend_states = legend_states
        self.legend_algebraic = legend_algebraic
        self.legend_voi = legend_voi
        self.legend_constants = legend_constants
        self.legend_rates = legend_rates

    def computeRates(self, voi, states, constants):
        return njitComputeRates(
            voi,
            states,
            constants,
            self.sizeStates,
            self.sizeAlgebraic,
            len(self.channel_list),
            self.init_algebraic[-len(self.channel_list) :],
        )

    def computeAlgebraic(self, voi, states, constants):
        algebraic = np.array([[0.0] * len(voi)] * self.sizeAlgebraic)
        states = np.array(states)
        voi = np.array(voi)
        num_channels = len(self.channel_list)
        if self.drug is not None:
            # Concentration is held constant, so the multipliers computed once
            # in init_drug apply at every time point.
            algebraic[-num_channels:] = self.init_algebraic[-num_channels:].reshape(
                (num_channels, 1)
            )
        else:
            algebraic[-num_channels:] = np.ones((num_channels, len(voi)))
        algebraic[3] = 1.00000 / (1.00000 + np.exp((states[0] + 87.6100) / 7.48800))
        algebraic[4] = 1.00000 / (1.00000 + np.exp((states[0] + 93.8100) / 7.48800))
        algebraic[9] = 1.00000 / (1.00000 + np.exp((states[0] + 18.0800) / 2.79160))
        algebraic[0] = 1.00000 / (
            np.power(1.00000 + np.exp(-(states[0] + 56.8600) / 9.03000), 2.00000)
        )
        algebraic[13] = 0.129200 * np.exp(
            -(np.power((states[0] + 45.7900) / 15.5400, 2.00000))
        ) + 0.0648700 * np.exp(-(np.power((states[0] - 4.82300) / 51.1200, 2.00000)))
        algebraic[2] = 1.00000 / (1.00000 + np.exp(-(states[0] + 42.8500) / 5.26400))
        algebraic[16] = 0.129200 * np.exp(
            -(np.power((states[0] + 45.7900) / 15.5400, 2.00000))
        ) + 0.0648700 * np.exp(-(np.power((states[0] - 4.82300) / 51.1200, 2.00000)))
        algebraic[5] = 1.00000 / (
            1.00000 + np.exp(-((states[0] + constants[44]) - 14.3400) / 14.8200)
        )
        algebraic[17] = 1.05150 / (
            1.00000
            / (
                1.20890
                * (1.00000 + np.exp(-((states[0] + constants[44]) - 18.4099) / 29.3814))
            )
            + 3.50000
            / (1.00000 + np.exp((states[0] + constants[44] + 100.000) / 29.3814))
        )
        algebraic[7] = super().custom_piecewise(
            [
                np.greater_equal(states[0], 31.4978),
                1.00000,
                True,
                1.07630 * np.exp(-1.00700 * np.exp(-0.0829000 * states[0])),
            ]
        )
        algebraic[22] = (
            constants[51]
            + 0.600000
            + 1.00000
            / (
                np.exp(-0.0500000 * (states[0] + constants[50] + 6.00000))
                + np.exp(0.0900000 * (states[0] + constants[50] + 14.0000))
            )
        )
        algebraic[8] = 1.00000 / (1.00000 + np.exp((states[0] + 19.5800) / 3.69600))
        algebraic[23] = 7.00000 + 1.00000 / (
            0.00450000 * np.exp(-(states[0] + 20.0000) / 10.0000)
            + 0.00450000 * np.exp((states[0] + 20.0000) / 10.0000)
        )
        algebraic[24] = 1000.00 + 1.00000 / (
            3.50000e-05 * np.exp(-(states[0] + 5.00000) / 4.00000)
            + 3.50000e-05 * np.exp((states[0] + 5.00000) / 6.00000)
        )
        algebraic[10] = states[31] * 1.00000
        algebraic[20] = 1.00000 / (
            constants[46] / algebraic[10]
            + np.power(1.00000 + constants[45] / states[2], 4.00000)
        )
        algebraic[21] = 1.00000 / (
            constants[46] / algebraic[10]
            + np.power(1.00000 + constants[45] / states[9], 4.00000)
        )
        algebraic[12] = 1.00000 / (1.00000 + np.exp(-(states[0] + 11.6000) / 8.93200))
        algebraic[27] = 817.300 + 1.00000 / (
            0.000232600 * np.exp((states[0] + 48.2800) / 17.8000)
            + 0.00129200 * np.exp(-(states[0] + 210.000) / 230.000)
        )
        algebraic[32] = 1.00000 / (
            1.00000 + np.exp(-((states[0] + constants[44]) - 24.3400) / 14.8200)
        )
        algebraic[19] = algebraic[8]
        algebraic[33] = 7.00000 + 1.00000 / (
            0.0400000 * np.exp(-(states[0] - 4.00000) / 7.00000)
            + 0.0400000 * np.exp((states[0] - 4.00000) / 7.00000)
        )
        algebraic[34] = 100.000 + 1.00000 / (
            0.000120000 * np.exp(-states[0] / 3.00000)
            + 0.000120000 * np.exp(states[0] / 7.00000)
        )
        algebraic[35] = 2.50000 * algebraic[23]
        algebraic[26] = algebraic[12]
        algebraic[36] = 1.00000 / (
            0.0100000 * np.exp((states[0] - 50.0000) / 20.0000)
            + 0.0193000 * np.exp(-(states[0] + 66.5400) / 31.0000)
        )
        algebraic[43] = (constants[22] * (1.00000 - states[1])) / (
            1.00000 + constants[23] / states[2]
        )
        algebraic[1] = 1.00000 / (
            np.power(1.00000 + np.exp((states[0] + 71.5500) / 7.43000), 2.00000)
        )
        algebraic[14] = super().custom_piecewise(
            [
                np.greater_equal(states[0], -40.0000),
                0.00000,
                True,
                0.0570000 * np.exp(-(states[0] + 80.0000) / 6.80000),
            ]
        )
        algebraic[29] = super().custom_piecewise(
            [
                np.greater_equal(states[0], -40.0000),
                0.770000
                / (0.130000 * (1.00000 + np.exp(-(states[0] + 10.6600) / 11.1000))),
                True,
                2.70000 * np.exp(0.0790000 * states[0])
                + 310000.0 * np.exp(0.348500 * states[0]),
            ]
        )
        algebraic[37] = 1.00000 / (algebraic[14] + algebraic[29])
        algebraic[40] = 2.50000 * algebraic[33]
        algebraic[38] = algebraic[1]
        algebraic[15] = super().custom_piecewise(
            [
                np.greater_equal(states[0], -40.0000),
                0.00000,
                True,
                (
                    (
                        -25428.0 * np.exp(0.244400 * states[0])
                        - 6.94800e-06 * np.exp(-0.0439100 * states[0])
                    )
                    * (states[0] + 37.7800)
                )
                / (1.00000 + np.exp(0.311000 * (states[0] + 79.2300))),
            ]
        )
        algebraic[30] = super().custom_piecewise(
            [
                np.greater_equal(states[0], -40.0000),
                (0.600000 * np.exp(0.0570000 * states[0]))
                / (1.00000 + np.exp(-0.100000 * (states[0] + 32.0000))),
                True,
                (0.0242400 * np.exp(-0.0105200 * states[0]))
                / (1.00000 + np.exp(-0.137800 * (states[0] + 40.1400))),
            ]
        )
        algebraic[44] = 1.00000 / (algebraic[15] + algebraic[30])
        algebraic[45] = 1.00000 / (
            np.power(1.00000 + np.exp((states[0] + 77.5500) / 7.43000), 2.00000)
        )
        algebraic[6] = 1.00000 / (
            1.00000 + np.exp((states[0] + constants[44] + 43.9400) / 5.71100)
        )
        algebraic[18] = super().custom_piecewise(
            [
                np.equal(constants[0], 1.00000),
                1.00000
                - 0.950000
                / (1.00000 + np.exp((states[0] + constants[44] + 70.0000) / 5.00000)),
                True,
                1.00000,
            ]
        )
        algebraic[31] = 4.56200 + 1.00000 / (
            0.393300 * np.exp(-(states[0] + constants[44] + 100.000) / 100.000)
            + 0.0800400 * np.exp((states[0] + constants[44] + 50.0000) / 16.5900)
        )
        algebraic[46] = algebraic[31] * algebraic[18]
        algebraic[28] = (states[0] * constants[7]) / (constants[5] * constants[6])
        algebraic[41] = 0.116100 * np.exp(0.299000 * algebraic[28])
        algebraic[47] = 0.244200 * np.exp(-1.60400 * algebraic[28])
        algebraic[50] = 1.46000 * algebraic[44]
        algebraic[39] = 23.6200 + 1.00000 / (
            0.00141600 * np.exp(-(states[0] + constants[44] + 96.5200) / 59.0500)
            + 1.78000e-08 * np.exp((states[0] + constants[44] + 114.100) / 8.07900)
        )
        algebraic[51] = algebraic[39] * algebraic[18]
        algebraic[42] = 0.0578000 * np.exp(0.971000 * algebraic[28])
        algebraic[48] = 0.000349000 * np.exp(-1.06200 * algebraic[28])
        algebraic[52] = 0.253300 * np.exp(0.595300 * algebraic[28])
        algebraic[55] = 0.0652500 * np.exp(-0.820900 * algebraic[28])
        algebraic[54] = 1.35400 + 0.000100000 / (
            np.exp(((states[0] + constants[44]) - 167.400) / 15.8900)
            + np.exp(-((states[0] + constants[44]) - 12.2300) / 0.215400)
        )
        algebraic[57] = 1.00000 - 0.500000 / (
            1.00000 + np.exp((states[0] + constants[44] + 70.0000) / 20.0000)
        )
        algebraic[60] = algebraic[54] * algebraic[57] * algebraic[46]
        algebraic[61] = algebraic[54] * algebraic[57] * algebraic[51]
        algebraic[58] = 5.20000e-05 * np.exp(1.52500 * algebraic[28])
        algebraic[62] = (algebraic[48] * algebraic[55] * algebraic[58]) / (
            algebraic[42] * algebraic[52]
        )
        algebraic[79] = constants[48] * states[27] + constants[119] * states[28]
        algebraic[80] = 0.300000 + 0.600000 / (
            1.00000 + np.exp((states[0] - 10.0000) / 10.0000)
        )
        algebraic[81] = 1.00000 - algebraic[80]
        algebraic[82] = algebraic[80] * states[29] + algebraic[81] * states[30]
        algebraic[83] = constants[48] * states[32] + constants[119] * states[28]
        algebraic[84] = algebraic[80] * states[33] + algebraic[81] * states[30]
        algebraic[25] = (states[0] * constants[7] * constants[7]) / (
            constants[5] * constants[6]
        )
        algebraic[85] = (
            0.500000 * (states[4] + states[6] + states[11] + 4.00000 * states[2])
        ) / 1000.00
        algebraic[86] = np.exp(
            -constants[133]
            * 4.00000
            * (
                (np.power(algebraic[85], 1.0 / 2))
                / (1.00000 + np.power(algebraic[85], 1.0 / 2))
                - 0.300000 * algebraic[85]
            )
        )
        algebraic[89] = (
            4.00000
            * algebraic[25]
            * (
                algebraic[86] * states[2] * np.exp(2.00000 * algebraic[28])
                - constants[138] * constants[2]
            )
        ) / (np.exp(2.00000 * algebraic[28]) - 1.00000)
        algebraic[49] = algebraic[43] + states[1]
        algebraic[92] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
        algebraic[93] = (
            algebraic[-7]
            * constants[53]
            * (
                (1.00000 - algebraic[92])
                * constants[120]
                * algebraic[89]
                * states[26]
                * (
                    algebraic[79] * (1.00000 - states[34])
                    + states[31] * algebraic[82] * states[34]
                )
                + algebraic[92]
                * constants[130]
                * algebraic[89]
                * states[26]
                * (
                    algebraic[83] * (1.00000 - states[34])
                    + states[31] * algebraic[84] * states[34]
                )
            )
        )
        algebraic[95] = ((-constants[126] * algebraic[93]) / 1.00000) / (
            1.00000 + np.power(constants[109] / states[8], 8.00000)
        )
        algebraic[98] = super().custom_piecewise(
            [
                np.equal(constants[0], 2.00000),
                algebraic[95] * 1.70000,
                True,
                algebraic[95],
            ]
        )
        algebraic[101] = constants[108] / (1.00000 + 0.0123000 / states[8])
        algebraic[104] = super().custom_piecewise(
            [np.less(algebraic[101], 0.00100000), 0.00100000, True, algebraic[101]]
        )
        algebraic[96] = ((-constants[134] * algebraic[93]) / 1.00000) / (
            1.00000 + np.power(constants[109] / states[8], 8.00000)
        )
        algebraic[99] = super().custom_piecewise(
            [
                np.equal(constants[0], 2.00000),
                algebraic[96] * 1.70000,
                True,
                algebraic[96],
            ]
        )
        algebraic[102] = constants[127] / (1.00000 + 0.0123000 / states[8])
        algebraic[105] = super().custom_piecewise(
            [np.less(algebraic[102], 0.00100000), 0.00100000, True, algebraic[102]]
        )
        algebraic[64] = (
            (constants[5] * constants[6]) / (constants[10] * constants[7])
        ) * np.log(constants[3] / states[5])
        algebraic[73] = 1.00000 / (
            1.00000 + np.exp(((states[0] + constants[44]) - 213.600) / 151.200)
        )
        algebraic[74] = 1.00000 - algebraic[73]
        algebraic[75] = algebraic[73] * states[21] + algebraic[74] * states[22]
        algebraic[76] = algebraic[73] * states[24] + algebraic[74] * states[25]
        algebraic[77] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
        algebraic[78] = (
            algebraic[-2]
            * constants[118]
            * (states[0] - algebraic[64])
            * (
                (1.00000 - algebraic[77]) * states[20] * algebraic[75]
                + algebraic[77] * states[23] * algebraic[76]
            )
        )
        algebraic[117] = (
            algebraic[-1]
            * constants[122]
            * (np.power(constants[3] / 5.00000, 1.0 / 2))
            * states[40]
            * (states[0] - algebraic[64])
        )
        algebraic[65] = (
            (constants[5] * constants[6]) / (constants[10] * constants[7])
        ) * np.log(
            (constants[3] + constants[34] * constants[1])
            / (states[5] + constants[34] * states[3])
        )
        algebraic[118] = 1.00000 + 0.600000 / (
            1.00000 + np.power(3.80000e-05 / states[9], 1.40000)
        )
        algebraic[119] = (
            algebraic[-5]
            * constants[123]
            * algebraic[118]
            * states[41]
            * states[42]
            * (states[0] - algebraic[65])
        )
        algebraic[120] = 4.09400 / (
            1.00000 + np.exp(0.121700 * ((states[0] - algebraic[64]) - 49.9340))
        )
        algebraic[121] = (
            15.7200 * np.exp(0.0674000 * ((states[0] - algebraic[64]) - 3.25700))
            + np.exp(0.0618000 * ((states[0] - algebraic[64]) - 594.310))
        ) / (1.00000 + np.exp(-0.162900 * ((states[0] - algebraic[64]) + 14.2070)))
        algebraic[122] = algebraic[120] / (algebraic[120] + algebraic[121])
        algebraic[123] = (
            algebraic[-6]
            * constants[124]
            * (np.power(constants[3] / 5.00000, 1.0 / 2))
            * algebraic[122]
            * (states[0] - algebraic[64])
        )
        algebraic[187] = constants[82] * np.exp(
            ((1.00000 - constants[83]) * algebraic[28]) / 3.00000
        )
        algebraic[191] = (
            constants[77] * (np.power(constants[3] / constants[85], 2.00000))
        ) / (
            (
                np.power(1.00000 + constants[1] / algebraic[187], 3.00000)
                + np.power(1.00000 + constants[3] / constants[85], 2.00000)
            )
            - 1.00000
        )
        algebraic[188] = constants[90] / (
            1.00000
            + constants[89] / constants[91]
            + states[3] / constants[92]
            + states[5] / constants[93]
        )
        algebraic[192] = (constants[78] * algebraic[188] * constants[89]) / (
            1.00000 + constants[87] / constants[88]
        )
        algebraic[186] = constants[81] * np.exp(
            (constants[83] * algebraic[28]) / 3.00000
        )
        algebraic[189] = (
            constants[73] * (np.power(states[3] / algebraic[186], 3.00000))
        ) / (
            (
                np.power(1.00000 + states[3] / algebraic[186], 3.00000)
                + np.power(1.00000 + states[5] / constants[84], 2.00000)
            )
            - 1.00000
        )
        algebraic[190] = (
            constants[76] * (np.power(constants[1] / algebraic[187], 3.00000))
        ) / (
            (
                np.power(1.00000 + constants[1] / algebraic[187], 3.00000)
                + np.power(1.00000 + constants[3] / constants[85], 2.00000)
            )
            - 1.00000
        )
        algebraic[193] = (
            constants[80] * (np.power(states[5] / constants[84], 2.00000))
        ) / (
            (
                np.power(1.00000 + states[3] / algebraic[186], 3.00000)
                + np.power(1.00000 + states[5] / constants[84], 2.00000)
            )
            - 1.00000
        )
        algebraic[194] = (
            constants[160] * algebraic[189] * constants[159]
            + algebraic[190] * algebraic[193] * algebraic[192]
            + constants[159] * algebraic[193] * algebraic[192]
            + algebraic[192] * algebraic[189] * constants[159]
        )
        algebraic[195] = (
            algebraic[190] * constants[158] * algebraic[193]
            + algebraic[189] * constants[159] * algebraic[191]
            + algebraic[191] * constants[158] * algebraic[193]
            + constants[159] * algebraic[191] * algebraic[193]
        )
        algebraic[196] = (
            constants[159] * algebraic[191] * constants[160]
            + algebraic[192] * algebraic[190] * constants[158]
            + algebraic[190] * constants[158] * constants[160]
            + algebraic[191] * constants[160] * constants[158]
        )
        algebraic[197] = (
            algebraic[193] * algebraic[192] * algebraic[190]
            + algebraic[191] * constants[160] * algebraic[189]
            + algebraic[190] * constants[160] * algebraic[189]
            + algebraic[192] * algebraic[190] * algebraic[189]
        )
        algebraic[198] = algebraic[194] / (
            algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
        )
        algebraic[199] = algebraic[195] / (
            algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
        )
        algebraic[202] = 3.00000 * (
            algebraic[198] * algebraic[191] - algebraic[199] * algebraic[192]
        )
        algebraic[200] = algebraic[196] / (
            algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
        )
        algebraic[201] = algebraic[197] / (
            algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
        )
        algebraic[203] = 2.00000 * (
            algebraic[201] * constants[158] - algebraic[200] * algebraic[189]
        )
        algebraic[204] = constants[161] * (
            constants[8] * algebraic[202] + constants[10] * algebraic[203]
        )
        algebraic[205] = 1.00000 / (1.00000 + np.exp(-(states[0] - 10.8968) / 23.9871))
        algebraic[206] = constants[125] * algebraic[205] * (states[0] - algebraic[64])
        algebraic[68] = (
            constants[36]
            * constants[35]
            * constants[114]
            * constants[115]
            * (states[0] - algebraic[64])
        )
        algebraic[11] = super().custom_piecewise(
            [
                np.greater_equal(voi, constants[14])
                & np.less_equal(
                    (voi - constants[14])
                    - np.floor((voi - constants[14]) / constants[17]) * constants[17],
                    constants[18],
                ),
                constants[16],
                True,
                0.00000,
            ]
        )
        algebraic[100] = (
            0.500000 * (states[3] + states[5] + states[10] + 4.00000 * states[9])
        ) / 1000.00
        algebraic[107] = np.exp(
            -constants[133]
            * 1.00000
            * (
                (np.power(algebraic[100], 1.0 / 2))
                / (1.00000 + np.power(algebraic[100], 1.0 / 2))
                - 0.300000 * algebraic[100]
            )
        )
        algebraic[110] = (
            1.00000
            * algebraic[25]
            * (
                algebraic[107] * states[5] * np.exp(1.00000 * algebraic[28])
                - constants[140] * constants[3]
            )
        ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
        algebraic[113] = (1.00000 - constants[53]) * (
            (1.00000 - algebraic[92])
            * constants[132]
            * algebraic[110]
            * states[26]
            * (
                algebraic[79] * (1.00000 - states[35])
                + states[31] * algebraic[82] * states[35]
            )
            + algebraic[92]
            * constants[137]
            * algebraic[110]
            * states[26]
            * (
                algebraic[83] * (1.00000 - states[35])
                + states[31] * algebraic[84] * states[35]
            )
        )
        algebraic[208] = (states[6] - states[5]) / constants[105]
        algebraic[88] = np.exp(
            -constants[133]
            * 1.00000
            * (
                (np.power(algebraic[85], 1.0 / 2))
                / (1.00000 + np.power(algebraic[85], 1.0 / 2))
                - 0.300000 * algebraic[85]
            )
        )
        algebraic[91] = (
            1.00000
            * algebraic[25]
            * (
                algebraic[88] * states[6] * np.exp(1.00000 * algebraic[28])
                - constants[140] * constants[3]
            )
        ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
        algebraic[97] = constants[53] * (
            (1.00000 - algebraic[92])
            * constants[132]
            * algebraic[91]
            * states[26]
            * (
                algebraic[79] * (1.00000 - states[34])
                + states[31] * algebraic[82] * states[34]
            )
            + algebraic[92]
            * constants[137]
            * algebraic[91]
            * states[26]
            * (
                algebraic[83] * (1.00000 - states[34])
                + states[31] * algebraic[84] * states[34]
            )
        )
        algebraic[63] = (
            (constants[5] * constants[6]) / (constants[8] * constants[7])
        ) * np.log(constants[1] / states[3])
        algebraic[69] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
        algebraic[70] = (
            algebraic[-4]
            * constants[40]
            * (states[0] - algebraic[63])
            * (np.power(states[12], 3.00000))
            * (
                (1.00000 - algebraic[69]) * states[13] * states[14]
                + algebraic[69] * states[15] * states[16]
            )
        )
        algebraic[71] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
        algebraic[72] = (
            algebraic[-3]
            * constants[117]
            * (states[0] - algebraic[63])
            * states[17]
            * ((1.00000 - algebraic[71]) * states[18] + algebraic[71] * states[19])
        )
        algebraic[152] = 1.00000 / (
            1.00000 + np.power(constants[71] / states[9], 2.00000)
        )
        algebraic[125] = np.exp(constants[69] * algebraic[28])
        algebraic[132] = 1.00000 + (constants[1] / constants[62]) * (
            1.00000 + 1.00000 / algebraic[125]
        )
        algebraic[133] = constants[1] / (
            constants[62] * algebraic[125] * algebraic[132]
        )
        algebraic[136] = algebraic[133] * constants[66]
        algebraic[126] = 1.00000 + (states[3] / constants[62]) * (
            1.00000 + algebraic[125]
        )
        algebraic[127] = (states[3] * algebraic[125]) / (constants[62] * algebraic[126])
        algebraic[139] = algebraic[127] * constants[66]
        algebraic[129] = 1.00000 + (states[3] / constants[60]) * (
            1.00000 + states[3] / constants[61]
        )
        algebraic[130] = (states[3] * states[3]) / (
            algebraic[129] * constants[60] * constants[61]
        )
        algebraic[142] = algebraic[130] * algebraic[127] * constants[64]
        algebraic[143] = algebraic[133] * constants[146] * constants[64]
        algebraic[134] = 1.00000 / algebraic[132]
        algebraic[135] = algebraic[134] * constants[65]
        algebraic[137] = algebraic[135] + algebraic[136]
        algebraic[124] = np.exp(constants[70] * algebraic[28])
        algebraic[128] = 1.00000 / algebraic[126]
        algebraic[138] = (algebraic[128] * constants[65]) / algebraic[124]
        algebraic[140] = algebraic[138] + algebraic[139]
        algebraic[131] = 1.00000 / algebraic[129]
        algebraic[141] = algebraic[131] * states[9] * constants[67]
        algebraic[144] = constants[149] * algebraic[140] * (
            algebraic[142] + algebraic[141]
        ) + constants[150] * algebraic[142] * (constants[149] + algebraic[137])
        algebraic[145] = constants[148] * algebraic[142] * (
            algebraic[140] + constants[150]
        ) + algebraic[140] * algebraic[141] * (constants[148] + algebraic[143])
        algebraic[146] = constants[148] * algebraic[137] * (
            algebraic[142] + algebraic[141]
        ) + algebraic[143] * algebraic[141] * (constants[149] + algebraic[137])
        algebraic[147] = constants[149] * algebraic[143] * (
            algebraic[140] + constants[150]
        ) + algebraic[137] * constants[150] * (constants[148] + algebraic[143])
        algebraic[148] = algebraic[144] / (
            algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
        )
        algebraic[149] = algebraic[145] / (
            algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
        )
        algebraic[150] = algebraic[146] / (
            algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
        )
        algebraic[151] = algebraic[147] / (
            algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
        )
        algebraic[153] = (
            3.00000
            * (algebraic[151] * algebraic[142] - algebraic[148] * algebraic[143])
            + algebraic[150] * algebraic[139]
        ) - algebraic[149] * algebraic[136]
        algebraic[154] = (
            algebraic[149] * constants[149] - algebraic[148] * constants[148]
        )
        algebraic[155] = (
            (1.00000 - constants[59])
            * constants[151]
            * algebraic[152]
            * (constants[8] * algebraic[153] + constants[9] * algebraic[154])
        )
        algebraic[207] = (
            constants[96]
            * algebraic[25]
            * (states[3] * np.exp(algebraic[28]) - constants[1])
        ) / (np.exp(algebraic[28]) - 1.00000)
        algebraic[106] = np.exp(
            -constants[133]
            * 1.00000
            * (
                (np.power(algebraic[100], 1.0 / 2))
                / (1.00000 + np.power(algebraic[100], 1.0 / 2))
                - 0.300000 * algebraic[100]
            )
        )
        algebraic[109] = (
            1.00000
            * algebraic[25]
            * (
                algebraic[106] * states[3] * np.exp(1.00000 * algebraic[28])
                - constants[139] * constants[1]
            )
        ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
        algebraic[112] = (1.00000 - constants[53]) * (
            (1.00000 - algebraic[92])
            * constants[131]
            * algebraic[109]
            * states[26]
            * (
                algebraic[79] * (1.00000 - states[35])
                + states[31] * algebraic[82] * states[35]
            )
            + algebraic[92]
            * constants[136]
            * algebraic[109]
            * states[26]
            * (
                algebraic[83] * (1.00000 - states[35])
                + states[31] * algebraic[84] * states[35]
            )
        )
        algebraic[210] = (states[4] - states[3]) / constants[104]
        algebraic[182] = 1.00000 / (
            1.00000 + np.power(constants[71] / states[2], 2.00000)
        )
        algebraic[162] = 1.00000 + (constants[1] / constants[62]) * (
            1.00000 + 1.00000 / algebraic[125]
        )
        algebraic[163] = constants[1] / (
            constants[62] * algebraic[125] * algebraic[162]
        )
        algebraic[166] = algebraic[163] * constants[66]
        algebraic[156] = 1.00000 + (states[4] / constants[62]) * (
            1.00000 + algebraic[125]
        )
        algebraic[157] = (states[4] * algebraic[125]) / (constants[62] * algebraic[156])
        algebraic[169] = algebraic[157] * constants[66]
        algebraic[159] = 1.00000 + (states[4] / constants[60]) * (
            1.00000 + states[4] / constants[61]
        )
        algebraic[160] = (states[4] * states[4]) / (
            algebraic[159] * constants[60] * constants[61]
        )
        algebraic[172] = algebraic[160] * algebraic[157] * constants[64]
        algebraic[173] = algebraic[163] * constants[153] * constants[64]
        algebraic[164] = 1.00000 / algebraic[162]
        algebraic[165] = algebraic[164] * constants[65]
        algebraic[167] = algebraic[165] + algebraic[166]
        algebraic[158] = 1.00000 / algebraic[156]
        algebraic[168] = (algebraic[158] * constants[65]) / algebraic[124]
        algebraic[170] = algebraic[168] + algebraic[169]
        algebraic[161] = 1.00000 / algebraic[159]
        algebraic[171] = algebraic[161] * states[2] * constants[67]
        algebraic[174] = constants[156] * algebraic[170] * (
            algebraic[172] + algebraic[171]
        ) + constants[157] * algebraic[172] * (constants[156] + algebraic[167])
        algebraic[175] = constants[155] * algebraic[172] * (
            algebraic[170] + constants[157]
        ) + algebraic[170] * algebraic[171] * (constants[155] + algebraic[173])
        algebraic[176] = constants[155] * algebraic[167] * (
            algebraic[172] + algebraic[171]
        ) + algebraic[173] * algebraic[171] * (constants[156] + algebraic[167])
        algebraic[177] = constants[156] * algebraic[173] * (
            algebraic[170] + constants[157]
        ) + algebraic[167] * constants[157] * (constants[155] + algebraic[173])
        algebraic[178] = algebraic[174] / (
            algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
        )
        algebraic[179] = algebraic[175] / (
            algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
        )
        algebraic[180] = algebraic[176] / (
            algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
        )
        algebraic[181] = algebraic[177] / (
            algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
        )
        algebraic[183] = (
            3.00000
            * (algebraic[181] * algebraic[172] - algebraic[178] * algebraic[173])
            + algebraic[180] * algebraic[169]
        ) - algebraic[179] * algebraic[166]
        algebraic[184] = (
            algebraic[179] * constants[156] - algebraic[178] * constants[155]
        )
        algebraic[185] = (
            constants[59]
            * constants[151]
            * algebraic[182]
            * (constants[8] * algebraic[183] + constants[9] * algebraic[184])
        )
        algebraic[87] = np.exp(
            -constants[133]
            * 1.00000
            * (
                (np.power(algebraic[85], 1.0 / 2))
                / (1.00000 + np.power(algebraic[85], 1.0 / 2))
                - 0.300000 * algebraic[85]
            )
        )
        algebraic[90] = (
            1.00000
            * algebraic[25]
            * (
                algebraic[87] * states[4] * np.exp(1.00000 * algebraic[28])
                - constants[139] * constants[1]
            )
        ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
        algebraic[94] = constants[53] * (
            (1.00000 - algebraic[92])
            * constants[131]
            * algebraic[90]
            * states[26]
            * (
                algebraic[79] * (1.00000 - states[34])
                + states[31] * algebraic[82] * states[34]
            )
            + algebraic[92]
            * constants[136]
            * algebraic[90]
            * states[26]
            * (
                algebraic[83] * (1.00000 - states[34])
                + states[31] * algebraic[84] * states[34]
            )
        )
        algebraic[213] = (states[2] - states[9]) / constants[106]
        algebraic[215] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
        algebraic[217] = constants[110] * (
            (1.00000 - algebraic[215]) * states[43] + algebraic[215] * states[44]
        )
        algebraic[56] = 1.00000 / (
            1.00000
            + (constants[28] * constants[29])
            / (np.power(constants[29] + states[2], 2.00000))
            + (constants[30] * constants[31])
            / (np.power(constants[31] + states[2], 2.00000))
        )
        algebraic[103] = np.exp(
            -constants[133]
            * 4.00000
            * (
                (np.power(algebraic[100], 1.0 / 2))
                / (1.00000 + np.power(algebraic[100], 1.0 / 2))
                - 0.300000 * algebraic[100]
            )
        )
        algebraic[108] = (
            4.00000
            * algebraic[25]
            * (
                algebraic[103] * states[9] * np.exp(2.00000 * algebraic[28])
                - constants[138] * constants[2]
            )
        ) / (np.exp(2.00000 * algebraic[28]) - 1.00000)
        algebraic[111] = (
            algebraic[-7]
            * (1.00000 - constants[53])
            * (
                (1.00000 - algebraic[92])
                * constants[120]
                * algebraic[108]
                * states[26]
                * (
                    algebraic[79] * (1.00000 - states[35])
                    + states[31] * algebraic[82] * states[35]
                )
                + algebraic[92]
                * constants[130]
                * algebraic[108]
                * states[26]
                * (
                    algebraic[83] * (1.00000 - states[35])
                    + states[31] * algebraic[84] * states[35]
                )
            )
        )
        algebraic[114] = algebraic[93] + algebraic[111]
        algebraic[115] = algebraic[94] + algebraic[112]
        algebraic[116] = algebraic[97] + algebraic[113]
        algebraic[211] = (constants[98] * states[9]) / (constants[99] + states[9])
        algebraic[209] = (
            constants[97]
            * 4.00000
            * algebraic[25]
            * (
                algebraic[103] * states[9] * np.exp(2.00000 * algebraic[28])
                - constants[138] * constants[2]
            )
        ) / (np.exp(2.00000 * algebraic[28]) - 1.00000)
        algebraic[66] = (
            (constants[5] * constants[6]) / (constants[11] * constants[7])
        ) * np.log(constants[4] / states[10])
        algebraic[214] = (
            ((1.00000 - constants[103]) * constants[100])
            / (1.00000 + constants[102] / states[9])
        ) * (states[0] - algebraic[66])
        algebraic[67] = (
            (constants[5] * constants[6]) / (constants[11] * constants[7])
        ) * np.log(constants[4] / states[11])
        algebraic[212] = (
            (constants[103] * constants[100]) / (1.00000 + constants[102] / states[2])
        ) * (states[0] - algebraic[67])
        algebraic[216] = algebraic[212] + algebraic[214]
        algebraic[218] = constants[101] * (states[0] - algebraic[66])
        algebraic[221] = (states[11] - states[10]) / constants[104]
        algebraic[219] = (constants[128] * 0.00542500 * states[9]) / (
            states[9] + 0.000920000
        )
        algebraic[220] = (constants[128] * 2.75000 * 0.00542500 * states[9]) / (
            (states[9] + 0.000920000) - 0.000170000
        )
        algebraic[222] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
        algebraic[223] = (0.00488250 * states[7]) / 15.0000
        algebraic[224] = constants[111] * (
            (
                (1.00000 - algebraic[222]) * algebraic[219]
                + algebraic[222] * algebraic[220]
            )
            - algebraic[223]
        )
        algebraic[53] = 1.00000 / (
            1.00000
            + (constants[113] * constants[25])
            / (np.power(constants[25] + states[9], 2.00000))
            + (constants[26] * constants[27])
            / (np.power(constants[27] + states[9], 2.00000))
        )
        algebraic[225] = (states[7] - states[8]) / 60.0000
        algebraic[59] = 1.00000 / (
            1.00000
            + (constants[32] * constants[33])
            / (np.power(constants[33] + states[8], 2.00000))
        )
        return algebraic


@njit
def njitComputeRates(
    voi, states, constants, sizeStates, sizeAlgebraic, num_channels, drug_multipliers
):
    """Compute the rates for the model with njit acceleration."""
    rates = np.zeros(sizeStates, dtype=np.float64)
    algebraic = algebraic = np.zeros(sizeAlgebraic, dtype=np.float64)
    algebraic[-num_channels:] = drug_multipliers
    algebraic[3] = 1.00000 / (1.00000 + np.exp((states[0] + 87.6100) / 7.48800))
    rates[18] = (algebraic[3] - states[18]) / constants[41]
    algebraic[4] = 1.00000 / (1.00000 + np.exp((states[0] + 93.8100) / 7.48800))
    rates[19] = (algebraic[4] - states[19]) / constants[116]
    algebraic[9] = 1.00000 / (1.00000 + np.exp((states[0] + 18.0800) / 2.79160))
    rates[31] = (algebraic[9] - states[31]) / constants[49]
    algebraic[0] = 1.00000 / (
        np.power(1.00000 + np.exp(-(states[0] + 56.8600) / 9.03000), 2.00000)
    )
    algebraic[13] = 0.129200 * np.exp(
        -(np.power((states[0] + 45.7900) / 15.5400, 2.00000))
    ) + 0.0648700 * np.exp(-(np.power((states[0] - 4.82300) / 51.1200, 2.00000)))
    rates[12] = (algebraic[0] - states[12]) / algebraic[13]
    algebraic[2] = 1.00000 / (1.00000 + np.exp(-(states[0] + 42.8500) / 5.26400))
    algebraic[16] = 0.129200 * np.exp(
        -(np.power((states[0] + 45.7900) / 15.5400, 2.00000))
    ) + 0.0648700 * np.exp(-(np.power((states[0] - 4.82300) / 51.1200, 2.00000)))
    rates[17] = (algebraic[2] - states[17]) / algebraic[16]
    algebraic[5] = 1.00000 / (
        1.00000 + np.exp(-((states[0] + constants[44]) - 14.3400) / 14.8200)
    )
    algebraic[17] = 1.05150 / (
        1.00000
        / (
            1.20890
            * (1.00000 + np.exp(-((states[0] + constants[44]) - 18.4099) / 29.3814))
        )
        + 3.50000 / (1.00000 + np.exp((states[0] + constants[44] + 100.000) / 29.3814))
    )
    rates[20] = (algebraic[5] - states[20]) / algebraic[17]
    algebraic[7] = (
        1.0
        if states[0] >= 31.4978
        else 1.07630 * np.exp(-1.00700 * np.exp(-0.0829000 * states[0]))
    )
    algebraic[22] = (
        constants[51]
        + 0.600000
        + 1.00000
        / (
            np.exp(-0.0500000 * (states[0] + constants[50] + 6.00000))
            + np.exp(0.0900000 * (states[0] + constants[50] + 14.0000))
        )
    )
    rates[26] = (algebraic[7] - states[26]) / algebraic[22]
    algebraic[8] = 1.00000 / (1.00000 + np.exp((states[0] + 19.5800) / 3.69600))
    algebraic[23] = 7.00000 + 1.00000 / (
        0.00450000 * np.exp(-(states[0] + 20.0000) / 10.0000)
        + 0.00450000 * np.exp((states[0] + 20.0000) / 10.0000)
    )
    rates[27] = (algebraic[8] - states[27]) / algebraic[23]
    algebraic[24] = 1000.00 + 1.00000 / (
        3.50000e-05 * np.exp(-(states[0] + 5.00000) / 4.00000)
        + 3.50000e-05 * np.exp((states[0] + 5.00000) / 6.00000)
    )
    rates[28] = (algebraic[8] - states[28]) / algebraic[24]
    algebraic[10] = states[31] * 1.00000
    algebraic[20] = 1.00000 / (
        constants[46] / algebraic[10]
        + np.power(1.00000 + constants[45] / states[2], 4.00000)
    )
    rates[34] = algebraic[20] * constants[46] - states[34] * algebraic[10]
    algebraic[21] = 1.00000 / (
        constants[46] / algebraic[10]
        + np.power(1.00000 + constants[45] / states[9], 4.00000)
    )
    rates[35] = algebraic[21] * constants[46] - states[35] * algebraic[10]
    algebraic[12] = 1.00000 / (1.00000 + np.exp(-(states[0] + 11.6000) / 8.93200))
    algebraic[27] = 817.300 + 1.00000 / (
        0.000232600 * np.exp((states[0] + 48.2800) / 17.8000)
        + 0.00129200 * np.exp(-(states[0] + 210.000) / 230.000)
    )
    rates[41] = (algebraic[12] - states[41]) / algebraic[27]
    algebraic[32] = 1.00000 / (
        1.00000 + np.exp(-((states[0] + constants[44]) - 24.3400) / 14.8200)
    )
    rates[23] = (algebraic[32] - states[23]) / algebraic[17]
    algebraic[19] = algebraic[8]
    algebraic[33] = 7.00000 + 1.00000 / (
        0.0400000 * np.exp(-(states[0] - 4.00000) / 7.00000)
        + 0.0400000 * np.exp((states[0] - 4.00000) / 7.00000)
    )
    rates[29] = (algebraic[19] - states[29]) / algebraic[33]
    algebraic[34] = 100.000 + 1.00000 / (
        0.000120000 * np.exp(-states[0] / 3.00000)
        + 0.000120000 * np.exp(states[0] / 7.00000)
    )
    rates[30] = (algebraic[19] - states[30]) / algebraic[34]
    algebraic[35] = 2.50000 * algebraic[23]
    rates[32] = (algebraic[8] - states[32]) / algebraic[35]
    algebraic[26] = algebraic[12]
    algebraic[36] = 1.00000 / (
        0.0100000 * np.exp((states[0] - 50.0000) / 20.0000)
        + 0.0193000 * np.exp(-(states[0] + 66.5400) / 31.0000)
    )
    rates[42] = (algebraic[26] - states[42]) / algebraic[36]
    algebraic[43] = (constants[22] * (1.00000 - states[1])) / (
        1.00000 + constants[23] / states[2]
    )
    rates[1] = (
        constants[20] * algebraic[43] * (algebraic[43] + states[1])
        - constants[21] * states[1]
    )
    algebraic[1] = 1.00000 / (
        np.power(1.00000 + np.exp((states[0] + 71.5500) / 7.43000), 2.00000)
    )
    algebraic[14] = (
        0.0 if states[0] >= -40.0 else 0.057 * np.exp(-(states[0] + 80.0) / 6.8)
    )
    algebraic[29] = (
        (0.77 / (0.13 * (1.0 + np.exp(-(states[0] + 10.66) / 11.1))))
        if states[0] >= -40.0
        else (2.7 * np.exp(0.079 * states[0]) + 310000.0 * np.exp(0.3485 * states[0]))
    )
    algebraic[37] = 1.00000 / (algebraic[14] + algebraic[29])
    rates[13] = (algebraic[1] - states[13]) / algebraic[37]
    algebraic[40] = 2.50000 * algebraic[33]
    rates[33] = (algebraic[19] - states[33]) / algebraic[40]
    algebraic[38] = algebraic[1]
    algebraic[15] = (
        0.0
        if states[0] >= -40.0
        else (
            (
                -25428.0 * np.exp(0.2444 * states[0])
                - 6.948e-06 * np.exp(-0.04391 * states[0])
            )
            * (states[0] + 37.78)
        )
        / (1.0 + np.exp(0.311 * (states[0] + 79.23)))
    )
    algebraic[30] = (
        (0.6 * np.exp(0.057 * states[0])) / (1.0 + np.exp(-0.1 * (states[0] + 32.0)))
        if states[0] >= -40.0
        else (0.02424 * np.exp(-0.01052 * states[0]))
        / (1.0 + np.exp(-0.1378 * (states[0] + 40.14)))
    )
    algebraic[44] = 1.00000 / (algebraic[15] + algebraic[30])
    rates[14] = (algebraic[38] - states[14]) / algebraic[44]
    algebraic[45] = 1.00000 / (
        np.power(1.00000 + np.exp((states[0] + 77.5500) / 7.43000), 2.00000)
    )
    rates[15] = (algebraic[45] - states[15]) / algebraic[37]
    algebraic[6] = 1.00000 / (
        1.00000 + np.exp((states[0] + constants[44] + 43.9400) / 5.71100)
    )
    algebraic[18] = (
        1.0 - 0.95 / (1.0 + np.exp((states[0] + constants[44] + 70.0) / 5.0))
        if constants[0] == 1.0
        else 1.0
    )
    algebraic[31] = 4.56200 + 1.00000 / (
        0.393300 * np.exp(-(states[0] + constants[44] + 100.000) / 100.000)
        + 0.0800400 * np.exp((states[0] + constants[44] + 50.0000) / 16.5900)
    )
    algebraic[46] = algebraic[31] * algebraic[18]
    rates[21] = (algebraic[6] - states[21]) / algebraic[46]
    algebraic[28] = (states[0] * constants[7]) / (constants[5] * constants[6])
    algebraic[41] = 0.116100 * np.exp(0.299000 * algebraic[28])
    algebraic[47] = 0.244200 * np.exp(-1.60400 * algebraic[28])
    rates[38] = algebraic[47] * states[37] - algebraic[41] * states[38]
    rates[37] = (algebraic[41] * states[38] + constants[56] * states[36]) - (
        algebraic[47] + constants[55]
    ) * states[37]
    algebraic[50] = 1.46000 * algebraic[44]
    rates[16] = (algebraic[38] - states[16]) / algebraic[50]
    algebraic[39] = 23.6200 + 1.00000 / (
        0.00141600 * np.exp(-(states[0] + constants[44] + 96.5200) / 59.0500)
        + 1.78000e-08 * np.exp((states[0] + constants[44] + 114.100) / 8.07900)
    )
    algebraic[51] = algebraic[39] * algebraic[18]
    rates[22] = (algebraic[6] - states[22]) / algebraic[51]
    algebraic[42] = 0.0578000 * np.exp(0.971000 * algebraic[28])
    algebraic[48] = 0.000349000 * np.exp(-1.06200 * algebraic[28])
    algebraic[52] = 0.253300 * np.exp(0.595300 * algebraic[28])
    algebraic[55] = 0.0652500 * np.exp(-0.820900 * algebraic[28])
    rates[40] = (algebraic[42] * states[36] + algebraic[55] * states[39]) - (
        algebraic[48] + algebraic[52]
    ) * states[40]
    algebraic[54] = 1.35400 + 0.000100000 / (
        np.exp(((states[0] + constants[44]) - 167.400) / 15.8900)
        + np.exp(-((states[0] + constants[44]) - 12.2300) / 0.215400)
    )
    algebraic[57] = 1.00000 - 0.500000 / (
        1.00000 + np.exp((states[0] + constants[44] + 70.0000) / 20.0000)
    )
    algebraic[60] = algebraic[54] * algebraic[57] * algebraic[46]
    rates[24] = (algebraic[6] - states[24]) / algebraic[60]
    algebraic[61] = algebraic[54] * algebraic[57] * algebraic[51]
    rates[25] = (algebraic[6] - states[25]) / algebraic[61]
    algebraic[58] = 5.20000e-05 * np.exp(1.52500 * algebraic[28])
    algebraic[62] = (algebraic[48] * algebraic[55] * algebraic[58]) / (
        algebraic[42] * algebraic[52]
    )
    rates[36] = (
        constants[55] * states[37]
        + algebraic[48] * states[40]
        + algebraic[62] * states[39]
    ) - (constants[56] + algebraic[42] + algebraic[58]) * states[36]
    rates[39] = (algebraic[58] * states[36] + algebraic[52] * states[40]) - (
        algebraic[62] + algebraic[55]
    ) * states[39]
    algebraic[79] = constants[48] * states[27] + constants[119] * states[28]
    algebraic[80] = 0.300000 + 0.600000 / (
        1.00000 + np.exp((states[0] - 10.0000) / 10.0000)
    )
    algebraic[81] = 1.00000 - algebraic[80]
    algebraic[82] = algebraic[80] * states[29] + algebraic[81] * states[30]
    algebraic[83] = constants[48] * states[32] + constants[119] * states[28]
    algebraic[84] = algebraic[80] * states[33] + algebraic[81] * states[30]
    algebraic[25] = (states[0] * constants[7] * constants[7]) / (
        constants[5] * constants[6]
    )
    algebraic[85] = (
        0.500000 * (states[4] + states[6] + states[11] + 4.00000 * states[2])
    ) / 1000.00
    algebraic[86] = np.exp(
        -constants[133]
        * 4.00000
        * (
            (np.power(algebraic[85], 1.0 / 2))
            / (1.00000 + np.power(algebraic[85], 1.0 / 2))
            - 0.300000 * algebraic[85]
        )
    )
    algebraic[89] = (
        4.00000
        * algebraic[25]
        * (
            algebraic[86] * states[2] * np.exp(2.00000 * algebraic[28])
            - constants[138] * constants[2]
        )
    ) / (np.exp(2.00000 * algebraic[28]) - 1.00000)
    algebraic[49] = algebraic[43] + states[1]
    algebraic[92] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
    algebraic[93] = (
        algebraic[-7]
        * constants[53]
        * (
            (1.00000 - algebraic[92])
            * constants[120]
            * algebraic[89]
            * states[26]
            * (
                algebraic[79] * (1.00000 - states[34])
                + states[31] * algebraic[82] * states[34]
            )
            + algebraic[92]
            * constants[130]
            * algebraic[89]
            * states[26]
            * (
                algebraic[83] * (1.00000 - states[34])
                + states[31] * algebraic[84] * states[34]
            )
        )
    )
    algebraic[95] = ((-constants[126] * algebraic[93]) / 1.00000) / (
        1.00000 + np.power(constants[109] / states[8], 8.00000)
    )
    algebraic[98] = algebraic[95] * 1.7 if constants[0] == 2.0 else algebraic[95]
    algebraic[101] = constants[108] / (1.00000 + 0.0123000 / states[8])
    algebraic[104] = 0.001 if algebraic[101] < 0.001 else algebraic[101]
    rates[43] = (algebraic[98] - states[43]) / algebraic[104]
    algebraic[96] = ((-constants[134] * algebraic[93]) / 1.00000) / (
        1.00000 + np.power(constants[109] / states[8], 8.00000)
    )
    algebraic[99] = algebraic[96] * 1.7 if constants[0] == 2.0 else algebraic[96]
    algebraic[102] = constants[127] / (1.00000 + 0.0123000 / states[8])
    algebraic[105] = 0.001 if algebraic[102] < 0.001 else algebraic[102]
    rates[44] = (algebraic[99] - states[44]) / algebraic[105]
    algebraic[64] = (
        (constants[5] * constants[6]) / (constants[10] * constants[7])
    ) * np.log(constants[3] / states[5])
    algebraic[73] = 1.00000 / (
        1.00000 + np.exp(((states[0] + constants[44]) - 213.600) / 151.200)
    )
    algebraic[74] = 1.00000 - algebraic[73]
    algebraic[75] = algebraic[73] * states[21] + algebraic[74] * states[22]
    algebraic[76] = algebraic[73] * states[24] + algebraic[74] * states[25]
    algebraic[77] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
    algebraic[78] = (
        algebraic[-2]
        * constants[118]
        * (states[0] - algebraic[64])
        * (
            (1.00000 - algebraic[77]) * states[20] * algebraic[75]
            + algebraic[77] * states[23] * algebraic[76]
        )
    )
    algebraic[117] = (
        algebraic[-1]
        * constants[122]
        * (np.power(constants[3] / 5.00000, 1.0 / 2))
        * states[40]
        * (states[0] - algebraic[64])
    )
    algebraic[65] = (
        (constants[5] * constants[6]) / (constants[10] * constants[7])
    ) * np.log(
        (constants[3] + constants[34] * constants[1])
        / (states[5] + constants[34] * states[3])
    )
    algebraic[118] = 1.00000 + 0.600000 / (
        1.00000 + np.power(3.80000e-05 / states[9], 1.40000)
    )
    algebraic[119] = (
        algebraic[-5]
        * constants[123]
        * algebraic[118]
        * states[41]
        * states[42]
        * (states[0] - algebraic[65])
    )
    algebraic[120] = 4.09400 / (
        1.00000 + np.exp(0.121700 * ((states[0] - algebraic[64]) - 49.9340))
    )
    algebraic[121] = (
        15.7200 * np.exp(0.0674000 * ((states[0] - algebraic[64]) - 3.25700))
        + np.exp(0.0618000 * ((states[0] - algebraic[64]) - 594.310))
    ) / (1.00000 + np.exp(-0.162900 * ((states[0] - algebraic[64]) + 14.2070)))
    algebraic[122] = algebraic[120] / (algebraic[120] + algebraic[121])
    algebraic[123] = (
        algebraic[-6]
        * constants[124]
        * (np.power(constants[3] / 5.00000, 1.0 / 2))
        * algebraic[122]
        * (states[0] - algebraic[64])
    )
    algebraic[187] = constants[82] * np.exp(
        ((1.00000 - constants[83]) * algebraic[28]) / 3.00000
    )
    algebraic[191] = (
        constants[77] * (np.power(constants[3] / constants[85], 2.00000))
    ) / (
        (
            np.power(1.00000 + constants[1] / algebraic[187], 3.00000)
            + np.power(1.00000 + constants[3] / constants[85], 2.00000)
        )
        - 1.00000
    )
    algebraic[188] = constants[90] / (
        1.00000
        + constants[89] / constants[91]
        + states[3] / constants[92]
        + states[5] / constants[93]
    )
    algebraic[192] = (constants[78] * algebraic[188] * constants[89]) / (
        1.00000 + constants[87] / constants[88]
    )
    algebraic[186] = constants[81] * np.exp((constants[83] * algebraic[28]) / 3.00000)
    algebraic[189] = (
        constants[73] * (np.power(states[3] / algebraic[186], 3.00000))
    ) / (
        (
            np.power(1.00000 + states[3] / algebraic[186], 3.00000)
            + np.power(1.00000 + states[5] / constants[84], 2.00000)
        )
        - 1.00000
    )
    algebraic[190] = (
        constants[76] * (np.power(constants[1] / algebraic[187], 3.00000))
    ) / (
        (
            np.power(1.00000 + constants[1] / algebraic[187], 3.00000)
            + np.power(1.00000 + constants[3] / constants[85], 2.00000)
        )
        - 1.00000
    )
    algebraic[193] = (
        constants[80] * (np.power(states[5] / constants[84], 2.00000))
    ) / (
        (
            np.power(1.00000 + states[3] / algebraic[186], 3.00000)
            + np.power(1.00000 + states[5] / constants[84], 2.00000)
        )
        - 1.00000
    )
    algebraic[194] = (
        constants[160] * algebraic[189] * constants[159]
        + algebraic[190] * algebraic[193] * algebraic[192]
        + constants[159] * algebraic[193] * algebraic[192]
        + algebraic[192] * algebraic[189] * constants[159]
    )
    algebraic[195] = (
        algebraic[190] * constants[158] * algebraic[193]
        + algebraic[189] * constants[159] * algebraic[191]
        + algebraic[191] * constants[158] * algebraic[193]
        + constants[159] * algebraic[191] * algebraic[193]
    )
    algebraic[196] = (
        constants[159] * algebraic[191] * constants[160]
        + algebraic[192] * algebraic[190] * constants[158]
        + algebraic[190] * constants[158] * constants[160]
        + algebraic[191] * constants[160] * constants[158]
    )
    algebraic[197] = (
        algebraic[193] * algebraic[192] * algebraic[190]
        + algebraic[191] * constants[160] * algebraic[189]
        + algebraic[190] * constants[160] * algebraic[189]
        + algebraic[192] * algebraic[190] * algebraic[189]
    )
    algebraic[198] = algebraic[194] / (
        algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
    )
    algebraic[199] = algebraic[195] / (
        algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
    )
    algebraic[202] = 3.00000 * (
        algebraic[198] * algebraic[191] - algebraic[199] * algebraic[192]
    )
    algebraic[200] = algebraic[196] / (
        algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
    )
    algebraic[201] = algebraic[197] / (
        algebraic[194] + algebraic[195] + algebraic[196] + algebraic[197]
    )
    algebraic[203] = 2.00000 * (
        algebraic[201] * constants[158] - algebraic[200] * algebraic[189]
    )
    algebraic[204] = constants[161] * (
        constants[8] * algebraic[202] + constants[10] * algebraic[203]
    )
    algebraic[205] = 1.00000 / (1.00000 + np.exp(-(states[0] - 10.8968) / 23.9871))
    algebraic[206] = constants[125] * algebraic[205] * (states[0] - algebraic[64])
    algebraic[68] = (
        constants[36]
        * constants[35]
        * constants[114]
        * constants[115]
        * (states[0] - algebraic[64])
    )
    algebraic[11] = (
        constants[16]
        if (
            voi >= constants[14]
            and (
                voi
                - constants[14]
                - float(math.floor((voi - constants[14]) / constants[17]))
                * constants[17]
            )
            <= constants[18]
        )
        else 0.0
    )
    algebraic[100] = (
        0.500000 * (states[3] + states[5] + states[10] + 4.00000 * states[9])
    ) / 1000.00
    algebraic[107] = np.exp(
        -constants[133]
        * 1.00000
        * (
            (np.power(algebraic[100], 1.0 / 2))
            / (1.00000 + np.power(algebraic[100], 1.0 / 2))
            - 0.300000 * algebraic[100]
        )
    )
    algebraic[110] = (
        1.00000
        * algebraic[25]
        * (
            algebraic[107] * states[5] * np.exp(1.00000 * algebraic[28])
            - constants[140] * constants[3]
        )
    ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
    algebraic[113] = (1.00000 - constants[53]) * (
        (1.00000 - algebraic[92])
        * constants[132]
        * algebraic[110]
        * states[26]
        * (
            algebraic[79] * (1.00000 - states[35])
            + states[31] * algebraic[82] * states[35]
        )
        + algebraic[92]
        * constants[137]
        * algebraic[110]
        * states[26]
        * (
            algebraic[83] * (1.00000 - states[35])
            + states[31] * algebraic[84] * states[35]
        )
    )
    algebraic[208] = (states[6] - states[5]) / constants[105]
    rates[5] = (
        -(
            (
                (
                    algebraic[78]
                    + algebraic[117]
                    + algebraic[119]
                    + algebraic[123]
                    + algebraic[206]
                    + algebraic[68]
                    + algebraic[11]
                )
                - 2.00000 * algebraic[204]
            )
            + algebraic[113]
        )
        * constants[135]
    ) / (constants[7] * constants[141]) + (algebraic[208] * constants[144]) / constants[
        141
    ]
    algebraic[88] = np.exp(
        -constants[133]
        * 1.00000
        * (
            (np.power(algebraic[85], 1.0 / 2))
            / (1.00000 + np.power(algebraic[85], 1.0 / 2))
            - 0.300000 * algebraic[85]
        )
    )
    algebraic[91] = (
        1.00000
        * algebraic[25]
        * (
            algebraic[88] * states[6] * np.exp(1.00000 * algebraic[28])
            - constants[140] * constants[3]
        )
    ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
    algebraic[97] = constants[53] * (
        (1.00000 - algebraic[92])
        * constants[132]
        * algebraic[91]
        * states[26]
        * (
            algebraic[79] * (1.00000 - states[34])
            + states[31] * algebraic[82] * states[34]
        )
        + algebraic[92]
        * constants[137]
        * algebraic[91]
        * states[26]
        * (
            algebraic[83] * (1.00000 - states[34])
            + states[31] * algebraic[84] * states[34]
        )
    )
    rates[6] = (-algebraic[97] * constants[135]) / (
        constants[7] * constants[144]
    ) - algebraic[208]
    algebraic[63] = (
        (constants[5] * constants[6]) / (constants[8] * constants[7])
    ) * np.log(constants[1] / states[3])
    algebraic[69] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
    algebraic[70] = (
        algebraic[-4]
        * constants[40]
        * (states[0] - algebraic[63])
        * (np.power(states[12], 3.00000))
        * (
            (1.00000 - algebraic[69]) * states[13] * states[14]
            + algebraic[69] * states[15] * states[16]
        )
    )
    algebraic[71] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
    algebraic[72] = (
        algebraic[-3]
        * constants[117]
        * (states[0] - algebraic[63])
        * states[17]
        * ((1.00000 - algebraic[71]) * states[18] + algebraic[71] * states[19])
    )
    algebraic[152] = 1.00000 / (1.00000 + np.power(constants[71] / states[9], 2.00000))
    algebraic[125] = np.exp(constants[69] * algebraic[28])
    algebraic[132] = 1.00000 + (constants[1] / constants[62]) * (
        1.00000 + 1.00000 / algebraic[125]
    )
    algebraic[133] = constants[1] / (constants[62] * algebraic[125] * algebraic[132])
    algebraic[136] = algebraic[133] * constants[66]
    algebraic[126] = 1.00000 + (states[3] / constants[62]) * (1.00000 + algebraic[125])
    algebraic[127] = (states[3] * algebraic[125]) / (constants[62] * algebraic[126])
    algebraic[139] = algebraic[127] * constants[66]
    algebraic[129] = 1.00000 + (states[3] / constants[60]) * (
        1.00000 + states[3] / constants[61]
    )
    algebraic[130] = (states[3] * states[3]) / (
        algebraic[129] * constants[60] * constants[61]
    )
    algebraic[142] = algebraic[130] * algebraic[127] * constants[64]
    algebraic[143] = algebraic[133] * constants[146] * constants[64]
    algebraic[134] = 1.00000 / algebraic[132]
    algebraic[135] = algebraic[134] * constants[65]
    algebraic[137] = algebraic[135] + algebraic[136]
    algebraic[124] = np.exp(constants[70] * algebraic[28])
    algebraic[128] = 1.00000 / algebraic[126]
    algebraic[138] = (algebraic[128] * constants[65]) / algebraic[124]
    algebraic[140] = algebraic[138] + algebraic[139]
    algebraic[131] = 1.00000 / algebraic[129]
    algebraic[141] = algebraic[131] * states[9] * constants[67]
    algebraic[144] = constants[149] * algebraic[140] * (
        algebraic[142] + algebraic[141]
    ) + constants[150] * algebraic[142] * (constants[149] + algebraic[137])
    algebraic[145] = constants[148] * algebraic[142] * (
        algebraic[140] + constants[150]
    ) + algebraic[140] * algebraic[141] * (constants[148] + algebraic[143])
    algebraic[146] = constants[148] * algebraic[137] * (
        algebraic[142] + algebraic[141]
    ) + algebraic[143] * algebraic[141] * (constants[149] + algebraic[137])
    algebraic[147] = constants[149] * algebraic[143] * (
        algebraic[140] + constants[150]
    ) + algebraic[137] * constants[150] * (constants[148] + algebraic[143])
    algebraic[148] = algebraic[144] / (
        algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
    )
    algebraic[149] = algebraic[145] / (
        algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
    )
    algebraic[150] = algebraic[146] / (
        algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
    )
    algebraic[151] = algebraic[147] / (
        algebraic[144] + algebraic[145] + algebraic[146] + algebraic[147]
    )
    algebraic[153] = (
        3.00000 * (algebraic[151] * algebraic[142] - algebraic[148] * algebraic[143])
        + algebraic[150] * algebraic[139]
    ) - algebraic[149] * algebraic[136]
    algebraic[154] = algebraic[149] * constants[149] - algebraic[148] * constants[148]
    algebraic[155] = (
        (1.00000 - constants[59])
        * constants[151]
        * algebraic[152]
        * (constants[8] * algebraic[153] + constants[9] * algebraic[154])
    )
    algebraic[207] = (
        constants[96]
        * algebraic[25]
        * (states[3] * np.exp(algebraic[28]) - constants[1])
    ) / (np.exp(algebraic[28]) - 1.00000)
    algebraic[106] = np.exp(
        -constants[133]
        * 1.00000
        * (
            (np.power(algebraic[100], 1.0 / 2))
            / (1.00000 + np.power(algebraic[100], 1.0 / 2))
            - 0.300000 * algebraic[100]
        )
    )
    algebraic[109] = (
        1.00000
        * algebraic[25]
        * (
            algebraic[106] * states[3] * np.exp(1.00000 * algebraic[28])
            - constants[139] * constants[1]
        )
    ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
    algebraic[112] = (1.00000 - constants[53]) * (
        (1.00000 - algebraic[92])
        * constants[131]
        * algebraic[109]
        * states[26]
        * (
            algebraic[79] * (1.00000 - states[35])
            + states[31] * algebraic[82] * states[35]
        )
        + algebraic[92]
        * constants[136]
        * algebraic[109]
        * states[26]
        * (
            algebraic[83] * (1.00000 - states[35])
            + states[31] * algebraic[84] * states[35]
        )
    )
    algebraic[210] = (states[4] - states[3]) / constants[104]
    rates[3] = (
        -(
            algebraic[70]
            + algebraic[72]
            + 3.00000 * algebraic[155]
            + algebraic[112]
            + 3.00000 * algebraic[204]
            + algebraic[207]
        )
        * constants[135]
    ) / (constants[7] * constants[141]) + (algebraic[210] * constants[144]) / constants[
        141
    ]
    algebraic[182] = 1.00000 / (1.00000 + np.power(constants[71] / states[2], 2.00000))
    algebraic[162] = 1.00000 + (constants[1] / constants[62]) * (
        1.00000 + 1.00000 / algebraic[125]
    )
    algebraic[163] = constants[1] / (constants[62] * algebraic[125] * algebraic[162])
    algebraic[166] = algebraic[163] * constants[66]
    algebraic[156] = 1.00000 + (states[4] / constants[62]) * (1.00000 + algebraic[125])
    algebraic[157] = (states[4] * algebraic[125]) / (constants[62] * algebraic[156])
    algebraic[169] = algebraic[157] * constants[66]
    algebraic[159] = 1.00000 + (states[4] / constants[60]) * (
        1.00000 + states[4] / constants[61]
    )
    algebraic[160] = (states[4] * states[4]) / (
        algebraic[159] * constants[60] * constants[61]
    )
    algebraic[172] = algebraic[160] * algebraic[157] * constants[64]
    algebraic[173] = algebraic[163] * constants[153] * constants[64]
    algebraic[164] = 1.00000 / algebraic[162]
    algebraic[165] = algebraic[164] * constants[65]
    algebraic[167] = algebraic[165] + algebraic[166]
    algebraic[158] = 1.00000 / algebraic[156]
    algebraic[168] = (algebraic[158] * constants[65]) / algebraic[124]
    algebraic[170] = algebraic[168] + algebraic[169]
    algebraic[161] = 1.00000 / algebraic[159]
    algebraic[171] = algebraic[161] * states[2] * constants[67]
    algebraic[174] = constants[156] * algebraic[170] * (
        algebraic[172] + algebraic[171]
    ) + constants[157] * algebraic[172] * (constants[156] + algebraic[167])
    algebraic[175] = constants[155] * algebraic[172] * (
        algebraic[170] + constants[157]
    ) + algebraic[170] * algebraic[171] * (constants[155] + algebraic[173])
    algebraic[176] = constants[155] * algebraic[167] * (
        algebraic[172] + algebraic[171]
    ) + algebraic[173] * algebraic[171] * (constants[156] + algebraic[167])
    algebraic[177] = constants[156] * algebraic[173] * (
        algebraic[170] + constants[157]
    ) + algebraic[167] * constants[157] * (constants[155] + algebraic[173])
    algebraic[178] = algebraic[174] / (
        algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
    )
    algebraic[179] = algebraic[175] / (
        algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
    )
    algebraic[180] = algebraic[176] / (
        algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
    )
    algebraic[181] = algebraic[177] / (
        algebraic[174] + algebraic[175] + algebraic[176] + algebraic[177]
    )
    algebraic[183] = (
        3.00000 * (algebraic[181] * algebraic[172] - algebraic[178] * algebraic[173])
        + algebraic[180] * algebraic[169]
    ) - algebraic[179] * algebraic[166]
    algebraic[184] = algebraic[179] * constants[156] - algebraic[178] * constants[155]
    algebraic[185] = (
        constants[59]
        * constants[151]
        * algebraic[182]
        * (constants[8] * algebraic[183] + constants[9] * algebraic[184])
    )
    algebraic[87] = np.exp(
        -constants[133]
        * 1.00000
        * (
            (np.power(algebraic[85], 1.0 / 2))
            / (1.00000 + np.power(algebraic[85], 1.0 / 2))
            - 0.300000 * algebraic[85]
        )
    )
    algebraic[90] = (
        1.00000
        * algebraic[25]
        * (
            algebraic[87] * states[4] * np.exp(1.00000 * algebraic[28])
            - constants[139] * constants[1]
        )
    ) / (np.exp(1.00000 * algebraic[28]) - 1.00000)
    algebraic[94] = constants[53] * (
        (1.00000 - algebraic[92])
        * constants[131]
        * algebraic[90]
        * states[26]
        * (
            algebraic[79] * (1.00000 - states[34])
            + states[31] * algebraic[82] * states[34]
        )
        + algebraic[92]
        * constants[136]
        * algebraic[90]
        * states[26]
        * (
            algebraic[83] * (1.00000 - states[34])
            + states[31] * algebraic[84] * states[34]
        )
    )
    rates[4] = (-(algebraic[94] + 3.00000 * algebraic[185]) * constants[135]) / (
        constants[7] * constants[144]
    ) - algebraic[210]
    algebraic[213] = (states[2] - states[9]) / constants[106]
    algebraic[215] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
    algebraic[217] = constants[110] * (
        (1.00000 - algebraic[215]) * states[43] + algebraic[215] * states[44]
    )
    algebraic[56] = 1.00000 / (
        1.00000
        + (constants[28] * constants[29])
        / (np.power(constants[29] + states[2], 2.00000))
        + (constants[30] * constants[31])
        / (np.power(constants[31] + states[2], 2.00000))
    )
    rates[2] = algebraic[56] * (
        (
            (-(algebraic[93] - 2.00000 * algebraic[185]) * constants[135])
            / (2.00000 * constants[7] * constants[144])
            + (algebraic[217] * constants[143]) / constants[144]
        )
        - algebraic[213]
    )
    algebraic[103] = np.exp(
        -constants[133]
        * 4.00000
        * (
            (np.power(algebraic[100], 1.0 / 2))
            / (1.00000 + np.power(algebraic[100], 1.0 / 2))
            - 0.300000 * algebraic[100]
        )
    )
    algebraic[108] = (
        4.00000
        * algebraic[25]
        * (
            algebraic[103] * states[9] * np.exp(2.00000 * algebraic[28])
            - constants[138] * constants[2]
        )
    ) / (np.exp(2.00000 * algebraic[28]) - 1.00000)
    algebraic[111] = (
        algebraic[-7]
        * (1.00000 - constants[53])
        * (
            (1.00000 - algebraic[92])
            * constants[120]
            * algebraic[108]
            * states[26]
            * (
                algebraic[79] * (1.00000 - states[35])
                + states[31] * algebraic[82] * states[35]
            )
            + algebraic[92]
            * constants[130]
            * algebraic[108]
            * states[26]
            * (
                algebraic[83] * (1.00000 - states[35])
                + states[31] * algebraic[84] * states[35]
            )
        )
    )
    algebraic[114] = algebraic[93] + algebraic[111]
    algebraic[115] = algebraic[94] + algebraic[112]
    algebraic[116] = algebraic[97] + algebraic[113]
    algebraic[211] = (constants[98] * states[9]) / (constants[99] + states[9])
    algebraic[209] = (
        constants[97]
        * 4.00000
        * algebraic[25]
        * (
            algebraic[103] * states[9] * np.exp(2.00000 * algebraic[28])
            - constants[138] * constants[2]
        )
    ) / (np.exp(2.00000 * algebraic[28]) - 1.00000)
    algebraic[66] = (
        (constants[5] * constants[6]) / (constants[11] * constants[7])
    ) * np.log(constants[4] / states[10])
    algebraic[214] = (
        ((1.00000 - constants[103]) * constants[100])
        / (1.00000 + constants[102] / states[9])
    ) * (states[0] - algebraic[66])
    algebraic[67] = (
        (constants[5] * constants[6]) / (constants[11] * constants[7])
    ) * np.log(constants[4] / states[11])
    algebraic[212] = (
        (constants[103] * constants[100]) / (1.00000 + constants[102] / states[2])
    ) * (states[0] - algebraic[67])
    algebraic[216] = algebraic[212] + algebraic[214]
    algebraic[218] = constants[101] * (states[0] - algebraic[66])
    rates[0] = -(
        algebraic[70]
        + algebraic[72]
        + algebraic[78]
        + algebraic[114]
        + algebraic[115]
        + algebraic[116]
        + algebraic[117]
        + algebraic[119]
        + algebraic[123]
        + algebraic[155]
        + algebraic[185]
        + algebraic[204]
        + algebraic[207]
        + algebraic[206]
        + algebraic[211]
        + algebraic[209]
        + algebraic[216]
        + algebraic[218]
        + algebraic[68]
        + algebraic[11]
    )
    algebraic[221] = (states[11] - states[10]) / constants[104]
    rates[10] = ((algebraic[218] + algebraic[214]) * constants[135]) / (
        constants[7] * constants[141]
    ) + (algebraic[221] * constants[144]) / constants[141]
    rates[11] = (algebraic[212] * constants[135]) / (
        constants[7] * constants[144]
    ) - algebraic[221]
    algebraic[219] = (constants[128] * 0.00542500 * states[9]) / (
        states[9] + 0.000920000
    )
    algebraic[220] = (constants[128] * 2.75000 * 0.00542500 * states[9]) / (
        (states[9] + 0.000920000) - 0.000170000
    )
    algebraic[222] = 1.00000 / (1.00000 + constants[19] / algebraic[49])
    algebraic[223] = (0.00488250 * states[7]) / 15.0000
    algebraic[224] = constants[111] * (
        ((1.00000 - algebraic[222]) * algebraic[219] + algebraic[222] * algebraic[220])
        - algebraic[223]
    )
    algebraic[53] = 1.00000 / (
        1.00000
        + (constants[113] * constants[25])
        / (np.power(constants[25] + states[9], 2.00000))
        + (constants[26] * constants[27])
        / (np.power(constants[27] + states[9], 2.00000))
    )
    rates[9] = algebraic[53] * (
        (
            (
                -(
                    (algebraic[111] + algebraic[211] + algebraic[209])
                    - 2.00000 * algebraic[155]
                )
                * constants[135]
            )
            / (2.00000 * constants[7] * constants[141])
            - (algebraic[224] * constants[142]) / constants[141]
        )
        + (algebraic[213] * constants[144]) / constants[141]
    )
    algebraic[225] = (states[7] - states[8]) / 60.0000
    rates[7] = algebraic[224] - (algebraic[225] * constants[143]) / constants[142]
    algebraic[59] = 1.00000 / (
        1.00000
        + (constants[32] * constants[33])
        / (np.power(constants[33] + states[8], 2.00000))
    )
    rates[8] = algebraic[59] * (algebraic[225] - algebraic[217])
    return rates
