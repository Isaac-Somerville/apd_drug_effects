import numpy as np
import time
from cell_models.base import CardiacCellModel
from numba import njit


class GuineaPigModel(CardiacCellModel):
    def __init__(
        self,
        drug=None,
        drug_amount=None,
        num_cycles_feasible_ap=None
    ):
        super().__init__(
            sizeStates=55 + 1,
            sizeAlgebraic=108 + 5,
            sizeConstants=130,
            species="Guinea Pig",
            time_unit="s",
            channel_list=["ICaL", "IK1", "IKs", "INa", "hERG"],
            state_idx_dict={"cai": 40, "nai": 2, "ki" : 3},
            default_t_max=0.3,
            num_cycles_feasible_ap=num_cycles_feasible_ap,
            ode_hyperparameters={"max_step": 0.001, "nsteps": 50000},
            num_cycles_limit_state=1000,
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
        constants[48] = 1
        constants[0] = 8310
        constants[1] = 310
        constants[2] = 96500

        states[0] = -8.5207812e1
        states[1] = -8.5208034e1
        states[2] = 1.1120279e1
        states[3] = 1.3678926e2
        states[4] = 3.4130493e-3
        states[5] = 8.2699973e-1
        states[6] = 1.3993239e2
        states[7] = 5.4140321e0
        states[8] = 3.4129472e-3
        states[9] = 8.2700551e-1
        states[10] = 9.9797984e-1
        states[11] = 7.5722514e-5
        states[12] = 2.1545646e-9
        states[13] = 2.7463178e-14
        states[14] = -5.7569102e-17
        states[15] = -1.3496934e-17
        states[16] = 1.9155065e-3
        states[17] = 5.8135383e-7
        states[18] = 6.615513e-11
        states[19] = 3.3426555e-15
        states[20] = 4.6640361e-19
        states[21] = 3.7986638e-22
        states[22] = 9.5977033e-5
        states[23] = 7.8134852e-1
        states[24] = 9.9798934e-1
        states[25] = 7.5720257e-5
        states[26] = 2.1544156e-9
        states[27] = 2.7459475e-14
        states[28] = -5.7367189e-17
        states[29] = -1.3449454e-17
        states[30] = 1.9155245e-3
        states[31] = 5.8133643e-7
        states[32] = 6.6150547e-11
        states[33] = 3.3423167e-15
        states[34] = 4.6165955e-19
        states[35] = 3.75926e-22
        states[36] = 1.8476402e0
        states[37] = 7.813477e-1
        states[38] = 2.0469344e-4
        states[39] = 2.0469344e-4
        states[40] = 8.8787034e-5
        states[41] = 6.1359896e-3
        states[42] = 6.1359896e-3
        states[43] = 9.625701e-1
        states[44] = 9.6250049e-1
        states[45] = 9.625701e-1
        states[46] = 9.6250049e-1
        states[47] = 1.0200296e0
        states[48] = 1.0326252e0
        states[49] = 8.053082e-1
        states[50] = 2.6639195e-4
        states[51] = 4.9455459e-10
        states[52] = 1.9442578e-1
        states[53] = 9.6700747e-1
        states[54] = 8.1740868e-2

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

        # Initialise derived constants
        constants[80] = (
            2.00000 * 3.14159 * (np.power(constants[10], 2.00000))
            + 2.00000 * 3.14159 * constants[10] * constants[11]
        )
        constants[83] = 3.14159 * (np.power(constants[10], 2.00000)) * constants[11]
        constants[84] = constants[83] * 0.680000 * 6.00000e-05
        constants[85] = constants[83] * 0.00316000
        constants[86] = constants[83] * 0.680000
        constants[87] = constants[83] * 0.0550000
        constants[88] = (
            constants[12] * 2.00000 * 3.14159 * constants[10] * constants[11]
        )
        constants[89] = ((constants[7] * constants[9]) / 2.00000) / (
            3.14159 * (np.power(constants[8], 2.00000)) * constants[88]
        )
        constants[90] = 2.00000 * 3.14159 * constants[8] * constants[9] * constants[88]
        constants[91] = (
            constants[13] * (constants[80] + constants[90]) * (1.00000 - constants[14])
        )
        constants[92] = (
            constants[15] * (constants[80] + constants[90]) * (1.00000 - constants[16])
        )
        constants[93] = (
            constants[17] * (constants[80] + constants[90]) * (1.00000 - constants[18])
        )
        constants[94] = (
            constants[19] * (constants[80] + constants[90]) * (1.00000 - constants[20])
        )
        constants[95] = (
            constants[21] * (constants[80] + constants[90]) * (1.00000 - constants[22])
        )
        constants[96] = (
            constants[23] * (constants[80] + constants[90]) * (1.00000 - constants[24])
        )
        constants[97] = (
            constants[25] * (constants[80] + constants[90]) * (1.00000 - constants[26])
        )
        constants[98] = (
            constants[27] * (constants[80] + constants[90]) * (1.00000 - constants[28])
        )
        constants[99] = (
            constants[29] * (constants[80] + constants[90]) * (1.00000 - constants[30])
        )
        constants[100] = (
            constants[31] * (constants[80] + constants[90]) * (1.00000 - constants[32])
        )
        constants[101] = (
            constants[33] * (constants[80] + constants[90]) * (1.00000 - constants[34])
        )
        constants[102] = (
            constants[46] * (constants[80] + constants[90]) * (1.00000 - constants[47])
        )
        constants[103] = (
            constants[44] * (constants[80] + constants[90]) * (1.00000 - constants[45])
        )
        constants[104] = (
            constants[42] * (constants[80] + constants[90]) * (1.00000 - constants[43])
        )
        constants[105] = (
            constants[35] * (constants[80] + constants[90]) * (1.00000 - constants[36])
        )
        constants[106] = (
            constants[37] * (constants[80] + constants[90]) * (1.00000 - constants[36])
        )
        constants[107] = (
            constants[38] * (constants[80] + constants[90]) * (1.00000 - constants[39])
        )
        constants[108] = (
            constants[40] * (constants[80] + constants[90]) * (1.00000 - constants[41])
        )
        constants[109] = constants[13] * (constants[80] + constants[90]) * constants[14]
        constants[110] = constants[15] * (constants[80] + constants[90]) * constants[16]
        constants[111] = constants[17] * (constants[80] + constants[90]) * constants[18]
        constants[112] = constants[19] * (constants[80] + constants[90]) * constants[20]
        constants[113] = constants[21] * (constants[80] + constants[90]) * constants[22]
        constants[114] = constants[23] * (constants[80] + constants[90]) * constants[24]
        constants[115] = constants[25] * (constants[80] + constants[90]) * constants[26]
        constants[116] = constants[27] * (constants[80] + constants[90]) * constants[28]
        constants[117] = constants[29] * (constants[80] + constants[90]) * constants[30]
        constants[118] = constants[31] * (constants[80] + constants[90]) * constants[32]
        constants[119] = constants[33] * (constants[80] + constants[90]) * constants[34]
        constants[120] = constants[46] * (constants[80] + constants[90]) * constants[47]
        constants[121] = constants[44] * (constants[80] + constants[90]) * constants[45]
        constants[122] = constants[42] * (constants[80] + constants[90]) * constants[43]
        constants[123] = constants[35] * (constants[80] + constants[90]) * constants[36]
        constants[124] = constants[37] * (constants[80] + constants[90]) * constants[36]
        constants[125] = constants[38] * (constants[80] + constants[90]) * constants[39]
        constants[126] = constants[40] * (constants[80] + constants[90]) * constants[41]
        constants[127] = (
            3.14159 * (np.power(constants[8], 2.00000)) * constants[9] * constants[88]
        )
        constants[128] = constants[90] * 1.00000
        constants[129] = constants[80] * 1.00000
        self.init_states = states
        self.constants = constants
        return states, constants

    def createLegends(self):
        legend_states = [""] * self.sizeStates
        legend_rates = [""] * self.sizeStates
        legend_algebraic = [""] * self.sizeAlgebraic
        legend_voi = ""
        legend_constants = [""] * self.sizeConstants
        legend_voi = "time in component environment (second)"
        legend_states[-1] = "drug concentration (nM)"
        legend_rates[-1] = "d/dt drug concentration (nM/ms)"
        for i in range(len(self.channel_list)):
            legend_algebraic[i - len(self.channel_list)] = (
                "proportion of "
                + self.channel_list[i]
                + " channel uninhibited (dimensionless)"
            )
        legend_constants[0] = (
            "R in component model_parameters (joule_per_kilomole_kelvin)"
        )
        legend_constants[1] = "T in component model_parameters (kelvin)"
        legend_constants[2] = "F in component model_parameters (coulomb_per_mole)"
        legend_constants[3] = "Na_e in component model_parameters (millimolar)"
        legend_constants[4] = "Ca_e in component model_parameters (millimolar)"
        legend_constants[5] = "K_e in component model_parameters (millimolar)"
        legend_constants[6] = "ATP_i in component model_parameters (millimolar)"
        legend_constants[127] = "Vt in component model_parameters (cm3)"
        legend_constants[84] = "Vd in component model_parameters (cm3)"
        legend_constants[86] = "Vmyo in component model_parameters (cm3)"
        legend_constants[80] = "Sms in component model_parameters (cm2)"
        legend_constants[90] = "Smt in component model_parameters (cm2)"
        legend_constants[129] = "Cms in component model_parameters (microF)"
        legend_constants[128] = "Cmt in component model_parameters (microF)"
        legend_constants[89] = "Rst in component model_parameters (ohm)"
        legend_constants[87] = "VSRup in component model_parameters (cm3)"
        legend_constants[85] = "VSRrel in component model_parameters (cm3)"
        legend_constants[83] = "Vc in component model_parameters (cm3)"
        legend_constants[88] = "pt in component model_parameters (dimensionless)"
        legend_constants[7] = "Rot in component model_parameters (ohm_cm)"
        legend_constants[8] = "rt in component model_parameters (cm)"
        legend_constants[9] = "Lt in component model_parameters (cm)"
        legend_constants[10] = "rc in component model_parameters (cm)"
        legend_constants[11] = "Lc in component model_parameters (cm)"
        legend_constants[12] = "ptcm in component model_parameters (per_cm2)"
        legend_algebraic[14] = "i_circ in component i_circ (microA)"
        legend_states[0] = "Vm_s in component Vm_s (millivolt)"
        legend_states[1] = "Vm_t in component Vm_t (millivolt)"
        legend_constants[91] = "g_Na_s in component membrane_permeabilities (milliS)"
        legend_constants[92] = "g_Naps_s in component membrane_permeabilities (milliS)"
        legend_constants[93] = "q_Kr_s in component membrane_permeabilities (milliS)"
        legend_constants[94] = "q_Ks_s in component membrane_permeabilities (milliS)"
        legend_constants[95] = "g_Kp_s in component membrane_permeabilities (milliS)"
        legend_constants[96] = "g_Kto_s in component membrane_permeabilities (milliS)"
        legend_constants[97] = "g_K1_s in component membrane_permeabilities (milliS)"
        legend_constants[98] = "g_KNa_s in component membrane_permeabilities (milliS)"
        legend_constants[99] = "g_KATP_s in component membrane_permeabilities (milliS)"
        legend_constants[100] = "g_Nab_s in component membrane_permeabilities (milliS)"
        legend_constants[101] = "g_Cab_s in component membrane_permeabilities (milliS)"
        legend_constants[102] = (
            "i_pCa_max_s in component membrane_permeabilities (microA)"
        )
        legend_constants[103] = (
            "i_NaK_max_s in component membrane_permeabilities (microA)"
        )
        legend_constants[104] = (
            "i_NaCa_max_s in component membrane_permeabilities (microA)"
        )
        legend_constants[105] = (
            "P_CaL_s in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[106] = (
            "P_KL_s in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[107] = (
            "P_nsNa_s in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[108] = (
            "P_nsK_s in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[109] = "g_Na_t in component membrane_permeabilities (milliS)"
        legend_constants[110] = "g_Naps_t in component membrane_permeabilities (milliS)"
        legend_constants[111] = "q_Kr_t in component membrane_permeabilities (milliS)"
        legend_constants[112] = "q_Ks_t in component membrane_permeabilities (milliS)"
        legend_constants[113] = "g_Kp_t in component membrane_permeabilities (milliS)"
        legend_constants[114] = "g_Kto_t in component membrane_permeabilities (milliS)"
        legend_constants[115] = "g_K1_t in component membrane_permeabilities (milliS)"
        legend_constants[116] = "g_KNa_t in component membrane_permeabilities (milliS)"
        legend_constants[117] = "g_KATP_t in component membrane_permeabilities (milliS)"
        legend_constants[118] = "g_Nab_t in component membrane_permeabilities (milliS)"
        legend_constants[119] = "g_Cab_t in component membrane_permeabilities (milliS)"
        legend_constants[120] = (
            "i_pCa_max_t in component membrane_permeabilities (microA)"
        )
        legend_constants[121] = (
            "i_NaK_max_t in component membrane_permeabilities (microA)"
        )
        legend_constants[122] = (
            "i_NaCa_max_t in component membrane_permeabilities (microA)"
        )
        legend_constants[123] = (
            "P_CaL_t in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[124] = (
            "P_KL_t in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[125] = (
            "P_nsNa_t in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[126] = (
            "P_nsK_t in component membrane_permeabilities (litre_per_second)"
        )
        legend_constants[13] = (
            "g_Na in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[14] = (
            "fNat in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[15] = (
            "g_Naps in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[16] = (
            "fNapst in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[17] = (
            "q_Kr in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[18] = (
            "fKrt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[19] = (
            "q_Ks in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[20] = (
            "fKst in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[21] = (
            "g_Kp in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[22] = (
            "fKpt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[23] = (
            "g_Kto in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[24] = (
            "fKtot in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[25] = (
            "g_K1 in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[26] = (
            "fK1t in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[27] = (
            "g_KNa in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[28] = (
            "fKNat in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[29] = (
            "g_KATP in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[30] = (
            "fKATPt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[31] = (
            "g_Nab in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[32] = (
            "fNabt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[33] = (
            "g_Cab in component membrane_permeabilities (milliS_per_cm2)"
        )
        legend_constants[34] = (
            "fCabt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[35] = (
            "P_CaL in component membrane_permeabilities (litre_per_second_cm2)"
        )
        legend_constants[36] = (
            "fCaLt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[37] = (
            "P_KL in component membrane_permeabilities (litre_per_second_cm2)"
        )
        legend_constants[38] = (
            "P_nsNa in component membrane_permeabilities (litre_per_second_cm2)"
        )
        legend_constants[39] = (
            "fnsNat in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[40] = (
            "P_nsK in component membrane_permeabilities (litre_per_second_cm2)"
        )
        legend_constants[41] = (
            "fnsKt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[42] = (
            "i_NaCa_max in component membrane_permeabilities (microA_per_cm2)"
        )
        legend_constants[43] = (
            "fNaCat in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[44] = (
            "i_NaK_max in component membrane_permeabilities (microA_per_cm2)"
        )
        legend_constants[45] = (
            "fNaKt in component membrane_permeabilities (dimensionless)"
        )
        legend_constants[46] = (
            "i_pCa_max in component membrane_permeabilities (microA_per_cm2)"
        )
        legend_constants[47] = (
            "fpCat in component membrane_permeabilities (dimensionless)"
        )
        legend_algebraic[29] = "i_Kext in component i_Kext (microA)"
        legend_constants[48] = "stim_Period in component i_Kext (second)"
        legend_algebraic[35] = "i_Na_s in component i_Na_s (microA)"
        legend_algebraic[32] = "E_Na_s in component i_Na_s (millivolt)"
        legend_states[2] = "Na_i in component ion_concentrations (millimolar)"
        legend_states[3] = "K_i in component ion_concentrations (millimolar)"
        legend_states[4] = "m in component i_Na_s_m_gate (dimensionless)"
        legend_states[5] = "h in component i_Na_s_h_gate (dimensionless)"
        legend_algebraic[0] = "m_infinity in component i_Na_s_m_gate (dimensionless)"
        legend_algebraic[15] = "tau_m in component i_Na_s_m_gate (second)"
        legend_algebraic[1] = "h_infinity in component i_Na_s_h_gate (dimensionless)"
        legend_algebraic[16] = "tau_h in component i_Na_s_h_gate (second)"
        legend_algebraic[39] = "i_Na_t in component i_Na_t (microA)"
        legend_algebraic[38] = "E_Na_t in component i_Na_t (millivolt)"
        legend_states[6] = "Na_t in component ion_concentrations (millimolar)"
        legend_states[7] = "K_t in component ion_concentrations (millimolar)"
        legend_states[8] = "m in component i_Na_t_m_gate (dimensionless)"
        legend_states[9] = "h in component i_Na_t_h_gate (dimensionless)"
        legend_algebraic[2] = "m_infinity in component i_Na_t_m_gate (dimensionless)"
        legend_algebraic[17] = "tau_m in component i_Na_t_m_gate (second)"
        legend_algebraic[3] = "h_infinity in component i_Na_t_h_gate (dimensionless)"
        legend_algebraic[18] = "tau_h in component i_Na_t_h_gate (second)"
        legend_algebraic[40] = "i_Naps_s in component i_Naps_s (microA)"
        legend_algebraic[41] = "i_Naps_t in component i_Naps_t (microA)"
        legend_algebraic[42] = "i_CaL_s in component i_CaL_s (microA)"
        legend_algebraic[44] = "i_KL_s in component i_CaL_s (microA)"
        legend_algebraic[4] = "alfas in component i_CaL_s (per_second)"
        legend_algebraic[19] = "betas in component i_CaL_s (per_second)"
        legend_algebraic[30] = "gama in component i_CaL_s (per_second)"
        legend_constants[49] = "omega in component i_CaL_s (per_second)"
        legend_constants[50] = "a in component i_CaL_s (dimensionless)"
        legend_constants[51] = "b in component i_CaL_s (dimensionless)"
        legend_constants[52] = "f in component i_CaL_s (per_second)"
        legend_constants[53] = "g in component i_CaL_s (per_second)"
        legend_constants[54] = "f2 in component i_CaL_s (per_second)"
        legend_constants[55] = "g2 in component i_CaL_s (per_second)"
        legend_algebraic[33] = "alfa2s in component i_CaL_s (per_second)"
        legend_algebraic[36] = "beta2s in component i_CaL_s (per_second)"
        legend_states[10] = "Cst in component i_CaL_s (dimensionless)"
        legend_states[11] = "C1 in component i_CaL_s (dimensionless)"
        legend_states[12] = "C2 in component i_CaL_s (dimensionless)"
        legend_states[13] = "C3 in component i_CaL_s (dimensionless)"
        legend_states[14] = "C4 in component i_CaL_s (dimensionless)"
        legend_states[15] = "Co in component i_CaL_s (dimensionless)"
        legend_states[16] = "Ccast in component i_CaL_s (dimensionless)"
        legend_states[17] = "Cca1 in component i_CaL_s (dimensionless)"
        legend_states[18] = "Cca2 in component i_CaL_s (dimensionless)"
        legend_states[19] = "Cca3 in component i_CaL_s (dimensionless)"
        legend_states[20] = "Cca4 in component i_CaL_s (dimensionless)"
        legend_states[21] = "Ccao in component i_CaL_s (dimensionless)"
        legend_states[22] = "Ca_ss in component ion_concentrations (millimolar)"
        legend_algebraic[43] = "i_CaL_t in component i_CaL_t (microA)"
        legend_states[23] = "y in component i_CaL_s_y_gate (dimensionless)"
        legend_algebraic[5] = "y_infinity in component i_CaL_s_y_gate (dimensionless)"
        legend_algebraic[20] = "tau_y in component i_CaL_s_y_gate (second)"
        legend_algebraic[45] = "i_KL_t in component i_CaL_t (microA)"
        legend_algebraic[21] = "alfat in component i_CaL_t (per_second)"
        legend_algebraic[31] = "betat in component i_CaL_t (per_second)"
        legend_algebraic[6] = "gama in component i_CaL_t (per_second)"
        legend_constants[56] = "omega in component i_CaL_t (per_second)"
        legend_constants[57] = "a in component i_CaL_t (dimensionless)"
        legend_constants[58] = "b in component i_CaL_t (dimensionless)"
        legend_constants[59] = "f in component i_CaL_t (per_second)"
        legend_constants[60] = "g in component i_CaL_t (per_second)"
        legend_constants[61] = "f2 in component i_CaL_t (per_second)"
        legend_constants[62] = "g2 in component i_CaL_t (per_second)"
        legend_algebraic[34] = "alfa2t in component i_CaL_t (per_second)"
        legend_algebraic[37] = "beta2t in component i_CaL_t (per_second)"
        legend_states[24] = "TCst in component i_CaL_t (dimensionless)"
        legend_states[25] = "TC1 in component i_CaL_t (dimensionless)"
        legend_states[26] = "TC2 in component i_CaL_t (dimensionless)"
        legend_states[27] = "TC3 in component i_CaL_t (dimensionless)"
        legend_states[28] = "TC4 in component i_CaL_t (dimensionless)"
        legend_states[29] = "TCo in component i_CaL_t (dimensionless)"
        legend_states[30] = "TCcast in component i_CaL_t (dimensionless)"
        legend_states[31] = "TCca1 in component i_CaL_t (dimensionless)"
        legend_states[32] = "TCca2 in component i_CaL_t (dimensionless)"
        legend_states[33] = "TCca3 in component i_CaL_t (dimensionless)"
        legend_states[34] = "TCca4 in component i_CaL_t (dimensionless)"
        legend_states[35] = "TCcao in component i_CaL_t (dimensionless)"
        legend_states[36] = "Ca_t in component ion_concentrations (millimolar)"
        legend_states[37] = "y in component i_CaL_t_y_gate (dimensionless)"
        legend_algebraic[7] = "y_infinity in component i_CaL_t_y_gate (dimensionless)"
        legend_algebraic[22] = "tau_y in component i_CaL_t_y_gate (second)"
        legend_algebraic[48] = "i_Kr_s in component i_Kr_s (microA)"
        legend_algebraic[46] = "E_Kr_s in component i_Kr_s (millivolt)"
        legend_states[38] = "xr in component i_Kr_s_xr_gate (dimensionless)"
        legend_algebraic[47] = "xri in component i_Kr_s_xri_gate (dimensionless)"
        legend_algebraic[8] = "xr_infinity in component i_Kr_s_xr_gate (dimensionless)"
        legend_algebraic[23] = "tau_xr in component i_Kr_s_xr_gate (second)"
        legend_algebraic[51] = "i_Kr_t in component i_Kr_t (microA)"
        legend_algebraic[49] = "E_Kr_t in component i_Kr_t (millivolt)"
        legend_states[39] = "xr in component i_Kr_t_xr_gate (dimensionless)"
        legend_algebraic[50] = "xri in component i_Kr_t_xri_gate (dimensionless)"
        legend_algebraic[9] = "xr_infinity in component i_Kr_t_xr_gate (dimensionless)"
        legend_algebraic[24] = "tau_xr in component i_Kr_t_xr_gate (second)"
        legend_algebraic[53] = "i_Ks_s in component i_Ks_s (microA)"
        legend_algebraic[52] = "E_Ks_s in component i_Ks_s (millivolt)"
        legend_constants[63] = "PRNaK in component i_Ks_s (dimensionless)"
        legend_states[40] = "Ca_i in component ion_concentrations (millimolar)"
        legend_states[41] = "xs in component i_Ks_s_xs_gate (dimensionless)"
        legend_algebraic[10] = "xs_infinity in component i_Ks_s_xs_gate (dimensionless)"
        legend_algebraic[25] = "tau_xs in component i_Ks_s_xs_gate (second)"
        legend_algebraic[55] = "i_Ks_t in component i_Ks_t (microA)"
        legend_algebraic[54] = "E_Ks_t in component i_Ks_t (millivolt)"
        legend_constants[64] = "PRNaK in component i_Ks_t (dimensionless)"
        legend_states[42] = "xs in component i_Ks_t_xs_gate (dimensionless)"
        legend_algebraic[11] = "xs_infinity in component i_Ks_t_xs_gate (dimensionless)"
        legend_algebraic[26] = "tau_xs in component i_Ks_t_xs_gate (second)"
        legend_algebraic[59] = "i_K1_s in component i_K1_s (microA)"
        legend_algebraic[56] = "E_K1_s in component i_K1_s (millivolt)"
        legend_algebraic[57] = "aK1s in component i_K1_s (dimensionless)"
        legend_algebraic[58] = "bK1s in component i_K1_s (dimensionless)"
        legend_algebraic[63] = "i_K1_t in component i_K1_t (microA)"
        legend_algebraic[60] = "E_K1_t in component i_K1_t (millivolt)"
        legend_algebraic[61] = "aK1t in component i_K1_t (dimensionless)"
        legend_algebraic[62] = "bK1t in component i_K1_t (dimensionless)"
        legend_algebraic[66] = "i_Kp_s in component i_Kp_s (microA)"
        legend_algebraic[65] = "kps in component i_Kp_s (dimensionless)"
        legend_algebraic[64] = "E_Kp_s in component i_Kp_s (millivolt)"
        legend_algebraic[69] = "i_Kp_t in component i_Kp_t (microA)"
        legend_algebraic[68] = "kpt in component i_Kp_t (dimensionless)"
        legend_algebraic[67] = "E_Kp_t in component i_Kp_t (millivolt)"
        legend_algebraic[71] = "i_Kto_s in component i_Kto_s (microA)"
        legend_constants[65] = "fr2 in component i_Kto_s (dimensionless)"
        legend_algebraic[70] = "E_Kto_s in component i_Kto_s (millivolt)"
        legend_states[43] = "r2 in component i_Kto_s_r2_gate (dimensionless)"
        legend_states[44] = "r3 in component i_Kto_s_r3_gate (dimensionless)"
        legend_constants[66] = "tau_r2 in component i_Kto_s_r2_gate (second)"
        legend_algebraic[12] = (
            "r2_infinity in component i_Kto_s_r2_gate (dimensionless)"
        )
        legend_constants[67] = "tau_r3 in component i_Kto_s_r3_gate (second)"
        legend_algebraic[27] = (
            "r3_infinity in component i_Kto_s_r3_gate (dimensionless)"
        )
        legend_algebraic[73] = "i_Kto_t in component i_Kto_t (microA)"
        legend_constants[68] = "fr2 in component i_Kto_t (dimensionless)"
        legend_algebraic[72] = "E_Kto_t in component i_Kto_t (millivolt)"
        legend_states[45] = "r2 in component i_Kto_t_r2_gate (dimensionless)"
        legend_states[46] = "r3 in component i_Kto_t_r3_gate (dimensionless)"
        legend_constants[69] = "tau_r2 in component i_Kto_t_r2_gate (second)"
        legend_algebraic[13] = (
            "r2_infinity in component i_Kto_t_r2_gate (dimensionless)"
        )
        legend_algebraic[28] = (
            "r3_infinity in component i_Kto_t_r3_gate (dimensionless)"
        )
        legend_constants[70] = "tau_r3 in component i_Kto_t_r3_gate (second)"
        legend_algebraic[75] = "i_KNa_s in component i_KNa_s (microA)"
        legend_algebraic[74] = "E_KNa_s in component i_KNa_s (millivolt)"
        legend_algebraic[77] = "i_KNa_t in component i_KNa_t (microA)"
        legend_algebraic[76] = "E_KNa_t in component i_KNa_t (millivolt)"
        legend_algebraic[78] = "i_nsNa_s in component i_nsNa_s (microA)"
        legend_algebraic[79] = "i_nsNa_t in component i_nsNa_t (microA)"
        legend_algebraic[80] = "i_nsK_s in component i_nsK_s (microA)"
        legend_algebraic[81] = "i_nsK_t in component i_nsK_t (microA)"
        legend_algebraic[82] = "i_Nab_s in component i_Nab_s (microA)"
        legend_algebraic[83] = "i_Nab_t in component i_Nab_t (microA)"
        legend_algebraic[85] = "i_Cab_s in component i_Cab_s (microA)"
        legend_algebraic[84] = "E_Ca_s in component i_Cab_s (millivolt)"
        legend_algebraic[87] = "i_Cab_t in component i_Cab_t (microA)"
        legend_algebraic[86] = "E_Ca_t in component i_Cab_t (millivolt)"
        legend_algebraic[88] = "i_NaCa_s in component i_NaCa_s (microA)"
        legend_algebraic[89] = "i_NaCa_t in component i_NaCa_t (microA)"
        legend_algebraic[90] = "i_NaK_s in component i_NaK_s (microA)"
        legend_algebraic[91] = "i_NaK_t in component i_NaK_t (microA)"
        legend_algebraic[93] = "i_pCa_s in component i_pCa_s (microA)"
        legend_algebraic[94] = "i_pCa_t in component i_pCa_t (microA)"
        legend_algebraic[98] = "i_KATP_s in component i_KATP_s (microA)"
        legend_algebraic[95] = "E_KATP_s in component i_KATP_s (millivolt)"
        legend_algebraic[102] = "i_KATP_t in component i_KATP_t (microA)"
        legend_algebraic[100] = "E_KATP_t in component i_KATP_t (millivolt)"
        legend_algebraic[92] = (
            "JteNa in component t_tubular_ion_fluxes (millimolar_per_second)"
        )
        legend_algebraic[96] = (
            "JteCa in component t_tubular_ion_fluxes (millimolar_per_second)"
        )
        legend_algebraic[104] = (
            "JteK in component t_tubular_ion_fluxes (millimolar_per_second)"
        )
        legend_constants[71] = "tau_Na in component t_tubular_ion_fluxes (second)"
        legend_constants[72] = "tau_Ca in component t_tubular_ion_fluxes (second)"
        legend_constants[73] = "tau_K in component t_tubular_ion_fluxes (second)"
        legend_algebraic[97] = "JCaSRup in component JCaSRup (millimolar_per_second)"
        legend_algebraic[99] = (
            "JCaSRleak in component JCaSRleak (millimolar_per_second)"
        )
        legend_states[47] = "CaSRup in component CaSRup (millimolar)"
        legend_algebraic[101] = "Jtr in component Jtr (millimolar_per_second)"
        legend_constants[74] = "tau_tr in component Jtr (second)"
        legend_states[48] = "CaSRrel in component CaSRrel (millimolar)"
        legend_algebraic[103] = "JCaSRrel in component JCaSRrel (millimolar_per_second)"
        legend_constants[81] = "kap in component JCaSRrel (per_millimolar4_per_second)"
        legend_constants[75] = "kam in component JCaSRrel (per_second)"
        legend_constants[82] = "kbp in component JCaSRrel (per_millimolar3_per_second)"
        legend_constants[76] = "kbm in component JCaSRrel (per_second)"
        legend_constants[77] = "kcp in component JCaSRrel (per_second)"
        legend_constants[78] = "kcm in component JCaSRrel (per_second)"
        legend_states[49] = "F1 in component JCaSRrel (dimensionless)"
        legend_states[50] = "F2 in component JCaSRrel (dimensionless)"
        legend_states[51] = "F3 in component JCaSRrel (dimensionless)"
        legend_states[52] = "F4 in component JCaSRrel (dimensionless)"
        legend_algebraic[105] = "JCad in component JCad (millimolar_per_second)"
        legend_constants[79] = "tau_d in component JCad (second)"
        legend_states[53] = "BTRH in component ion_concentrations (millimolar)"
        legend_states[54] = "BTRL in component ion_concentrations (millimolar)"
        legend_algebraic[106] = (
            "dBTRH in component ion_concentrations (millimolar_per_second)"
        )
        legend_algebraic[107] = (
            "dBTRL in component ion_concentrations (millimolar_per_second)"
        )
        legend_rates[4] = "d/dt m in component i_Na_s_m_gate (dimensionless)"
        legend_rates[5] = "d/dt h in component i_Na_s_h_gate (dimensionless)"
        legend_rates[8] = "d/dt m in component i_Na_t_m_gate (dimensionless)"
        legend_rates[9] = "d/dt h in component i_Na_t_h_gate (dimensionless)"
        legend_rates[10] = "d/dt Cst in component i_CaL_s (dimensionless)"
        legend_rates[11] = "d/dt C1 in component i_CaL_s (dimensionless)"
        legend_rates[12] = "d/dt C2 in component i_CaL_s (dimensionless)"
        legend_rates[13] = "d/dt C3 in component i_CaL_s (dimensionless)"
        legend_rates[14] = "d/dt C4 in component i_CaL_s (dimensionless)"
        legend_rates[15] = "d/dt Co in component i_CaL_s (dimensionless)"
        legend_rates[16] = "d/dt Ccast in component i_CaL_s (dimensionless)"
        legend_rates[17] = "d/dt Cca1 in component i_CaL_s (dimensionless)"
        legend_rates[18] = "d/dt Cca2 in component i_CaL_s (dimensionless)"
        legend_rates[19] = "d/dt Cca3 in component i_CaL_s (dimensionless)"
        legend_rates[20] = "d/dt Cca4 in component i_CaL_s (dimensionless)"
        legend_rates[21] = "d/dt Ccao in component i_CaL_s (dimensionless)"
        legend_rates[23] = "d/dt y in component i_CaL_s_y_gate (dimensionless)"
        legend_rates[24] = "d/dt TCst in component i_CaL_t (dimensionless)"
        legend_rates[25] = "d/dt TC1 in component i_CaL_t (dimensionless)"
        legend_rates[26] = "d/dt TC2 in component i_CaL_t (dimensionless)"
        legend_rates[27] = "d/dt TC3 in component i_CaL_t (dimensionless)"
        legend_rates[28] = "d/dt TC4 in component i_CaL_t (dimensionless)"
        legend_rates[29] = "d/dt TCo in component i_CaL_t (dimensionless)"
        legend_rates[30] = "d/dt TCcast in component i_CaL_t (dimensionless)"
        legend_rates[31] = "d/dt TCca1 in component i_CaL_t (dimensionless)"
        legend_rates[32] = "d/dt TCca2 in component i_CaL_t (dimensionless)"
        legend_rates[33] = "d/dt TCca3 in component i_CaL_t (dimensionless)"
        legend_rates[34] = "d/dt TCca4 in component i_CaL_t (dimensionless)"
        legend_rates[35] = "d/dt TCcao in component i_CaL_t (dimensionless)"
        legend_rates[37] = "d/dt y in component i_CaL_t_y_gate (dimensionless)"
        legend_rates[38] = "d/dt xr in component i_Kr_s_xr_gate (dimensionless)"
        legend_rates[39] = "d/dt xr in component i_Kr_t_xr_gate (dimensionless)"
        legend_rates[41] = "d/dt xs in component i_Ks_s_xs_gate (dimensionless)"
        legend_rates[42] = "d/dt xs in component i_Ks_t_xs_gate (dimensionless)"
        legend_rates[43] = "d/dt r2 in component i_Kto_s_r2_gate (dimensionless)"
        legend_rates[44] = "d/dt r3 in component i_Kto_s_r3_gate (dimensionless)"
        legend_rates[45] = "d/dt r2 in component i_Kto_t_r2_gate (dimensionless)"
        legend_rates[46] = "d/dt r3 in component i_Kto_t_r3_gate (dimensionless)"
        legend_rates[49] = "d/dt F1 in component JCaSRrel (dimensionless)"
        legend_rates[50] = "d/dt F2 in component JCaSRrel (dimensionless)"
        legend_rates[51] = "d/dt F3 in component JCaSRrel (dimensionless)"
        legend_rates[52] = "d/dt F4 in component JCaSRrel (dimensionless)"
        legend_rates[48] = "d/dt CaSRrel in component CaSRrel (millimolar)"
        legend_rates[47] = "d/dt CaSRup in component CaSRup (millimolar)"
        legend_rates[22] = "d/dt Ca_ss in component ion_concentrations (millimolar)"
        legend_rates[40] = "d/dt Ca_i in component ion_concentrations (millimolar)"
        legend_rates[53] = "d/dt BTRH in component ion_concentrations (millimolar)"
        legend_rates[54] = "d/dt BTRL in component ion_concentrations (millimolar)"
        legend_rates[2] = "d/dt Na_i in component ion_concentrations (millimolar)"
        legend_rates[3] = "d/dt K_i in component ion_concentrations (millimolar)"
        legend_rates[6] = "d/dt Na_t in component ion_concentrations (millimolar)"
        legend_rates[36] = "d/dt Ca_t in component ion_concentrations (millimolar)"
        legend_rates[7] = "d/dt K_t in component ion_concentrations (millimolar)"
        legend_rates[0] = "d/dt Vm_s in component Vm_s (millivolt)"
        legend_rates[1] = "d/dt Vm_t in component Vm_t (millivolt)"
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
        algebraic[12] = 1.00600 / (
            1.00000 + np.exp((states[0] + 36.6900 + 10.0000) / 12.4300)
        )
        algebraic[13] = 1.00600 / (
            1.00000 + np.exp((states[1] + 36.6900 + 10.0000) / 12.4300)
        )
        algebraic[0] = 1.00000 / (
            1.00000 + np.exp(((states[0] + 52.2000) - 9.00000) / -7.40000)
        )
        algebraic[15] = (
            0.00100000
            / (
                101.600 * np.exp(states[0] * 0.113500)
                + 0.0226800 * np.exp(-0.0717000 * states[0])
            )
            + 0.000100000
        ) / 4.83000
        algebraic[1] = 1.00000 / (
            1.00000 + np.exp(((states[0] + 85.6000) - 9.00000) / 5.50000)
        )
        algebraic[16] = (
            0.00100000
            / (
                1.13810e-06 * np.exp(-0.101700 * states[0])
                + 6.53700 * np.exp(states[0] * 0.0801600)
            )
            + 0.000500000
        ) / 4.83000
        algebraic[2] = 1.00000 / (
            1.00000 + np.exp(((states[1] + 52.2000) - 9.00000) / -7.40000)
        )
        algebraic[17] = (
            0.00100000
            / (
                101.600 * np.exp(states[1] * 0.113500)
                + 0.0226800 * np.exp(-0.0717000 * states[1])
            )
            + 0.000100000
        ) / 4.83000
        algebraic[3] = 1.00000 / (
            1.00000 + np.exp(((states[1] + 85.6000) - 9.00000) / 5.50000)
        )
        algebraic[18] = (
            0.00100000
            / (
                1.13810e-06 * np.exp(-0.101700 * states[1])
                + 6.53700 * np.exp(states[1] * 0.0801600)
            )
            + 0.000500000
        ) / 4.83000
        algebraic[5] = 1.00000 / (1.00000 + np.exp((states[0] + 35.0000) / 6.00000))
        algebraic[20] = 0.00100000 / (
            0.0200000
            + 0.0197000
            * np.exp(-(np.power((states[0] + 10.0000) * 0.0337000, 2.00000)))
        ) + 0.550000 / (1.00000 + np.exp(((states[0] + 40.0000) / 9.50000) * 4.00000))
        algebraic[7] = 1.00000 / (1.00000 + np.exp((states[1] + 35.0000) / 6.00000))
        algebraic[22] = 0.00100000 / (
            0.0200000
            + 0.0197000
            * np.exp(-(np.power((states[1] + 10.0000) * 0.0337000, 2.00000)))
        ) + 0.550000 / (1.00000 + np.exp(((states[1] + 40.0000) / 9.50000) * 4.00000))
        algebraic[8] = 1.00000 / (1.00000 + np.exp(-(states[0] + 21.5000) / 7.50000))
        algebraic[23] = 0.00100000 / (
            (0.00138000 * (states[0] + 14.2000))
            / (1.00000 - np.exp(-0.123000 * (states[0] + 14.2000)))
            + (0.000610000 * (states[0] + 38.9000))
            / (np.exp(0.145000 * (states[0] + 38.9000)) - 1.00000)
        )
        algebraic[9] = 1.00000 / (1.00000 + np.exp(-(states[1] + 21.5000) / 7.50000))
        algebraic[24] = 0.00100000 / (
            (0.00138000 * (states[1] + 14.2000))
            / (1.00000 - np.exp(-0.123000 * (states[1] + 14.2000)))
            + (0.000610000 * (states[1] + 38.9000))
            / (np.exp(0.145000 * (states[1] + 38.9000)) - 1.00000)
        )
        algebraic[10] = 1.00000 / (1.00000 + np.exp(-(states[0] - 1.50000) / 16.7000))
        algebraic[25] = 0.00100000 / (
            (7.19000e-05 * (states[0] + 30.0000))
            / (1.00000 - np.exp(-0.148000 * (states[0] + 30.0000)))
            + (0.000131000 * (states[0] + 30.0000))
            / (np.exp(0.0687000 * (states[0] + 30.0000)) - 1.00000)
        )
        algebraic[11] = 1.00000 / (1.00000 + np.exp(-(states[1] - 1.50000) / 16.7000))
        algebraic[26] = 0.00100000 / (
            (7.19000e-05 * (states[1] + 30.0000))
            / (1.00000 - np.exp(-0.148000 * (states[1] + 30.0000)))
            + (0.000131000 * (states[1] + 30.0000))
            / (np.exp(0.0687000 * (states[1] + 30.0000)) - 1.00000)
        )
        algebraic[27] = algebraic[12]
        algebraic[28] = algebraic[13]
        algebraic[4] = 400.000 * np.exp((states[0] + 12.0000) / 10.0000)
        algebraic[19] = 50.0000 * np.exp(-(states[0] + 12.0000) / 13.0000)
        algebraic[30] = 187.500 * states[22]
        algebraic[21] = 400.000 * np.exp((states[1] + 12.0000) / 10.0000)
        algebraic[31] = 50.0000 * np.exp(-(states[1] + 12.0000) / 13.0000)
        algebraic[6] = 187.500 * states[22]
        algebraic[33] = algebraic[4] * constants[50]
        algebraic[36] = algebraic[19] / constants[51]
        algebraic[34] = algebraic[21] * constants[57]
        algebraic[37] = algebraic[31] / constants[58]
        algebraic[35] = (
            algebraic[-2]
            * constants[91]
            * (np.power(states[4], 3.00000))
            * states[5]
            * (
                states[0]
                - (
                    np.log(
                        (constants[3] + 0.120000 * constants[5])
                        / (states[2] + 0.120000 * states[3])
                    )
                    * constants[0]
                    * constants[1]
                )
                / constants[2]
            )
        )
        algebraic[39] = (
            algebraic[-2]
            * constants[109]
            * (np.power(states[8], 3.00000))
            * states[9]
            * (
                states[1]
                - (
                    np.log(
                        (states[6] + 0.120000 * states[7])
                        / (states[2] + 0.120000 * states[3])
                    )
                    * constants[0]
                    * constants[1]
                )
                / constants[2]
            )
        )
        algebraic[40] = (
            constants[92] / (1.00000 + np.exp((-54.0000 - states[0]) / 8.00000))
        ) * (
            states[0]
            - (
                np.log(
                    (constants[3] + 0.120000 * constants[5])
                    / (states[2] + 0.120000 * states[3])
                )
                * constants[0]
                * constants[1]
            )
            / constants[2]
        )
        algebraic[41] = (
            constants[110] / (1.00000 + np.exp((-54.0000 - states[1]) / 8.00000))
        ) * (
            states[1]
            - (
                np.log(
                    (states[6] + 0.120000 * states[7])
                    / (states[2] + 0.120000 * states[3])
                )
                * constants[0]
                * constants[1]
            )
            / constants[2]
        )
        algebraic[78] = (
            (
                (
                    (
                        1.00000
                        * constants[107]
                        * states[0]
                        * (np.power(constants[2], 2.00000))
                    )
                    / (constants[0] * constants[1])
                )
                * (
                    0.750000
                    * states[2]
                    * np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                    - 0.750000 * constants[3]
                )
            )
            / (
                np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                - 1.00000
            )
        ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
        algebraic[79] = (
            (
                (
                    (
                        1.00000
                        * constants[125]
                        * states[1]
                        * (np.power(constants[2], 2.00000))
                    )
                    / (constants[0] * constants[1])
                )
                * (
                    0.750000
                    * states[2]
                    * np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                    - 0.750000 * states[6]
                )
            )
            / (
                np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                - 1.00000
            )
        ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
        algebraic[32] = (
            np.log(constants[3] / states[2]) * constants[0] * constants[1]
        ) / constants[2]
        algebraic[82] = constants[100] * (states[0] - algebraic[32])
        algebraic[38] = (
            np.log(states[6] / states[2]) * constants[0] * constants[1]
        ) / constants[2]
        algebraic[83] = constants[118] * (states[1] - algebraic[38])
        algebraic[88] = (
            constants[104]
            * np.exp(
                (-0.850000 * states[0] * constants[2]) / (constants[0] * constants[1])
            )
            * (
                np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                * (np.power(states[2], 3.00000))
                * constants[4]
                - (np.power(constants[3], 3.00000)) * states[40]
            )
        ) / (
            1.00000
            + 0.000100000
            * np.exp(
                (-0.850000 * states[0] * constants[2]) / (constants[0] * constants[1])
            )
            * (
                np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                * (np.power(states[2], 3.00000))
                * constants[4]
                + (np.power(constants[3], 3.00000)) * states[40]
            )
        )
        algebraic[89] = (
            constants[122]
            * np.exp(
                (-0.850000 * states[1] * constants[2]) / (constants[0] * constants[1])
            )
            * (
                np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                * (np.power(states[2], 3.00000))
                * states[36]
                - (np.power(states[6], 3.00000)) * states[40]
            )
        ) / (
            1.00000
            + 0.000100000
            * np.exp(
                (-0.850000 * states[1] * constants[2]) / (constants[0] * constants[1])
            )
            * (
                np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                * (np.power(states[2], 3.00000))
                * states[36]
                + (np.power(states[6], 3.00000)) * states[40]
            )
        )
        algebraic[90] = (
            (
                (
                    (
                        constants[103]
                        / (
                            1.00000
                            + 0.124500
                            * np.exp(
                                (-0.100000 * states[0] * constants[2])
                                / (constants[0] * constants[1])
                            )
                            + ((0.0365000 * 1.00000) / 7.00000)
                            * (np.exp(constants[3] / 67.3000) - 1.00000)
                            * np.exp(
                                (-states[0] * constants[2])
                                / (constants[0] * constants[1])
                            )
                        )
                    )
                    * 1.00000
                )
                / (1.00000 + np.power(10.0000 / states[2], 1.50000))
            )
            * constants[5]
        ) / (constants[5] + 1.50000)
        algebraic[91] = (
            (
                (
                    (
                        constants[121]
                        / (
                            1.00000
                            + 0.124500
                            * np.exp(
                                (-0.100000 * states[1] * constants[2])
                                / (constants[0] * constants[1])
                            )
                            + ((0.0365000 * 1.00000) / 7.00000)
                            * (np.exp(states[6] / 67.3000) - 1.00000)
                            * np.exp(
                                (-states[1] * constants[2])
                                / (constants[0] * constants[1])
                            )
                        )
                    )
                    * 1.00000
                )
                / (1.00000 + np.power(10.0000 / states[2], 1.50000))
            )
            * states[7]
        ) / (states[7] + 1.50000)
        algebraic[92] = ((1.00000 * constants[127]) / constants[71]) * (
            states[6] - constants[3]
        )
        algebraic[43] = (
            algebraic[-5]
            * (
                (
                    (
                        1.00000
                        * constants[123]
                        * 4.00000
                        * (states[29] + states[35])
                        * states[37]
                        * states[1]
                        * (np.power(constants[2], 2.00000))
                    )
                    / (constants[0] * constants[1])
                )
                * (
                    0.00100000
                    * np.exp(
                        (2.00000 * states[1] * constants[2])
                        / (constants[0] * constants[1])
                    )
                    - 0.341000 * states[36]
                )
            )
            / (
                np.exp(
                    (2.00000 * states[1] * constants[2]) / (constants[0] * constants[1])
                )
                - 1.00000
            )
        )
        algebraic[86] = (
            np.log(states[36] / states[40]) * constants[0] * constants[1]
        ) / (2.00000 * constants[2])
        algebraic[87] = constants[119] * (states[1] - algebraic[86])
        algebraic[94] = (constants[120] * states[40]) / (0.000500000 + states[40])
        algebraic[96] = ((1.00000 * constants[127]) / constants[72]) * (
            states[36] - constants[4]
        )
        algebraic[14] = 1000.00 * (
            states[1] / constants[89] - states[0] / constants[89]
        )
        algebraic[29] = super().custom_piecewise(
            [
                np.greater_equal(
                    voi - np.floor(voi / constants[48]) * constants[48], 0.00000
                )
                & np.less_equal(
                    voi - np.floor(voi / constants[48]) * constants[48], 0.00100000
                ),
                45.0000 * (constants[80] + constants[90]),
                True,
                0.00000,
            ]
        )
        algebraic[42] = (
            algebraic[-5]
            * (
                (
                    (
                        1.00000
                        * constants[105]
                        * 4.00000
                        * (states[15] + states[21])
                        * states[23]
                        * states[0]
                        * (np.power(constants[2], 2.00000))
                    )
                    / (constants[0] * constants[1])
                )
                * (
                    0.00100000
                    * np.exp(
                        (2.00000 * states[0] * constants[2])
                        / (constants[0] * constants[1])
                    )
                    - 0.341000 * constants[4]
                )
            )
            / (
                np.exp(
                    (2.00000 * states[0] * constants[2]) / (constants[0] * constants[1])
                )
                - 1.00000
            )
        )
        algebraic[44] = (
            (
                (
                    (
                        (1.00000 * constants[106])
                        / (
                            1.00000
                            - (algebraic[42] + algebraic[43])
                            / (0.458000 * (constants[80] + constants[90]))
                        )
                    )
                    * (states[15] + states[21])
                    * states[23]
                    * states[0]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                states[3]
                * np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                - constants[5]
            )
        ) / (
            np.exp((states[0] * constants[2]) / (constants[0] * constants[1])) - 1.00000
        )
        algebraic[46] = (
            np.log(constants[5] / states[3]) * constants[0] * constants[1]
        ) / constants[2]
        algebraic[47] = 1.00000 / (1.00000 + np.exp((states[0] + 9.00000) / 22.4000))
        algebraic[48] = (
            algebraic[-1]
            * constants[93]
            * 0.0261400
            * (np.power(constants[5] / 5.40000, 1.0 / 2))
            * states[38]
            * algebraic[47]
            * (states[0] - algebraic[46])
        )
        algebraic[52] = (
            np.log(
                (constants[5] + constants[63] * constants[3])
                / (states[3] + constants[63] * states[2])
            )
            * constants[0]
            * constants[1]
        ) / constants[2]
        algebraic[53] = (
            algebraic[-3]
            * constants[94]
            * (
                0.230800
                + 0.769200
                / (
                    1.00000
                    + np.exp((-np.log10(1.00000 * states[40]) - 4.20000) / 0.600000)
                )
            )
            * (np.power(states[41], 2.00000))
            * (states[0] - algebraic[52])
        )
        algebraic[56] = algebraic[46]
        algebraic[57] = 1020.00 / (
            1.00000 + np.exp(0.238500 * ((states[0] - algebraic[56]) - 59.2150))
        )
        algebraic[58] = (
            1000.00
            * (
                0.491240 * np.exp(0.0803200 * ((states[0] - algebraic[56]) + 5.47600))
                + np.exp(0.0617500 * ((states[0] - algebraic[56]) - 594.310))
            )
        ) / (1.00000 + np.exp(-0.514300 * ((states[0] - algebraic[56]) + 4.75300)))
        algebraic[59] = (
            algebraic[-4]
            * (
                (
                    constants[97]
                    * (np.power(constants[5] / 5.40000, 1.0 / 2))
                    * algebraic[57]
                )
                / (algebraic[57] + algebraic[58])
            )
            * (states[0] - algebraic[56])
        )
        algebraic[65] = 1.00000 / (1.00000 + np.exp((20.0000 - states[0]) / 5.00000))
        algebraic[64] = algebraic[46]
        algebraic[66] = constants[95] * algebraic[65] * (states[0] - algebraic[64])
        algebraic[70] = algebraic[46]
        algebraic[71] = (
            (
                constants[96]
                / (
                    (1.00000 + np.exp((states[0] + 57.5300) / -5.86300))
                    * (1.00000 + np.exp((states[0] - 45.8000) / 25.8700))
                )
            )
            * (states[43] * constants[65] + states[44] * (1.00000 - constants[65]))
            * (states[0] - algebraic[70])
        )
        algebraic[74] = algebraic[46]
        algebraic[75] = (
            (
                constants[98]
                * (
                    0.800000
                    - 0.650000 / (1.00000 + np.exp((states[0] + 125.000) / 15.0000))
                )
                * 0.850000
            )
            / (1.00000 + np.power(66.0000 / states[2], 2.80000))
        ) * (states[0] - algebraic[74])
        algebraic[80] = (
            (
                (
                    (
                        1.00000
                        * constants[108]
                        * states[0]
                        * (np.power(constants[2], 2.00000))
                    )
                    / (constants[0] * constants[1])
                )
                * (
                    0.750000
                    * states[3]
                    * np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                    - 0.750000 * constants[5]
                )
            )
            / (
                np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                - 1.00000
            )
        ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
        algebraic[84] = (
            np.log(constants[4] / states[40]) * constants[0] * constants[1]
        ) / (2.00000 * constants[2])
        algebraic[85] = constants[101] * (states[0] - algebraic[84])
        algebraic[93] = (constants[102] * states[40]) / (0.000500000 + states[40])
        algebraic[95] = algebraic[46]
        algebraic[98] = (
            (
                (constants[99] * 1.00000)
                / (1.00000 + np.power(constants[6] / 0.114000, 2.00000))
            )
            * (np.power(constants[5] / 4.00000, 0.240000))
            * (states[0] - algebraic[95])
        )
        algebraic[97] = (
            1.00000
            * 1000.00
            * constants[86]
            * 0.00180000
            * (np.power(states[40], 2.00000))
        ) / (np.power(states[40], 2.00000) + np.power(0.000500000, 2.00000))
        algebraic[99] = (
            1250.00 * constants[86] * 5.80000e-05 * (states[47] - states[40])
        )
        algebraic[101] = ((1.00000 * constants[85]) / constants[74]) * (
            states[47] - states[48]
        )
        algebraic[103] = (
            1800.00
            * constants[85]
            * (states[50] + states[51])
            * (states[48] - states[22])
        )
        algebraic[45] = (
            (
                (
                    (
                        (1.00000 * constants[124])
                        / (
                            1.00000
                            - (algebraic[42] + algebraic[43])
                            / (0.458000 * (constants[80] + constants[90]))
                        )
                    )
                    * (states[29] + states[35])
                    * states[37]
                    * states[1]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                states[3]
                * np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                - states[7]
            )
        ) / (
            np.exp((states[1] * constants[2]) / (constants[0] * constants[1])) - 1.00000
        )
        algebraic[49] = (
            np.log(states[7] / states[3]) * constants[0] * constants[1]
        ) / constants[2]
        algebraic[50] = 1.00000 / (1.00000 + np.exp((states[1] + 9.00000) / 22.4000))
        algebraic[51] = (
            algebraic[-1]
            * constants[111]
            * 0.0261400
            * (np.power(states[7] / 5.40000, 1.0 / 2))
            * states[39]
            * algebraic[50]
            * (states[1] - algebraic[49])
        )
        algebraic[54] = (
            np.log(
                (states[7] + constants[64] * states[6])
                / (states[3] + constants[64] * states[2])
            )
            * constants[0]
            * constants[1]
        ) / constants[2]
        algebraic[55] = (
            algebraic[-3]
            * constants[112]
            * (
                0.230800
                + 0.769200
                / (
                    1.00000
                    + np.exp((-np.log10(1.00000 * states[40]) - 4.20000) / 0.600000)
                )
            )
            * (np.power(states[42], 2.00000))
            * (states[1] - algebraic[54])
        )
        algebraic[60] = algebraic[49]
        algebraic[61] = 1020.00 / (
            1.00000 + np.exp(0.238500 * ((states[1] - algebraic[60]) - 59.2150))
        )
        algebraic[62] = (
            1000.00
            * (
                0.491240 * np.exp(0.0803200 * ((states[1] - algebraic[60]) + 5.47600))
                + np.exp(0.0617500 * ((states[1] - algebraic[60]) - 594.310))
            )
        ) / (1.00000 + np.exp(-0.514300 * ((states[1] - algebraic[60]) + 4.75300)))
        algebraic[63] = (
            algebraic[-4]
            * (
                (
                    constants[115]
                    * (np.power(states[7] / 5.40000, 1.0 / 2))
                    * algebraic[61]
                )
                / (algebraic[61] + algebraic[62])
            )
            * (states[1] - algebraic[60])
        )
        algebraic[68] = 1.00000 / (1.00000 + np.exp((20.0000 - states[1]) / 5.00000))
        algebraic[67] = algebraic[49]
        algebraic[69] = constants[113] * algebraic[68] * (states[1] - algebraic[67])
        algebraic[72] = algebraic[49]
        algebraic[73] = (
            (
                constants[114]
                / (
                    (1.00000 + np.exp((states[1] + 57.5300) / -5.86300))
                    * (1.00000 + np.exp((states[1] - 45.8000) / 25.8700))
                )
            )
            * (states[45] * constants[68] + states[46] * (1.00000 - constants[68]))
            * (states[1] - algebraic[72])
        )
        algebraic[76] = algebraic[49]
        algebraic[77] = (
            (
                constants[116]
                * (
                    0.800000
                    - 0.650000 / (1.00000 + np.exp((states[1] + 125.000) / 15.0000))
                )
                * 0.850000
            )
            / (1.00000 + np.power(66.0000 / states[2], 2.80000))
        ) * (states[1] - algebraic[76])
        algebraic[81] = (
            (
                (
                    (
                        1.00000
                        * constants[126]
                        * states[1]
                        * (np.power(constants[2], 2.00000))
                    )
                    / (constants[0] * constants[1])
                )
                * (
                    0.750000
                    * states[3]
                    * np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                    - 0.750000 * states[7]
                )
            )
            / (
                np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                - 1.00000
            )
        ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
        algebraic[100] = algebraic[49]
        algebraic[102] = (
            (
                (constants[117] * 1.00000)
                / (1.00000 + np.power(constants[6] / 0.114000, 2.00000))
            )
            * (np.power(states[7] / 4.00000, 0.240000))
            * (states[1] - algebraic[100])
        )
        algebraic[105] = (
            1.00000 * (states[22] - states[40]) * constants[86]
        ) / constants[79]
        algebraic[104] = ((1.00000 * constants[127]) / constants[73]) * (
            states[7] - constants[5]
        )
        algebraic[106] = (
            20000.0 * states[40] * (1.00000 - states[53]) - 0.0700000 * states[53]
        )
        algebraic[107] = (
            40000.0 * states[40] * (1.00000 - states[54]) - states[54] * 40.0000
        )
        return algebraic


@njit
def njitComputeRates(
    voi, states, constants, sizeStates, sizeAlgebraic, num_channels, drug_multipliers
):
    """Compute ODEs for the model."""
    rates = np.zeros(sizeStates, dtype=np.float64)
    algebraic = np.zeros(sizeAlgebraic, dtype=np.float64)
    algebraic[-num_channels:] = drug_multipliers
    rates[15] = constants[52] * states[14] - constants[53] * states[15]
    rates[21] = constants[54] * states[20] - constants[55] * states[21]
    rates[29] = constants[59] * states[28] - constants[60] * states[29]
    rates[35] = constants[61] * states[34] - constants[62] * states[35]
    rates[49] = states[50] * constants[75] - states[49] * constants[81] * (
        np.power(states[22], 4.00000)
    )
    rates[50] = (
        states[49] * constants[81] * (np.power(states[22], 4.00000))
        + states[51] * constants[76]
        + states[52] * constants[78]
    ) - states[50] * (
        constants[75] + constants[82] * (np.power(states[22], 3.00000)) + constants[77]
    )
    rates[51] = (
        states[50] * constants[82] * (np.power(states[22], 3.00000))
        - states[51] * constants[76]
    )
    rates[52] = states[50] * constants[77] - states[52] * constants[78]
    algebraic[12] = 1.00600 / (
        1.00000 + np.exp((states[0] + 36.6900 + 10.0000) / 12.4300)
    )
    rates[43] = (algebraic[12] - states[43]) / constants[66]
    algebraic[13] = 1.00600 / (
        1.00000 + np.exp((states[1] + 36.6900 + 10.0000) / 12.4300)
    )
    rates[45] = (algebraic[13] - states[45]) / constants[69]
    algebraic[0] = 1.00000 / (
        1.00000 + np.exp(((states[0] + 52.2000) - 9.00000) / -7.40000)
    )
    algebraic[15] = (
        0.00100000
        / (
            101.600 * np.exp(states[0] * 0.113500)
            + 0.0226800 * np.exp(-0.0717000 * states[0])
        )
        + 0.000100000
    ) / 4.83000
    rates[4] = (algebraic[0] - states[4]) / algebraic[15]
    algebraic[1] = 1.00000 / (
        1.00000 + np.exp(((states[0] + 85.6000) - 9.00000) / 5.50000)
    )
    algebraic[16] = (
        0.00100000
        / (
            1.13810e-06 * np.exp(-0.101700 * states[0])
            + 6.53700 * np.exp(states[0] * 0.0801600)
        )
        + 0.000500000
    ) / 4.83000
    rates[5] = (algebraic[1] - states[5]) / algebraic[16]
    algebraic[2] = 1.00000 / (
        1.00000 + np.exp(((states[1] + 52.2000) - 9.00000) / -7.40000)
    )
    algebraic[17] = (
        0.00100000
        / (
            101.600 * np.exp(states[1] * 0.113500)
            + 0.0226800 * np.exp(-0.0717000 * states[1])
        )
        + 0.000100000
    ) / 4.83000
    rates[8] = (algebraic[2] - states[8]) / algebraic[17]
    algebraic[3] = 1.00000 / (
        1.00000 + np.exp(((states[1] + 85.6000) - 9.00000) / 5.50000)
    )
    algebraic[18] = (
        0.00100000
        / (
            1.13810e-06 * np.exp(-0.101700 * states[1])
            + 6.53700 * np.exp(states[1] * 0.0801600)
        )
        + 0.000500000
    ) / 4.83000
    rates[9] = (algebraic[3] - states[9]) / algebraic[18]
    algebraic[5] = 1.00000 / (1.00000 + np.exp((states[0] + 35.0000) / 6.00000))
    algebraic[20] = 0.00100000 / (
        0.0200000
        + 0.0197000 * np.exp(-(np.power((states[0] + 10.0000) * 0.0337000, 2.00000)))
    ) + 0.550000 / (1.00000 + np.exp(((states[0] + 40.0000) / 9.50000) * 4.00000))
    rates[23] = (algebraic[5] - states[23]) / algebraic[20]
    algebraic[7] = 1.00000 / (1.00000 + np.exp((states[1] + 35.0000) / 6.00000))
    algebraic[22] = 0.00100000 / (
        0.0200000
        + 0.0197000 * np.exp(-(np.power((states[1] + 10.0000) * 0.0337000, 2.00000)))
    ) + 0.550000 / (1.00000 + np.exp(((states[1] + 40.0000) / 9.50000) * 4.00000))
    rates[37] = (algebraic[7] - states[37]) / algebraic[22]
    algebraic[8] = 1.00000 / (1.00000 + np.exp(-(states[0] + 21.5000) / 7.50000))
    algebraic[23] = 0.00100000 / (
        (0.00138000 * (states[0] + 14.2000))
        / (1.00000 - np.exp(-0.123000 * (states[0] + 14.2000)))
        + (0.000610000 * (states[0] + 38.9000))
        / (np.exp(0.145000 * (states[0] + 38.9000)) - 1.00000)
    )
    rates[38] = (algebraic[8] - states[38]) / algebraic[23]
    algebraic[9] = 1.00000 / (1.00000 + np.exp(-(states[1] + 21.5000) / 7.50000))
    algebraic[24] = 0.00100000 / (
        (0.00138000 * (states[1] + 14.2000))
        / (1.00000 - np.exp(-0.123000 * (states[1] + 14.2000)))
        + (0.000610000 * (states[1] + 38.9000))
        / (np.exp(0.145000 * (states[1] + 38.9000)) - 1.00000)
    )
    rates[39] = (algebraic[9] - states[39]) / algebraic[24]
    algebraic[10] = 1.00000 / (1.00000 + np.exp(-(states[0] - 1.50000) / 16.7000))
    algebraic[25] = 0.00100000 / (
        (7.19000e-05 * (states[0] + 30.0000))
        / (1.00000 - np.exp(-0.148000 * (states[0] + 30.0000)))
        + (0.000131000 * (states[0] + 30.0000))
        / (np.exp(0.0687000 * (states[0] + 30.0000)) - 1.00000)
    )
    rates[41] = (algebraic[10] - states[41]) / algebraic[25]
    algebraic[11] = 1.00000 / (1.00000 + np.exp(-(states[1] - 1.50000) / 16.7000))
    algebraic[26] = 0.00100000 / (
        (7.19000e-05 * (states[1] + 30.0000))
        / (1.00000 - np.exp(-0.148000 * (states[1] + 30.0000)))
        + (0.000131000 * (states[1] + 30.0000))
        / (np.exp(0.0687000 * (states[1] + 30.0000)) - 1.00000)
    )
    rates[42] = (algebraic[11] - states[42]) / algebraic[26]
    algebraic[27] = algebraic[12]
    rates[44] = (algebraic[27] - states[44]) / constants[67]
    algebraic[28] = algebraic[13]
    rates[46] = (algebraic[28] - states[46]) / constants[70]
    algebraic[4] = 400.000 * np.exp((states[0] + 12.0000) / 10.0000)
    algebraic[19] = 50.0000 * np.exp(-(states[0] + 12.0000) / 13.0000)
    algebraic[30] = 187.500 * states[22]
    rates[10] = (algebraic[19] * states[11] + constants[49] * states[16]) - (
        4.00000 * algebraic[4] + algebraic[30]
    ) * states[10]
    rates[11] = (
        4.00000 * algebraic[4] * states[10]
        + 2.00000 * algebraic[19] * states[12]
        + (constants[49] / constants[51]) * states[17]
    ) - (
        algebraic[19] + 3.00000 * algebraic[4] + algebraic[30] * constants[50]
    ) * states[11]
    rates[12] = (
        3.00000 * algebraic[4] * states[11]
        + 3.00000 * algebraic[19] * states[13]
        + (constants[49] / (np.power(constants[51], 2.00000))) * states[18]
    ) - (
        2.00000 * algebraic[19]
        + 2.00000 * algebraic[4]
        + algebraic[30] * (np.power(constants[50], 2.00000))
    ) * states[12]
    rates[13] = (
        2.00000 * algebraic[4] * states[12]
        + 4.00000 * algebraic[19] * states[14]
        + (constants[49] / (np.power(constants[51], 3.00000))) * states[19]
    ) - (
        3.00000 * algebraic[19]
        + algebraic[4]
        + algebraic[30] * (np.power(constants[50], 3.00000))
    ) * states[13]
    rates[14] = (
        algebraic[4] * states[13]
        + constants[53] * states[15]
        + (constants[49] / (np.power(constants[51], 4.00000))) * states[20]
    ) - (
        4.00000 * algebraic[19]
        + constants[52]
        + algebraic[30] * (np.power(constants[50], 4.00000))
    ) * states[14]
    algebraic[21] = 400.000 * np.exp((states[1] + 12.0000) / 10.0000)
    algebraic[31] = 50.0000 * np.exp(-(states[1] + 12.0000) / 13.0000)
    algebraic[6] = 187.500 * states[22]
    rates[24] = (algebraic[31] * states[25] + constants[56] * states[30]) - (
        4.00000 * algebraic[21] + algebraic[6]
    ) * states[24]
    rates[25] = (
        4.00000 * algebraic[21] * states[24]
        + 2.00000 * algebraic[31] * states[26]
        + (constants[56] / constants[58]) * states[31]
    ) - (
        algebraic[31] + 3.00000 * algebraic[21] + algebraic[6] * constants[57]
    ) * states[25]
    rates[26] = (
        3.00000 * algebraic[21] * states[25]
        + 3.00000 * algebraic[31] * states[27]
        + (constants[56] / (np.power(constants[58], 2.00000))) * states[32]
    ) - (
        2.00000 * algebraic[31]
        + 2.00000 * algebraic[21]
        + algebraic[6] * (np.power(constants[57], 2.00000))
    ) * states[26]
    rates[27] = (
        2.00000 * algebraic[21] * states[26]
        + 4.00000 * algebraic[31] * states[28]
        + (constants[56] / (np.power(constants[58], 3.00000))) * states[33]
    ) - (
        3.00000 * algebraic[31]
        + algebraic[21]
        + algebraic[6] * (np.power(constants[57], 3.00000))
    ) * states[27]
    rates[28] = (
        algebraic[21] * states[27]
        + constants[60] * states[29]
        + (constants[56] / (np.power(constants[58], 4.00000))) * states[34]
    ) - (
        4.00000 * algebraic[31]
        + constants[59]
        + algebraic[6] * (np.power(constants[57], 4.00000))
    ) * states[28]
    algebraic[33] = algebraic[4] * constants[50]
    algebraic[36] = algebraic[19] / constants[51]
    rates[16] = (algebraic[36] * states[17] + algebraic[30] * states[10]) - (
        4.00000 * algebraic[33] + constants[49]
    ) * states[16]
    rates[17] = (
        4.00000 * algebraic[33] * states[16]
        + 2.00000 * algebraic[36] * states[18]
        + algebraic[30] * constants[50] * states[11]
    ) - (
        algebraic[36] + 3.00000 * algebraic[33] + constants[49] / constants[51]
    ) * states[17]
    rates[18] = (
        3.00000 * algebraic[33] * states[17]
        + 3.00000 * algebraic[36] * states[19]
        + algebraic[30] * (np.power(constants[50], 2.00000)) * states[12]
    ) - (
        2.00000 * algebraic[36]
        + 2.00000 * algebraic[33]
        + constants[49] / (np.power(constants[51], 2.00000))
    ) * states[18]
    rates[19] = (
        2.00000 * algebraic[33] * states[18]
        + 4.00000 * algebraic[36] * states[20]
        + algebraic[30] * (np.power(constants[50], 3.00000)) * states[13]
    ) - (
        3.00000 * algebraic[36]
        + algebraic[33]
        + constants[49] / (np.power(constants[51], 3.00000))
    ) * states[19]
    rates[20] = (
        algebraic[33] * states[19]
        + constants[55] * states[21]
        + algebraic[30] * (np.power(constants[50], 4.00000)) * states[14]
    ) - (
        4.00000 * algebraic[36]
        + constants[54]
        + constants[49] / (np.power(constants[51], 4.00000))
    ) * states[20]
    algebraic[34] = algebraic[21] * constants[57]
    algebraic[37] = algebraic[31] / constants[58]
    rates[30] = (algebraic[37] * states[31] + algebraic[6] * states[24]) - (
        4.00000 * algebraic[34] + constants[56]
    ) * states[30]
    rates[31] = (
        4.00000 * algebraic[34] * states[30]
        + 2.00000 * algebraic[37] * states[32]
        + algebraic[6] * constants[57] * states[25]
    ) - (
        algebraic[37] + 3.00000 * algebraic[34] + constants[56] / constants[58]
    ) * states[31]
    rates[32] = (
        3.00000 * algebraic[34] * states[31]
        + 3.00000 * algebraic[37] * states[33]
        + algebraic[6] * (np.power(constants[57], 2.00000)) * states[26]
    ) - (
        2.00000 * algebraic[37]
        + 2.00000 * algebraic[34]
        + constants[56] / (np.power(constants[58], 2.00000))
    ) * states[32]
    rates[33] = (
        2.00000 * algebraic[34] * states[32]
        + 4.00000 * algebraic[37] * states[34]
        + algebraic[6] * (np.power(constants[57], 3.00000)) * states[27]
    ) - (
        3.00000 * algebraic[37]
        + algebraic[34]
        + constants[56] / (np.power(constants[58], 3.00000))
    ) * states[33]
    rates[34] = (
        algebraic[34] * states[33]
        + constants[62] * states[35]
        + algebraic[6] * (np.power(constants[57], 4.00000)) * states[28]
    ) - (
        4.00000 * algebraic[37]
        + constants[61]
        + constants[56] / (np.power(constants[58], 4.00000))
    ) * states[34]
    algebraic[35] = (
        algebraic[-2]
        * constants[91]
        * (np.power(states[4], 3.00000))
        * states[5]
        * (
            states[0]
            - (
                np.log(
                    (constants[3] + 0.120000 * constants[5])
                    / (states[2] + 0.120000 * states[3])
                )
                * constants[0]
                * constants[1]
            )
            / constants[2]
        )
    )
    algebraic[39] = (
        algebraic[-2]
        * constants[109]
        * (np.power(states[8], 3.00000))
        * states[9]
        * (
            states[1]
            - (
                np.log(
                    (states[6] + 0.120000 * states[7])
                    / (states[2] + 0.120000 * states[3])
                )
                * constants[0]
                * constants[1]
            )
            / constants[2]
        )
    )
    algebraic[40] = (
        constants[92] / (1.00000 + np.exp((-54.0000 - states[0]) / 8.00000))
    ) * (
        states[0]
        - (
            np.log(
                (constants[3] + 0.120000 * constants[5])
                / (states[2] + 0.120000 * states[3])
            )
            * constants[0]
            * constants[1]
        )
        / constants[2]
    )
    algebraic[41] = (
        constants[110] / (1.00000 + np.exp((-54.0000 - states[1]) / 8.00000))
    ) * (
        states[1]
        - (
            np.log(
                (states[6] + 0.120000 * states[7]) / (states[2] + 0.120000 * states[3])
            )
            * constants[0]
            * constants[1]
        )
        / constants[2]
    )
    algebraic[78] = (
        (
            (
                (
                    1.00000
                    * constants[107]
                    * states[0]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                0.750000
                * states[2]
                * np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                - 0.750000 * constants[3]
            )
        )
        / (np.exp((states[0] * constants[2]) / (constants[0] * constants[1])) - 1.00000)
    ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
    algebraic[79] = (
        (
            (
                (
                    1.00000
                    * constants[125]
                    * states[1]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                0.750000
                * states[2]
                * np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                - 0.750000 * states[6]
            )
        )
        / (np.exp((states[1] * constants[2]) / (constants[0] * constants[1])) - 1.00000)
    ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
    algebraic[32] = (
        np.log(constants[3] / states[2]) * constants[0] * constants[1]
    ) / constants[2]
    algebraic[82] = constants[100] * (states[0] - algebraic[32])
    algebraic[38] = (
        np.log(states[6] / states[2]) * constants[0] * constants[1]
    ) / constants[2]
    algebraic[83] = constants[118] * (states[1] - algebraic[38])
    algebraic[88] = (
        constants[104]
        * np.exp((-0.850000 * states[0] * constants[2]) / (constants[0] * constants[1]))
        * (
            np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
            * (np.power(states[2], 3.00000))
            * constants[4]
            - (np.power(constants[3], 3.00000)) * states[40]
        )
    ) / (
        1.00000
        + 0.000100000
        * np.exp((-0.850000 * states[0] * constants[2]) / (constants[0] * constants[1]))
        * (
            np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
            * (np.power(states[2], 3.00000))
            * constants[4]
            + (np.power(constants[3], 3.00000)) * states[40]
        )
    )
    algebraic[89] = (
        constants[122]
        * np.exp((-0.850000 * states[1] * constants[2]) / (constants[0] * constants[1]))
        * (
            np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
            * (np.power(states[2], 3.00000))
            * states[36]
            - (np.power(states[6], 3.00000)) * states[40]
        )
    ) / (
        1.00000
        + 0.000100000
        * np.exp((-0.850000 * states[1] * constants[2]) / (constants[0] * constants[1]))
        * (
            np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
            * (np.power(states[2], 3.00000))
            * states[36]
            + (np.power(states[6], 3.00000)) * states[40]
        )
    )
    algebraic[90] = (
        (
            (
                (
                    constants[103]
                    / (
                        1.00000
                        + 0.124500
                        * np.exp(
                            (-0.100000 * states[0] * constants[2])
                            / (constants[0] * constants[1])
                        )
                        + ((0.0365000 * 1.00000) / 7.00000)
                        * (np.exp(constants[3] / 67.3000) - 1.00000)
                        * np.exp(
                            (-states[0] * constants[2]) / (constants[0] * constants[1])
                        )
                    )
                )
                * 1.00000
            )
            / (1.00000 + np.power(10.0000 / states[2], 1.50000))
        )
        * constants[5]
    ) / (constants[5] + 1.50000)
    algebraic[91] = (
        (
            (
                (
                    constants[121]
                    / (
                        1.00000
                        + 0.124500
                        * np.exp(
                            (-0.100000 * states[1] * constants[2])
                            / (constants[0] * constants[1])
                        )
                        + ((0.0365000 * 1.00000) / 7.00000)
                        * (np.exp(states[6] / 67.3000) - 1.00000)
                        * np.exp(
                            (-states[1] * constants[2]) / (constants[0] * constants[1])
                        )
                    )
                )
                * 1.00000
            )
            / (1.00000 + np.power(10.0000 / states[2], 1.50000))
        )
        * states[7]
    ) / (states[7] + 1.50000)
    rates[2] = -(
        algebraic[35]
        + algebraic[39]
        + algebraic[40]
        + algebraic[41]
        + algebraic[78]
        + algebraic[79]
        + algebraic[82]
        + algebraic[83]
        + 3.00000 * algebraic[88]
        + 3.00000 * algebraic[89]
        + 3.00000 * algebraic[90]
        + 3.00000 * algebraic[91]
    ) / (constants[2] * constants[86])
    algebraic[92] = ((1.00000 * constants[127]) / constants[71]) * (
        states[6] - constants[3]
    )
    rates[6] = (
        (
            algebraic[39]
            + algebraic[41]
            + algebraic[79]
            + algebraic[83]
            + 3.00000 * algebraic[89]
            + 3.00000 * algebraic[91]
        )
        / constants[2]
        - 1.00000 * algebraic[92]
    ) / constants[127]
    algebraic[43] = (
        algebraic[-5]
        * (
            (
                (
                    1.00000
                    * constants[123]
                    * 4.00000
                    * (states[29] + states[35])
                    * states[37]
                    * states[1]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                0.00100000
                * np.exp(
                    (2.00000 * states[1] * constants[2]) / (constants[0] * constants[1])
                )
                - 0.341000 * states[36]
            )
        )
        / (
            np.exp((2.00000 * states[1] * constants[2]) / (constants[0] * constants[1]))
            - 1.00000
        )
    )
    algebraic[86] = (np.log(states[36] / states[40]) * constants[0] * constants[1]) / (
        2.00000 * constants[2]
    )
    algebraic[87] = constants[119] * (states[1] - algebraic[86])
    algebraic[94] = (constants[120] * states[40]) / (0.000500000 + states[40])
    algebraic[96] = ((1.00000 * constants[127]) / constants[72]) * (
        states[36] - constants[4]
    )
    rates[36] = (
        (-2.00000 * algebraic[89] + algebraic[43] + algebraic[87] + algebraic[94])
        / (2.00000 * constants[2])
        - 1.00000 * algebraic[96]
    ) / constants[127]
    algebraic[14] = 1000.00 * (states[1] / constants[89] - states[0] / constants[89])
    algebraic[29] = (
        45.0000 * (constants[80] + constants[90])
        if (
            voi - np.floor(voi / constants[48]) * constants[48] <= 0.00100000
            and voi - np.floor(voi / constants[48]) * constants[48] >= 0.00000
        )
        else 0.00000
    )
    algebraic[42] = (
        algebraic[-5]
        * (
            (
                (
                    1.00000
                    * constants[105]
                    * 4.00000
                    * (states[15] + states[21])
                    * states[23]
                    * states[0]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                0.00100000
                * np.exp(
                    (2.00000 * states[0] * constants[2]) / (constants[0] * constants[1])
                )
                - 0.341000 * constants[4]
            )
        )
        / (
            np.exp((2.00000 * states[0] * constants[2]) / (constants[0] * constants[1]))
            - 1.00000
        )
    )
    algebraic[44] = (
        (
            (
                (
                    (1.00000 * constants[106])
                    / (
                        1.00000
                        - (algebraic[42] + algebraic[43])
                        / (0.458000 * (constants[80] + constants[90]))
                    )
                )
                * (states[15] + states[21])
                * states[23]
                * states[0]
                * (np.power(constants[2], 2.00000))
            )
            / (constants[0] * constants[1])
        )
        * (
            states[3]
            * np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
            - constants[5]
        )
    ) / (np.exp((states[0] * constants[2]) / (constants[0] * constants[1])) - 1.00000)
    algebraic[46] = (
        np.log(constants[5] / states[3]) * constants[0] * constants[1]
    ) / constants[2]
    algebraic[47] = 1.00000 / (1.00000 + np.exp((states[0] + 9.00000) / 22.4000))
    algebraic[48] = (
        algebraic[-1]
        * constants[93]
        * 0.0261400
        * (np.power(constants[5] / 5.40000, 1.0 / 2))
        * states[38]
        * algebraic[47]
        * (states[0] - algebraic[46])
    )
    algebraic[52] = (
        np.log(
            (constants[5] + constants[63] * constants[3])
            / (states[3] + constants[63] * states[2])
        )
        * constants[0]
        * constants[1]
    ) / constants[2]
    algebraic[53] = (
        algebraic[-3]
        * constants[94]
        * (
            0.230800
            + 0.769200
            / (1.00000 + np.exp((-np.log10(1.00000 * states[40]) - 4.20000) / 0.600000))
        )
        * (np.power(states[41], 2.00000))
        * (states[0] - algebraic[52])
    )
    algebraic[56] = algebraic[46]
    algebraic[57] = 1020.00 / (
        1.00000 + np.exp(0.238500 * ((states[0] - algebraic[56]) - 59.2150))
    )
    algebraic[58] = (
        1000.00
        * (
            0.491240 * np.exp(0.0803200 * ((states[0] - algebraic[56]) + 5.47600))
            + np.exp(0.0617500 * ((states[0] - algebraic[56]) - 594.310))
        )
    ) / (1.00000 + np.exp(-0.514300 * ((states[0] - algebraic[56]) + 4.75300)))
    algebraic[59] = (
        algebraic[-4]
        * (
            (
                constants[97]
                * (np.power(constants[5] / 5.40000, 1.0 / 2))
                * algebraic[57]
            )
            / (algebraic[57] + algebraic[58])
        )
        * (states[0] - algebraic[56])
    )
    algebraic[65] = 1.00000 / (1.00000 + np.exp((20.0000 - states[0]) / 5.00000))
    algebraic[64] = algebraic[46]
    algebraic[66] = constants[95] * algebraic[65] * (states[0] - algebraic[64])
    algebraic[70] = algebraic[46]
    algebraic[71] = (
        (
            constants[96]
            / (
                (1.00000 + np.exp((states[0] + 57.5300) / -5.86300))
                * (1.00000 + np.exp((states[0] - 45.8000) / 25.8700))
            )
        )
        * (states[43] * constants[65] + states[44] * (1.00000 - constants[65]))
        * (states[0] - algebraic[70])
    )
    algebraic[74] = algebraic[46]
    algebraic[75] = (
        (
            constants[98]
            * (
                0.800000
                - 0.650000 / (1.00000 + np.exp((states[0] + 125.000) / 15.0000))
            )
            * 0.850000
        )
        / (1.00000 + np.power(66.0000 / states[2], 2.80000))
    ) * (states[0] - algebraic[74])
    algebraic[80] = (
        (
            (
                (
                    1.00000
                    * constants[108]
                    * states[0]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                0.750000
                * states[3]
                * np.exp((states[0] * constants[2]) / (constants[0] * constants[1]))
                - 0.750000 * constants[5]
            )
        )
        / (np.exp((states[0] * constants[2]) / (constants[0] * constants[1])) - 1.00000)
    ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
    algebraic[84] = (
        np.log(constants[4] / states[40]) * constants[0] * constants[1]
    ) / (2.00000 * constants[2])
    algebraic[85] = constants[101] * (states[0] - algebraic[84])
    algebraic[93] = (constants[102] * states[40]) / (0.000500000 + states[40])
    algebraic[95] = algebraic[46]
    algebraic[98] = (
        (
            (constants[99] * 1.00000)
            / (1.00000 + np.power(constants[6] / 0.114000, 2.00000))
        )
        * (np.power(constants[5] / 4.00000, 0.240000))
        * (states[0] - algebraic[95])
    )
    rates[0] = (
        1000.00
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                (
                                                                                    algebraic[
                                                                                        29
                                                                                    ]
                                                                                    + algebraic[
                                                                                        14
                                                                                    ]
                                                                                )
                                                                                - algebraic[
                                                                                    35
                                                                                ]
                                                                            )
                                                                            - algebraic[
                                                                                40
                                                                            ]
                                                                        )
                                                                        - algebraic[42]
                                                                    )
                                                                    - algebraic[44]
                                                                )
                                                                - algebraic[53]
                                                            )
                                                            - algebraic[48]
                                                        )
                                                        - algebraic[59]
                                                    )
                                                    - algebraic[66]
                                                )
                                                - algebraic[75]
                                            )
                                            - algebraic[98]
                                        )
                                        - algebraic[78]
                                    )
                                    - algebraic[80]
                                )
                                - algebraic[82]
                            )
                            - algebraic[85]
                        )
                        - algebraic[88]
                    )
                    - algebraic[90]
                )
                - algebraic[93]
            )
            - algebraic[71]
        )
    ) / constants[129]
    algebraic[97] = (
        1.00000 * 1000.00 * constants[86] * 0.00180000 * (np.power(states[40], 2.00000))
    ) / (np.power(states[40], 2.00000) + np.power(0.000500000, 2.00000))
    algebraic[99] = 1250.00 * constants[86] * 5.80000e-05 * (states[47] - states[40])
    algebraic[101] = ((1.00000 * constants[85]) / constants[74]) * (
        states[47] - states[48]
    )
    rates[47] = ((algebraic[97] - algebraic[99]) - algebraic[101]) / (
        1.00000 * constants[87]
    )
    algebraic[103] = (
        1800.00 * constants[85] * (states[50] + states[51]) * (states[48] - states[22])
    )
    rates[48] = (
        (
            1.00000
            / (
                1.00000
                + (15.0000 * 0.800000) / (np.power(0.800000 + states[48], 2.00000))
            )
        )
        * (algebraic[101] - algebraic[103])
    ) / constants[85]
    algebraic[45] = (
        (
            (
                (
                    (1.00000 * constants[124])
                    / (
                        1.00000
                        - (algebraic[42] + algebraic[43])
                        / (0.458000 * (constants[80] + constants[90]))
                    )
                )
                * (states[29] + states[35])
                * states[37]
                * states[1]
                * (np.power(constants[2], 2.00000))
            )
            / (constants[0] * constants[1])
        )
        * (
            states[3]
            * np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
            - states[7]
        )
    ) / (np.exp((states[1] * constants[2]) / (constants[0] * constants[1])) - 1.00000)
    algebraic[49] = (
        np.log(states[7] / states[3]) * constants[0] * constants[1]
    ) / constants[2]
    algebraic[50] = 1.00000 / (1.00000 + np.exp((states[1] + 9.00000) / 22.4000))
    algebraic[51] = (
        algebraic[-1]
        * constants[111]
        * 0.0261400
        * (np.power(states[7] / 5.40000, 1.0 / 2))
        * states[39]
        * algebraic[50]
        * (states[1] - algebraic[49])
    )
    algebraic[54] = (
        np.log(
            (states[7] + constants[64] * states[6])
            / (states[3] + constants[64] * states[2])
        )
        * constants[0]
        * constants[1]
    ) / constants[2]
    algebraic[55] = (
        algebraic[-3]
        * constants[112]
        * (
            0.230800
            + 0.769200
            / (1.00000 + np.exp((-np.log10(1.00000 * states[40]) - 4.20000) / 0.600000))
        )
        * (np.power(states[42], 2.00000))
        * (states[1] - algebraic[54])
    )
    algebraic[60] = algebraic[49]
    algebraic[61] = 1020.00 / (
        1.00000 + np.exp(0.238500 * ((states[1] - algebraic[60]) - 59.2150))
    )
    algebraic[62] = (
        1000.00
        * (
            0.491240 * np.exp(0.0803200 * ((states[1] - algebraic[60]) + 5.47600))
            + np.exp(0.0617500 * ((states[1] - algebraic[60]) - 594.310))
        )
    ) / (1.00000 + np.exp(-0.514300 * ((states[1] - algebraic[60]) + 4.75300)))
    algebraic[63] = (
        algebraic[-4]
        * (
            (constants[115] * (np.power(states[7] / 5.40000, 1.0 / 2)) * algebraic[61])
            / (algebraic[61] + algebraic[62])
        )
        * (states[1] - algebraic[60])
    )
    algebraic[68] = 1.00000 / (1.00000 + np.exp((20.0000 - states[1]) / 5.00000))
    algebraic[67] = algebraic[49]
    algebraic[69] = constants[113] * algebraic[68] * (states[1] - algebraic[67])
    algebraic[72] = algebraic[49]
    algebraic[73] = (
        (
            constants[114]
            / (
                (1.00000 + np.exp((states[1] + 57.5300) / -5.86300))
                * (1.00000 + np.exp((states[1] - 45.8000) / 25.8700))
            )
        )
        * (states[45] * constants[68] + states[46] * (1.00000 - constants[68]))
        * (states[1] - algebraic[72])
    )
    algebraic[76] = algebraic[49]
    algebraic[77] = (
        (
            constants[116]
            * (
                0.800000
                - 0.650000 / (1.00000 + np.exp((states[1] + 125.000) / 15.0000))
            )
            * 0.850000
        )
        / (1.00000 + np.power(66.0000 / states[2], 2.80000))
    ) * (states[1] - algebraic[76])
    algebraic[81] = (
        (
            (
                (
                    1.00000
                    * constants[126]
                    * states[1]
                    * (np.power(constants[2], 2.00000))
                )
                / (constants[0] * constants[1])
            )
            * (
                0.750000
                * states[3]
                * np.exp((states[1] * constants[2]) / (constants[0] * constants[1]))
                - 0.750000 * states[7]
            )
        )
        / (np.exp((states[1] * constants[2]) / (constants[0] * constants[1])) - 1.00000)
    ) / (1.00000 + np.power(0.00250000 / states[40], 3.00000))
    algebraic[100] = algebraic[49]
    algebraic[102] = (
        (
            (constants[117] * 1.00000)
            / (1.00000 + np.power(constants[6] / 0.114000, 2.00000))
        )
        * (np.power(states[7] / 4.00000, 0.240000))
        * (states[1] - algebraic[100])
    )
    rates[3] = -(
        (
            (
                (
                    -algebraic[29]
                    + algebraic[53]
                    + algebraic[55]
                    + algebraic[48]
                    + algebraic[51]
                    + algebraic[59]
                    + algebraic[63]
                    + algebraic[66]
                    + algebraic[69]
                    + algebraic[44]
                    + algebraic[45]
                    + algebraic[75]
                    + algebraic[77]
                    + algebraic[80]
                    + algebraic[81]
                )
                - 2.00000 * algebraic[90]
            )
            - 2.00000 * algebraic[91]
        )
        + algebraic[98]
        + algebraic[102]
        + algebraic[71]
        + algebraic[73]
    ) / (constants[2] * constants[86])
    rates[1] = (
        1000.00
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                -algebraic[
                                                                                    14
                                                                                ]
                                                                                - algebraic[
                                                                                    39
                                                                                ]
                                                                            )
                                                                            - algebraic[
                                                                                41
                                                                            ]
                                                                        )
                                                                        - algebraic[43]
                                                                    )
                                                                    - algebraic[45]
                                                                )
                                                                - algebraic[55]
                                                            )
                                                            - algebraic[51]
                                                        )
                                                        - algebraic[63]
                                                    )
                                                    - algebraic[69]
                                                )
                                                - algebraic[77]
                                            )
                                            - algebraic[102]
                                        )
                                        - algebraic[79]
                                    )
                                    - algebraic[81]
                                )
                                - algebraic[83]
                            )
                            - algebraic[87]
                        )
                        - algebraic[89]
                    )
                    - algebraic[91]
                )
                - algebraic[94]
            )
            - algebraic[73]
        )
    ) / constants[128]
    algebraic[105] = (1.00000 * (states[22] - states[40]) * constants[86]) / constants[
        79
    ]
    rates[22] = (
        1.00000
        / (
            1.00000
            + (0.0500000 * 0.00238000) / (np.power(0.00238000 + states[22], 2.00000))
        )
    ) * (
        (
            -(algebraic[42] + algebraic[43]) / (2.00000 * constants[2] * constants[84])
            + (1.00000 * algebraic[103]) / constants[84]
        )
        - (1.00000 * algebraic[105]) / constants[84]
    )
    algebraic[104] = ((1.00000 * constants[127]) / constants[73]) * (
        states[7] - constants[5]
    )
    rates[7] = (
        (
            (
                (
                    algebraic[55]
                    + algebraic[51]
                    + algebraic[63]
                    + algebraic[69]
                    + algebraic[77]
                    + algebraic[45]
                    + algebraic[81]
                )
                - 2.00000 * algebraic[91]
            )
            + algebraic[102]
            + algebraic[73]
        )
        / constants[2]
        - 1.00000 * algebraic[104]
    ) / constants[127]
    algebraic[106] = (
        20000.0 * states[40] * (1.00000 - states[53]) - 0.0700000 * states[53]
    )
    rates[53] = algebraic[106]
    algebraic[107] = (
        40000.0 * states[40] * (1.00000 - states[54]) - states[54] * 40.0000
    )
    rates[40] = (
        1.00000
        / (
            1.00000
            + (0.0500000 * 0.00238000) / (np.power(0.00238000 + states[40], 2.00000))
        )
    ) * (
        (
            (
                (
                    -2.00000 * (algebraic[88] + algebraic[89])
                    + algebraic[85]
                    + algebraic[87]
                    + algebraic[93]
                    + algebraic[94]
                )
                / (-2.00000 * constants[2] * constants[86])
                + (1.00000 * ((algebraic[105] + algebraic[99]) - algebraic[97]))
                / constants[86]
            )
            - algebraic[106] * 0.140000
        )
        - algebraic[107] * 0.0700000
    )
    rates[54] = algebraic[107]
    return rates
