import numpy as np
import time
from cell_models.base import CardiacCellModel
from numba import njit
import pandas as pd


class MouseModel(CardiacCellModel):
    def __init__(
        self,
        drug=None,
        drug_amount=None,
        num_cycles_feasible_ap=3000,
    ):
        super().__init__(
            sizeStates=36 + 1,
            sizeAlgebraic=81 + 6,
            sizeConstants=77,
            species="Mouse",
            channel_list=["ICaL", "IK1", "IKs", "INa", "Ito", "hERG"],
            state_idx_dict={"cai": 1, "nai": 13, "ki": 22},
            default_t_max=240,
            num_cycles_feasible_ap=num_cycles_feasible_ap,
            num_cycles_limit_state=3000,
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
        constants[9] = 8.314
        constants[10] = 308
        constants[11] = 96.5
        constants[12] = 100
        constants[13] = 1000
        constants[60] = 0.0
        constants[76] = 0.0

        # Initialise states
        states[0] = -85.64004
        states[1] = 0.1040595
        states[2] = 0.1043777
        states[3] = 730.0589
        states[4] = 841.106
        states[5] = 2.290355e-9
        states[6] = 0.08989079
        states[7] = 0.003825599
        states[8] = 1.835831e-8
        states[9] = 0.3797679
        states[10] = 4.373318e-6
        states[11] = 0.009171979
        states[12] = 0.8876797
        states[13] = 16522.45
        states[14] = 2.639399e-7
        states[15] = 0.0001581035
        states[16] = 0.01702105
        states[17] = 0.00001799179
        states[18] = 0.000005460299
        states[19] = 0.0000556206
        states[20] = 0.005985434
        states[21] = 0.2543133
        states[22] = 141474
        states[23] = 0.001937245
        states[24] = 0.9999985
        states[25] = 0.02000568
        states[26] = 0.9308568
        states[27] = 0.002206261
        states[28] = 0.02000568
        states[29] = 0.9822006
        states[30] = 0.8883113
        states[31] = 1
        states[32] = 0.0004858865
        states[33] = 0.0007799137
        states[34] = 0.0005301217
        states[35] = 0.00007519518

        # Initialise constants that may be perturbed
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
        constants[74] = constants[43] / constants[42]
        constants[75] = (1.00000 / 7.00000) * (np.exp(constants[7] / 67300.0) - 1.00000)

        self.init_states = states
        self.constants = constants
        return states, constants

    def createLegends(self):
        legend_states = [""] * self.sizeStates
        legend_rates = [""] * self.sizeStates
        legend_algebraic = [""] * self.sizeAlgebraic
        legend_voi = ""
        legend_constants = [""] * self.sizeConstants
        legend_voi = "time in component environment (millisecond)"
        legend_states[-1] = "drug concentration (nM)"
        legend_rates[-1] = "d/dt drug concentration (nM/ms)"
        for i in range(len(self.channel_list)):
            legend_algebraic[i - len(self.channel_list)] = (
                "proportion of "
                + self.channel_list[i]
                + " channel uninhibited (dimensionless)"
            )
        legend_states[0] = "V in component membrane (millivolt)"
        legend_constants[0] = "Cm in component membrane (microF_per_cm2)"
        legend_constants[1] = "Vmyo in component membrane (microlitre)"
        legend_constants[2] = "VJSR in component membrane (microlitre)"
        legend_constants[3] = "VNSR in component membrane (microlitre)"
        legend_constants[4] = "Vss in component membrane (microlitre)"
        legend_constants[5] = "Acap in component membrane (cm2)"
        legend_constants[6] = "Ko in component membrane (micromolar)"
        legend_constants[7] = "Nao in component membrane (micromolar)"
        legend_constants[8] = "Cao in component membrane (micromolar)"
        legend_constants[9] = "R in component membrane (joule_per_mole_kelvin)"
        legend_constants[10] = "T in component membrane (kelvin)"
        legend_constants[11] = "F in component membrane (coulomb_per_millimole)"
        legend_algebraic[58] = (
            "i_CaL in component L_type_calcium_current (picoA_per_picoF)"
        )
        legend_algebraic[59] = (
            "i_pCa in component calcium_pump_current (picoA_per_picoF)"
        )
        legend_algebraic[60] = (
            "i_NaCa in component sodium_calcium_exchange_current (picoA_per_picoF)"
        )
        legend_algebraic[64] = (
            "i_Cab in component calcium_background_current (picoA_per_picoF)"
        )
        legend_algebraic[67] = "i_Na in component fast_sodium_current (picoA_per_picoF)"
        legend_algebraic[68] = (
            "i_Nab in component sodium_background_current (picoA_per_picoF)"
        )
        legend_algebraic[78] = (
            "i_NaK in component sodium_potassium_pump_current (picoA_per_picoF)"
        )
        legend_algebraic[70] = (
            "i_Kto_f in component fast_transient_outward_K_I (picoA_per_picoF)"
        )
        legend_algebraic[71] = (
            "i_Kto_s in component slow_transient_outward_K_I (picoA_per_picoF)"
        )
        legend_algebraic[72] = (
            "i_K1 in component time_independent_K_I (picoA_per_picoF)"
        )
        legend_algebraic[73] = (
            "i_Ks in component slow_delayed_rectifier_K_I (picoA_per_picoF)"
        )
        legend_algebraic[74] = (
            "i_Kur in component ultra_rapidly_activating_delayed_rectifier_K_I (picoA_per_picoF)"
        )
        legend_algebraic[75] = (
            "i_Kss in component non_inactivating_steady_state_K_I (picoA_per_picoF)"
        )
        legend_algebraic[80] = (
            "i_ClCa in component calcium_activated_chloride_current (picoA_per_picoF)"
        )
        legend_algebraic[76] = (
            "i_Kr in component rapid_delayed_rectifier_K_I (picoA_per_picoF)"
        )
        legend_constants[12] = "stim_offset in component membrane (millisecond)"
        legend_constants[13] = "stim_period in component membrane (millisecond)"
        legend_constants[14] = "stim_duration in component membrane (millisecond)"
        legend_constants[15] = "stim_amplitude in component membrane (picoA_per_picoF)"
        legend_algebraic[12] = "i_Stim in component membrane (picoA_per_picoF)"
        legend_algebraic[0] = "past in component membrane (millisecond)"
        legend_states[1] = "Cai in component calcium_concentration (micromolar)"
        legend_states[2] = "Cass in component calcium_concentration (micromolar)"
        legend_states[3] = "CaJSR in component calcium_concentration (micromolar)"
        legend_states[4] = "CaNSR in component calcium_concentration (micromolar)"
        legend_algebraic[26] = "Bi in component calcium_concentration (dimensionless)"
        legend_algebraic[31] = "Bss in component calcium_concentration (dimensionless)"
        legend_algebraic[35] = "BJSR in component calcium_concentration (dimensionless)"
        legend_constants[16] = "Bmax in component calcium_concentration (micromolar)"
        legend_constants[17] = (
            "CSQN_tot in component calcium_concentration (micromolar)"
        )
        legend_constants[18] = "Kd in component calcium_concentration (micromolar)"
        legend_constants[19] = "Km_CSQN in component calcium_concentration (micromolar)"
        legend_algebraic[44] = (
            "J_leak in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_algebraic[38] = (
            "J_rel in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_algebraic[52] = (
            "J_serca in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_algebraic[40] = (
            "J_tr in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_algebraic[42] = (
            "J_xfer in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_states[5] = "P_RyR in component calcium_fluxes (dimensionless)"
        legend_constants[20] = "v1 in component calcium_fluxes (per_millisecond)"
        legend_constants[21] = "tau_tr in component calcium_fluxes (millisecond)"
        legend_constants[22] = "v2 in component calcium_fluxes (per_millisecond)"
        legend_constants[23] = "tau_xfer in component calcium_fluxes (millisecond)"
        legend_algebraic[46] = "CaMKb in component calcium_fluxes (dimensionless)"
        legend_states[6] = "CaMKt in component calcium_fluxes (dimensionless)"
        legend_algebraic[48] = "CaMKa in component calcium_fluxes (dimensionless)"
        legend_constants[24] = "on_rate in component calcium_fluxes (per_millisecond)"
        legend_constants[25] = "off_rate in component calcium_fluxes (per_millisecond)"
        legend_algebraic[50] = (
            "vmup in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_constants[26] = "Km_up in component calcium_fluxes (micromolar)"
        legend_constants[27] = (
            "i_CaL_max in component L_type_calcium_current (picoA_per_picoF)"
        )
        legend_states[7] = "P_O1 in component ryanodine_receptors (dimensionless)"
        legend_states[8] = "P_O2 in component ryanodine_receptors (dimensionless)"
        legend_constants[28] = (
            "vmup_init in component calcium_fluxes (micromolar_per_millisecond)"
        )
        legend_constants[29] = (
            "P_ryr_const1 in component calcium_fluxes (per_millisecond)"
        )
        legend_constants[30] = (
            "P_ryr_const2 in component calcium_fluxes (per_millisecond)"
        )
        legend_algebraic[1] = "P_C1 in component ryanodine_receptors (dimensionless)"
        legend_states[9] = "P_C2 in component ryanodine_receptors (dimensionless)"
        legend_constants[31] = (
            "k_plus_a in component ryanodine_receptors (micromolar4_per_millisecond)"
        )
        legend_constants[32] = (
            "k_minus_a in component ryanodine_receptors (per_millisecond)"
        )
        legend_constants[33] = (
            "k_plus_b in component ryanodine_receptors (micromolar3_per_millisecond)"
        )
        legend_constants[34] = (
            "k_minus_b in component ryanodine_receptors (per_millisecond)"
        )
        legend_constants[35] = (
            "k_plus_c in component ryanodine_receptors (per_millisecond)"
        )
        legend_constants[36] = (
            "k_minus_c in component ryanodine_receptors (per_millisecond)"
        )
        legend_constants[37] = "m in component ryanodine_receptors (dimensionless)"
        legend_constants[38] = "n in component ryanodine_receptors (dimensionless)"
        legend_constants[39] = (
            "P_CaL in component L_type_calcium_current (per_millisecond)"
        )
        legend_states[10] = "O in component L_type_calcium_current (dimensionless)"
        legend_algebraic[36] = "C in component L_type_calcium_current (dimensionless)"
        legend_states[11] = "I in component L_type_calcium_current (dimensionless)"
        legend_states[12] = "y_gate in component L_type_calcium_current (dimensionless)"
        legend_algebraic[2] = (
            "y_gate_inf in component L_type_calcium_current (dimensionless)"
        )
        legend_algebraic[13] = (
            "y_gate_tau in component L_type_calcium_current (millisecond)"
        )
        legend_algebraic[14] = (
            "alpha_p in component L_type_calcium_current (per_millisecond)"
        )
        legend_constants[74] = (
            "alpha_m in component L_type_calcium_current (per_millisecond)"
        )
        legend_algebraic[27] = (
            "epsilon_p in component L_type_calcium_current (per_micromolar_millisecond)"
        )
        legend_algebraic[32] = (
            "epsilon_m in component L_type_calcium_current (per_millisecond)"
        )
        legend_constants[40] = "V_L in component L_type_calcium_current (millivolt)"
        legend_constants[41] = (
            "delta_V_L in component L_type_calcium_current (millivolt)"
        )
        legend_constants[42] = "t_L in component L_type_calcium_current (millisecond)"
        legend_constants[43] = (
            "phi_L in component L_type_calcium_current (dimensionless)"
        )
        legend_constants[44] = "a in component L_type_calcium_current (dimensionless)"
        legend_constants[45] = "b in component L_type_calcium_current (dimensionless)"
        legend_constants[46] = "tau_L in component L_type_calcium_current (millisecond)"
        legend_constants[47] = "K_L in component L_type_calcium_current (micromolar)"
        legend_algebraic[3] = (
            "expVL in component L_type_calcium_current (dimensionless)"
        )
        legend_algebraic[54] = (
            "FVRT in component L_type_calcium_current (dimensionless)"
        )
        legend_algebraic[56] = (
            "FVRT_Ca in component L_type_calcium_current (dimensionless)"
        )
        legend_constants[48] = "const5 in component L_type_calcium_current (millivolt)"
        legend_constants[49] = (
            "i_pCa_max in component calcium_pump_current (picoA_per_picoF)"
        )
        legend_constants[50] = "Km_pCa in component calcium_pump_current (micromolar)"
        legend_algebraic[61] = (
            "J_pCa in component calcium_pump_current (micromolar_per_millisecond)"
        )
        legend_algebraic[63] = (
            "J_ncx in component sodium_calcium_exchange_current (micromolar_per_millisecond)"
        )
        legend_constants[51] = (
            "k_NaCa in component sodium_calcium_exchange_current (picoA_per_picoF)"
        )
        legend_constants[52] = (
            "K_mNa in component sodium_calcium_exchange_current (micromolar)"
        )
        legend_constants[53] = (
            "K_mCa in component sodium_calcium_exchange_current (micromolar)"
        )
        legend_constants[54] = (
            "k_sat in component sodium_calcium_exchange_current (dimensionless)"
        )
        legend_constants[55] = (
            "eta in component sodium_calcium_exchange_current (dimensionless)"
        )
        legend_states[13] = "Nai in component sodium_concentration (micromolar)"
        legend_constants[56] = (
            "g_Cab in component calcium_background_current (milliS_per_microF)"
        )
        legend_algebraic[62] = (
            "E_CaN in component calcium_background_current (millivolt)"
        )
        legend_algebraic[66] = (
            "J_Cab in component calcium_background_current (micromolar_per_millisecond)"
        )
        legend_algebraic[65] = "E_Na in component fast_sodium_current (millivolt)"
        legend_constants[57] = (
            "g_Na in component fast_sodium_current (milliS_per_microF)"
        )
        legend_states[14] = "O_Na in component fast_sodium_current (dimensionless)"
        legend_states[15] = "C_Na1 in component fast_sodium_current (dimensionless)"
        legend_states[16] = "C_Na2 in component fast_sodium_current (dimensionless)"
        legend_algebraic[4] = "C_Na3 in component fast_sodium_current (dimensionless)"
        legend_states[17] = "I1_Na in component fast_sodium_current (dimensionless)"
        legend_states[18] = "I2_Na in component fast_sodium_current (dimensionless)"
        legend_states[19] = "IF_Na in component fast_sodium_current (dimensionless)"
        legend_states[20] = "IC_Na2 in component fast_sodium_current (dimensionless)"
        legend_states[21] = "IC_Na3 in component fast_sodium_current (dimensionless)"
        legend_algebraic[15] = (
            "alpha_Na11 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[37] = (
            "beta_Na11 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[28] = (
            "alpha_Na12 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[39] = (
            "beta_Na12 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[33] = (
            "alpha_Na13 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[41] = (
            "beta_Na13 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[43] = (
            "alpha_Na3 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[45] = (
            "beta_Na3 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[47] = (
            "alpha_Na2 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[49] = (
            "beta_Na2 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[51] = (
            "alpha_Na4 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[53] = (
            "beta_Na4 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[55] = (
            "alpha_Na5 in component fast_sodium_current (per_millisecond)"
        )
        legend_algebraic[57] = (
            "beta_Na5 in component fast_sodium_current (per_millisecond)"
        )
        legend_states[22] = "Ki in component potassium_concentration (micromolar)"
        legend_constants[58] = (
            "g_Nab in component sodium_background_current (milliS_per_microF)"
        )
        legend_algebraic[69] = "E_K in component fast_transient_outward_K_I (millivolt)"
        legend_constants[59] = (
            "g_Kto_f in component fast_transient_outward_K_I (milliS_per_microF)"
        )
        legend_states[23] = (
            "ato_f in component fast_transient_outward_K_I (dimensionless)"
        )
        legend_states[24] = (
            "ito_f in component fast_transient_outward_K_I (dimensionless)"
        )
        legend_algebraic[5] = (
            "alpha_a in component fast_transient_outward_K_I (per_millisecond)"
        )
        legend_algebraic[16] = (
            "beta_a in component fast_transient_outward_K_I (per_millisecond)"
        )
        legend_algebraic[6] = (
            "alpha_i in component fast_transient_outward_K_I (per_millisecond)"
        )
        legend_algebraic[17] = (
            "beta_i in component fast_transient_outward_K_I (per_millisecond)"
        )
        legend_algebraic[7] = (
            "ass in component slow_transient_outward_K_I (dimensionless)"
        )
        legend_algebraic[8] = (
            "iss in component slow_transient_outward_K_I (dimensionless)"
        )
        legend_constants[60] = (
            "g_Kto_s in component slow_transient_outward_K_I (milliS_per_microF)"
        )
        legend_states[25] = (
            "ato_s in component slow_transient_outward_K_I (dimensionless)"
        )
        legend_states[26] = (
            "ito_s in component slow_transient_outward_K_I (dimensionless)"
        )
        legend_algebraic[18] = (
            "tau_ta_s in component slow_transient_outward_K_I (millisecond)"
        )
        legend_algebraic[19] = (
            "tau_ti_s in component slow_transient_outward_K_I (millisecond)"
        )
        legend_constants[61] = (
            "g_K1 in component time_independent_K_I (milliS_per_microF)"
        )
        legend_constants[62] = (
            "g_Ks in component slow_delayed_rectifier_K_I (milliS_per_microF)"
        )
        legend_states[27] = (
            "nKs in component slow_delayed_rectifier_K_I (dimensionless)"
        )
        legend_algebraic[9] = (
            "alpha_n in component slow_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_algebraic[20] = (
            "beta_n in component slow_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_constants[63] = (
            "g_Kur in component ultra_rapidly_activating_delayed_rectifier_K_I (milliS_per_microF)"
        )
        legend_states[28] = (
            "aur in component ultra_rapidly_activating_delayed_rectifier_K_I (dimensionless)"
        )
        legend_states[29] = (
            "iur in component ultra_rapidly_activating_delayed_rectifier_K_I (dimensionless)"
        )
        legend_algebraic[21] = (
            "tau_aur in component ultra_rapidly_activating_delayed_rectifier_K_I (millisecond)"
        )
        legend_algebraic[22] = (
            "tau_iur in component ultra_rapidly_activating_delayed_rectifier_K_I (millisecond)"
        )
        legend_constants[64] = (
            "g_Kss in component non_inactivating_steady_state_K_I (milliS_per_microF)"
        )
        legend_states[30] = (
            "aKss in component non_inactivating_steady_state_K_I (dimensionless)"
        )
        legend_states[31] = (
            "iKss in component non_inactivating_steady_state_K_I (dimensionless)"
        )
        legend_algebraic[23] = (
            "tau_Kss in component non_inactivating_steady_state_K_I (millisecond)"
        )
        legend_constants[65] = (
            "g_Kr in component rapid_delayed_rectifier_K_I (milliS_per_microF)"
        )
        legend_states[32] = (
            "O_K in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_states[33] = (
            "C_K1 in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_states[34] = (
            "C_K2 in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_algebraic[10] = (
            "C_K0 in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_states[35] = (
            "I_K in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_algebraic[24] = (
            "alpha_a0 in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_algebraic[29] = (
            "beta_a0 in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_constants[66] = (
            "kb in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_constants[67] = (
            "kf in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_algebraic[11] = (
            "alpha_a1 in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_algebraic[25] = (
            "beta_a1 in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_algebraic[30] = (
            "alpha_i in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_algebraic[34] = (
            "beta_i in component rapid_delayed_rectifier_K_I (per_millisecond)"
        )
        legend_constants[68] = (
            "i_NaK_max in component sodium_potassium_pump_current (picoA_per_picoF)"
        )
        legend_constants[69] = (
            "Km_Nai in component sodium_potassium_pump_current (micromolar)"
        )
        legend_constants[70] = (
            "Km_Ko in component sodium_potassium_pump_current (micromolar)"
        )
        legend_algebraic[77] = (
            "f_NaK in component sodium_potassium_pump_current (dimensionless)"
        )
        legend_constants[75] = (
            "sigma in component sodium_potassium_pump_current (dimensionless)"
        )
        legend_constants[71] = (
            "g_ClCa in component calcium_activated_chloride_current (milliS_per_microF)"
        )
        legend_algebraic[79] = (
            "O_ClCa in component calcium_activated_chloride_current (dimensionless)"
        )
        legend_constants[72] = (
            "E_Cl in component calcium_activated_chloride_current (millivolt)"
        )
        legend_constants[73] = (
            "Km_Cl in component calcium_activated_chloride_current (micromolar)"
        )
        legend_rates[0] = "d/dt V in component membrane (millivolt)"
        legend_rates[1] = "d/dt Cai in component calcium_concentration (micromolar)"
        legend_rates[2] = "d/dt Cass in component calcium_concentration (micromolar)"
        legend_rates[3] = "d/dt CaJSR in component calcium_concentration (micromolar)"
        legend_rates[4] = "d/dt CaNSR in component calcium_concentration (micromolar)"
        legend_rates[6] = "d/dt CaMKt in component calcium_fluxes (dimensionless)"
        legend_rates[5] = "d/dt P_RyR in component calcium_fluxes (dimensionless)"
        legend_rates[7] = "d/dt P_O1 in component ryanodine_receptors (dimensionless)"
        legend_rates[8] = "d/dt P_O2 in component ryanodine_receptors (dimensionless)"
        legend_rates[9] = "d/dt P_C2 in component ryanodine_receptors (dimensionless)"
        legend_rates[10] = "d/dt O in component L_type_calcium_current (dimensionless)"
        legend_rates[11] = "d/dt I in component L_type_calcium_current (dimensionless)"
        legend_rates[12] = (
            "d/dt y_gate in component L_type_calcium_current (dimensionless)"
        )
        legend_rates[13] = "d/dt Nai in component sodium_concentration (micromolar)"
        legend_rates[16] = "d/dt C_Na2 in component fast_sodium_current (dimensionless)"
        legend_rates[15] = "d/dt C_Na1 in component fast_sodium_current (dimensionless)"
        legend_rates[14] = "d/dt O_Na in component fast_sodium_current (dimensionless)"
        legend_rates[19] = "d/dt IF_Na in component fast_sodium_current (dimensionless)"
        legend_rates[17] = "d/dt I1_Na in component fast_sodium_current (dimensionless)"
        legend_rates[18] = "d/dt I2_Na in component fast_sodium_current (dimensionless)"
        legend_rates[20] = (
            "d/dt IC_Na2 in component fast_sodium_current (dimensionless)"
        )
        legend_rates[21] = (
            "d/dt IC_Na3 in component fast_sodium_current (dimensionless)"
        )
        legend_rates[22] = "d/dt Ki in component potassium_concentration (micromolar)"
        legend_rates[23] = (
            "d/dt ato_f in component fast_transient_outward_K_I (dimensionless)"
        )
        legend_rates[24] = (
            "d/dt ito_f in component fast_transient_outward_K_I (dimensionless)"
        )
        legend_rates[25] = (
            "d/dt ato_s in component slow_transient_outward_K_I (dimensionless)"
        )
        legend_rates[26] = (
            "d/dt ito_s in component slow_transient_outward_K_I (dimensionless)"
        )
        legend_rates[27] = (
            "d/dt nKs in component slow_delayed_rectifier_K_I (dimensionless)"
        )
        legend_rates[28] = (
            "d/dt aur in component ultra_rapidly_activating_delayed_rectifier_K_I (dimensionless)"
        )
        legend_rates[29] = (
            "d/dt iur in component ultra_rapidly_activating_delayed_rectifier_K_I (dimensionless)"
        )
        legend_rates[30] = (
            "d/dt aKss in component non_inactivating_steady_state_K_I (dimensionless)"
        )
        legend_rates[31] = (
            "d/dt iKss in component non_inactivating_steady_state_K_I (dimensionless)"
        )
        legend_rates[34] = (
            "d/dt C_K2 in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_rates[33] = (
            "d/dt C_K1 in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_rates[32] = (
            "d/dt O_K in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
        legend_rates[35] = (
            "d/dt I_K in component rapid_delayed_rectifier_K_I (dimensionless)"
        )
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
        algebraic[1] = 1.00000 - (states[9] + states[7] + states[8])
        algebraic[2] = 1.00000 / (
            1.00000 + np.exp((states[0] + 16.6577) / constants[48])
        ) + 0.100000 / (1.00000 + np.exp((-states[0] + 40.0000) / 6.00000))
        algebraic[13] = 20.0000 + 600.000 / (
            1.00000 + np.exp((states[0] + 30.0000) / 9.60000)
        )
        algebraic[5] = 0.180640 * np.exp(0.0357700 * (states[0] + 30.0000))
        algebraic[16] = 0.395600 * np.exp(-0.0623700 * (states[0] + 30.0000))
        algebraic[6] = (0.000152000 * np.exp(-(states[0] + 13.5000) / 7.00000)) / (
            0.00670830 * np.exp(-(states[0] + 33.5000) / 7.00000) + 1.00000
        )
        algebraic[17] = (0.000950000 * np.exp((states[0] + 33.5000) / 7.00000)) / (
            0.0513350 * np.exp((states[0] + 33.5000) / 7.00000) + 1.00000
        )
        algebraic[7] = 1.00000 / (1.00000 + np.exp(-(states[0] + 22.5000) / 7.70000))
        algebraic[18] = 0.493000 * np.exp(-0.0629000 * states[0]) + 2.05800
        algebraic[8] = 1.00000 / (1.00000 + np.exp((states[0] + 45.2000) / 5.70000))
        algebraic[19] = 270.000 + 1050.00 / (
            1.00000 + np.exp((states[0] + 45.2000) / 5.70000)
        )
        algebraic[9] = (4.81333e-06 * (states[0] + 26.5000)) / (
            1.00000 - np.exp(-0.128000 * (states[0] + 26.5000))
        )
        algebraic[20] = 9.53333e-05 * np.exp(-0.0380000 * (states[0] + 26.5000))
        algebraic[21] = 0.493000 * np.exp(-0.0629000 * states[0]) + 2.05800
        algebraic[22] = 1200.00 - 170.000 / (
            1.00000 + np.exp((states[0] + 45.2000) / 5.70000)
        )
        algebraic[23] = 39.3000 * np.exp(-0.0862000 * states[0]) + 13.1700
        algebraic[11] = 0.0137330 * np.exp(0.0381980 * states[0])
        algebraic[25] = 6.89000e-05 * np.exp(-0.0417800 * states[0])
        algebraic[10] = 1.00000 - (states[33] + states[34] + states[32] + states[35])
        algebraic[24] = 0.0223480 * np.exp(0.0117600 * states[0])
        algebraic[29] = 0.0470020 * np.exp(-0.0631000 * states[0])
        algebraic[30] = 0.0908210 * np.exp(0.0233910 * (states[0] + 5.00000))
        algebraic[34] = 0.00649700 * np.exp(-0.0326800 * (states[0] + 5.00000))
        algebraic[36] = (1.00000 - states[10]) - states[11]
        algebraic[3] = np.exp((states[0] - constants[40]) / constants[41])
        algebraic[14] = algebraic[3] / (constants[42] * (algebraic[3] + 1.00000))
        algebraic[27] = (algebraic[3] + constants[44]) / (
            constants[46] * constants[47] * (algebraic[3] + 1.00000)
        )
        algebraic[32] = (constants[45] * (algebraic[3] + constants[44])) / (
            constants[46] * (constants[45] * algebraic[3] + constants[44])
        )
        algebraic[35] = np.power(
            1.00000
            + (constants[17] * constants[19])
            / (np.power(constants[19] + states[3], 2.00000)),
            -1.00000,
        )
        algebraic[38] = (
            constants[20]
            * (states[7] + states[8])
            * (states[3] - states[2])
            * states[5]
        )
        algebraic[40] = (states[4] - states[3]) / constants[21]
        algebraic[4] = 1.00000 - (
            states[14]
            + states[15]
            + states[16]
            + states[19]
            + states[17]
            + states[18]
            + states[20]
            + states[21]
        )
        algebraic[15] = 3.80200 / (
            0.102700 * np.exp(-(states[0] + 2.50000) / 17.0000)
            + 0.200000 * np.exp(-(states[0] + 2.50000) / 150.000)
        )
        algebraic[37] = 0.191700 * np.exp(-(states[0] + 2.50000) / 20.3000)
        algebraic[28] = 3.80200 / (
            0.102700 * np.exp(-(states[0] + 2.50000) / 15.0000)
            + 0.230000 * np.exp(-(states[0] + 2.50000) / 150.000)
        )
        algebraic[39] = 0.200000 * np.exp(-(states[0] - 2.50000) / 20.3000)
        algebraic[43] = 7.00000e-07 * np.exp(-(states[0] + 7.00000) / 7.70000)
        algebraic[45] = 0.00840000 + 2.00000e-05 * (states[0] + 7.00000)
        algebraic[33] = 3.80200 / (
            0.102700 * np.exp(-(states[0] + 2.50000) / 12.0000)
            + 0.250000 * np.exp(-(states[0] + 2.50000) / 150.000)
        )
        algebraic[41] = 0.220000 * np.exp(-(states[0] - 7.50000) / 20.3000)
        algebraic[46] = (0.0500000 * (1.00000 - states[6]) * 1.00000) / (
            1.00000 + 0.700000 / states[2]
        )
        algebraic[47] = 1.00000 / (
            0.188495 * np.exp(-(states[0] + 7.00000) / 16.6000) + 0.393956
        )
        algebraic[49] = (algebraic[33] * algebraic[47] * algebraic[43]) / (
            algebraic[41] * algebraic[45]
        )
        algebraic[44] = constants[22] * (states[4] - states[1])
        algebraic[48] = algebraic[46] + states[6]
        algebraic[50] = (
            (3.15120 * (np.power(algebraic[48], 2.20620)))
            / (np.power(0.158800, 2.20620) + np.power(algebraic[48], 2.20620))
            + 1.00000
        ) * constants[28]
        algebraic[52] = (algebraic[50] * (np.power(states[1], 2.00000))) / (
            np.power(constants[26], 2.00000) + np.power(states[1], 2.00000)
        )
        algebraic[51] = algebraic[47] / 1000.00
        algebraic[53] = algebraic[43]
        algebraic[55] = algebraic[47] / 95000.0
        algebraic[57] = algebraic[43] / 50.0000
        algebraic[54] = (constants[11] * states[0]) / (constants[9] * constants[10])
        algebraic[56] = 2.00000 * algebraic[54]
        algebraic[58] = algebraic[-6] * self.custom_piecewise(
            [
                np.greater(np.fabs(algebraic[56]), 1.00000e-05),
                (
                    (
                        (
                            (-constants[39] * 2.00000 * constants[4] * constants[11])
                            / (constants[5] * constants[0])
                        )
                        * states[10]
                        * states[12]
                        * algebraic[56]
                    )
                    / (1.00000 - np.exp(-algebraic[56]))
                )
                * (constants[8] * np.exp(-algebraic[56]) - states[2]),
                True,
                (
                    (
                        (
                            (-constants[39] * 2.00000 * constants[4] * constants[11])
                            / (constants[5] * constants[0])
                        )
                        * states[10]
                        * states[12]
                        * 1.00000e-05
                    )
                    / (1.00000 - np.exp(-1.00000e-05))
                )
                * (constants[8] * np.exp(-1.00000e-05) - states[2]),
            ]
        )
        algebraic[31] = np.power(
            1.00000
            + (constants[16] * constants[18])
            / (np.power(constants[18] + states[2], 2.00000)),
            -1.00000,
        )
        algebraic[42] = (states[2] - states[1]) / constants[23]
        algebraic[59] = (constants[49] * (np.power(states[1], 2.00000))) / (
            np.power(constants[50], 2.00000) + np.power(states[1], 2.00000)
        )
        algebraic[60] = (
            (
                (
                    (
                        (
                            (constants[51] * 1.00000)
                            / (
                                np.power(constants[52], 3.00000)
                                + np.power(constants[7], 3.00000)
                            )
                        )
                        * 1.00000
                    )
                    / (constants[53] + constants[8])
                )
                * 1.00000
            )
            / (
                1.00000
                + constants[54]
                * np.exp(
                    ((constants[55] - 1.00000) * states[0] * constants[11])
                    / (constants[9] * constants[10])
                )
            )
        ) * (
            np.exp(
                (constants[55] * states[0] * constants[11])
                / (constants[9] * constants[10])
            )
            * (np.power(states[13], 3.00000))
            * constants[8]
            - np.exp(
                ((constants[55] - 1.00000) * states[0] * constants[11])
                / (constants[9] * constants[10])
            )
            * (np.power(constants[7], 3.00000))
            * states[1]
        )
        algebraic[62] = (
            (constants[9] * constants[10]) / (2.00000 * constants[11])
        ) * np.log(constants[8] / states[1])
        algebraic[64] = constants[56] * (states[0] - algebraic[62])
        algebraic[26] = np.power(
            1.00000
            + (constants[16] * constants[18])
            / (np.power(constants[18] + states[1], 2.00000)),
            -1.00000,
        )
        algebraic[65] = ((constants[9] * constants[10]) / constants[11]) * np.log(
            (0.900000 * constants[7] + 0.100000 * constants[6])
            / (0.900000 * states[13] + 0.100000 * states[22])
        )
        algebraic[67] = (
            algebraic[-3] * constants[57] * states[14] * (states[0] - algebraic[65])
        )
        algebraic[68] = constants[58] * (states[0] - algebraic[65])
        algebraic[77] = 1.00000 / (
            1.00000
            + 0.124500
            * np.exp(
                (-0.100000 * states[0] * constants[11]) / (constants[9] * constants[10])
            )
            + 0.0365000
            * constants[75]
            * np.exp((-states[0] * constants[11]) / (constants[9] * constants[10]))
        )
        algebraic[78] = (
            (
                (constants[68] * algebraic[77] * 1.00000)
                / (1.00000 + np.power(constants[69] / states[13], 1.50000))
            )
            * constants[6]
        ) / (constants[6] + constants[70])
        algebraic[69] = ((constants[9] * constants[10]) / constants[11]) * np.log(
            constants[6] / states[22]
        )
        algebraic[70] = (
            algebraic[-2]
            * constants[59]
            * (np.power(states[23], 3.00000))
            * states[24]
            * (states[0] - algebraic[69])
        )
        algebraic[71] = (
            constants[60] * states[25] * states[26] * (states[0] - algebraic[69])
        )
        algebraic[72] = (
            algebraic[-5]
            * (
                ((constants[61] * constants[6]) / (constants[6] + 210.000))
                * (states[0] - algebraic[69])
            )
            / (1.00000 + np.exp(0.0896000 * (states[0] - algebraic[69])))
        )
        algebraic[73] = (
            algebraic[-4]
            * constants[62]
            * (np.power(states[27], 2.00000))
            * (states[0] - algebraic[69])
        )
        algebraic[74] = (
            constants[63] * states[28] * states[29] * (states[0] - algebraic[69])
        )
        algebraic[75] = (
            constants[64] * states[30] * states[31] * (states[0] - algebraic[69])
        )
        algebraic[76] = (
            algebraic[-1]
            * constants[65]
            * states[32]
            * (
                states[0]
                - ((constants[9] * constants[10]) / constants[11])
                * np.log(
                    (0.980000 * constants[6] + 0.0200000 * constants[7])
                    / (0.980000 * states[22] + 0.0200000 * states[13])
                )
            )
        )
        algebraic[0] = np.floor(voi / constants[13]) * constants[13]
        algebraic[12] = self.custom_piecewise(
            [
                np.greater_equal(voi - algebraic[0], constants[12])
                & np.less_equal(voi - algebraic[0], constants[12] + constants[14]),
                constants[15],
                True,
                0.00000,
            ]
        )
        algebraic[79] = 0.200000 / (1.00000 + np.exp(-(states[0] - 46.7000) / 7.80000))
        algebraic[80] = (
            (constants[71] * algebraic[79] * states[1]) / (states[1] + constants[73])
        ) * (states[0] - constants[72])
        algebraic[61] = (-algebraic[59] * constants[5] * constants[0]) / (
            2.00000 * constants[1] * constants[11]
        )
        algebraic[63] = (algebraic[60] * constants[5] * constants[0]) / (
            constants[1] * constants[11]
        )
        algebraic[66] = (-algebraic[64] * constants[5] * constants[0]) / (
            2.00000 * constants[1] * constants[11]
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
    rates[31] = constants[76]
    rates[8] = (
        constants[33] * (np.power(states[2], constants[37])) * states[7]
        - constants[34] * states[8]
    )
    rates[9] = constants[35] * states[7] - constants[36] * states[9]
    algebraic[1] = 1.00000 - (states[9] + states[7] + states[8])
    rates[7] = (
        constants[31] * (np.power(states[2], constants[38])) * algebraic[1]
        + constants[34] * states[8]
        + constants[36] * states[9]
    ) - (
        constants[32] * states[7]
        + constants[33] * (np.power(states[2], constants[37])) * states[7]
        + constants[35] * states[7]
    )
    algebraic[2] = 1.00000 / (
        1.00000 + np.exp((states[0] + 16.6577) / constants[48])
    ) + 0.100000 / (1.00000 + np.exp((-states[0] + 40.0000) / 6.00000))
    algebraic[13] = 20.0000 + 600.000 / (
        1.00000 + np.exp((states[0] + 30.0000) / 9.60000)
    )
    rates[12] = (algebraic[2] - states[12]) / algebraic[13]
    algebraic[5] = 0.180640 * np.exp(0.0357700 * (states[0] + 30.0000))
    algebraic[16] = 0.395600 * np.exp(-0.0623700 * (states[0] + 30.0000))
    rates[23] = algebraic[5] * (1.00000 - states[23]) - algebraic[16] * states[23]
    algebraic[6] = (0.000152000 * np.exp(-(states[0] + 13.5000) / 7.00000)) / (
        0.00670830 * np.exp(-(states[0] + 33.5000) / 7.00000) + 1.00000
    )
    algebraic[17] = (0.000950000 * np.exp((states[0] + 33.5000) / 7.00000)) / (
        0.0513350 * np.exp((states[0] + 33.5000) / 7.00000) + 1.00000
    )
    rates[24] = algebraic[6] * (1.00000 - states[24]) - algebraic[17] * states[24]
    algebraic[7] = 1.00000 / (1.00000 + np.exp(-(states[0] + 22.5000) / 7.70000))
    algebraic[18] = 0.493000 * np.exp(-0.0629000 * states[0]) + 2.05800
    rates[25] = (algebraic[7] - states[25]) / algebraic[18]
    algebraic[8] = 1.00000 / (1.00000 + np.exp((states[0] + 45.2000) / 5.70000))
    algebraic[19] = 270.000 + 1050.00 / (
        1.00000 + np.exp((states[0] + 45.2000) / 5.70000)
    )
    rates[26] = (algebraic[8] - states[26]) / algebraic[19]
    algebraic[9] = (4.81333e-06 * (states[0] + 26.5000)) / (
        1.00000 - np.exp(-0.128000 * (states[0] + 26.5000))
    )
    algebraic[20] = 9.53333e-05 * np.exp(-0.0380000 * (states[0] + 26.5000))
    rates[27] = algebraic[9] * (1.00000 - states[27]) - algebraic[20] * states[27]
    algebraic[21] = 0.493000 * np.exp(-0.0629000 * states[0]) + 2.05800
    rates[28] = (algebraic[7] - states[28]) / algebraic[21]
    algebraic[22] = 1200.00 - 170.000 / (
        1.00000 + np.exp((states[0] + 45.2000) / 5.70000)
    )
    rates[29] = (algebraic[8] - states[29]) / algebraic[22]
    algebraic[23] = 39.3000 * np.exp(-0.0862000 * states[0]) + 13.1700
    rates[30] = (algebraic[7] - states[30]) / algebraic[23]
    algebraic[11] = 0.0137330 * np.exp(0.0381980 * states[0])
    algebraic[25] = 6.89000e-05 * np.exp(-0.0417800 * states[0])
    rates[34] = (constants[67] * states[33] + algebraic[25] * states[32]) - (
        constants[66] * states[34] + algebraic[11] * states[34]
    )
    algebraic[10] = 1.00000 - (states[33] + states[34] + states[32] + states[35])
    algebraic[24] = 0.0223480 * np.exp(0.0117600 * states[0])
    algebraic[29] = 0.0470020 * np.exp(-0.0631000 * states[0])
    rates[33] = (algebraic[24] * algebraic[10] + constants[66] * states[34]) - (
        algebraic[29] * states[33] + constants[67] * states[33]
    )
    algebraic[30] = 0.0908210 * np.exp(0.0233910 * (states[0] + 5.00000))
    algebraic[34] = 0.00649700 * np.exp(-0.0326800 * (states[0] + 5.00000))
    rates[32] = (algebraic[11] * states[34] + algebraic[34] * states[35]) - (
        algebraic[25] * states[32] + algebraic[30] * states[32]
    )
    rates[35] = algebraic[30] * states[32] - algebraic[34] * states[35]
    algebraic[36] = (1.00000 - states[10]) - states[11]
    algebraic[3] = np.exp((states[0] - constants[40]) / constants[41])
    algebraic[14] = algebraic[3] / (constants[42] * (algebraic[3] + 1.00000))
    rates[10] = algebraic[14] * algebraic[36] - constants[74] * states[10]
    algebraic[27] = (algebraic[3] + constants[44]) / (
        constants[46] * constants[47] * (algebraic[3] + 1.00000)
    )
    algebraic[32] = (constants[45] * (algebraic[3] + constants[44])) / (
        constants[46] * (constants[45] * algebraic[3] + constants[44])
    )
    rates[11] = algebraic[27] * states[2] * algebraic[36] - algebraic[32] * states[11]
    algebraic[35] = np.power(
        1.00000
        + (constants[17] * constants[19])
        / (np.power(constants[19] + states[3], 2.00000)),
        -1.00000,
    )
    algebraic[38] = (
        constants[20] * (states[7] + states[8]) * (states[3] - states[2]) * states[5]
    )
    algebraic[40] = (states[4] - states[3]) / constants[21]
    rates[3] = algebraic[35] * (algebraic[40] - algebraic[38])
    algebraic[4] = 1.00000 - (
        states[14]
        + states[15]
        + states[16]
        + states[19]
        + states[17]
        + states[18]
        + states[20]
        + states[21]
    )
    algebraic[15] = 3.80200 / (
        0.102700 * np.exp(-(states[0] + 2.50000) / 17.0000)
        + 0.200000 * np.exp(-(states[0] + 2.50000) / 150.000)
    )
    algebraic[37] = 0.191700 * np.exp(-(states[0] + 2.50000) / 20.3000)
    algebraic[28] = 3.80200 / (
        0.102700 * np.exp(-(states[0] + 2.50000) / 15.0000)
        + 0.230000 * np.exp(-(states[0] + 2.50000) / 150.000)
    )
    algebraic[39] = 0.200000 * np.exp(-(states[0] - 2.50000) / 20.3000)
    algebraic[43] = 7.00000e-07 * np.exp(-(states[0] + 7.00000) / 7.70000)
    algebraic[45] = 0.00840000 + 2.00000e-05 * (states[0] + 7.00000)
    rates[16] = (
        algebraic[15] * algebraic[4]
        + algebraic[39] * states[15]
        + algebraic[43] * states[20]
    ) - (
        algebraic[37] * states[16]
        + algebraic[28] * states[16]
        + algebraic[45] * states[16]
    )
    algebraic[33] = 3.80200 / (
        0.102700 * np.exp(-(states[0] + 2.50000) / 12.0000)
        + 0.250000 * np.exp(-(states[0] + 2.50000) / 150.000)
    )
    algebraic[41] = 0.220000 * np.exp(-(states[0] - 7.50000) / 20.3000)
    rates[15] = (
        algebraic[28] * states[16]
        + algebraic[41] * states[14]
        + algebraic[43] * states[19]
    ) - (
        algebraic[39] * states[15]
        + algebraic[33] * states[15]
        + algebraic[45] * states[15]
    )
    rates[20] = (
        algebraic[15] * states[21]
        + algebraic[39] * states[19]
        + algebraic[45] * states[20]
    ) - (
        algebraic[37] * states[20]
        + algebraic[28] * states[20]
        + algebraic[43] * states[20]
    )
    rates[21] = (algebraic[37] * states[20] + algebraic[45] * algebraic[4]) - (
        algebraic[15] * states[21] + algebraic[43] * states[21]
    )
    algebraic[46] = (0.0500000 * (1.00000 - states[6]) * 1.00000) / (
        1.00000 + 0.700000 / states[2]
    )
    rates[6] = (
        constants[24] * algebraic[46] * (algebraic[46] + states[6])
        - constants[25] * states[6]
    )
    algebraic[47] = 1.00000 / (
        0.188495 * np.exp(-(states[0] + 7.00000) / 16.6000) + 0.393956
    )
    algebraic[49] = (algebraic[33] * algebraic[47] * algebraic[43]) / (
        algebraic[41] * algebraic[45]
    )
    rates[14] = (algebraic[33] * states[15] + algebraic[49] * states[19]) - (
        algebraic[41] * states[14] + algebraic[47] * states[14]
    )
    algebraic[44] = constants[22] * (states[4] - states[1])
    algebraic[48] = algebraic[46] + states[6]
    algebraic[50] = (
        (3.15120 * (np.power(algebraic[48], 2.20620)))
        / (np.power(0.158800, 2.20620) + np.power(algebraic[48], 2.20620))
        + 1.00000
    ) * constants[28]
    algebraic[52] = (algebraic[50] * (np.power(states[1], 2.00000))) / (
        np.power(constants[26], 2.00000) + np.power(states[1], 2.00000)
    )
    rates[4] = ((algebraic[52] - algebraic[44]) * constants[1]) / constants[3] - (
        algebraic[40] * constants[2]
    ) / constants[3]
    algebraic[51] = algebraic[47] / 1000.00
    algebraic[53] = algebraic[43]
    rates[19] = (
        algebraic[47] * states[14]
        + algebraic[45] * states[15]
        + algebraic[53] * states[17]
        + algebraic[28] * states[20]
    ) - (
        algebraic[49] * states[19]
        + algebraic[43] * states[19]
        + algebraic[51] * states[19]
        + algebraic[39] * states[19]
    )
    algebraic[55] = algebraic[47] / 95000.0
    algebraic[57] = algebraic[43] / 50.0000
    rates[17] = (algebraic[51] * states[19] + algebraic[57] * states[18]) - (
        algebraic[53] * states[17] + algebraic[55] * states[17]
    )
    rates[18] = algebraic[55] * states[17] - algebraic[57] * states[18]
    algebraic[54] = (constants[11] * states[0]) / (constants[9] * constants[10])
    algebraic[56] = 2.00000 * algebraic[54]
    if np.fabs(algebraic[56]) > 1.00000e-05:
        algebraic[58] = (
            algebraic[-6]
            * (
                (
                    (
                        (-constants[39] * 2.00000 * constants[4] * constants[11])
                        / (constants[5] * constants[0])
                    )
                    * states[10]
                    * states[12]
                    * algebraic[56]
                )
                / (1.00000 - np.exp(-algebraic[56]))
            )
            * (constants[8] * np.exp(-algebraic[56]) - states[2])
        )
    else:
        algebraic[58] = (
            algebraic[-6]
            * (
                (
                    (
                        (-constants[39] * 2.00000 * constants[4] * constants[11])
                        / (constants[5] * constants[0])
                    )
                    * states[10]
                    * states[12]
                    * 1.00000e-05
                )
                / (1.00000 - np.exp(-1.00000e-05))
            )
            * (constants[8] * np.exp(-1.00000e-05) - states[2])
        )
    algebraic[31] = np.power(
        1.00000
        + (constants[16] * constants[18])
        / (np.power(constants[18] + states[2], 2.00000)),
        -1.00000,
    )
    algebraic[42] = (states[2] - states[1]) / constants[23]
    rates[2] = algebraic[31] * (
        (algebraic[38] * constants[2]) / constants[4]
        - (
            (algebraic[42] * constants[1]) / constants[4]
            + (algebraic[58] * constants[5] * constants[0])
            / (2.00000 * constants[4] * constants[11])
        )
    )
    rates[5] = constants[29] * states[5] + (
        (constants[30] * algebraic[58]) / constants[27]
    ) * np.exp(-(np.power(states[0] - 5.00000, 2.00000)) / 648.000)
    algebraic[59] = (constants[49] * (np.power(states[1], 2.00000))) / (
        np.power(constants[50], 2.00000) + np.power(states[1], 2.00000)
    )
    algebraic[60] = (
        (
            (
                (
                    (
                        (constants[51] * 1.00000)
                        / (
                            np.power(constants[52], 3.00000)
                            + np.power(constants[7], 3.00000)
                        )
                    )
                    * 1.00000
                )
                / (constants[53] + constants[8])
            )
            * 1.00000
        )
        / (
            1.00000
            + constants[54]
            * np.exp(
                ((constants[55] - 1.00000) * states[0] * constants[11])
                / (constants[9] * constants[10])
            )
        )
    ) * (
        np.exp(
            (constants[55] * states[0] * constants[11]) / (constants[9] * constants[10])
        )
        * (np.power(states[13], 3.00000))
        * constants[8]
        - np.exp(
            ((constants[55] - 1.00000) * states[0] * constants[11])
            / (constants[9] * constants[10])
        )
        * (np.power(constants[7], 3.00000))
        * states[1]
    )
    algebraic[62] = (
        (constants[9] * constants[10]) / (2.00000 * constants[11])
    ) * np.log(constants[8] / states[1])
    algebraic[64] = constants[56] * (states[0] - algebraic[62])
    algebraic[26] = np.power(
        1.00000
        + (constants[16] * constants[18])
        / (np.power(constants[18] + states[1], 2.00000)),
        -1.00000,
    )
    rates[1] = algebraic[26] * (
        (
            (algebraic[44] + algebraic[42])
            - (
                ((algebraic[64] + algebraic[59]) - 2.00000 * algebraic[60])
                * constants[5]
                * constants[0]
            )
            / (2.00000 * constants[1] * constants[11])
        )
        - algebraic[52]
    )
    algebraic[65] = ((constants[9] * constants[10]) / constants[11]) * np.log(
        (0.900000 * constants[7] + 0.100000 * constants[6])
        / (0.900000 * states[13] + 0.100000 * states[22])
    )
    algebraic[67] = (
        algebraic[-3] * constants[57] * states[14] * (states[0] - algebraic[65])
    )
    algebraic[68] = constants[58] * (states[0] - algebraic[65])
    algebraic[77] = 1.00000 / (
        1.00000
        + 0.124500
        * np.exp(
            (-0.100000 * states[0] * constants[11]) / (constants[9] * constants[10])
        )
        + 0.0365000
        * constants[75]
        * np.exp((-states[0] * constants[11]) / (constants[9] * constants[10]))
    )
    algebraic[78] = (
        (
            (constants[68] * algebraic[77] * 1.00000)
            / (1.00000 + np.power(constants[69] / states[13], 1.50000))
        )
        * constants[6]
    ) / (constants[6] + constants[70])
    rates[13] = (
        -(
            algebraic[67]
            + algebraic[68]
            + 3.00000 * algebraic[78]
            + 3.00000 * algebraic[60]
        )
        * constants[5]
        * constants[0]
    ) / (constants[1] * constants[11])
    algebraic[69] = ((constants[9] * constants[10]) / constants[11]) * np.log(
        constants[6] / states[22]
    )
    algebraic[70] = (
        algebraic[-2]
        * constants[59]
        * (np.power(states[23], 3.00000))
        * states[24]
        * (states[0] - algebraic[69])
    )
    algebraic[71] = (
        constants[60] * states[25] * states[26] * (states[0] - algebraic[69])
    )
    algebraic[72] = (
        algebraic[-5]
        * (
            ((constants[61] * constants[6]) / (constants[6] + 210.000))
            * (states[0] - algebraic[69])
        )
        / (1.00000 + np.exp(0.0896000 * (states[0] - algebraic[69])))
    )
    algebraic[73] = (
        algebraic[-4]
        * constants[62]
        * (np.power(states[27], 2.00000))
        * (states[0] - algebraic[69])
    )
    algebraic[74] = (
        constants[63] * states[28] * states[29] * (states[0] - algebraic[69])
    )
    algebraic[75] = (
        constants[64] * states[30] * states[31] * (states[0] - algebraic[69])
    )
    algebraic[76] = (
        algebraic[-1]
        * constants[65]
        * states[32]
        * (
            states[0]
            - ((constants[9] * constants[10]) / constants[11])
            * np.log(
                (0.980000 * constants[6] + 0.0200000 * constants[7])
                / (0.980000 * states[22] + 0.0200000 * states[13])
            )
        )
    )
    algebraic[0] = np.floor(voi / constants[13]) * constants[13]
    if (voi - algebraic[0] >= constants[12]) and (
        voi - algebraic[0] <= constants[12] + constants[14]
    ):
        algebraic[12] = constants[15]
    else:
        algebraic[12] = 0.00000
    rates[22] = (
        -(
            (
                algebraic[12]
                + algebraic[70]
                + algebraic[71]
                + algebraic[72]
                + algebraic[73]
                + algebraic[75]
                + algebraic[74]
                + algebraic[76]
            )
            - 2.00000 * algebraic[78]
        )
        * constants[5]
        * constants[0]
    ) / (constants[1] * constants[11])
    algebraic[79] = 0.200000 / (1.00000 + np.exp(-(states[0] - 46.7000) / 7.80000))
    algebraic[80] = (
        (constants[71] * algebraic[79] * states[1]) / (states[1] + constants[73])
    ) * (states[0] - constants[72])
    rates[0] = -(
        algebraic[58]
        + algebraic[59]
        + algebraic[60]
        + algebraic[64]
        + algebraic[67]
        + algebraic[68]
        + algebraic[78]
        + algebraic[70]
        + algebraic[71]
        + algebraic[72]
        + algebraic[73]
        + algebraic[74]
        + algebraic[75]
        + algebraic[76]
        # + algebraic[80]
        + algebraic[12]
    )
    return rates
