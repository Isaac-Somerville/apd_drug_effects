import numpy as np
import math
from cell_models.base import CardiacCellModel
from numba import njit


class DogModel(CardiacCellModel):
    def __init__(
        self,
        drug=None,
        drug_amount=None,
        num_cycles_feasible_ap=None,
    ):
        super().__init__(
            sizeStates=29 + 1,
            sizeAlgebraic=92 + 7,
            sizeConstants=69,
            species="Dog",
            state_idx_dict={"cai": 12, "nai": 18, "ki": 23, "cli": 19},
            num_cycles_feasible_ap=num_cycles_feasible_ap,
            num_cycles_limit_state=10_000,
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

        constants = np.zeros(self.sizeConstants)
        states = np.zeros(self.sizeStates)

        # Initialise constants that are never perturbed
        constants[0] = 96485
        constants[5] = 0.0374358835078
        constants[6] = 0
        constants[9] = 0
        constants[10] = 1000
        constants[11] = 3
        constants[12] = -20

        # Initialise states
        states[0] = -85.781844107117
        states[1] = 0.987317750543
        states[2] = 0.001356538159
        states[3] = 0.991924983076
        states[4] = 0.00012271265
        states[5] = 0.00000164013
        states[6] = 8.98230672628
        states[7] = 0.999961508634
        states[8] = 0.97836624923
        states[9] = 0.893052931249
        states[10] = 0.992234519148
        states[11] = 0.00000724074
        states[12] = 0.00012131666
        states[13] = 0.019883138161
        states[14] = 0.019890650554
        states[15] = 0.013970786703
        states[16] = 0.99996472752
        states[17] = 0.829206149767
        states[18] = 12.972433387269
        states[19] = 15.59207157178
        states[20] = 0.000816605172
        states[21] = 0.001356538159
        states[22] = 0.26130711759
        states[23] = 135.469546216758
        states[24] = 1.737580994071
        states[25] = 0.021123704774
        states[26] = 0.0
        states[27] = 0.862666650318
        states[28] = 1.832822335168

        # Perturb constants with parameter-specific CVs first
        base_value_idxs = list(self.constant_base_value_dict.keys())
        different_cv_idxs = set()
        for param in self.constant_idx_dict:
            if param not in self.parameter_CVs:
                continue
            idx = self.constant_idx_dict[param]
            different_cv_idxs.add(int(idx))
            cv = self.parameter_CVs[param]
            reference_value = self.constant_base_value_dict[str(idx)]
            constants[idx] = self.perturb_constant(
                reference_value=reference_value, perturbation=perturbation, cv=cv
            )
        # Perturb remaining base-value constants with default lognormal CV (95% CI = [0.5, 2x])
        for idx in base_value_idxs:
            if int(idx) in different_cv_idxs:
                continue
            cv = np.sqrt(np.exp((np.log(2) / 1.96) ** 2) - 1)
            constants[int(idx)] = self.perturb_constant(
                reference_value=self.constant_base_value_dict[str(idx)],
                perturbation=perturbation,
                cv=cv,
            )

        # Initialise derived constants (constants[57] = GNa already set by perturbation)
        constants[56] = 1000.0 * math.pi * constants[8] * constants[8] * constants[7]
        constants[58] = 0.0138542 * (np.power(constants[1] / 5.40000, 1.0 / 2))
        constants[59] = (np.exp(constants[3] / 67.3000) - 1.00000) / 7.00000
        constants[60] = (
            2.00000 * math.pi * constants[8] * constants[8]
            + 2.00000 * math.pi * constants[8] * constants[7]
        )
        constants[61] = constants[56] * 0.260000
        constants[62] = constants[56] * 0.0600000
        constants[63] = constants[60] * 2.00000
        constants[64] = constants[56] * 0.680000
        constants[65] = constants[56] * 0.0552000
        constants[66] = constants[56] * 0.00480000
        constants[67] = constants[56] * 0.0200000
        constants[68] = constants[63] / constants[0]

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
        legend_voi = "time in component Environment (ms)"
        legend_constants[0] = "F in component Environment (C_per_mole)"
        legend_constants[1] = "K_o in component Environment (mM)"
        legend_constants[2] = "Ca_o in component Environment (mM)"
        legend_constants[3] = "Na_o in component Environment (mM)"
        legend_constants[4] = "Cl_o in component Environment (mM)"
        legend_constants[5] = "FonRT in component Environment (per_mV)"
        legend_constants[6] = "tissue in component Environment (dimensionless)"
        legend_states[0] = "V in component cell (mV)"
        legend_algebraic[62] = "INa in component INa (uA_per_uF)"
        legend_algebraic[31] = "ICaL in component ICaL (uA_per_uF)"
        legend_algebraic[77] = "IK1 in component IK1 (uA_per_uF)"
        legend_algebraic[83] = "IKp in component IKp (uA_per_uF)"
        legend_algebraic[85] = "IKs in component IKs (uA_per_uF)"
        legend_algebraic[81] = "IKr in component IKr (uA_per_uF)"
        legend_algebraic[49] = "IpCa in component IpCa (uA_per_uF)"
        legend_algebraic[50] = "ICab in component ICab (uA_per_uF)"
        legend_algebraic[48] = "INaCa in component INaCa (uA_per_uF)"
        legend_algebraic[41] = "INaK in component INaK (uA_per_uF)"
        legend_algebraic[82] = "Ito in component Ito (uA_per_uF)"
        legend_algebraic[55] = "Ito2 in component Ito2 (uA_per_uF)"
        legend_algebraic[88] = "IClb in component IClb (uA_per_uF)"
        legend_algebraic[65] = "INal in component INal (uA_per_uF)"
        legend_algebraic[51] = "caiont in component cell (uA_per_uF)"
        legend_algebraic[68] = "naiont in component cell (uA_per_uF)"
        legend_algebraic[86] = "kiont in component cell (uA_per_uF)"
        legend_algebraic[89] = "clont in component cell (uA_per_uF)"
        legend_constants[7] = "l in component cell (cm)"
        legend_constants[8] = "a in component cell (cm)"
        legend_constants[56] = "vcell in component cell (uL)"
        legend_constants[60] = "ageo in component cell (cm2)"
        legend_constants[63] = "Acap in component cell (uF)"
        legend_constants[64] = "vmyo in component cell (uL)"
        legend_constants[61] = "vmito in component cell (uL)"
        legend_constants[62] = "vsr in component cell (uL)"
        legend_constants[65] = "vnsr in component cell (uL)"
        legend_constants[66] = "vjsr in component cell (uL)"
        legend_constants[67] = "vss in component cell (uL)"
        legend_constants[68] = "AF in component cell (uF_mole_per_C)"
        legend_constants[9] = "stim_offset in component cell (ms)"
        legend_constants[10] = "stim_period in component cell (ms)"
        legend_constants[11] = "stim_duration in component cell (ms)"
        legend_constants[12] = "stim_amplitude in component cell (uA_per_uF)"
        legend_algebraic[16] = "i_Stim in component cell (uA_per_uF)"
        legend_algebraic[0] = "past in component cell (ms)"
        legend_algebraic[59] = "ENa in component reversal_potentials (mV)"
        legend_constants[57] = "GNa in component INa (mS_per_uF)"
        legend_algebraic[29] = "gNa in component INa (mS_per_uF)"
        legend_states[1] = "H in component INa (dimensionless)"
        legend_states[2] = "m in component INa (dimensionless)"
        legend_states[3] = "J in component INa (dimensionless)"
        legend_algebraic[1] = "am in component INa (per_ms)"
        legend_algebraic[17] = "bm in component INa (per_ms)"
        legend_algebraic[2] = "ah in component INa (per_ms)"
        legend_algebraic[18] = "bh in component INa (per_ms)"
        legend_algebraic[3] = "aj in component INa (per_ms)"
        legend_algebraic[19] = "bj in component INa (per_ms)"
        legend_states[4] = "Ca_ss in component Ca (mM)"
        legend_states[5] = "d in component ICaL (dimensionless)"
        legend_states[6] = "dp in component ICaL (dimensionless)"
        legend_states[7] = "f in component ICaL (dimensionless)"
        legend_states[8] = "fca in component ICaL (dimensionless)"
        legend_states[9] = "fca2 in component ICaL (dimensionless)"
        legend_states[10] = "f2 in component ICaL (dimensionless)"
        legend_constants[13] = "pca in component ICaL (L_per_F_ms)"
        legend_constants[14] = "gacai in component ICaL (dimensionless)"
        legend_constants[15] = "gacao in component ICaL (dimensionless)"
        legend_algebraic[54] = "CaMKactive in component Irel (dimensionless)"
        legend_algebraic[30] = "ibarca in component ICaL (uA_per_uF)"
        legend_algebraic[4] = "dss in component ICaL (dimensionless)"
        legend_algebraic[20] = "taud in component ICaL (ms)"
        legend_algebraic[5] = "fss in component ICaL (dimensionless)"
        legend_algebraic[6] = "f2ss in component ICaL (dimensionless)"
        legend_algebraic[21] = "tauf in component ICaL (ms)"
        legend_algebraic[22] = "tauf2 in component ICaL (ms)"
        legend_algebraic[7] = "dpss in component ICaL (dimensionless)"
        legend_algebraic[32] = "fcass in component ICaL (dimensionless)"
        legend_algebraic[33] = "fca2ss in component ICaL (dimensionless)"
        legend_algebraic[56] = "taufca in component ICaL (ms)"
        legend_algebraic[35] = "taufca2 in component ICaL (ms)"
        legend_algebraic[70] = "EK in component reversal_potentials (mV)"
        legend_algebraic[72] = "ak1 in component IK1 (per_ms)"
        legend_algebraic[74] = "bk1 in component IK1 (per_ms)"
        legend_constants[58] = "gkr in component IKr (mS_per_uF)"
        legend_algebraic[37] = "r in component IKr (dimensionless)"
        legend_states[11] = "xr in component IKr (dimensionless)"
        legend_algebraic[8] = "xrss in component IKr (dimensionless)"
        legend_algebraic[23] = "tauxr in component IKr (ms)"
        legend_states[12] = "Ca_i in component Ca (mM)"
        legend_algebraic[38] = "gks in component IKs (mS_per_uF)"
        legend_algebraic[84] = "EKs in component reversal_potentials (mV)"
        legend_algebraic[9] = "xss in component IKs (dimensionless)"
        legend_algebraic[24] = "tauxs in component IKs (ms)"
        legend_states[13] = "xs1 in component IKs (dimensionless)"
        legend_states[14] = "xs2 in component IKs (dimensionless)"
        legend_constants[16] = "gitodv in component Ito (mS_per_uF)"
        legend_algebraic[39] = "rv in component Ito (dimensionless)"
        legend_algebraic[10] = "ay in component Ito (per_ms)"
        legend_algebraic[25] = "by in component Ito (per_ms)"
        legend_algebraic[11] = "ay2 in component Ito (per_ms)"
        legend_algebraic[26] = "by2 in component Ito (per_ms)"
        legend_algebraic[12] = "ay3 in component Ito (per_ms)"
        legend_algebraic[27] = "by3 in component Ito (per_ms)"
        legend_states[15] = "ydv in component Ito (dimensionless)"
        legend_states[16] = "ydv2 in component Ito (dimensionless)"
        legend_states[17] = "zdv in component Ito (dimensionless)"
        legend_states[18] = "Na_i in component Na (mM)"
        legend_constants[17] = "kmnai in component INaK (mM)"
        legend_constants[18] = "kmko in component INaK (mM)"
        legend_constants[19] = "ibarnak in component INaK (uA_per_uF)"
        legend_constants[59] = "sigma in component INaK (dimensionless)"
        legend_algebraic[40] = "fnak in component INaK (dimensionless)"
        legend_algebraic[42] = "ca_i_NaCa in component INaCa (mM)"
        legend_constants[20] = "KmCa in component INaCa (mM)"
        legend_algebraic[43] = "allo in component INaCa (dimensionless)"
        legend_constants[21] = "NCXmax in component INaCa (uA_per_uF)"
        legend_constants[22] = "ksat in component INaCa (dimensionless)"
        legend_constants[23] = "eta in component INaCa (dimensionless)"
        legend_constants[24] = "KmNai in component INaCa (mM)"
        legend_constants[25] = "KmNao in component INaCa (mM)"
        legend_constants[26] = "KmCai in component INaCa (mM)"
        legend_constants[27] = "KmCao in component INaCa (mM)"
        legend_algebraic[44] = "num in component INaCa (mM4)"
        legend_algebraic[45] = "denom1 in component INaCa (dimensionless)"
        legend_algebraic[46] = "denom2 in component INaCa (mM4)"
        legend_algebraic[47] = "denom3 in component INaCa (mM4)"
        legend_constants[28] = "ibarpca in component IpCa (uA_per_uF)"
        legend_constants[29] = "kmpca in component IpCa (mM)"
        legend_states[19] = "Cl_i in component Cl (mM)"
        legend_constants[30] = "PCl in component Ito2 (L_per_F_ms)"
        legend_states[20] = "AA in component Ito2 (dimensionless)"
        legend_algebraic[53] = "Ito2_max in component Ito2 (uA_per_uF)"
        legend_algebraic[13] = "AAss in component Ito2 (dimensionless)"
        legend_constants[31] = "Kmto2 in component Ito2 (mM)"
        legend_algebraic[87] = "ECl in component reversal_potentials (mV)"
        legend_constants[32] = "GClb in component IClb (mS_per_uF)"
        legend_constants[33] = "GNaL in component INal (mS_per_uF)"
        legend_states[21] = "mL in component INal (dimensionless)"
        legend_states[22] = "hL in component INal (dimensionless)"
        legend_algebraic[14] = "amL in component INal (per_ms)"
        legend_algebraic[28] = "bmL in component INal (per_ms)"
        legend_algebraic[15] = "hLss in component INal (dimensionless)"
        legend_states[23] = "K_i in component K (mM)"
        legend_constants[34] = "prnak in component reversal_potentials (dimensionless)"
        legend_states[24] = "Ca_jsr in component Ca (mM)"
        legend_algebraic[61] = "Grel in component Irel (per_ms)"
        legend_algebraic[34] = "dro_inf in component Irel (dimensionless)"
        legend_constants[35] = "dtau_rel_max in component Irel (ms)"
        legend_algebraic[60] = "dtau_rel in component Irel (ms)"
        legend_algebraic[36] = "ross in component Irel (dimensionless)"
        legend_algebraic[63] = "riss in component Irel (dimensionless)"
        legend_algebraic[66] = "tauri in component Irel (ms)"
        legend_algebraic[64] = "irelcicr in component Irel (mM_per_ms)"
        legend_constants[36] = "CaMK0 in component Irel (dimensionless)"
        legend_constants[37] = "Km in component Irel (mM)"
        legend_constants[38] = "KmCaMK in component Irel (dimensionless)"
        legend_algebraic[52] = "CaMKbound in component Irel (dimensionless)"
        legend_states[25] = "CaMKtrap in component Irel (dimensionless)"
        legend_states[26] = "ro in component Irel (dimensionless)"
        legend_states[27] = "ri in component Irel (dimensionless)"
        legend_algebraic[58] = "vg in component Irel (dimensionless)"
        legend_algebraic[57] = "cafac in component Irel (dimensionless)"
        legend_constants[39] = "dKmPLBmax in component Iup_Ileak (mM)"
        legend_constants[40] = "dJupmax in component Iup_Ileak (dimensionless)"
        legend_algebraic[67] = "dKmPLB in component Iup_Ileak (mM)"
        legend_algebraic[69] = "dJup in component Iup_Ileak (dimensionless)"
        legend_constants[41] = "iupmax in component Iup_Ileak (mM_per_ms)"
        legend_constants[42] = "Kmup in component Iup_Ileak (mM)"
        legend_constants[43] = "nsrmax in component Iup_Ileak (mM)"
        legend_algebraic[71] = "iup in component Iup_Ileak (mM_per_ms)"
        legend_algebraic[73] = "ileak in component Iup_Ileak (mM_per_ms)"
        legend_states[28] = "Ca_nsr in component Ca (mM)"
        legend_algebraic[76] = "idiff in component Idiff_Itr (mM_per_ms)"
        legend_algebraic[75] = "itr in component Idiff_Itr (mM_per_ms)"
        legend_algebraic[90] = "CTNaCl in component Na (mM_per_ms)"
        legend_constants[44] = "CTNaClmax in component Na (mM_per_ms)"
        legend_algebraic[91] = "CTKCl in component K (mM_per_ms)"
        legend_constants[45] = "CTKClmax in component K (mM_per_ms)"
        legend_constants[46] = "kmt in component Ca (mM)"
        legend_constants[47] = "kmc in component Ca (mM)"
        legend_constants[48] = "tbar in component Ca (mM)"
        legend_constants[49] = "cbar in component Ca (mM)"
        legend_constants[50] = "kmcsqn in component Ca (mM)"
        legend_constants[51] = "csqnbar in component Ca (mM)"
        legend_algebraic[78] = "bcsqn in component Ca (dimensionless)"
        legend_algebraic[79] = "bmyo in component Ca (dimensionless)"
        legend_constants[52] = "BSRmax in component Ca (mM)"
        legend_constants[53] = "KmBSR in component Ca (mM)"
        legend_constants[54] = "BSLmax in component Ca (mM)"
        legend_constants[55] = "KmBSL in component Ca (mM)"
        legend_algebraic[80] = "bss in component Ca (dimensionless)"
        legend_rates[0] = "d/dt V in component cell (mV)"
        legend_rates[1] = "d/dt H in component INa (dimensionless)"
        legend_rates[2] = "d/dt m in component INa (dimensionless)"
        legend_rates[3] = "d/dt J in component INa (dimensionless)"
        legend_rates[5] = "d/dt d in component ICaL (dimensionless)"
        legend_rates[6] = "d/dt dp in component ICaL (dimensionless)"
        legend_rates[7] = "d/dt f in component ICaL (dimensionless)"
        legend_rates[10] = "d/dt f2 in component ICaL (dimensionless)"
        legend_rates[8] = "d/dt fca in component ICaL (dimensionless)"
        legend_rates[9] = "d/dt fca2 in component ICaL (dimensionless)"
        legend_rates[11] = "d/dt xr in component IKr (dimensionless)"
        legend_rates[13] = "d/dt xs1 in component IKs (dimensionless)"
        legend_rates[14] = "d/dt xs2 in component IKs (dimensionless)"
        legend_rates[15] = "d/dt ydv in component Ito (dimensionless)"
        legend_rates[16] = "d/dt ydv2 in component Ito (dimensionless)"
        legend_rates[17] = "d/dt zdv in component Ito (dimensionless)"
        legend_rates[20] = "d/dt AA in component Ito2 (dimensionless)"
        legend_rates[21] = "d/dt mL in component INal (dimensionless)"
        legend_rates[22] = "d/dt hL in component INal (dimensionless)"
        legend_rates[26] = "d/dt ro in component Irel (dimensionless)"
        legend_rates[27] = "d/dt ri in component Irel (dimensionless)"
        legend_rates[25] = "d/dt CaMKtrap in component Irel (dimensionless)"
        legend_rates[18] = "d/dt Na_i in component Na (mM)"
        legend_rates[23] = "d/dt K_i in component K (mM)"
        legend_rates[19] = "d/dt Cl_i in component Cl (mM)"
        legend_rates[12] = "d/dt Ca_i in component Ca (mM)"
        legend_rates[4] = "d/dt Ca_ss in component Ca (mM)"
        legend_rates[28] = "d/dt Ca_nsr in component Ca (mM)"
        legend_rates[24] = "d/dt Ca_jsr in component Ca (mM)"
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
            self.init_algebraic[-len(self.channel_list):],
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
        algebraic[7] = 9.00000 - 8.00000 / (
            1.00000 + np.exp(-(states[0] + 65.0000) / 3.40000)
        )
        algebraic[13] = 1.00000 / (1.00000 + constants[31] / states[4])
        algebraic[15] = 1.00000 / (
            1.00000 + np.exp((states[0] + 91.0000) / 6.10000)
        )
        algebraic[2] = super().custom_piecewise([
            np.greater_equal(states[0], -40.0000),
            0.00000,
            True,
            0.135000 * np.exp((80.0000 + states[0]) / -6.80000),
        ])
        algebraic[18] = super().custom_piecewise([
            np.greater_equal(states[0], -40.0000),
            1.00000 / (0.130000 * (1.00000 + np.exp((states[0] + 10.6600) / -11.1000))),
            True,
            3.56000 * np.exp(0.0790000 * states[0])
            + 310000.0 * np.exp(0.350000 * states[0]),
        ])
        algebraic[1] = (0.320000 * 1.00000 * (states[0] + 47.1300)) / (
            1.00000 - np.exp(-0.100000 * (states[0] + 47.1300))
        )
        algebraic[17] = 0.0800000 * np.exp(-states[0] / 11.0000)
        algebraic[3] = super().custom_piecewise([
            np.greater_equal(states[0], -40.0000),
            0.00000,
            True,
            (
                (
                    -127140.0 * np.exp(0.244400 * states[0])
                    - 3.47400e-05 * np.exp(-0.0439100 * states[0])
                )
                * 1.00000
                * (states[0] + 37.7800)
            )
            / (1.00000 + np.exp(0.311000 * (states[0] + 79.2300))),
        ])
        algebraic[19] = super().custom_piecewise([
            np.greater_equal(states[0], -40.0000),
            (0.300000 * np.exp(-2.53500e-07 * states[0]))
            / (1.00000 + np.exp(-0.100000 * (states[0] + 32.0000))),
            True,
            (0.121200 * np.exp(-0.0105200 * states[0]))
            / (1.00000 + np.exp(-0.137800 * (states[0] + 40.1400))),
        ])
        algebraic[4] = 1.00000 / (
            1.00000 + np.exp(-(states[0] - 4.00000) / 6.74000)
        )
        algebraic[20] = 0.590000 + (
            0.800000 * np.exp(0.0520000 * (states[0] + 13.0000))
        ) / (1.00000 + np.exp(0.132000 * (states[0] + 13.0000)))
        algebraic[5] = (
            0.700000 / (1.00000 + np.exp((states[0] + 17.1200) / 7.00000))
            + 0.300000
        )
        algebraic[21] = 1.00000 / (
            0.241100
            * np.exp(-(np.power(0.0450000 * (states[0] - 9.69140), 2.00000)))
            + 0.0529000
        )
        algebraic[6] = (
            0.770000 / (1.00000 + np.exp((states[0] + 17.1200) / 7.00000))
            + 0.230000
        )
        algebraic[22] = 1.00000 / (
            0.0423000
            * np.exp(-(np.power(0.0590000 * (states[0] - 18.5726), 2.00000)))
            + 0.00540000
        )
        algebraic[8] = 1.00000 / (
            1.00000 + np.exp(-(states[0] + 10.0850) / 4.25000)
        )
        algebraic[23] = 1.00000 / (
            (0.000600000 * (states[0] - 1.73840))
            / (1.00000 - np.exp(-0.136000 * (states[0] - 1.73840)))
            + (0.000300000 * (states[0] + 38.3608))
            / (np.exp(0.152200 * (states[0] + 38.3608)) - 1.00000)
        )
        algebraic[9] = 1.00000 / (
            1.00000 + np.exp(-(states[0] - 10.5000) / 24.7000)
        )
        algebraic[24] = 1.00000 / (
            (7.61000e-05 * (states[0] + 44.6000))
            / (1.00000 - np.exp(-9.97000 * (states[0] + 44.6000)))
            + (0.000360000 * (states[0] - 0.550000))
            / (np.exp(0.128000 * (states[0] - 0.550000)) - 1.00000)
        )
        algebraic[10] = (25.0000 * np.exp((states[0] - 40.0000) / 25.0000)) / (
            1.00000 + np.exp((states[0] - 40.0000) / 25.0000)
        )
        algebraic[25] = (25.0000 * np.exp(-(states[0] + 90.0000) / 25.0000)) / (
            1.00000 + np.exp(-(states[0] + 90.0000) / 25.0000)
        )
        algebraic[11] = 0.0300000 / (
            1.00000 + np.exp((states[0] + 60.0000) / 5.00000)
        )
        algebraic[26] = (0.200000 * np.exp((states[0] + 25.0000) / 5.00000)) / (
            1.00000 + np.exp((states[0] + 25.0000) / 5.00000)
        )
        algebraic[12] = 0.00390000 / (
            1.00000 + np.exp((states[0] + 63.0000) / 5.00000)
        )
        algebraic[27] = (0.100000 * np.exp((states[0] + 25.0000) / 5.00000)) / (
            1.00000 + np.exp((states[0] + 25.0000) / 5.00000)
        )
        algebraic[14] = (0.320000 * 1.00000 * (states[0] + 47.1300)) / (
            1.00000 - np.exp(-0.100000 * (states[0] + 47.1300))
        )
        algebraic[28] = 0.0800000 * np.exp(-states[0] / 11.0000)
        algebraic[30] = (
            constants[13]
            * 4.00000
            * (states[0] - 15.0000)
            * constants[0]
            * constants[5]
            * (
                constants[14]
                * states[4]
                * np.exp(2.00000 * (states[0] - 15.0000) * constants[5])
                - constants[15] * constants[2]
            )
        ) / (np.exp(2.00000 * (states[0] - 15.0000) * constants[5]) - 1.00000)
        algebraic[31] = (
            algebraic[-7]
            * np.power(states[5], states[6])
            * states[7]
            * states[10]
            * states[8]
            * states[9]
            * algebraic[30]
        )
        algebraic[33] = 1.00000 / (1.00000 - algebraic[31] / 0.0100000)
        algebraic[35] = (
            300.000 / (1.00000 + np.exp((-algebraic[31] - 0.175000) / 0.0400000))
            + 125.000
        )
        algebraic[34] = np.power(states[24], 1.90000) / (
            np.power(states[24], 1.90000)
            + np.power(
                (49.2800 * states[4]) / (states[4] + 0.00280000), 1.90000
            )
        )
        algebraic[36] = algebraic[34] / (
            np.power(1.00000 / algebraic[31], 2.00000) + 1.00000
        )
        algebraic[52] = (constants[36] * (1.00000 - states[25])) / (
            1.00000 + constants[37] / states[4]
        )
        algebraic[54] = algebraic[52] + states[25]
        algebraic[32] = (
            0.300000 / (1.00000 - algebraic[31] / 0.0500000)
            + 0.550000 / (1.00000 + states[4] / 0.00300000)
            + 0.150000
        )
        algebraic[56] = (
            (10.0000 * algebraic[54]) / (0.150000 + algebraic[54])
            + 1.00000 / (1.00000 + states[4] / 0.00300000)
            + 0.500000
        )
        algebraic[57] = 1.00000 / (
            1.00000 + np.exp((algebraic[31] + 0.0500000) / 0.0150000)
        )
        algebraic[63] = 1.00000 / (
            1.00000
            + np.exp(
                ((states[4] - 0.000400000) + 0.00200000 * algebraic[57])
                / 2.50000e-05
            )
        )
        algebraic[60] = (constants[35] * algebraic[54]) / (
            constants[38] + algebraic[54]
        )
        algebraic[66] = (
            3.00000
            + algebraic[60]
            + (350.000 - algebraic[60])
            / (
                1.00000
                + np.exp(
                    ((states[4] - 0.00300000) + 0.00300000 * algebraic[57])
                    / 0.000200000
                )
            )
        )
        algebraic[67] = (constants[39] * algebraic[54]) / (
            constants[38] + algebraic[54]
        )
        algebraic[69] = (constants[40] * algebraic[54]) / (
            constants[38] + algebraic[54]
        )
        algebraic[71] = (
            (algebraic[69] + 1.00000) * constants[41] * states[12]
        ) / ((states[12] + constants[42]) - algebraic[67])
        algebraic[73] = (constants[41] * states[28]) / constants[43]
        algebraic[75] = (states[28] - states[24]) / 120.000
        algebraic[49] = (constants[28] * states[12]) / (
            constants[29] + states[12]
        )
        algebraic[50] = (
            1.99508e-07
            * 4.00000
            * states[0]
            * constants[0]
            * constants[5]
            * (
                states[12] * np.exp(2.00000 * states[0] * constants[5])
                - 0.341000 * constants[2]
            )
        ) / (np.exp(2.00000 * states[0] * constants[5]) - 1.00000)
        algebraic[42] = 1.50000 * states[12]
        algebraic[43] = 1.00000 / (
            1.00000 + np.power(constants[20] / algebraic[42], 2.00000)
        )
        algebraic[44] = (
            np.power(states[18], 3.00000)
            * constants[2]
            * np.exp(constants[23] * states[0] * constants[5])
            - np.power(constants[3], 3.00000)
            * algebraic[42]
            * np.exp((constants[23] - 1.00000) * states[0] * constants[5])
        )
        algebraic[45] = 1.00000 + constants[22] * np.exp(
            (constants[23] - 1.00000) * states[0] * constants[5]
        )
        algebraic[46] = (
            constants[27] * np.power(states[18], 3.00000)
            + np.power(constants[25], 3.00000) * algebraic[42]
            + np.power(constants[24], 3.00000)
            * constants[2]
            * (1.00000 + algebraic[42] / constants[26])
        )
        algebraic[47] = (
            constants[26]
            * np.power(constants[3], 3.00000)
            * (1.00000 + np.power(states[18] / constants[24], 3.00000))
            + np.power(states[18], 3.00000) * constants[2]
            + np.power(constants[3], 3.00000) * algebraic[42]
        )
        algebraic[48] = (constants[21] * algebraic[43] * algebraic[44]) / (
            algebraic[45] * (algebraic[46] + algebraic[47])
        )
        algebraic[76] = (states[4] - states[12]) / 0.200000
        algebraic[79] = 1.00000 / (
            1.00000
            + (constants[49] * constants[47])
            / np.power(states[12] + constants[47], 2.00000)
            + (constants[46] * constants[48])
            / np.power(states[12] + constants[46], 2.00000)
        )
        algebraic[58] = 1.00000 / (
            1.00000
            + np.exp((algebraic[-7] * algebraic[30] + 13.0000) / 5.00000)
        )
        algebraic[61] = 3000.00 * algebraic[58]
        algebraic[64] = (
            algebraic[61] * states[26] * states[27] * (states[24] - states[4])
        )
        algebraic[80] = 1.00000 / (
            1.00000
            + (constants[52] * constants[53])
            / np.power(constants[53] + states[4], 2.00000)
            + (constants[54] * constants[55])
            / np.power(constants[55] + states[4], 2.00000)
        )
        algebraic[78] = 1.00000 / (
            1.00000
            + (constants[50] * constants[51])
            / np.power(states[24] + constants[50], 2.00000)
        )
        algebraic[51] = (
            (algebraic[31] + algebraic[50] + algebraic[49]) - 2.00000 * algebraic[48]
        )
        algebraic[59] = np.log(constants[3] / states[18]) / constants[5]
        algebraic[29] = (
            constants[57] * states[2] * states[2] * states[2] * states[1] * states[3]
        )
        algebraic[62] = algebraic[-4] * algebraic[29] * (states[0] - algebraic[59])
        algebraic[40] = 1.00000 / (
            1.00000
            + 0.124500 * np.exp(-0.100000 * states[0] * constants[5])
            + 0.0365000 * constants[59] * np.exp(-states[0] * constants[5])
        )
        algebraic[41] = (
            (
                (constants[19] * algebraic[40] * 1.00000)
                / (1.00000 + np.power(constants[17] / states[18], 2.00000))
            )
            * constants[1]
        ) / (constants[1] + constants[18])
        algebraic[65] = (
            algebraic[-3]
            * constants[33]
            * np.power(states[21], 3.00000)
            * states[22]
            * (states[0] - algebraic[59])
        )
        algebraic[68] = (
            algebraic[62]
            + 3.00000 * algebraic[48]
            + 3.00000 * algebraic[41]
            + algebraic[65]
        )
        algebraic[70] = np.log(constants[1] / states[23]) / constants[5]
        algebraic[72] = 1.02000 / (
            1.00000
            + np.exp(0.238500 * ((states[0] - algebraic[70]) - 59.2150))
        )
        algebraic[74] = (
            0.491240 * np.exp(0.0803200 * ((states[0] - algebraic[70]) + 5.47600))
            + 1.00000 * np.exp(0.0617500 * ((states[0] - algebraic[70]) - 594.310))
        ) / (1.00000 + np.exp(-0.514300 * ((states[0] - algebraic[70]) + 4.75300)))
        algebraic[77] = (
            algebraic[-6]
            * (
                (0.500000 * np.power(constants[1] / 5.40000, 1.0 / 2) * algebraic[72])
                / (algebraic[72] + algebraic[74])
            )
            * (states[0] - algebraic[70])
        )
        algebraic[83] = (0.00276000 * (states[0] - algebraic[70])) / (
            1.00000 + np.exp((7.48800 - states[0]) / 5.98000)
        )
        algebraic[38] = 0.0248975 * (
            1.00000
            + 0.600000 / (1.00000 + np.power(3.80000e-05 / states[12], 1.40000))
        )
        algebraic[84] = (
            np.log(
                (constants[1] + constants[34] * constants[3])
                / (states[23] + constants[34] * states[18])
            )
            / constants[5]
        )
        algebraic[85] = (
            algebraic[-5]
            * algebraic[38]
            * states[13]
            * states[14]
            * (states[0] - algebraic[84])
        )
        algebraic[37] = 1.00000 / (
            1.00000 + np.exp((states[0] + 10.0000) / 15.4000)
        )
        algebraic[81] = (
            algebraic[-1]
            * constants[58]
            * states[11]
            * algebraic[37]
            * (states[0] - algebraic[70])
        )
        algebraic[39] = np.exp(states[0] / 300.000)
        algebraic[82] = (
            algebraic[-2]
            * constants[16]
            * np.power(states[15], 3.00000)
            * states[16]
            * states[17]
            * algebraic[39]
            * (states[0] - algebraic[70])
        )
        algebraic[0] = np.floor(voi / constants[10]) * constants[10]
        algebraic[16] = super().custom_piecewise([
            np.greater_equal(voi - algebraic[0], constants[9])
            & np.less_equal(voi - algebraic[0], constants[9] + constants[11]),
            constants[12],
            True,
            0.00000,
        ])
        algebraic[86] = (
            (algebraic[81] + algebraic[85] + algebraic[77] + algebraic[83])
            - 2.00000 * algebraic[41]
        ) + algebraic[82] + 0.500000 * algebraic[16]
        algebraic[53] = (
            constants[30]
            * states[0]
            * constants[0]
            * constants[5]
            * (states[19] - constants[4] * np.exp(states[0] * constants[5]))
        ) / (1.00000 - np.exp(states[0] * constants[5]))
        algebraic[55] = algebraic[53] * states[20]
        algebraic[87] = -np.log(constants[4] / states[19]) / constants[5]
        algebraic[88] = constants[32] * (states[0] - algebraic[87])
        algebraic[89] = algebraic[88] + algebraic[55] + 0.500000 * algebraic[16]
        algebraic[90] = (
            constants[44] * np.power(algebraic[59] - algebraic[87], 4.00000)
        ) / (
            np.power(algebraic[59] - algebraic[87], 4.00000)
            + np.power(87.8251, 4.00000)
        )
        algebraic[91] = (constants[45] * (algebraic[70] - algebraic[87])) / (
            (algebraic[70] - algebraic[87]) + 87.8251
        )
        return algebraic


@njit
def njitComputeRates(
    voi, states, constants, sizeStates, sizeAlgebraic, num_channels, drug_multipliers
):
    """Compute ODEs for the Benson 2008 epicardial dog model."""
    rates = np.zeros(sizeStates, dtype=np.float64)
    algebraic = np.zeros(sizeAlgebraic, dtype=np.float64)
    algebraic[-num_channels:] = drug_multipliers
    algebraic[7] = 9.00000 - 8.00000 / (
        1.00000 + np.exp(-(states[0] + 65.0000) / 3.40000)
    )
    rates[6] = (algebraic[7] - states[6]) / 10.0000
    algebraic[13] = 1.00000 / (1.00000 + constants[31] / states[4])
    rates[20] = (algebraic[13] - states[20]) / 1.00000
    algebraic[15] = 1.00000 / (1.00000 + np.exp((states[0] + 91.0000) / 6.10000))
    rates[22] = (algebraic[15] - states[22]) / 600.000
    algebraic[2] = (
        0.00000
        if states[0] >= -40.0000
        else 0.135000 * np.exp((80.0000 + states[0]) / -6.80000)
    )
    algebraic[18] = (
        1.00000 / (0.130000 * (1.00000 + np.exp((states[0] + 10.6600) / -11.1000)))
        if states[0] >= -40.0000
        else 3.56000 * np.exp(0.0790000 * states[0])
        + 310000.0 * np.exp(0.350000 * states[0])
    )
    rates[1] = algebraic[2] * (1.00000 - states[1]) - algebraic[18] * states[1]
    algebraic[1] = (0.320000 * 1.00000 * (states[0] + 47.1300)) / (
        1.00000 - np.exp(-0.100000 * (states[0] + 47.1300))
    )
    algebraic[17] = 0.0800000 * np.exp(-states[0] / 11.0000)
    rates[2] = algebraic[1] * (1.00000 - states[2]) - algebraic[17] * states[2]
    algebraic[3] = (
        0.00000
        if states[0] >= -40.0000
        else (
            (
                -127140.0 * np.exp(0.244400 * states[0])
                - 3.47400e-05 * np.exp(-0.0439100 * states[0])
            )
            * 1.00000
            * (states[0] + 37.7800)
        )
        / (1.00000 + np.exp(0.311000 * (states[0] + 79.2300)))
    )
    algebraic[19] = (
        (0.300000 * np.exp(-2.53500e-07 * states[0]))
        / (1.00000 + np.exp(-0.100000 * (states[0] + 32.0000)))
        if states[0] >= -40.0000
        else (0.121200 * np.exp(-0.0105200 * states[0]))
        / (1.00000 + np.exp(-0.137800 * (states[0] + 40.1400)))
    )
    rates[3] = algebraic[3] * (1.00000 - states[3]) - algebraic[19] * states[3]
    algebraic[4] = 1.00000 / (1.00000 + np.exp(-(states[0] - 4.00000) / 6.74000))
    algebraic[20] = 0.590000 + (
        0.800000 * np.exp(0.0520000 * (states[0] + 13.0000))
    ) / (1.00000 + np.exp(0.132000 * (states[0] + 13.0000)))
    rates[5] = (algebraic[4] - states[5]) / algebraic[20]
    algebraic[5] = (
        0.700000 / (1.00000 + np.exp((states[0] + 17.1200) / 7.00000)) + 0.300000
    )
    algebraic[21] = 1.00000 / (
        0.241100
        * np.exp(-(np.power(0.0450000 * (states[0] - 9.69140), 2.00000)))
        + 0.0529000
    )
    rates[7] = (algebraic[5] - states[7]) / algebraic[21]
    algebraic[6] = (
        0.770000 / (1.00000 + np.exp((states[0] + 17.1200) / 7.00000)) + 0.230000
    )
    algebraic[22] = 1.00000 / (
        0.0423000
        * np.exp(-(np.power(0.0590000 * (states[0] - 18.5726), 2.00000)))
        + 0.00540000
    )
    rates[10] = (algebraic[6] - states[10]) / algebraic[22]
    algebraic[8] = 1.00000 / (
        1.00000 + np.exp(-(states[0] + 10.0850) / 4.25000)
    )
    algebraic[23] = 1.00000 / (
        (0.000600000 * (states[0] - 1.73840))
        / (1.00000 - np.exp(-0.136000 * (states[0] - 1.73840)))
        + (0.000300000 * (states[0] + 38.3608))
        / (np.exp(0.152200 * (states[0] + 38.3608)) - 1.00000)
    )
    rates[11] = (algebraic[8] - states[11]) / algebraic[23]
    algebraic[9] = 1.00000 / (
        1.00000 + np.exp(-(states[0] - 10.5000) / 24.7000)
    )
    algebraic[24] = 1.00000 / (
        (7.61000e-05 * (states[0] + 44.6000))
        / (1.00000 - np.exp(-9.97000 * (states[0] + 44.6000)))
        + (0.000360000 * (states[0] - 0.550000))
        / (np.exp(0.128000 * (states[0] - 0.550000)) - 1.00000)
    )
    rates[13] = (algebraic[9] - states[13]) / algebraic[24]
    rates[14] = ((algebraic[9] - states[14]) / algebraic[24]) / 2.00000
    algebraic[10] = (25.0000 * np.exp((states[0] - 40.0000) / 25.0000)) / (
        1.00000 + np.exp((states[0] - 40.0000) / 25.0000)
    )
    algebraic[25] = (25.0000 * np.exp(-(states[0] + 90.0000) / 25.0000)) / (
        1.00000 + np.exp(-(states[0] + 90.0000) / 25.0000)
    )
    rates[15] = algebraic[10] * (1.00000 - states[15]) - algebraic[25] * states[15]
    algebraic[11] = 0.0300000 / (1.00000 + np.exp((states[0] + 60.0000) / 5.00000))
    algebraic[26] = (0.200000 * np.exp((states[0] + 25.0000) / 5.00000)) / (
        1.00000 + np.exp((states[0] + 25.0000) / 5.00000)
    )
    rates[16] = algebraic[11] * (1.00000 - states[16]) - algebraic[26] * states[16]
    algebraic[12] = 0.00390000 / (
        1.00000 + np.exp((states[0] + 63.0000) / 5.00000)
    )
    algebraic[27] = (0.100000 * np.exp((states[0] + 25.0000) / 5.00000)) / (
        1.00000 + np.exp((states[0] + 25.0000) / 5.00000)
    )
    rates[17] = algebraic[12] * (1.00000 - states[17]) - algebraic[27] * states[17]
    algebraic[14] = (0.320000 * 1.00000 * (states[0] + 47.1300)) / (
        1.00000 - np.exp(-0.100000 * (states[0] + 47.1300))
    )
    algebraic[28] = 0.0800000 * np.exp(-states[0] / 11.0000)
    rates[21] = algebraic[14] * (1.00000 - states[21]) - algebraic[28] * states[21]
    algebraic[30] = (
        constants[13]
        * 4.00000
        * (states[0] - 15.0000)
        * constants[0]
        * constants[5]
        * (
            constants[14]
            * states[4]
            * np.exp(2.00000 * (states[0] - 15.0000) * constants[5])
            - constants[15] * constants[2]
        )
    ) / (np.exp(2.00000 * (states[0] - 15.0000) * constants[5]) - 1.00000)
    algebraic[31] = (
        algebraic[-7]
        * np.power(states[5], states[6])
        * states[7]
        * states[10]
        * states[8]
        * states[9]
        * algebraic[30]
    )
    algebraic[33] = 1.00000 / (1.00000 - algebraic[31] / 0.0100000)
    algebraic[35] = (
        300.000 / (1.00000 + np.exp((-algebraic[31] - 0.175000) / 0.0400000))
        + 125.000
    )
    rates[9] = (algebraic[33] - states[9]) / algebraic[35]
    algebraic[34] = np.power(states[24], 1.90000) / (
        np.power(states[24], 1.90000)
        + np.power(
            (49.2800 * states[4]) / (states[4] + 0.00280000), 1.90000
        )
    )
    algebraic[36] = algebraic[34] / (
        np.power(1.00000 / algebraic[31], 2.00000) + 1.00000
    )
    rates[26] = (algebraic[36] - states[26]) / 3.00000
    algebraic[52] = (constants[36] * (1.00000 - states[25])) / (
        1.00000 + constants[37] / states[4]
    )
    algebraic[54] = algebraic[52] + states[25]
    rates[25] = (
        0.0500000 * algebraic[54] * (algebraic[54] - states[25])
        - 0.000680000 * states[25]
    )
    algebraic[32] = (
        0.300000 / (1.00000 - algebraic[31] / 0.0500000)
        + 0.550000 / (1.00000 + states[4] / 0.00300000)
        + 0.150000
    )
    algebraic[56] = (
        (10.0000 * algebraic[54]) / (0.150000 + algebraic[54])
        + 1.00000 / (1.00000 + states[4] / 0.00300000)
        + 0.500000
    )
    rates[8] = (algebraic[32] - states[8]) / algebraic[56]
    algebraic[57] = 1.00000 / (
        1.00000 + np.exp((algebraic[31] + 0.0500000) / 0.0150000)
    )
    algebraic[63] = 1.00000 / (
        1.00000
        + np.exp(
            ((states[4] - 0.000400000) + 0.00200000 * algebraic[57]) / 2.50000e-05
        )
    )
    algebraic[60] = (constants[35] * algebraic[54]) / (constants[38] + algebraic[54])
    algebraic[66] = (
        3.00000
        + algebraic[60]
        + (350.000 - algebraic[60])
        / (
            1.00000
            + np.exp(
                ((states[4] - 0.00300000) + 0.00300000 * algebraic[57]) / 0.000200000
            )
        )
    )
    rates[27] = (algebraic[63] - states[27]) / algebraic[66]
    algebraic[67] = (constants[39] * algebraic[54]) / (constants[38] + algebraic[54])
    algebraic[69] = (constants[40] * algebraic[54]) / (constants[38] + algebraic[54])
    algebraic[71] = (
        (algebraic[69] + 1.00000) * constants[41] * states[12]
    ) / ((states[12] + constants[42]) - algebraic[67])
    algebraic[73] = (constants[41] * states[28]) / constants[43]
    algebraic[75] = (states[28] - states[24]) / 120.000
    rates[28] = (
        algebraic[71] - (algebraic[75] * constants[66]) / constants[65]
    ) - algebraic[73]
    algebraic[49] = (constants[28] * states[12]) / (constants[29] + states[12])
    algebraic[50] = (
        1.99508e-07
        * 4.00000
        * states[0]
        * constants[0]
        * constants[5]
        * (
            states[12] * np.exp(2.00000 * states[0] * constants[5])
            - 0.341000 * constants[2]
        )
    ) / (np.exp(2.00000 * states[0] * constants[5]) - 1.00000)
    algebraic[42] = 1.50000 * states[12]
    algebraic[43] = 1.00000 / (
        1.00000 + np.power(constants[20] / algebraic[42], 2.00000)
    )
    algebraic[44] = (
        np.power(states[18], 3.00000)
        * constants[2]
        * np.exp(constants[23] * states[0] * constants[5])
        - np.power(constants[3], 3.00000)
        * algebraic[42]
        * np.exp((constants[23] - 1.00000) * states[0] * constants[5])
    )
    algebraic[45] = 1.00000 + constants[22] * np.exp(
        (constants[23] - 1.00000) * states[0] * constants[5]
    )
    algebraic[46] = (
        constants[27] * np.power(states[18], 3.00000)
        + np.power(constants[25], 3.00000) * algebraic[42]
        + np.power(constants[24], 3.00000)
        * constants[2]
        * (1.00000 + algebraic[42] / constants[26])
    )
    algebraic[47] = (
        constants[26]
        * np.power(constants[3], 3.00000)
        * (1.00000 + np.power(states[18] / constants[24], 3.00000))
        + np.power(states[18], 3.00000) * constants[2]
        + np.power(constants[3], 3.00000) * algebraic[42]
    )
    algebraic[48] = (constants[21] * algebraic[43] * algebraic[44]) / (
        algebraic[45] * (algebraic[46] + algebraic[47])
    )
    algebraic[76] = (states[4] - states[12]) / 0.200000
    algebraic[79] = 1.00000 / (
        1.00000
        + (constants[49] * constants[47])
        / np.power(states[12] + constants[47], 2.00000)
        + (constants[46] * constants[48])
        / np.power(states[12] + constants[46], 2.00000)
    )
    rates[12] = algebraic[79] * (
        (
            -((algebraic[50] + algebraic[49]) - 2.00000 * algebraic[48])
            * constants[68]
        )
        / (constants[64] * 2.00000)
        + ((algebraic[73] - algebraic[71]) * constants[65]) / constants[64]
        + (algebraic[76] * constants[67]) / constants[64]
    )
    algebraic[58] = 1.00000 / (
        1.00000 + np.exp((algebraic[-7] * algebraic[30] + 13.0000) / 5.00000)
    )
    algebraic[61] = 3000.00 * algebraic[58]
    algebraic[64] = algebraic[61] * states[26] * states[27] * (states[24] - states[4])
    algebraic[80] = 1.00000 / (
        1.00000
        + (constants[52] * constants[53])
        / np.power(constants[53] + states[4], 2.00000)
        + (constants[54] * constants[55])
        / np.power(constants[55] + states[4], 2.00000)
    )
    rates[4] = algebraic[80] * (
        (-algebraic[31] * constants[68]) / (constants[67] * 2.00000)
        + (algebraic[64] * constants[66]) / constants[67]
        - algebraic[76]
    )
    algebraic[78] = 1.00000 / (
        1.00000
        + (constants[50] * constants[51])
        / np.power(states[24] + constants[50], 2.00000)
    )
    rates[24] = algebraic[78] * (algebraic[75] - algebraic[64])
    algebraic[51] = (
        (algebraic[31] + algebraic[50] + algebraic[49]) - 2.00000 * algebraic[48]
    )
    algebraic[59] = np.log(constants[3] / states[18]) / constants[5]
    algebraic[29] = (
        constants[57] * states[2] * states[2] * states[2] * states[1] * states[3]
    )
    algebraic[62] = algebraic[-4] * algebraic[29] * (states[0] - algebraic[59])
    algebraic[40] = 1.00000 / (
        1.00000
        + 0.124500 * np.exp(-0.100000 * states[0] * constants[5])
        + 0.0365000 * constants[59] * np.exp(-states[0] * constants[5])
    )
    algebraic[41] = (
        (
            (constants[19] * algebraic[40] * 1.00000)
            / (1.00000 + np.power(constants[17] / states[18], 2.00000))
        )
        * constants[1]
    ) / (constants[1] + constants[18])
    algebraic[65] = (
        algebraic[-3]
        * constants[33]
        * np.power(states[21], 3.00000)
        * states[22]
        * (states[0] - algebraic[59])
    )
    algebraic[68] = (
        algebraic[62]
        + 3.00000 * algebraic[48]
        + 3.00000 * algebraic[41]
        + algebraic[65]
    )
    algebraic[70] = np.log(constants[1] / states[23]) / constants[5]
    algebraic[72] = 1.02000 / (
        1.00000 + np.exp(0.238500 * ((states[0] - algebraic[70]) - 59.2150))
    )
    algebraic[74] = (
        0.491240 * np.exp(0.0803200 * ((states[0] - algebraic[70]) + 5.47600))
        + 1.00000 * np.exp(0.0617500 * ((states[0] - algebraic[70]) - 594.310))
    ) / (1.00000 + np.exp(-0.514300 * ((states[0] - algebraic[70]) + 4.75300)))
    algebraic[77] = (
        algebraic[-6]
        * (
            (0.500000 * np.power(constants[1] / 5.40000, 1.0 / 2) * algebraic[72])
            / (algebraic[72] + algebraic[74])
        )
        * (states[0] - algebraic[70])
    )
    algebraic[83] = (0.00276000 * (states[0] - algebraic[70])) / (
        1.00000 + np.exp((7.48800 - states[0]) / 5.98000)
    )
    algebraic[38] = 0.0248975 * (
        1.00000
        + 0.600000 / (1.00000 + np.power(3.80000e-05 / states[12], 1.40000))
    )
    algebraic[84] = (
        np.log(
            (constants[1] + constants[34] * constants[3])
            / (states[23] + constants[34] * states[18])
        )
        / constants[5]
    )
    algebraic[85] = (
        algebraic[-5]
        * algebraic[38]
        * states[13]
        * states[14]
        * (states[0] - algebraic[84])
    )
    algebraic[37] = 1.00000 / (
        1.00000 + np.exp((states[0] + 10.0000) / 15.4000)
    )
    algebraic[81] = (
        algebraic[-1]
        * constants[58]
        * states[11]
        * algebraic[37]
        * (states[0] - algebraic[70])
    )
    algebraic[39] = np.exp(states[0] / 300.000)
    algebraic[82] = (
        algebraic[-2]
        * constants[16]
        * np.power(states[15], 3.00000)
        * states[16]
        * states[17]
        * algebraic[39]
        * (states[0] - algebraic[70])
    )
    algebraic[0] = np.floor(voi / constants[10]) * constants[10]
    algebraic[16] = (
        constants[12]
        if (
            voi - algebraic[0] >= constants[9]
            and voi - algebraic[0] <= constants[9] + constants[11]
        )
        else 0.00000
    )
    algebraic[86] = (
        (algebraic[81] + algebraic[85] + algebraic[77] + algebraic[83])
        - 2.00000 * algebraic[41]
    ) + algebraic[82] + 0.500000 * algebraic[16]
    algebraic[53] = (
        constants[30]
        * states[0]
        * constants[0]
        * constants[5]
        * (states[19] - constants[4] * np.exp(states[0] * constants[5]))
    ) / (1.00000 - np.exp(states[0] * constants[5]))
    algebraic[55] = algebraic[53] * states[20]
    algebraic[87] = -np.log(constants[4] / states[19]) / constants[5]
    algebraic[88] = constants[32] * (states[0] - algebraic[87])
    algebraic[89] = algebraic[88] + algebraic[55] + 0.500000 * algebraic[16]
    rates[0] = -(algebraic[68] + algebraic[86] + algebraic[51] + algebraic[89])
    algebraic[90] = (
        constants[44] * np.power(algebraic[59] - algebraic[87], 4.00000)
    ) / (
        np.power(algebraic[59] - algebraic[87], 4.00000)
        + np.power(87.8251, 4.00000)
    )
    rates[18] = (-algebraic[68] * constants[68]) / constants[64] + algebraic[90]
    algebraic[91] = (constants[45] * (algebraic[70] - algebraic[87])) / (
        (algebraic[70] - algebraic[87]) + 87.8251
    )
    rates[23] = (-algebraic[86] * constants[68]) / constants[64] + algebraic[91]
    rates[19] = (
        (algebraic[89] * constants[68]) / constants[64]
        + algebraic[90]
        + algebraic[91]
    )
    return rates
