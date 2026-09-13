import numpy as np
import matplotlib.pyplot as plt
import time
from cell_models.base import CardiacCellModel
from numba import njit


class PigModel(CardiacCellModel):
    def __init__(
        self,
        drug=None,
        drug_amount=None,
        num_cycles_feasible_ap=None,
    ):
        super().__init__(
            sizeStates=29 + 1,
            sizeAlgebraic=193 + 6,
            sizeConstants=119,
            species="Pig",
            state_idx_dict={"v": 12, "cai": 4, "cai2": 5, "nai": 11, "ki": 10},
            rate_idx_dict={"dvdt": 12},
            algebraic_idx_dict={"cai_avg": 13},
            channel_list=["ICaL", "IK1", "IKs", "INa", "INaL", "hERG"],
            num_cycles_feasible_ap=num_cycles_feasible_ap,
            ode_hyperparameters={"max_step": 0.1},
            num_cycles_limit_state=2000,
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
        constants[0] = 96485.0
        constants[10] = 8314.0
        constants[11] = 310.0
        constants[73] = 3.14
        constants[77] = 20.0
        constants[78] = 1000.0
        constants[75] = 0.5
        constants[35] = 1
        constants[20] = 2
        constants[56] = 1

        # Initialise states
        states[0] = 100.0
        states[1] = 9.71e-22
        states[2] = 8.59e-10
        states[3] = 1.3
        states[4] = 6.75e-05
        states[5] = 6.81e-05
        states[6] = 1.31
        states[7] = 6.75e-05
        states[8] = 100.0
        states[9] = 8.23910999999999914e-03
        states[10] = 140.76
        states[11] = 6.43
        states[12] = -87.3
        states[13] = 4.42e-07
        states[14] = 0.988
        states[15] = 0.0
        states[16] = 0.0
        states[17] = 140.76
        states[18] = 6.43
        states[19] = 0.153
        states[20] = 0.054
        states[21] = 0.046
        states[22] = 0.301
        states[23] = 0.001
        states[24] = 0.995
        states[25] = 0.631
        states[26] = 0.631
        states[27] = 0.0022
        states[28] = 1.37

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
        constants[79] = ((constants[10] * constants[11]) / constants[0]) * np.log(
            constants[14] / constants[15]
        )
        constants[80] = 0.000357400 * constants[18]
        constants[81] = 0.00125000 * constants[18]
        constants[90] = (constants[24] + 1.00000) + (constants[17] / constants[27]) * (
            1.00000 + constants[17] / constants[28]
        )
        constants[91] = (constants[17] * constants[17]) / (
            (constants[90] * constants[27]) * constants[28]
        )
        constants[92] = 1.00000 / constants[90]
        constants[93] = constants[25]
        constants[94] = constants[25]
        constants[95] = (constants[24] + 1.00000) + (constants[17] / constants[27]) * (
            1.00000 + constants[17] / constants[28]
        )
        constants[96] = (constants[17] * constants[17]) / (
            (constants[95] * constants[27]) * constants[28]
        )
        constants[97] = 1.00000 / constants[95]
        constants[98] = constants[25]
        constants[99] = constants[25]
        constants[101] = constants[51]
        constants[102] = ((constants[55] * constants[46]) / constants[40]) / (
            1.00000 + constants[46] / constants[40]
        )
        constants[103] = constants[48] * constants[45]
        constants[104] = ((2.00000 * constants[73]) * constants[74]) * constants[74] + (
            (2.00000 * constants[73]) * constants[74]
        ) * constants[72]
        constants[105] = 2.00000 * constants[104]
        constants[107] = (constants[92] * constants[106]) * constants[26]
        constants[108] = (constants[97] * constants[106]) * constants[26]
        constants[109] = (
            ((1000.00 * constants[73]) * constants[74]) * constants[74]
        ) * constants[72]
        constants[110] = (0.00480000 * constants[109]) / 2.00000
        constants[111] = constants[110]
        constants[112] = 0.680000 * constants[109]
        constants[113] = constants[112] * constants[75]
        constants[114] = constants[112] - constants[113]
        constants[115] = 0.0552000 * constants[109]
        constants[116] = constants[115] / 2.00000
        constants[117] = constants[115] / 2.00000
        constants[118] = (0.0200000 * constants[109]) / 2.00000

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
        legend_states[0] = "A in component CICR (dimensionless)"
        legend_constants[105] = "Acap in component cell (dimensionless)"
        legend_algebraic[38] = "CaMKa in component CaMK (dimensionless)"
        legend_constants[0] = "F in component cell (dimensionless)"
        legend_algebraic[57] = "ICaL in component ICaL (dimensionless)"
        legend_algebraic[131] = "INaCa_ss in component INaCa_ss (dimensionless)"
        legend_algebraic[180] = "Jdiff in component diffusion (dimensionless)"
        legend_algebraic[0] = "Jgap in component CICR (dimensionless)"
        legend_algebraic[31] = "Jrel in component CICR (dimensionless)"
        legend_states[1] = "Jrel1 in component CICR (dimensionless)"
        legend_algebraic[188] = "Jrel1_inf in component CICR (dimensionless)"
        legend_states[2] = "Jrel2 in component CICR (dimensionless)"
        legend_algebraic[189] = "Jrel2_inf in component CICR (dimensionless)"
        legend_algebraic[24] = "Jrelol in component CICR (dimensionless)"
        legend_algebraic[181] = "Jup2 in component SR_uptake (dimensionless)"
        legend_constants[1] = "KmCaMK in component CaMK (dimensionless)"
        legend_algebraic[182] = "Rel1 in component CICR (dimensionless)"
        legend_algebraic[183] = "Rel2 in component CICR (dimensionless)"
        legend_constants[2] = "SOICR in component CICR (dimensionless)"
        legend_states[3] = "cacsr in component ionic_concentrations (dimensionless)"
        legend_states[4] = "cai in component ionic_concentrations (dimensionless)"
        legend_states[5] = "cai2 in component ionic_concentrations (dimensionless)"
        legend_states[6] = "cajsr in component ionic_concentrations (dimensionless)"
        legend_states[7] = "cass in component ionic_concentrations (dimensionless)"
        legend_algebraic[1] = "diff_A in component CICR (dimensionless)"
        legend_algebraic[2] = "diff_tjsrol in component CICR (dimensionless)"
        legend_constants[3] = "grelbarjsrol in component CICR (dimensionless)"
        legend_algebraic[14] = "greljsrol in component CICR (dimensionless)"
        legend_voi = "t in component stimulus (dimensionless)"
        legend_algebraic[44] = "tau_Jrel1 in component CICR (dimensionless)"
        legend_algebraic[46] = "tau_Jrel2 in component CICR (dimensionless)"
        legend_constants[4] = "tau_gap in component CICR (dimensionless)"
        legend_constants[5] = "tauoff in component CICR (dimensionless)"
        legend_constants[6] = "tauon in component CICR (dimensionless)"
        legend_states[8] = "tjsrol in component CICR (dimensionless)"
        legend_algebraic[42] = "trel1factor in component CICR (dimensionless)"
        legend_algebraic[45] = "trel2factor in component CICR (dimensionless)"
        legend_constants[111] = "vcsr in component cell (dimensionless)"
        legend_constants[110] = "vjsr in component cell (dimensionless)"
        legend_constants[114] = "vmyo2 in component cell (dimensionless)"
        legend_constants[117] = "vnsr2 in component cell (dimensionless)"
        legend_constants[118] = "vss in component cell (dimensionless)"
        legend_algebraic[47] = "CaMK_f in component CaMK (dimensionless)"
        legend_algebraic[36] = "CaMKb in component CaMK (dimensionless)"
        legend_constants[7] = "CaMKo in component CaMK (dimensionless)"
        legend_states[9] = "CaMKt in component CaMK (dimensionless)"
        legend_constants[79] = "ECl in component CaMK (dimensionless)"
        legend_algebraic[48] = "EK in component CaMK (dimensionless)"
        legend_algebraic[49] = "EKs in component CaMK (dimensionless)"
        legend_algebraic[50] = "ENa in component CaMK (dimensionless)"
        legend_constants[8] = "KmCaM in component CaMK (dimensionless)"
        legend_constants[9] = "PKNa in component CaMK (dimensionless)"
        legend_constants[10] = "R in component cell (dimensionless)"
        legend_constants[11] = "T in component cell (dimensionless)"
        legend_constants[12] = "aCaMK in component CaMK (dimensionless)"
        legend_constants[13] = "bCaMK in component CaMK (dimensionless)"
        legend_constants[14] = "cli in component cell (dimensionless)"
        legend_constants[15] = "clo in component cell (dimensionless)"
        legend_algebraic[39] = "diff_CaMKt in component CaMK (dimensionless)"
        legend_states[10] = "ki in component ionic_concentrations (dimensionless)"
        legend_constants[16] = "ko in component cell (dimensionless)"
        legend_states[11] = "nai in component ionic_concentrations (dimensionless)"
        legend_constants[17] = "nao in component cell (dimensionless)"
        legend_states[12] = "v in component cell (dimensionless)"
        legend_algebraic[3] = "vffrt in component CaMK (dimensionless)"
        legend_algebraic[51] = "vfrt in component CaMK (dimensionless)"
        legend_algebraic[56] = "ICaK in component ICaL (dimensionless)"
        legend_algebraic[59] = "ICaNa in component ICaL (dimensionless)"
        legend_constants[18] = "PCa in component ICaL (dimensionless)"
        legend_constants[80] = "PCaK in component ICaL (dimensionless)"
        legend_constants[81] = "PCaNa in component ICaL (dimensionless)"
        legend_algebraic[52] = "PhiCaK in component ICaL (dimensionless)"
        legend_algebraic[53] = "PhiCaL in component ICaL (dimensionless)"
        legend_algebraic[54] = "PhiCaNa in component ICaL (dimensionless)"
        legend_constants[106] = "cao in component cell (dimensionless)"
        legend_states[13] = "d in component ICaL (dimensionless)"
        legend_algebraic[4] = "d_inf in component ICaL (dimensionless)"
        legend_algebraic[55] = "f in component ICaL (dimensionless)"
        legend_algebraic[5] = "f_inf in component ICaL (dimensionless)"
        legend_states[14] = "fca in component ICaL (dimensionless)"
        legend_algebraic[58] = "fca_inf in component ICaL (dimensionless)"
        legend_states[15] = "ff in component ICaL (dimensionless)"
        legend_algebraic[15] = "ff_inf in component ICaL (dimensionless)"
        legend_states[16] = "fs in component ICaL (dimensionless)"
        legend_algebraic[16] = "fs_inf in component ICaL (dimensionless)"
        legend_states[17] = "kss in component ionic_concentrations (dimensionless)"
        legend_states[18] = "nass in component ionic_concentrations (dimensionless)"
        legend_algebraic[17] = "tau_d in component ICaL (dimensionless)"
        legend_algebraic[60] = "tau_fca in component ICaL (dimensionless)"
        legend_algebraic[25] = "tau_ff in component ICaL (dimensionless)"
        legend_algebraic[26] = "tau_fs in component ICaL (dimensionless)"
        legend_constants[19] = "vhalf_d in component ICaL (dimensionless)"
        legend_constants[82] = "vhalff in component ICaL (dimensionless)"
        legend_constants[20] = "zca in component ICaL (dimensionless)"
        legend_algebraic[61] = "ICab in component ICab (dimensionless)"
        legend_constants[83] = "PCab in component ICab (dimensionless)"
        legend_constants[84] = "GK1 in component IK1 (dimensionless)"
        legend_algebraic[63] = "IK1 in component IK1 (dimensionless)"
        legend_algebraic[62] = "rk1 in component IK1 (dimensionless)"
        legend_constants[21] = "GKb in component IKb (dimensionless)"
        legend_algebraic[65] = "IKb in component IKb (dimensionless)"
        legend_algebraic[64] = "xkb in component IKb (dimensionless)"
        legend_constants[85] = "GKr in component IKr (dimensionless)"
        legend_algebraic[67] = "IKr in component IKr (dimensionless)"
        legend_algebraic[66] = "rkr in component IKr (dimensionless)"
        legend_algebraic[6] = "tau_xr in component IKr (dimensionless)"
        legend_states[19] = "xr in component IKr (dimensionless)"
        legend_algebraic[18] = "xr_inf in component IKr (dimensionless)"
        legend_constants[22] = "GKs in component IKs (dimensionless)"
        legend_algebraic[69] = "IKs in component IKs (dimensionless)"
        legend_algebraic[68] = "KsCa in component IKs (dimensionless)"
        legend_algebraic[7] = "tau_xs1 in component IKs (dimensionless)"
        legend_algebraic[19] = "tau_xs2 in component IKs (dimensionless)"
        legend_states[20] = "xs1 in component IKs (dimensionless)"
        legend_algebraic[27] = "xs1_inf in component IKs (dimensionless)"
        legend_states[21] = "xs2 in component IKs (dimensionless)"
        legend_algebraic[32] = "xs2_inf in component IKs (dimensionless)"
        legend_algebraic[95] = "E1 in component INaCa_i (dimensionless)"
        legend_algebraic[96] = "E2 in component INaCa_i (dimensionless)"
        legend_algebraic[97] = "E3 in component INaCa_i (dimensionless)"
        legend_algebraic[98] = "E4 in component INaCa_i (dimensionless)"
        legend_constants[87] = "Gncx in component INaCa_i (dimensionless)"
        legend_algebraic[101] = "INaCa_i in component INaCa_i (dimensionless)"
        legend_algebraic[99] = "JncxCa in component INaCa_i (dimensionless)"
        legend_algebraic[100] = "JncxNa in component INaCa_i (dimensionless)"
        legend_constants[23] = "KmCaAct in component INaCa_i (dimensionless)"
        legend_algebraic[70] = "allo in component INaCa_i (dimensionless)"
        legend_algebraic[76] = "h1 in component INaCa_i (dimensionless)"
        legend_constants[90] = "h10 in component INaCa_i (dimensionless)"
        legend_constants[91] = "h11 in component INaCa_i (dimensionless)"
        legend_constants[92] = "h12 in component INaCa_i (dimensionless)"
        legend_algebraic[77] = "h2 in component INaCa_i (dimensionless)"
        legend_algebraic[78] = "h3 in component INaCa_i (dimensionless)"
        legend_algebraic[71] = "h4 in component INaCa_i (dimensionless)"
        legend_algebraic[72] = "h5 in component INaCa_i (dimensionless)"
        legend_algebraic[73] = "h6 in component INaCa_i (dimensionless)"
        legend_algebraic[79] = "h7 in component INaCa_i (dimensionless)"
        legend_algebraic[80] = "h8 in component INaCa_i (dimensionless)"
        legend_algebraic[81] = "h9 in component INaCa_i (dimensionless)"
        legend_algebraic[74] = "hca in component INaCa_i (dimensionless)"
        legend_algebraic[75] = "hna in component INaCa_i (dimensionless)"
        legend_constants[107] = "k1 in component INaCa_i (dimensionless)"
        legend_constants[93] = "k2 in component INaCa_i (dimensionless)"
        legend_algebraic[84] = "k3 in component INaCa_i (dimensionless)"
        legend_algebraic[82] = "k3p in component INaCa_i (dimensionless)"
        legend_algebraic[83] = "k3pp in component INaCa_i (dimensionless)"
        legend_algebraic[87] = "k4 in component INaCa_i (dimensionless)"
        legend_algebraic[85] = "k4p in component INaCa_i (dimensionless)"
        legend_algebraic[86] = "k4pp in component INaCa_i (dimensionless)"
        legend_constants[94] = "k5 in component INaCa_i (dimensionless)"
        legend_algebraic[88] = "k6 in component INaCa_i (dimensionless)"
        legend_algebraic[89] = "k7 in component INaCa_i (dimensionless)"
        legend_algebraic[90] = "k8 in component INaCa_i (dimensionless)"
        legend_constants[24] = "kasymm in component INaCa_i (dimensionless)"
        legend_constants[25] = "kcaoff in component INaCa_i (dimensionless)"
        legend_constants[26] = "kcaon in component INaCa_i (dimensionless)"
        legend_constants[27] = "kna1 in component INaCa_i (dimensionless)"
        legend_constants[28] = "kna2 in component INaCa_i (dimensionless)"
        legend_constants[29] = "kna3 in component INaCa_i (dimensionless)"
        legend_constants[30] = "qca in component INaCa_i (dimensionless)"
        legend_constants[31] = "qna in component INaCa_i (dimensionless)"
        legend_constants[32] = "wca in component INaCa_i (dimensionless)"
        legend_constants[33] = "wna in component INaCa_i (dimensionless)"
        legend_constants[34] = "wnaca in component INaCa_i (dimensionless)"
        legend_algebraic[91] = "x1 in component INaCa_i (dimensionless)"
        legend_algebraic[92] = "x2 in component INaCa_i (dimensionless)"
        legend_algebraic[93] = "x3 in component INaCa_i (dimensionless)"
        legend_algebraic[94] = "x4 in component INaCa_i (dimensionless)"
        legend_constants[35] = "zna in component INaCa_i (dimensionless)"
        legend_algebraic[125] = "E11 in component INaCa_ss (dimensionless)"
        legend_algebraic[126] = "E21 in component INaCa_ss (dimensionless)"
        legend_algebraic[127] = "E31 in component INaCa_ss (dimensionless)"
        legend_algebraic[128] = "E41 in component INaCa_ss (dimensionless)"
        legend_algebraic[132] = "INaCa in component INaCa_ss (dimensionless)"
        legend_algebraic[129] = "JncxCa1 in component INaCa_ss (dimensionless)"
        legend_algebraic[130] = "JncxNa1 in component INaCa_ss (dimensionless)"
        legend_algebraic[102] = "allo1 in component INaCa_ss (dimensionless)"
        legend_constants[95] = "h101 in component INaCa_ss (dimensionless)"
        legend_algebraic[103] = "h111 in component INaCa_ss (dimensionless)"
        legend_constants[96] = "h1111 in component INaCa_ss (dimensionless)"
        legend_constants[97] = "h121 in component INaCa_ss (dimensionless)"
        legend_algebraic[104] = "h21 in component INaCa_ss (dimensionless)"
        legend_algebraic[105] = "h31 in component INaCa_ss (dimensionless)"
        legend_algebraic[106] = "h41 in component INaCa_ss (dimensionless)"
        legend_algebraic[107] = "h51 in component INaCa_ss (dimensionless)"
        legend_algebraic[108] = "h61 in component INaCa_ss (dimensionless)"
        legend_algebraic[109] = "h71 in component INaCa_ss (dimensionless)"
        legend_algebraic[110] = "h81 in component INaCa_ss (dimensionless)"
        legend_algebraic[111] = "h91 in component INaCa_ss (dimensionless)"
        legend_constants[108] = "k11 in component INaCa_ss (dimensionless)"
        legend_constants[98] = "k21 in component INaCa_ss (dimensionless)"
        legend_algebraic[114] = "k31 in component INaCa_ss (dimensionless)"
        legend_algebraic[112] = "k3p1 in component INaCa_ss (dimensionless)"
        legend_algebraic[113] = "k3pp1 in component INaCa_ss (dimensionless)"
        legend_algebraic[117] = "k41 in component INaCa_ss (dimensionless)"
        legend_algebraic[115] = "k4p1 in component INaCa_ss (dimensionless)"
        legend_algebraic[116] = "k4pp1 in component INaCa_ss (dimensionless)"
        legend_constants[99] = "k51 in component INaCa_ss (dimensionless)"
        legend_algebraic[118] = "k61 in component INaCa_ss (dimensionless)"
        legend_algebraic[119] = "k71 in component INaCa_ss (dimensionless)"
        legend_algebraic[120] = "k81 in component INaCa_ss (dimensionless)"
        legend_algebraic[121] = "x11 in component INaCa_ss (dimensionless)"
        legend_algebraic[122] = "x21 in component INaCa_ss (dimensionless)"
        legend_algebraic[123] = "x31 in component INaCa_ss (dimensionless)"
        legend_algebraic[124] = "x41 in component INaCa_ss (dimensionless)"
        legend_algebraic[145] = "E12 in component INaK (dimensionless)"
        legend_algebraic[146] = "E22 in component INaK (dimensionless)"
        legend_algebraic[147] = "E32 in component INaK (dimensionless)"
        legend_algebraic[148] = "E42 in component INaK (dimensionless)"
        legend_constants[36] = "H in component INaK (dimensionless)"
        legend_algebraic[151] = "INaK in component INaK (dimensionless)"
        legend_algebraic[149] = "JnakK in component INaK (dimensionless)"
        legend_algebraic[150] = "JnakNa in component INaK (dimensionless)"
        legend_constants[37] = "Khp in component INaK (dimensionless)"
        legend_constants[38] = "Kki in component INaK (dimensionless)"
        legend_constants[39] = "Kko in component INaK (dimensionless)"
        legend_constants[40] = "Kmgatp in component INaK (dimensionless)"
        legend_algebraic[133] = "Knai in component INaK (dimensionless)"
        legend_constants[41] = "Knai0 in component INaK (dimensionless)"
        legend_algebraic[134] = "Knao in component INaK (dimensionless)"
        legend_constants[42] = "Knao0 in component INaK (dimensionless)"
        legend_constants[43] = "Knap in component INaK (dimensionless)"
        legend_constants[44] = "Kxkur in component INaK (dimensionless)"
        legend_constants[45] = "MgADP in component INaK (dimensionless)"
        legend_constants[46] = "MgATP in component INaK (dimensionless)"
        legend_algebraic[135] = "P in component INaK (dimensionless)"
        legend_constants[100] = "Pnak in component INaK (dimensionless)"
        legend_algebraic[136] = "a1 in component INaK (dimensionless)"
        legend_constants[101] = "a2 in component INaK (dimensionless)"
        legend_algebraic[137] = "a3 in component INaK (dimensionless)"
        legend_constants[102] = "a4 in component INaK (dimensionless)"
        legend_constants[103] = "b1 in component INaK (dimensionless)"
        legend_algebraic[138] = "b2 in component INaK (dimensionless)"
        legend_algebraic[139] = "b3 in component INaK (dimensionless)"
        legend_algebraic[140] = "b4 in component INaK (dimensionless)"
        legend_constants[86] = "delta in component INaK (dimensionless)"
        legend_constants[47] = "eP in component INaK (dimensionless)"
        legend_constants[48] = "k1m in component INaK (dimensionless)"
        legend_constants[49] = "k1p in component INaK (dimensionless)"
        legend_constants[50] = "k2m in component INaK (dimensionless)"
        legend_constants[51] = "k2p in component INaK (dimensionless)"
        legend_constants[52] = "k3m in component INaK (dimensionless)"
        legend_constants[53] = "k3p2 in component INaK (dimensionless)"
        legend_constants[54] = "k4m in component INaK (dimensionless)"
        legend_constants[55] = "k4p2 in component INaK (dimensionless)"
        legend_algebraic[141] = "x12 in component INaK (dimensionless)"
        legend_algebraic[142] = "x22 in component INaK (dimensionless)"
        legend_algebraic[143] = "x32 in component INaK (dimensionless)"
        legend_algebraic[144] = "x42 in component INaK (dimensionless)"
        legend_constants[56] = "zk in component INaK (dimensionless)"
        legend_constants[57] = "GNaL in component INaL (dimensionless)"
        legend_algebraic[152] = "INaL in component INaL (dimensionless)"
        legend_algebraic[9] = "aml in component INaL (dimensionless)"
        legend_algebraic[20] = "bml in component INaL (dimensionless)"
        legend_states[22] = "hl in component INaL (dimensionless)"
        legend_algebraic[8] = "hl_inf in component INaL (dimensionless)"
        legend_states[23] = "ml in component INaL (dimensionless)"
        legend_algebraic[28] = "ml_inf in component INaL (dimensionless)"
        legend_constants[58] = "tau_hl in component INaL (dimensionless)"
        legend_algebraic[33] = "tau_ml in component INaL (dimensionless)"
        legend_algebraic[154] = "INab in component INab (dimensionless)"
        legend_constants[59] = "PNab in component INab (dimensionless)"
        legend_constants[88] = "Gto in component ITo (dimensionless)"
        legend_algebraic[159] = "ITo in component ITo (dimensionless)"
        legend_states[24] = "aa in component ITo (dimensionless)"
        legend_algebraic[10] = "alpha_aa in component ITo (dimensionless)"
        legend_algebraic[21] = "beta_aa in component ITo (dimensionless)"
        legend_algebraic[157] = "kito2 in component ITo (dimensionless)"
        legend_algebraic[158] = "rito2 in component ITo (dimensionless)"
        legend_constants[60] = "GNa in component I_Na (dimensionless)"
        legend_algebraic[160] = "INa in component I_Na (dimensionless)"
        legend_algebraic[11] = "aa_h in component I_Na (dimensionless)"
        legend_algebraic[22] = "aa_j in component I_Na (dimensionless)"
        legend_algebraic[12] = "aa_m in component I_Na (dimensionless)"
        legend_algebraic[29] = "bb_h in component I_Na (dimensionless)"
        legend_algebraic[34] = "bb_j in component I_Na (dimensionless)"
        legend_algebraic[23] = "bb_m in component I_Na (dimensionless)"
        legend_states[25] = "h in component I_Na (dimensionless)"
        legend_algebraic[37] = "h_inf in component I_Na (dimensionless)"
        legend_states[26] = "j in component I_Na (dimensionless)"
        legend_algebraic[40] = "j_inf in component I_Na (dimensionless)"
        legend_states[27] = "m in component I_Na (dimensionless)"
        legend_algebraic[30] = "m_inf in component I_Na (dimensionless)"
        legend_algebraic[41] = "tau_h in component I_Na (dimensionless)"
        legend_algebraic[43] = "tau_j in component I_Na (dimensionless)"
        legend_algebraic[35] = "tau_m in component I_Na (dimensionless)"
        legend_constants[61] = "GpCa in component IpCa (dimensionless)"
        legend_algebraic[163] = "Iion in component IpCa (dimensionless)"
        legend_algebraic[161] = "IpCa in component IpCa (dimensionless)"
        legend_constants[62] = "BSLmax in component SR_uptake (dimensionless)"
        legend_constants[63] = "BSRmax in component SR_uptake (dimensionless)"
        legend_algebraic[164] = "Jleak in component SR_uptake (dimensionless)"
        legend_algebraic[167] = "Jtr in component SR_uptake (dimensionless)"
        legend_algebraic[169] = "Jtr2 in component SR_uptake (dimensionless)"
        legend_algebraic[179] = "Jup in component SR_uptake (dimensionless)"
        legend_algebraic[171] = "Jupnp in component SR_uptake (dimensionless)"
        legend_algebraic[174] = "Jupnp2 in component SR_uptake (dimensionless)"
        legend_algebraic[176] = "Jupp in component SR_uptake (dimensionless)"
        legend_algebraic[177] = "Jupp2 in component SR_uptake (dimensionless)"
        legend_constants[64] = "KmBSL in component SR_uptake (dimensionless)"
        legend_constants[65] = "KmBSR in component SR_uptake (dimensionless)"
        legend_states[28] = "cansr in component ionic_concentrations (dimensionless)"
        legend_constants[66] = "cmdnmax in component SR_uptake (dimensionless)"
        legend_constants[67] = "csqnmax in component SR_uptake (dimensionless)"
        legend_algebraic[178] = "fJupp in component SR_uptake (dimensionless)"
        legend_constants[68] = "kmcmdn in component SR_uptake (dimensionless)"
        legend_constants[69] = "kmcsqn in component SR_uptake (dimensionless)"
        legend_constants[70] = "kmtrpn in component SR_uptake (dimensionless)"
        legend_constants[71] = "trpnmax in component SR_uptake (dimensionless)"
        legend_constants[104] = "Ageo in component cell (dimensionless)"
        legend_algebraic[168] = "I_stim in component stimulus (dimensionless)"
        legend_constants[72] = "L in component cell (dimensionless)"
        legend_constants[73] = "pi in component cell (dimensionless)"
        legend_constants[74] = "rad in component cell (dimensionless)"
        legend_constants[109] = "vcell in component cell (dimensionless)"
        legend_constants[112] = "vmyo in component cell (dimensionless)"
        legend_constants[113] = "vmyo1 in component cell (dimensionless)"
        legend_constants[75] = "vmyo1frac in component cell (dimensionless)"
        legend_constants[115] = "vnsr in component cell (dimensionless)"
        legend_constants[116] = "vnsr1 in component cell (dimensionless)"
        legend_algebraic[153] = "JdiffK in component diffusion (dimensionless)"
        legend_algebraic[162] = "JdiffNa in component diffusion (dimensionless)"
        legend_algebraic[172] = (
            "Bcacsr in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[184] = "Bcai in component ionic_concentrations (dimensionless)"
        legend_algebraic[185] = (
            "Bcai2 in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[170] = (
            "Bcajsr in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[187] = (
            "Bcass in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[13] = (
            "cai_avg in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[175] = (
            "diff_cacsr in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[190] = (
            "diff_cai in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[191] = (
            "diff_cai2 in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[173] = (
            "diff_cajsr in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[186] = (
            "diff_cansr in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[192] = (
            "diff_cass in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[155] = (
            "diff_ki in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[156] = (
            "diff_kss in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[165] = (
            "diff_nai in component ionic_concentrations (dimensionless)"
        )
        legend_algebraic[166] = (
            "diff_nass in component ionic_concentrations (dimensionless)"
        )
        legend_constants[76] = "duration in component stimulus (dimensionless)"
        legend_constants[77] = "offset in component stimulus (dimensionless)"
        legend_constants[78] = "period in component stimulus (dimensionless)"
        legend_constants[89] = (
            "stimulus_amplitude in component stimulus (dimensionless)"
        )
        legend_rates[0] = "d/dt A in component CICR (dimensionless)"
        legend_rates[1] = "d/dt Jrel1 in component CICR (dimensionless)"
        legend_rates[2] = "d/dt Jrel2 in component CICR (dimensionless)"
        legend_rates[8] = "d/dt tjsrol in component CICR (dimensionless)"
        legend_rates[9] = "d/dt CaMKt in component CaMK (dimensionless)"
        legend_rates[13] = "d/dt d in component ICaL (dimensionless)"
        legend_rates[14] = "d/dt fca in component ICaL (dimensionless)"
        legend_rates[15] = "d/dt ff in component ICaL (dimensionless)"
        legend_rates[16] = "d/dt fs in component ICaL (dimensionless)"
        legend_rates[19] = "d/dt xr in component IKr (dimensionless)"
        legend_rates[20] = "d/dt xs1 in component IKs (dimensionless)"
        legend_rates[21] = "d/dt xs2 in component IKs (dimensionless)"
        legend_rates[22] = "d/dt hl in component INaL (dimensionless)"
        legend_rates[23] = "d/dt ml in component INaL (dimensionless)"
        legend_rates[24] = "d/dt aa in component ITo (dimensionless)"
        legend_rates[25] = "d/dt h in component I_Na (dimensionless)"
        legend_rates[26] = "d/dt j in component I_Na (dimensionless)"
        legend_rates[27] = "d/dt m in component I_Na (dimensionless)"
        legend_rates[12] = "d/dt v in component cell (dimensionless)"
        legend_rates[3] = "d/dt cacsr in component ionic_concentrations (dimensionless)"
        legend_rates[4] = "d/dt cai in component ionic_concentrations (dimensionless)"
        legend_rates[5] = "d/dt cai2 in component ionic_concentrations (dimensionless)"
        legend_rates[6] = "d/dt cajsr in component ionic_concentrations (dimensionless)"
        legend_rates[28] = (
            "d/dt cansr in component ionic_concentrations (dimensionless)"
        )
        legend_rates[7] = "d/dt cass in component ionic_concentrations (dimensionless)"
        legend_rates[10] = "d/dt ki in component ionic_concentrations (dimensionless)"
        legend_rates[17] = "d/dt kss in component ionic_concentrations (dimensionless)"
        legend_rates[11] = "d/dt nai in component ionic_concentrations (dimensionless)"
        legend_rates[18] = "d/dt nass in component ionic_concentrations (dimensionless)"
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
        algebraic[1] = super().custom_piecewise(
            [np.less(states[8], 5.00000), -states[0] / 0.100000, True, 1.00000]
        )
        algebraic[2] = super().custom_piecewise(
            [
                np.greater(states[6], constants[2]) & np.greater(states[0], 45.0000),
                -states[8] / 0.00100000,
                True,
                1.00000,
            ]
        )
        algebraic[8] = 1.00000 / (1.00000 + np.exp((states[12] + 91.0000) / 6.10000))
        algebraic[4] = 1.00000 / (
            1.00000 + np.exp(-(states[12] - constants[19]) / 6.20000)
        )
        algebraic[17] = 0.600000 + 1.00000 / (
            np.exp(-0.0500000 * (states[12] + 6.00000))
            + np.exp(0.0900000 * (states[12] + 14.0000))
        )
        algebraic[6] = 12.9800 + 1.00000 / (
            0.365200 * np.exp((states[12] - 31.6600) / 3.86900)
            + 4.12300e-05 * np.exp(-(states[12] - 47.7800) / 20.3800)
        )
        algebraic[18] = 1.00000 / (1.00000 + np.exp(-(states[12] + 56.8000) / 17.8000))
        algebraic[10] = 0.0250000 / (1.00000 + np.exp((states[12] + 58.0000) / 5.00000))
        algebraic[21] = 1.00000 / (
            5.00000 * (1.00000 + np.exp((states[12] + 19.0000) / -9.00000))
        )
        algebraic[5] = 1.00000 / (
            1.00000 + np.exp((states[12] - constants[82]) / 4.90000)
        ) + 0.350000 / (1.00000 + np.exp((45.0000 - states[12]) / 20.0000))
        algebraic[15] = algebraic[5]
        algebraic[25] = 7.00000 + 1.00000 / (
            0.00450000 * np.exp(-(states[12] + 20.0000) / 10.0000)
            + 0.00450000 * np.exp((states[12] + 20.0000) / 10.0000)
        )
        algebraic[16] = algebraic[5]
        algebraic[26] = 70.0000 + 1.00000 / (
            3.50000e-05 * np.exp(-(states[12] + 5.00000) / 4.00000)
            + 3.50000e-05 * np.exp((states[12] + 5.00000) / 6.00000)
        )
        algebraic[7] = 300.000 + 1.00000 / (
            1.00000e-06 * np.exp((states[12] + 50.0000) / 20.0000)
            + 0.0400000 * np.exp(-(states[12] + 50.0000) / 20.0000)
        )
        algebraic[27] = 1.00000 / (1.00000 + np.exp(-(states[12] - 25.1000) / 37.1000))
        algebraic[19] = 1.00000 / (
            0.0100000 * np.exp((states[12] - 50.0000) / 20.0000)
            + 0.0193000 * np.exp(-(states[12] + 66.5400) / 31.0000)
        )
        algebraic[32] = algebraic[27]
        algebraic[9] = 0.320000 * (
            super().custom_piecewise(
                [
                    np.equal(states[12], -47.1300),
                    10.0000,
                    True,
                    -(states[12] + 47.1300)
                    / (np.exp(-0.100000 * (states[12] + 47.1300)) - 1.00000),
                ]
            )
        )
        algebraic[20] = 0.0800000 * np.exp(-states[12] / 11.0000)
        algebraic[28] = algebraic[9] / (algebraic[9] + algebraic[20])
        algebraic[33] = 1.00000 / (algebraic[9] + algebraic[20])
        algebraic[30] = 1.00000 / (
            (1.00000 + np.exp((-56.8600 - states[12]) / 9.03000))
            * (1.00000 + np.exp((-56.8600 - states[12]) / 9.03000))
        )
        algebraic[12] = 1.00000 / (1.00000 + np.exp((-60.0000 - states[12]) / 5.00000))
        algebraic[23] = 0.100000 / (
            1.00000 + np.exp((states[12] + 35.0000) / 5.00000)
        ) + 0.100000 / (1.00000 + np.exp((states[12] - 50.0000) / 200.000))
        algebraic[35] = algebraic[12] * algebraic[23]
        algebraic[36] = (constants[7] * (1.00000 - states[9])) / (
            1.00000 + constants[8] / states[7]
        )
        algebraic[39] = (constants[12] * algebraic[36]) * (
            algebraic[36] + states[9]
        ) - constants[13] * states[9]
        algebraic[37] = 1.00000 / (
            (1.00000 + np.exp((states[12] + 71.5500) / 7.43000))
            * (1.00000 + np.exp((states[12] + 71.5500) / 7.43000))
        )
        algebraic[11] = super().custom_piecewise(
            [
                np.greater_equal(states[12], -40.0000),
                0.00000,
                True,
                0.0570000 * np.exp(-(states[12] + 80.0000) / 6.80000),
            ]
        )
        algebraic[29] = super().custom_piecewise(
            [
                np.greater_equal(states[12], -40.0000),
                0.770000
                / (0.130000 * (1.00000 + np.exp(-(states[12] + 10.6600) / 11.1000))),
                True,
                2.70000 * np.exp(0.0790000 * states[12])
                + 310000.0 * np.exp(0.348500 * states[12]),
            ]
        )
        algebraic[41] = 1.00000 / (algebraic[11] + algebraic[29])
        algebraic[40] = algebraic[37]
        algebraic[22] = super().custom_piecewise(
            [
                np.greater_equal(states[12], -40.0000),
                0.00000,
                True,
                (
                    (
                        -25428.0 * np.exp(0.244400 * states[12])
                        - 6.94800e-06 * np.exp(-0.0439100 * states[12])
                    )
                    * (states[12] + 37.7800)
                )
                / (1.00000 + np.exp(0.311000 * (states[12] + 79.2300))),
            ]
        )
        algebraic[34] = super().custom_piecewise(
            [
                np.greater_equal(states[12], -40.0000),
                (0.600000 * np.exp(0.0570000 * states[12]))
                / (1.00000 + np.exp(-0.100000 * (states[12] + 32.0000))),
                True,
                (0.0242400 * np.exp(-0.0105200 * states[12]))
                / (1.00000 + np.exp(-0.137800 * (states[12] + 40.1400))),
            ]
        )
        algebraic[43] = 1.00000 / (algebraic[22] + algebraic[34])
        algebraic[51] = (states[12] * constants[0]) / (constants[10] * constants[11])
        algebraic[53] = (
            (2.00000 * constants[0])
            * (
                states[7]
                * np.exp(
                    2.00000
                    * ((states[12] * constants[0]) / (constants[10] * constants[11]))
                )
                - 0.341000 * constants[106]
            )
        ) * (
            super().custom_piecewise(
                [
                    np.equal(states[12], 0.00000),
                    1.00000,
                    True,
                    (2.00000 * algebraic[51])
                    / (np.exp(2.00000 * algebraic[51]) - 1.00000),
                ]
            )
        )
        algebraic[55] = states[15] * states[16]
        algebraic[57] = (
            algebraic[-6]
            * (((constants[18] * algebraic[53]) * states[13]) * algebraic[55])
            * states[14]
        )
        algebraic[58] = (
            0.300000
            / (
                1.00000
                - (
                    super().custom_piecewise(
                        [
                            np.greater(algebraic[57], 0.00000),
                            0.00000,
                            True,
                            algebraic[57] / 0.0500000,
                        ]
                    )
                )
            )
            + 0.550000 / (1.00000 + states[7] / 0.00300000)
        ) + 0.150000
        algebraic[38] = algebraic[36] + states[9]
        algebraic[47] = 1.00000 / (1.00000 + constants[1] / algebraic[38])
        algebraic[60] = (10.0000 * algebraic[47] + 0.500000) + 1.00000 / (
            1.00000 + states[7] / 0.00300000
        )
        algebraic[48] = ((constants[10] * constants[11]) / constants[0]) * np.log(
            constants[16] / states[10]
        )
        algebraic[62] = 1.00000 / (
            1.00000
            + np.exp(((states[12] + 79.3000) - 2.60000 * constants[16]) / 19.6000)
        )
        algebraic[63] = (
            algebraic[-5]
            * (
                (constants[84] * (np.power(constants[16] / 5.40000, 1.0 / 2)))
                * algebraic[62]
            )
            * (states[12] - algebraic[48])
        )
        algebraic[64] = 1.00000 / (1.00000 + np.exp(-(states[12] - 14.4800) / 18.3400))
        algebraic[65] = (constants[21] * algebraic[64]) * (states[12] - algebraic[48])
        algebraic[66] = 1.00000 / (1.00000 + np.exp((states[12] + 22.0000) / 15.0000))
        algebraic[67] = (
            algebraic[-1]
            * (
                (
                    (constants[85] * (np.power(constants[16] / 5.40000, 1.0 / 2)))
                    * states[19]
                )
                * algebraic[66]
            )
            * (states[12] - algebraic[48])
        )
        algebraic[49] = ((constants[10] * constants[11]) / constants[0]) * np.log(
            (constants[16] + constants[9] * constants[17])
            / (states[10] + constants[9] * states[11])
        )
        algebraic[68] = 1.00000 + 0.600000 / (
            1.00000 + np.power(3.80000e-05 / states[4], 1.40000)
        )
        algebraic[69] = (
            algebraic[-4]
            * (((constants[22] * algebraic[68]) * states[20]) * states[21])
            * (states[12] - algebraic[49])
        )
        algebraic[133] = constants[41] * np.exp(
            ((constants[86] * states[12]) * constants[0])
            / ((3.00000 * constants[10]) * constants[11])
        )
        algebraic[136] = (
            constants[49]
            * (
                ((states[11] / algebraic[133]) * (states[11] / algebraic[133]))
                * (states[11] / algebraic[133])
            )
        ) / (
            (
                (
                    (1.00000 + states[11] / algebraic[133])
                    * (1.00000 + states[11] / algebraic[133])
                )
                * (1.00000 + states[11] / algebraic[133])
                + (1.00000 + states[10] / constants[38])
                * (1.00000 + states[10] / constants[38])
            )
            - 1.00000
        )
        algebraic[134] = constants[42] * np.exp(
            (((1.00000 - constants[86]) * states[12]) * constants[0])
            / ((3.00000 * constants[10]) * constants[11])
        )
        algebraic[138] = (
            constants[50]
            * (
                ((constants[17] / algebraic[134]) * (constants[17] / algebraic[134]))
                * (constants[17] / algebraic[134])
            )
        ) / (
            (
                (
                    (1.00000 + constants[17] / algebraic[134])
                    * (1.00000 + constants[17] / algebraic[134])
                )
                * (1.00000 + constants[17] / algebraic[134])
                + (1.00000 + constants[16] / constants[39])
                * (1.00000 + constants[16] / constants[39])
            )
            - 1.00000
        )
        algebraic[135] = constants[47] / (
            ((1.00000 + constants[36] / constants[37]) + states[11] / constants[43])
            + states[10] / constants[44]
        )
        algebraic[139] = ((constants[52] * algebraic[135]) * constants[36]) / (
            1.00000 + constants[46] / constants[40]
        )
        algebraic[140] = (
            constants[54]
            * ((states[10] / constants[38]) * (states[10] / constants[38]))
        ) / (
            (
                (
                    (1.00000 + states[11] / algebraic[133])
                    * (1.00000 + states[11] / algebraic[133])
                )
                * (1.00000 + states[11] / algebraic[133])
                + (1.00000 + states[10] / constants[38])
                * (1.00000 + states[10] / constants[38])
            )
            - 1.00000
        )
        algebraic[141] = (
            (
                (constants[102] * algebraic[136]) * constants[101]
                + (algebraic[138] * algebraic[140]) * algebraic[139]
            )
            + (constants[101] * algebraic[140]) * algebraic[139]
        ) + (algebraic[139] * algebraic[136]) * constants[101]
        algebraic[137] = (
            constants[53]
            * ((constants[16] / constants[39]) * (constants[16] / constants[39]))
        ) / (
            (
                (
                    (1.00000 + constants[17] / algebraic[134])
                    * (1.00000 + constants[17] / algebraic[134])
                )
                * (1.00000 + constants[17] / algebraic[134])
                + (1.00000 + constants[16] / constants[39])
                * (1.00000 + constants[16] / constants[39])
            )
            - 1.00000
        )
        algebraic[142] = (
            (
                (algebraic[138] * constants[103]) * algebraic[140]
                + (algebraic[136] * constants[101]) * algebraic[137]
            )
            + (algebraic[137] * constants[103]) * algebraic[140]
        ) + (constants[101] * algebraic[137]) * algebraic[140]
        algebraic[143] = (
            (
                (constants[101] * algebraic[137]) * constants[102]
                + (algebraic[139] * algebraic[138]) * constants[103]
            )
            + (algebraic[138] * constants[103]) * constants[102]
        ) + (algebraic[137] * constants[102]) * constants[103]
        algebraic[144] = (
            (
                (algebraic[140] * algebraic[139]) * algebraic[138]
                + (algebraic[137] * constants[102]) * algebraic[136]
            )
            + (algebraic[138] * constants[102]) * algebraic[136]
        ) + (algebraic[139] * algebraic[138]) * algebraic[136]
        algebraic[147] = algebraic[143] / (
            ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
        )
        algebraic[148] = algebraic[144] / (
            ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
        )
        algebraic[149] = 2.00000 * (
            algebraic[148] * constants[103] - algebraic[147] * algebraic[136]
        )
        algebraic[145] = algebraic[141] / (
            ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
        )
        algebraic[146] = algebraic[142] / (
            ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
        )
        algebraic[150] = 3.00000 * (
            algebraic[145] * algebraic[137] - algebraic[146] * algebraic[139]
        )
        algebraic[151] = constants[100] * (
            constants[35] * algebraic[150] + constants[56] * algebraic[149]
        )
        algebraic[153] = (states[17] - states[10]) / 2.00000
        algebraic[168] = (
            super().custom_piecewise(
                [
                    np.less(
                        (voi - constants[77])
                        - constants[78]
                        * np.floor((voi - constants[77]) / constants[78]),
                        constants[76],
                    ),
                    1.00000,
                    True,
                    0.00000,
                ]
            )
        ) * constants[89]
        algebraic[155] = (
            -(
                (
                    ((algebraic[168] + algebraic[67] + algebraic[69]) + algebraic[63])
                    + algebraic[65]
                )
                - 2.00000 * algebraic[151]
            )
            * constants[105]
        ) / ((2.00000 * constants[0]) * constants[112]) + (
            algebraic[153] * constants[118]
        ) / constants[112]
        algebraic[52] = (
            constants[0]
            * (
                (0.750000 * states[17]) * np.exp(algebraic[51])
                - 0.750000 * constants[16]
            )
        ) * (
            super().custom_piecewise(
                [
                    np.equal(states[12], 0.00000),
                    1.00000,
                    True,
                    algebraic[51] / (np.exp(algebraic[51]) - 1.00000),
                ]
            )
        )
        algebraic[56] = (
            ((constants[80] * algebraic[52]) * states[13]) * algebraic[55]
        ) * states[14]
        algebraic[156] = (-algebraic[56] * constants[105]) / (
            (2.00000 * constants[0]) * constants[118]
        ) - algebraic[153]
        algebraic[75] = np.exp(
            ((constants[31] * states[12]) * constants[0])
            / (constants[10] * constants[11])
        )
        algebraic[79] = 1.00000 + (constants[17] / constants[29]) * (
            1.00000 + 1.00000 / algebraic[75]
        )
        algebraic[81] = 1.00000 / algebraic[79]
        algebraic[82] = algebraic[81] * constants[32]
        algebraic[80] = constants[17] / (
            (constants[29] * algebraic[75]) * algebraic[79]
        )
        algebraic[83] = algebraic[80] * constants[34]
        algebraic[84] = algebraic[82] + algebraic[83]
        algebraic[76] = 1.00000 + (states[11] / constants[29]) * (
            1.00000 + algebraic[75]
        )
        algebraic[78] = 1.00000 / algebraic[76]
        algebraic[74] = np.exp(
            ((constants[30] * states[12]) * constants[0])
            / (constants[10] * constants[11])
        )
        algebraic[85] = (algebraic[78] * constants[32]) / algebraic[74]
        algebraic[77] = (states[11] * algebraic[75]) / (constants[29] * algebraic[76])
        algebraic[86] = algebraic[77] * constants[34]
        algebraic[87] = algebraic[85] + algebraic[86]
        algebraic[71] = 1.00000 + (states[11] / constants[27]) * (
            1.00000 + states[11] / constants[28]
        )
        algebraic[73] = 1.00000 / algebraic[71]
        algebraic[88] = (algebraic[73] * states[4]) * constants[26]
        algebraic[72] = (states[11] * states[11]) / (
            (algebraic[71] * constants[27]) * constants[28]
        )
        algebraic[89] = (algebraic[72] * algebraic[77]) * constants[33]
        algebraic[91] = (constants[93] * algebraic[87]) * (
            algebraic[89] + algebraic[88]
        ) + (constants[94] * algebraic[89]) * (constants[93] + algebraic[84])
        algebraic[90] = (algebraic[80] * constants[91]) * constants[33]
        algebraic[92] = (constants[107] * algebraic[89]) * (
            algebraic[87] + constants[94]
        ) + (algebraic[87] * algebraic[88]) * (constants[107] + algebraic[90])
        algebraic[93] = (constants[107] * algebraic[84]) * (
            algebraic[89] + algebraic[88]
        ) + (algebraic[90] * algebraic[88]) * (constants[93] + algebraic[84])
        algebraic[94] = (constants[93] * algebraic[90]) * (
            algebraic[87] + constants[94]
        ) + (algebraic[84] * constants[94]) * (constants[107] + algebraic[90])
        algebraic[95] = algebraic[91] / (
            ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
        )
        algebraic[96] = algebraic[92] / (
            ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
        )
        algebraic[99] = algebraic[96] * constants[93] - algebraic[95] * constants[107]
        algebraic[97] = algebraic[93] / (
            ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
        )
        algebraic[98] = algebraic[94] / (
            ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
        )
        algebraic[100] = (
            3.00000 * (algebraic[98] * algebraic[89] - algebraic[95] * algebraic[90])
            + algebraic[97] * algebraic[86]
        ) - algebraic[96] * algebraic[83]
        algebraic[70] = 1.00000 / (
            1.00000 + (constants[23] / states[4]) * (constants[23] / states[4])
        )
        algebraic[101] = ((0.800000 * constants[87]) * algebraic[70]) * (
            constants[35] * algebraic[100] + constants[20] * algebraic[99]
        )
        algebraic[50] = ((constants[10] * constants[11]) / constants[0]) * np.log(
            constants[17] / states[11]
        )
        algebraic[152] = (
            algebraic[-2]
            * ((((constants[57] * states[23]) * states[23]) * states[23]) * states[22])
            * (states[12] - algebraic[50])
        )
        algebraic[154] = (
            (constants[59] * constants[0])
            * (states[11] * np.exp(algebraic[51]) - constants[17])
        ) * (
            super().custom_piecewise(
                [
                    np.equal(states[12], 0.00000),
                    1.00000,
                    True,
                    algebraic[51] / (np.exp(algebraic[51]) - 1.00000),
                ]
            )
        )
        algebraic[160] = (
            algebraic[-3]
            * (
                (
                    (((constants[60] * states[27]) * states[27]) * states[27])
                    * states[25]
                )
                * states[26]
            )
            * (states[12] - algebraic[50])
        )
        algebraic[162] = (states[18] - states[11]) / 2.00000
        algebraic[165] = (
            -(
                (
                    ((algebraic[160] + algebraic[152]) + 3.00000 * algebraic[101])
                    + 3.00000 * algebraic[151]
                )
                + algebraic[154]
            )
            * constants[105]
        ) / ((2.00000 * constants[0]) * constants[112]) + (
            algebraic[162] * constants[118]
        ) / constants[112]
        algebraic[109] = 1.00000 + (constants[17] / constants[29]) * (
            1.00000 + 1.00000 / algebraic[75]
        )
        algebraic[111] = 1.00000 / algebraic[109]
        algebraic[112] = algebraic[111] * constants[32]
        algebraic[110] = constants[17] / (
            (constants[29] * algebraic[75]) * algebraic[109]
        )
        algebraic[113] = algebraic[110] * constants[34]
        algebraic[114] = algebraic[112] + algebraic[113]
        algebraic[103] = 1.00000 + (states[18] / constants[29]) * (
            1.00000 + algebraic[75]
        )
        algebraic[105] = 1.00000 / algebraic[103]
        algebraic[115] = (algebraic[105] * constants[32]) / algebraic[74]
        algebraic[104] = (states[18] * algebraic[75]) / (constants[29] * algebraic[103])
        algebraic[116] = algebraic[104] * constants[34]
        algebraic[117] = algebraic[115] + algebraic[116]
        algebraic[106] = 1.00000 + (states[18] / constants[27]) * (
            1.00000 + states[18] / constants[28]
        )
        algebraic[108] = 1.00000 / algebraic[106]
        algebraic[118] = (algebraic[108] * states[7]) * constants[26]
        algebraic[107] = (states[18] * states[18]) / (
            (algebraic[106] * constants[27]) * constants[28]
        )
        algebraic[119] = (algebraic[107] * algebraic[104]) * constants[33]
        algebraic[121] = (constants[98] * algebraic[117]) * (
            algebraic[119] + algebraic[118]
        ) + (constants[99] * algebraic[119]) * (constants[98] + algebraic[114])
        algebraic[120] = (algebraic[110] * constants[96]) * constants[33]
        algebraic[122] = (constants[108] * algebraic[119]) * (
            algebraic[117] + constants[99]
        ) + (algebraic[117] * algebraic[118]) * (constants[108] + algebraic[120])
        algebraic[123] = (constants[108] * algebraic[114]) * (
            algebraic[119] + algebraic[118]
        ) + (algebraic[120] * algebraic[118]) * (constants[98] + algebraic[114])
        algebraic[124] = (constants[98] * algebraic[120]) * (
            algebraic[117] + constants[99]
        ) + (algebraic[114] * constants[99]) * (constants[108] + algebraic[120])
        algebraic[125] = algebraic[121] / (
            ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
        )
        algebraic[126] = algebraic[122] / (
            ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
        )
        algebraic[129] = (
            algebraic[126] * constants[98] - algebraic[125] * constants[108]
        )
        algebraic[127] = algebraic[123] / (
            ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
        )
        algebraic[128] = algebraic[124] / (
            ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
        )
        algebraic[130] = (
            3.00000
            * (algebraic[128] * algebraic[119] - algebraic[125] * algebraic[120])
            + algebraic[127] * algebraic[116]
        ) - algebraic[126] * algebraic[113]
        algebraic[102] = 1.00000 / (
            1.00000 + (constants[23] / states[7]) * (constants[23] / states[7])
        )
        algebraic[131] = ((0.200000 * constants[87]) * algebraic[102]) * (
            constants[35] * algebraic[130] + constants[20] * algebraic[129]
        )
        algebraic[54] = (
            constants[0]
            * (
                (0.750000 * states[18]) * np.exp(algebraic[51])
                - 0.750000 * constants[17]
            )
        ) * (
            super().custom_piecewise(
                [
                    np.equal(states[12], 0.00000),
                    1.00000,
                    True,
                    algebraic[51] / (np.exp(algebraic[51]) - 1.00000),
                ]
            )
        )
        algebraic[59] = (
            ((constants[81] * algebraic[54]) * states[13]) * algebraic[55]
        ) * states[14]
        algebraic[166] = (
            -(algebraic[59] + 3.00000 * algebraic[131]) * constants[105]
        ) / ((2.00000 * constants[0]) * constants[118]) - algebraic[162]
        algebraic[61] = (
            ((constants[83] * 2.00000) * constants[0])
            * (states[4] * np.exp(2.00000 * algebraic[51]) - 0.341000 * constants[106])
        ) * (
            super().custom_piecewise(
                [
                    np.equal(states[12], 0.00000),
                    1.00000,
                    True,
                    (2.00000 * algebraic[51])
                    / (np.exp(2.00000 * algebraic[51]) - 1.00000),
                ]
            )
        )
        algebraic[14] = (
            constants[3] * (1.00000 - np.exp(-states[8] / constants[6]))
        ) * np.exp(-states[8] / constants[5])
        algebraic[24] = algebraic[14] * (states[6] - states[7])
        algebraic[31] = states[1] + algebraic[24]
        algebraic[157] = 1.00000 - 1.00000 / (
            1.00000 + (algebraic[31] / 0.400000) * (algebraic[31] / 0.400000)
        )
        algebraic[158] = 1.00000 / (1.00000 + np.exp(-(states[12] + 10.0000) / 5.00000))
        algebraic[159] = (
            ((constants[88] * states[24]) * algebraic[158]) * algebraic[157]
        ) * (states[12] - constants[79])
        algebraic[161] = (constants[61] * states[4]) / (0.000500000 + states[4])
        algebraic[163] = (
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
                                                                algebraic[160]
                                                                + algebraic[152]
                                                            )
                                                            + algebraic[57]
                                                        )
                                                        + algebraic[59]
                                                    )
                                                    + algebraic[56]
                                                )
                                                + algebraic[67]
                                            )
                                            + algebraic[69]
                                        )
                                        + algebraic[63]
                                    )
                                    + algebraic[159]
                                )
                                + algebraic[101]
                            )
                            + algebraic[131]
                        )
                        + algebraic[151]
                    )
                    + algebraic[154]
                )
                + algebraic[65]
            )
            + algebraic[161]
        ) + algebraic[61]
        algebraic[167] = (states[28] - states[6]) / 100.000
        algebraic[170] = 1.00000 / (
            1.00000
            + (constants[67] * constants[69])
            / ((constants[69] + states[6]) * (constants[69] + states[6]))
        )
        algebraic[173] = algebraic[170] * (algebraic[167] - algebraic[31])
        algebraic[169] = (states[28] - states[3]) / 100.000
        algebraic[172] = 1.00000 / (
            1.00000
            + (constants[67] * constants[69])
            / ((constants[69] + states[3]) * (constants[69] + states[3]))
        )
        algebraic[175] = algebraic[172] * (algebraic[169] - states[2])
        algebraic[164] = (0.00500000 * states[28]) / 15.0000
        algebraic[174] = ((1.00000 * 0.00500000) * states[5]) / (states[5] + 0.00100000)
        algebraic[177] = (((1.00000 * 3.00000) * 0.00500000) * states[5]) / (
            (states[5] + 0.00100000) - 0.000200000
        )
        algebraic[178] = 1.00000 / (1.00000 + constants[1] / algebraic[38])
        algebraic[181] = (
            (1.00000 - algebraic[178]) * algebraic[174]
            + algebraic[178] * algebraic[177]
        ) - algebraic[164]
        algebraic[171] = ((1.00000 * 0.00500000) * states[4]) / (states[4] + 0.00100000)
        algebraic[176] = (((1.00000 * 3.00000) * 0.00500000) * states[4]) / (
            (states[4] + 0.00100000) - 0.000200000
        )
        algebraic[179] = (
            (1.00000 - algebraic[178]) * algebraic[171]
            + algebraic[178] * algebraic[176]
        ) - algebraic[164]
        algebraic[186] = (
            (
                algebraic[179] * (constants[116] / constants[115])
                + algebraic[181] * (constants[117] / constants[115])
            )
            - (algebraic[167] * constants[110]) / constants[115]
        ) - (algebraic[169] * constants[111]) / constants[115]
        algebraic[180] = (states[7] - states[4]) / 0.200000
        algebraic[182] = (
            (-algebraic[57] + 2.00000 * algebraic[131])
            * (constants[105] / ((2.00000 * constants[118]) * constants[0]))
            + algebraic[31] * (constants[110] / constants[118])
        ) - algebraic[180]
        algebraic[188] = super().custom_piecewise(
            [
                np.greater(algebraic[182], 0.00000),
                (
                    ((1.00000 * 60.0000) * algebraic[182])
                    * (
                        1.00000
                        + 1.00000
                        / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
                    )
                )
                / (1.00000 + np.power(0.750000 / states[6], 8.00000)),
                True,
                0.00000,
            ]
        )
        algebraic[42] = (
            (1.00000 * 20.0000)
            * (
                1.00000
                + 1.00000 / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
            )
        ) / (1.00000 + np.power(0.500000 / states[6], 8.00000))
        algebraic[44] = super().custom_piecewise(
            [np.greater(0.00100000, algebraic[42]), 0.00100000, True, algebraic[42]]
        )
        algebraic[0] = (states[4] - states[5]) / constants[4]
        algebraic[183] = (
            algebraic[0] + states[2] * (constants[111] / constants[114])
        ) - algebraic[181] * (constants[117] / constants[114])
        algebraic[189] = super().custom_piecewise(
            [
                np.greater(algebraic[183], 0.00000),
                (
                    1.00000
                    * (
                        (250.000 * algebraic[183])
                        * (
                            1.00000
                            + 1.00000
                            / (
                                1.00000
                                + np.power(constants[1] / algebraic[38], 8.00000)
                            )
                        )
                    )
                )
                / (1.00000 + np.power(0.750000 / states[3], 8.00000)),
                True,
                0.00000,
            ]
        )
        algebraic[45] = (
            50.0000
            * (
                1.00000
                + 1.00000 / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
            )
        ) / (1.00000 + np.power(0.500000 / states[3], 8.00000))
        algebraic[46] = super().custom_piecewise(
            [np.greater(0.00100000, algebraic[45]), 0.00100000, True, algebraic[45]]
        )
        algebraic[184] = 1.00000 / (
            (
                1.00000
                + (constants[66] * constants[68])
                / ((constants[68] + states[4]) * (constants[68] + states[4]))
            )
            + (constants[71] * constants[70])
            / ((constants[70] + states[4]) * (constants[70] + states[4]))
        )
        algebraic[190] = algebraic[184] * (
            (
                (
                    (
                        -((algebraic[161] + algebraic[61]) - 2.00000 * algebraic[101])
                        * constants[105]
                    )
                    / (((2.00000 * 2.00000) * constants[0]) * constants[113])
                    - (algebraic[179] * constants[116]) / constants[113]
                )
                + (algebraic[180] * constants[118]) / constants[113]
            )
            - algebraic[0]
        )
        algebraic[185] = 1.00000 / (
            (
                1.00000
                + (constants[66] * constants[68])
                / ((constants[68] + states[5]) * (constants[68] + states[5]))
            )
            + (constants[71] * constants[70])
            / ((constants[70] + states[5]) * (constants[70] + states[5]))
        )
        algebraic[191] = algebraic[185] * (
            (
                states[2] * (constants[111] / constants[114])
                + (algebraic[0] * constants[114]) / constants[113]
            )
            - (algebraic[181] * constants[117]) / constants[114]
        )
        algebraic[187] = 1.00000 / (
            (
                1.00000
                + (constants[63] * constants[65])
                / ((constants[65] + states[7]) * (constants[65] + states[7]))
            )
            + (constants[62] * constants[64])
            / ((constants[64] + states[7]) * (constants[64] + states[7]))
        )
        algebraic[192] = algebraic[187] * (
            (
                (
                    -(algebraic[159] + algebraic[57] - 2.00000 * algebraic[131])
                    * constants[105]
                )
                / (((2.00000 * 2.00000) * constants[0]) * constants[118])
                + (algebraic[31] * constants[110]) / constants[118]
            )
            - algebraic[180]
        )
        algebraic[3] = ((states[12] * constants[0]) * constants[0]) / (
            constants[10] * constants[11]
        )
        algebraic[13] = (
            constants[75] * states[4] + (1.00000 - constants[75]) * states[5]
        )
        algebraic[132] = algebraic[101] + algebraic[131]
        return algebraic

    def plot_ctd(
        self, voi, states=None, algebraic=None, percent_decay=90, time_idx_range=None
    ):
        """Bespoke plot_ctd function PigModel since cai_avg is used as algebraic variable instead of state variable."""
        if time_idx_range is None:
            time_idx_range = [0, len(voi)]

        time = voi[time_idx_range[0] : time_idx_range[1]]
        calcium = algebraic[self.algebraic_idx_dict["cai_avg"]][
            time_idx_range[0] : time_idx_range[1]
        ]

        ctd_start_time, ctd_frac = self.detect_ctd(time, calcium, percent_decay)
        # print(f"ctd{percent_decay} = ", ctd_frac)
        plt.plot(time, calcium, label="Cytosolic Calcium")
        # print(ctd_start_time)
        # print(ctd_frac)
        if ctd_frac is not None:
            plt.axvline(
                ctd_frac + ctd_start_time,
                color="red",
                label=f"CTD_{percent_decay} Endpoint",
            )
            plt.axvline(
                ctd_start_time, color="red", linestyle="--", label="CTD Startpoint"
            )
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Calcium Concentration (mM)")
        plt.title(self.species + " Calcium Transient")
        plt.legend()
        plt.show()

    def calculate_ap_features(
        self, voi, states, rates, algebraic=None, time_idx_range=None
    ):
        """Bespoke ap_features function PigModel since cai_avg is used as algebraic variable instead of state variable."""
        if time_idx_range is None:
            time_idx_range = [0, len(voi)]

        dvdt = list(rates[self.rate_idx_dict["dvdt"]])[
            time_idx_range[0] : time_idx_range[1]
        ]
        # print(dvdt)
        time = voi[time_idx_range[0] : time_idx_range[1]]
        potential = states[self.state_idx_dict["v"]][
            time_idx_range[0] : time_idx_range[1]
        ]
        calcium = algebraic[self.algebraic_idx_dict["cai_avg"]][
            time_idx_range[0] : time_idx_range[1]
        ]

        ap_measures_dict = {}
        for percent_repolarisation in [40, 50, 90]:
            ap_measures_dict[f"APD_{percent_repolarisation}"] = self.detect_apd(
                time, potential, dvdt, percent_repolarisation
            )[1]

        if ap_measures_dict["APD_90"] is None or ap_measures_dict["APD_40"] is None:
            print("APD_90 or APD_40 not reached, cannot calculate tri_90_40!")
            ap_measures_dict["tri_90_40"] = None
        else:
            ap_measures_dict["tri_90_40"] = (
                ap_measures_dict["APD_90"] - ap_measures_dict["APD_40"]
            )
        ap_measures_dict["dvdt_max"] = max(dvdt)
        ap_measures_dict["v_peak"] = max(potential)
        ap_measures_dict["RMP"] = min(potential)

        for percent_decay in [50, 90]:
            ap_measures_dict[f"CTD_{percent_decay}"] = self.detect_ctd(
                time, calcium, percent_decay
            )[1]
        # print("AP Measures: ", ap_measures_dict)

        return ap_measures_dict


@njit
def njitComputeRates(
    voi, states, constants, sizeStates, sizeAlgebraic, num_channels, drug_multipliers
):
    """Compute the rates for the model with njit acceleration."""
    rates = np.zeros(sizeStates, dtype=np.float64)
    algebraic = algebraic = np.zeros(sizeAlgebraic, dtype=np.float64)
    algebraic[-num_channels:] = drug_multipliers
    algebraic[1] = -states[0] / 0.100000 if states[8] < 5.00000 else 1.00000
    rates[0] = algebraic[1]
    algebraic[2] = (
        -states[8] / 0.00100000
        if (states[6] > constants[2] and states[0] > 45.0000)
        else 1.00000
    )
    rates[8] = algebraic[2]
    algebraic[8] = 1.00000 / (1.00000 + np.exp((states[12] + 91.0000) / 6.10000))
    rates[22] = (algebraic[8] - states[22]) / constants[58]
    algebraic[4] = 1.00000 / (1.00000 + np.exp(-(states[12] - constants[19]) / 6.20000))
    algebraic[17] = 0.600000 + 1.00000 / (
        np.exp(-0.0500000 * (states[12] + 6.00000))
        + np.exp(0.0900000 * (states[12] + 14.0000))
    )
    rates[13] = (algebraic[4] - states[13]) / algebraic[17]
    algebraic[6] = 12.9800 + 1.00000 / (
        0.365200 * np.exp((states[12] - 31.6600) / 3.86900)
        + 4.12300e-05 * np.exp(-(states[12] - 47.7800) / 20.3800)
    )
    algebraic[18] = 1.00000 / (1.00000 + np.exp(-(states[12] + 56.8000) / 17.8000))
    rates[19] = (algebraic[18] - states[19]) / algebraic[6]
    algebraic[10] = 0.0250000 / (1.00000 + np.exp((states[12] + 58.0000) / 5.00000))
    algebraic[21] = 1.00000 / (
        5.00000 * (1.00000 + np.exp((states[12] + 19.0000) / -9.00000))
    )
    rates[24] = algebraic[10] * (1.00000 - states[24]) - algebraic[21] * states[24]
    algebraic[5] = 1.00000 / (
        1.00000 + np.exp((states[12] - constants[82]) / 4.90000)
    ) + 0.350000 / (1.00000 + np.exp((45.0000 - states[12]) / 20.0000))
    algebraic[15] = algebraic[5]
    algebraic[25] = 7.00000 + 1.00000 / (
        0.00450000 * np.exp(-(states[12] + 20.0000) / 10.0000)
        + 0.00450000 * np.exp((states[12] + 20.0000) / 10.0000)
    )
    rates[15] = (algebraic[15] - states[15]) / algebraic[25]
    algebraic[16] = algebraic[5]
    algebraic[26] = 70.0000 + 1.00000 / (
        3.50000e-05 * np.exp(-(states[12] + 5.00000) / 4.00000)
        + 3.50000e-05 * np.exp((states[12] + 5.00000) / 6.00000)
    )
    rates[16] = (algebraic[16] - states[16]) / algebraic[26]
    algebraic[7] = 300.000 + 1.00000 / (
        1.00000e-06 * np.exp((states[12] + 50.0000) / 20.0000)
        + 0.0400000 * np.exp(-(states[12] + 50.0000) / 20.0000)
    )
    algebraic[27] = 1.00000 / (1.00000 + np.exp(-(states[12] - 25.1000) / 37.1000))
    rates[20] = (algebraic[27] - states[20]) / algebraic[7]
    algebraic[19] = 1.00000 / (
        0.0100000 * np.exp((states[12] - 50.0000) / 20.0000)
        + 0.0193000 * np.exp(-(states[12] + 66.5400) / 31.0000)
    )
    algebraic[32] = algebraic[27]
    rates[21] = (algebraic[32] - states[21]) / algebraic[19]
    algebraic[9] = 0.320000 * (
        10.0000
        if states[12] == -47.1300
        else -(states[12] + 47.1300)
        / (np.exp(-0.100000 * (states[12] + 47.1300)) - 1.00000)
    )
    algebraic[20] = 0.0800000 * np.exp(-states[12] / 11.0000)
    algebraic[28] = algebraic[9] / (algebraic[9] + algebraic[20])
    algebraic[33] = 1.00000 / (algebraic[9] + algebraic[20])
    rates[23] = (algebraic[28] - states[23]) / algebraic[33]
    algebraic[30] = 1.00000 / (
        (1.00000 + np.exp((-56.8600 - states[12]) / 9.03000))
        * (1.00000 + np.exp((-56.8600 - states[12]) / 9.03000))
    )
    algebraic[12] = 1.00000 / (1.00000 + np.exp((-60.0000 - states[12]) / 5.00000))
    algebraic[23] = 0.100000 / (
        1.00000 + np.exp((states[12] + 35.0000) / 5.00000)
    ) + 0.100000 / (1.00000 + np.exp((states[12] - 50.0000) / 200.000))
    algebraic[35] = algebraic[12] * algebraic[23]
    rates[27] = (algebraic[30] - states[27]) / algebraic[35]
    algebraic[36] = (constants[7] * (1.00000 - states[9])) / (
        1.00000 + constants[8] / states[7]
    )
    algebraic[39] = (constants[12] * algebraic[36]) * (
        algebraic[36] + states[9]
    ) - constants[13] * states[9]
    rates[9] = algebraic[39]
    algebraic[37] = 1.00000 / (
        (1.00000 + np.exp((states[12] + 71.5500) / 7.43000))
        * (1.00000 + np.exp((states[12] + 71.5500) / 7.43000))
    )
    algebraic[11] = (
        0.0
        if states[12] >= -40.0
        else 0.0570000 * np.exp(-(states[12] + 80.0000) / 6.80000)
    )
    algebraic[29] = (
        0.770000 / (0.130000 * (1.00000 + np.exp(-(states[12] + 10.6600) / 11.1000)))
        if states[12] >= -40.0
        else 2.70000 * np.exp(0.0790000 * states[12])
        + 310000.0 * np.exp(0.348500 * states[12])
    )
    algebraic[41] = 1.00000 / (algebraic[11] + algebraic[29])
    rates[25] = (algebraic[37] - states[25]) / algebraic[41]
    algebraic[40] = algebraic[37]
    algebraic[22] = (
        0.0
        if states[12] >= -40.0
        else (
            (
                -25428.0 * np.exp(0.244400 * states[12])
                - 6.94800e-06 * np.exp(-0.0439100 * states[12])
            )
            * (states[12] + 37.7800)
        )
        / (1.00000 + np.exp(0.311000 * (states[12] + 79.2300)))
    )
    algebraic[34] = (
        (0.600000 * np.exp(0.0570000 * states[12]))
        / (1.00000 + np.exp(-0.100000 * (states[12] + 32.0000)))
        if states[12] >= -40.0
        else (0.0242400 * np.exp(-0.0105200 * states[12]))
        / (1.00000 + np.exp(-0.137800 * (states[12] + 40.1400)))
    )
    algebraic[43] = 1.00000 / (algebraic[22] + algebraic[34])
    rates[26] = (algebraic[40] - states[26]) / algebraic[43]
    algebraic[51] = (states[12] * constants[0]) / (constants[10] * constants[11])
    algebraic[53] = (
        (2.00000 * constants[0])
        * (
            states[7]
            * np.exp(
                2.00000
                * ((states[12] * constants[0]) / (constants[10] * constants[11]))
            )
            - 0.341000 * constants[106]
        )
    ) * (
        1.00000
        if states[12] == 0.00000
        else ((2.00000 * algebraic[51]) / (np.exp(2.00000 * algebraic[51]) - 1.00000))
    )
    algebraic[55] = states[15] * states[16]
    algebraic[57] = (
        algebraic[-6]
        * (((constants[18] * algebraic[53]) * states[13]) * algebraic[55])
        * states[14]
    )
    algebraic[58] = (
        0.300000
        / (1.00000 - (0.0 if algebraic[57] > 0.0 else algebraic[57] / 0.0500000))
        + 0.550000 / (1.00000 + states[7] / 0.00300000)
        + 0.150000
    )
    algebraic[38] = algebraic[36] + states[9]
    algebraic[47] = 1.00000 / (1.00000 + constants[1] / algebraic[38])
    algebraic[60] = (10.0000 * algebraic[47] + 0.500000) + 1.00000 / (
        1.00000 + states[7] / 0.00300000
    )
    rates[14] = (algebraic[58] - states[14]) / algebraic[60]
    algebraic[48] = ((constants[10] * constants[11]) / constants[0]) * np.log(
        constants[16] / states[10]
    )
    algebraic[62] = 1.00000 / (
        1.00000 + np.exp(((states[12] + 79.3000) - 2.60000 * constants[16]) / 19.6000)
    )
    algebraic[63] = (
        algebraic[-5]
        * (
            (constants[84] * (np.power(constants[16] / 5.40000, 1.0 / 2)))
            * algebraic[62]
        )
        * (states[12] - algebraic[48])
    )
    algebraic[64] = 1.00000 / (1.00000 + np.exp(-(states[12] - 14.4800) / 18.3400))
    algebraic[65] = (constants[21] * algebraic[64]) * (states[12] - algebraic[48])
    algebraic[66] = 1.00000 / (1.00000 + np.exp((states[12] + 22.0000) / 15.0000))
    algebraic[67] = (
        algebraic[-1]
        * (
            (
                (constants[85] * (np.power(constants[16] / 5.40000, 1.0 / 2)))
                * states[19]
            )
            * algebraic[66]
        )
        * (states[12] - algebraic[48])
    )
    algebraic[49] = ((constants[10] * constants[11]) / constants[0]) * np.log(
        (constants[16] + constants[9] * constants[17])
        / (states[10] + constants[9] * states[11])
    )
    algebraic[68] = 1.00000 + 0.600000 / (
        1.00000 + np.power(3.80000e-05 / states[4], 1.40000)
    )
    algebraic[69] = (
        algebraic[-4]
        * (((constants[22] * algebraic[68]) * states[20]) * states[21])
        * (states[12] - algebraic[49])
    )
    algebraic[133] = constants[41] * np.exp(
        ((constants[86] * states[12]) * constants[0])
        / ((3.00000 * constants[10]) * constants[11])
    )
    algebraic[136] = (
        constants[49]
        * (
            ((states[11] / algebraic[133]) * (states[11] / algebraic[133]))
            * (states[11] / algebraic[133])
        )
    ) / (
        (
            (
                (1.00000 + states[11] / algebraic[133])
                * (1.00000 + states[11] / algebraic[133])
            )
            * (1.00000 + states[11] / algebraic[133])
            + (1.00000 + states[10] / constants[38])
            * (1.00000 + states[10] / constants[38])
        )
        - 1.00000
    )
    algebraic[134] = constants[42] * np.exp(
        (((1.00000 - constants[86]) * states[12]) * constants[0])
        / ((3.00000 * constants[10]) * constants[11])
    )
    algebraic[138] = (
        constants[50]
        * (
            ((constants[17] / algebraic[134]) * (constants[17] / algebraic[134]))
            * (constants[17] / algebraic[134])
        )
    ) / (
        (
            (
                (1.00000 + constants[17] / algebraic[134])
                * (1.00000 + constants[17] / algebraic[134])
            )
            * (1.00000 + constants[17] / algebraic[134])
            + (1.00000 + constants[16] / constants[39])
            * (1.00000 + constants[16] / constants[39])
        )
        - 1.00000
    )
    algebraic[135] = constants[47] / (
        ((1.00000 + constants[36] / constants[37]) + states[11] / constants[43])
        + states[10] / constants[44]
    )
    algebraic[139] = ((constants[52] * algebraic[135]) * constants[36]) / (
        1.00000 + constants[46] / constants[40]
    )
    algebraic[140] = (
        constants[54] * ((states[10] / constants[38]) * (states[10] / constants[38]))
    ) / (
        (
            (
                (1.00000 + states[11] / algebraic[133])
                * (1.00000 + states[11] / algebraic[133])
            )
            * (1.00000 + states[11] / algebraic[133])
            + (1.00000 + states[10] / constants[38])
            * (1.00000 + states[10] / constants[38])
        )
        - 1.00000
    )
    algebraic[141] = (
        (
            (constants[102] * algebraic[136]) * constants[101]
            + (algebraic[138] * algebraic[140]) * algebraic[139]
        )
        + (constants[101] * algebraic[140]) * algebraic[139]
    ) + (algebraic[139] * algebraic[136]) * constants[101]
    algebraic[137] = (
        constants[53]
        * ((constants[16] / constants[39]) * (constants[16] / constants[39]))
    ) / (
        (
            (
                (1.00000 + constants[17] / algebraic[134])
                * (1.00000 + constants[17] / algebraic[134])
            )
            * (1.00000 + constants[17] / algebraic[134])
            + (1.00000 + constants[16] / constants[39])
            * (1.00000 + constants[16] / constants[39])
        )
        - 1.00000
    )
    algebraic[142] = (
        (
            (algebraic[138] * constants[103]) * algebraic[140]
            + (algebraic[136] * constants[101]) * algebraic[137]
        )
        + (algebraic[137] * constants[103]) * algebraic[140]
    ) + (constants[101] * algebraic[137]) * algebraic[140]
    algebraic[143] = (
        (
            (constants[101] * algebraic[137]) * constants[102]
            + (algebraic[139] * algebraic[138]) * constants[103]
        )
        + (algebraic[138] * constants[103]) * constants[102]
    ) + (algebraic[137] * constants[102]) * constants[103]
    algebraic[144] = (
        (
            (algebraic[140] * algebraic[139]) * algebraic[138]
            + (algebraic[137] * constants[102]) * algebraic[136]
        )
        + (algebraic[138] * constants[102]) * algebraic[136]
    ) + (algebraic[139] * algebraic[138]) * algebraic[136]
    algebraic[147] = algebraic[143] / (
        ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
    )
    algebraic[148] = algebraic[144] / (
        ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
    )
    algebraic[149] = 2.00000 * (
        algebraic[148] * constants[103] - algebraic[147] * algebraic[136]
    )
    algebraic[145] = algebraic[141] / (
        ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
    )
    algebraic[146] = algebraic[142] / (
        ((algebraic[141] + algebraic[142]) + algebraic[143]) + algebraic[144]
    )
    algebraic[150] = 3.00000 * (
        algebraic[145] * algebraic[137] - algebraic[146] * algebraic[139]
    )
    algebraic[151] = constants[100] * (
        constants[35] * algebraic[150] + constants[56] * algebraic[149]
    )
    algebraic[153] = (states[17] - states[10]) / 2.00000
    algebraic[168] = constants[89] * (
        1.00000
        if (voi - constants[77])
        - constants[78] * np.floor((voi - constants[77]) / constants[78])
        < constants[76]
        else 0.00000
    )
    algebraic[155] = (
        -(
            (
                ((algebraic[168] + algebraic[67] + algebraic[69]) + algebraic[63])
                + algebraic[65]
            )
            - 2.00000 * algebraic[151]
        )
        * constants[105]
    ) / ((2.00000 * constants[0]) * constants[112]) + (
        algebraic[153] * constants[118]
    ) / constants[112]
    rates[10] = algebraic[155]
    algebraic[52] = (
        constants[0]
        * ((0.750000 * states[17]) * np.exp(algebraic[51]) - 0.750000 * constants[16])
    ) * (
        1.00000
        if states[12] == 0.00000
        else algebraic[51] / (np.exp(algebraic[51]) - 1.00000)
    )
    algebraic[56] = (
        ((constants[80] * algebraic[52]) * states[13]) * algebraic[55]
    ) * states[14]
    algebraic[156] = (-algebraic[56] * constants[105]) / (
        (2.00000 * constants[0]) * constants[118]
    ) - algebraic[153]
    rates[17] = algebraic[156]
    algebraic[75] = np.exp(
        ((constants[31] * states[12]) * constants[0]) / (constants[10] * constants[11])
    )
    algebraic[79] = 1.00000 + (constants[17] / constants[29]) * (
        1.00000 + 1.00000 / algebraic[75]
    )
    algebraic[81] = 1.00000 / algebraic[79]
    algebraic[82] = algebraic[81] * constants[32]
    algebraic[80] = constants[17] / ((constants[29] * algebraic[75]) * algebraic[79])
    algebraic[83] = algebraic[80] * constants[34]
    algebraic[84] = algebraic[82] + algebraic[83]
    algebraic[76] = 1.00000 + (states[11] / constants[29]) * (1.00000 + algebraic[75])
    algebraic[78] = 1.00000 / algebraic[76]
    algebraic[74] = np.exp(
        ((constants[30] * states[12]) * constants[0]) / (constants[10] * constants[11])
    )
    algebraic[85] = (algebraic[78] * constants[32]) / algebraic[74]
    algebraic[77] = (states[11] * algebraic[75]) / (constants[29] * algebraic[76])
    algebraic[86] = algebraic[77] * constants[34]
    algebraic[87] = algebraic[85] + algebraic[86]
    algebraic[71] = 1.00000 + (states[11] / constants[27]) * (
        1.00000 + states[11] / constants[28]
    )
    algebraic[73] = 1.00000 / algebraic[71]
    algebraic[88] = (algebraic[73] * states[4]) * constants[26]
    algebraic[72] = (states[11] * states[11]) / (
        (algebraic[71] * constants[27]) * constants[28]
    )
    algebraic[89] = (algebraic[72] * algebraic[77]) * constants[33]
    algebraic[91] = (constants[93] * algebraic[87]) * (
        algebraic[89] + algebraic[88]
    ) + (constants[94] * algebraic[89]) * (constants[93] + algebraic[84])
    algebraic[90] = (algebraic[80] * constants[91]) * constants[33]
    algebraic[92] = (constants[107] * algebraic[89]) * (
        algebraic[87] + constants[94]
    ) + (algebraic[87] * algebraic[88]) * (constants[107] + algebraic[90])
    algebraic[93] = (constants[107] * algebraic[84]) * (
        algebraic[89] + algebraic[88]
    ) + (algebraic[90] * algebraic[88]) * (constants[93] + algebraic[84])
    algebraic[94] = (constants[93] * algebraic[90]) * (
        algebraic[87] + constants[94]
    ) + (algebraic[84] * constants[94]) * (constants[107] + algebraic[90])
    algebraic[95] = algebraic[91] / (
        ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
    )
    algebraic[96] = algebraic[92] / (
        ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
    )
    algebraic[99] = algebraic[96] * constants[93] - algebraic[95] * constants[107]
    algebraic[97] = algebraic[93] / (
        ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
    )
    algebraic[98] = algebraic[94] / (
        ((algebraic[91] + algebraic[92]) + algebraic[93]) + algebraic[94]
    )
    algebraic[100] = (
        3.00000 * (algebraic[98] * algebraic[89] - algebraic[95] * algebraic[90])
        + algebraic[97] * algebraic[86]
    ) - algebraic[96] * algebraic[83]
    algebraic[70] = 1.00000 / (
        1.00000 + (constants[23] / states[4]) * (constants[23] / states[4])
    )
    algebraic[101] = ((0.800000 * constants[87]) * algebraic[70]) * (
        constants[35] * algebraic[100] + constants[20] * algebraic[99]
    )
    algebraic[50] = ((constants[10] * constants[11]) / constants[0]) * np.log(
        constants[17] / states[11]
    )
    algebraic[152] = (
        algebraic[-2]
        * ((((constants[57] * states[23]) * states[23]) * states[23]) * states[22])
        * (states[12] - algebraic[50])
    )
    algebraic[154] = (
        (constants[59] * constants[0])
        * (states[11] * np.exp(algebraic[51]) - constants[17])
    ) * (
        1.00000
        if states[12] == 0.00000
        else algebraic[51] / (np.exp(algebraic[51]) - 1.00000)
    )
    algebraic[160] = (
        algebraic[-3]
        * (
            ((((constants[60] * states[27]) * states[27]) * states[27]) * states[25])
            * states[26]
        )
        * (states[12] - algebraic[50])
    )
    algebraic[162] = (states[18] - states[11]) / 2.00000
    algebraic[165] = (
        -(
            (
                ((algebraic[160] + algebraic[152]) + 3.00000 * algebraic[101])
                + 3.00000 * algebraic[151]
            )
            + algebraic[154]
        )
        * constants[105]
    ) / ((2.00000 * constants[0]) * constants[112]) + (
        algebraic[162] * constants[118]
    ) / constants[112]
    rates[11] = algebraic[165]
    algebraic[109] = 1.00000 + (constants[17] / constants[29]) * (
        1.00000 + 1.00000 / algebraic[75]
    )
    algebraic[111] = 1.00000 / algebraic[109]
    algebraic[112] = algebraic[111] * constants[32]
    algebraic[110] = constants[17] / ((constants[29] * algebraic[75]) * algebraic[109])
    algebraic[113] = algebraic[110] * constants[34]
    algebraic[114] = algebraic[112] + algebraic[113]
    algebraic[103] = 1.00000 + (states[18] / constants[29]) * (1.00000 + algebraic[75])
    algebraic[105] = 1.00000 / algebraic[103]
    algebraic[115] = (algebraic[105] * constants[32]) / algebraic[74]
    algebraic[104] = (states[18] * algebraic[75]) / (constants[29] * algebraic[103])
    algebraic[116] = algebraic[104] * constants[34]
    algebraic[117] = algebraic[115] + algebraic[116]
    algebraic[106] = 1.00000 + (states[18] / constants[27]) * (
        1.00000 + states[18] / constants[28]
    )
    algebraic[108] = 1.00000 / algebraic[106]
    algebraic[118] = (algebraic[108] * states[7]) * constants[26]
    algebraic[107] = (states[18] * states[18]) / (
        (algebraic[106] * constants[27]) * constants[28]
    )
    algebraic[119] = (algebraic[107] * algebraic[104]) * constants[33]
    algebraic[121] = (constants[98] * algebraic[117]) * (
        algebraic[119] + algebraic[118]
    ) + (constants[99] * algebraic[119]) * (constants[98] + algebraic[114])
    algebraic[120] = (algebraic[110] * constants[96]) * constants[33]
    algebraic[122] = (constants[108] * algebraic[119]) * (
        algebraic[117] + constants[99]
    ) + (algebraic[117] * algebraic[118]) * (constants[108] + algebraic[120])
    algebraic[123] = (constants[108] * algebraic[114]) * (
        algebraic[119] + algebraic[118]
    ) + (algebraic[120] * algebraic[118]) * (constants[98] + algebraic[114])
    algebraic[124] = (constants[98] * algebraic[120]) * (
        algebraic[117] + constants[99]
    ) + (algebraic[114] * constants[99]) * (constants[108] + algebraic[120])
    algebraic[125] = algebraic[121] / (
        ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
    )
    algebraic[126] = algebraic[122] / (
        ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
    )
    algebraic[129] = algebraic[126] * constants[98] - algebraic[125] * constants[108]
    algebraic[127] = algebraic[123] / (
        ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
    )
    algebraic[128] = algebraic[124] / (
        ((algebraic[121] + algebraic[122]) + algebraic[123]) + algebraic[124]
    )
    algebraic[130] = (
        3.00000 * (algebraic[128] * algebraic[119] - algebraic[125] * algebraic[120])
        + algebraic[127] * algebraic[116]
    ) - algebraic[126] * algebraic[113]
    algebraic[102] = 1.00000 / (
        1.00000 + (constants[23] / states[7]) * (constants[23] / states[7])
    )
    algebraic[131] = ((0.200000 * constants[87]) * algebraic[102]) * (
        constants[35] * algebraic[130] + constants[20] * algebraic[129]
    )
    algebraic[54] = (
        constants[0]
        * ((0.750000 * states[18]) * np.exp(algebraic[51]) - 0.750000 * constants[17])
    ) * (
        1.00000
        if states[12] == 0.00000
        else algebraic[51] / (np.exp(algebraic[51]) - 1.00000)
    )
    algebraic[59] = (
        ((constants[81] * algebraic[54]) * states[13]) * algebraic[55]
    ) * states[14]
    algebraic[166] = (-(algebraic[59] + 3.00000 * algebraic[131]) * constants[105]) / (
        (2.00000 * constants[0]) * constants[118]
    ) - algebraic[162]
    rates[18] = algebraic[166]
    algebraic[61] = (
        ((constants[83] * 2.00000) * constants[0])
        * (states[4] * np.exp(2.00000 * algebraic[51]) - 0.341000 * constants[106])
    ) * (
        1.00000
        if states[12] == 0.00000
        else (2.00000 * algebraic[51]) / (np.exp(2.00000 * algebraic[51]) - 1.00000)
    )
    algebraic[14] = (
        constants[3] * (1.00000 - np.exp(-states[8] / constants[6]))
    ) * np.exp(-states[8] / constants[5])
    algebraic[24] = algebraic[14] * (states[6] - states[7])
    algebraic[31] = states[1] + algebraic[24]
    algebraic[157] = 1.00000 - 1.00000 / (
        1.00000 + (algebraic[31] / 0.400000) * (algebraic[31] / 0.400000)
    )
    algebraic[158] = 1.00000 / (1.00000 + np.exp(-(states[12] + 10.0000) / 5.00000))
    algebraic[159] = (
        ((constants[88] * states[24]) * algebraic[158]) * algebraic[157]
    ) * (states[12] - constants[79])
    algebraic[161] = (constants[61] * states[4]) / (0.000500000 + states[4])
    algebraic[163] = (
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
                                                            algebraic[160]
                                                            + algebraic[152]
                                                        )
                                                        + algebraic[57]
                                                    )
                                                    + algebraic[59]
                                                )
                                                + algebraic[56]
                                            )
                                            + algebraic[67]
                                        )
                                        + algebraic[69]
                                    )
                                    + algebraic[63]
                                )
                                + algebraic[159]
                            )
                            + algebraic[101]
                        )
                        + algebraic[131]
                    )
                    + algebraic[151]
                )
                + algebraic[154]
            )
            + algebraic[65]
        )
        + algebraic[161]
    ) + algebraic[61]
    rates[12] = -algebraic[163] - algebraic[168]
    algebraic[167] = (states[28] - states[6]) / 100.000
    algebraic[170] = 1.00000 / (
        1.00000
        + (constants[67] * constants[69])
        / ((constants[69] + states[6]) * (constants[69] + states[6]))
    )
    algebraic[173] = algebraic[170] * (algebraic[167] - algebraic[31])
    rates[6] = algebraic[173]
    algebraic[169] = (states[28] - states[3]) / 100.000
    algebraic[172] = 1.00000 / (
        1.00000
        + (constants[67] * constants[69])
        / ((constants[69] + states[3]) * (constants[69] + states[3]))
    )
    algebraic[175] = algebraic[172] * (algebraic[169] - states[2])
    rates[3] = algebraic[175]
    algebraic[164] = (0.00500000 * states[28]) / 15.0000
    algebraic[174] = ((1.00000 * 0.00500000) * states[5]) / (states[5] + 0.00100000)
    algebraic[177] = (((1.00000 * 3.00000) * 0.00500000) * states[5]) / (
        (states[5] + 0.00100000) - 0.000200000
    )
    algebraic[178] = 1.00000 / (1.00000 + constants[1] / algebraic[38])
    algebraic[181] = (
        (1.00000 - algebraic[178]) * algebraic[174] + algebraic[178] * algebraic[177]
    ) - algebraic[164]
    algebraic[171] = ((1.00000 * 0.00500000) * states[4]) / (states[4] + 0.00100000)
    algebraic[176] = (((1.00000 * 3.00000) * 0.00500000) * states[4]) / (
        (states[4] + 0.00100000) - 0.000200000
    )
    algebraic[179] = (
        (1.00000 - algebraic[178]) * algebraic[171] + algebraic[178] * algebraic[176]
    ) - algebraic[164]
    algebraic[186] = (
        (
            algebraic[179] * (constants[116] / constants[115])
            + algebraic[181] * (constants[117] / constants[115])
        )
        - (algebraic[167] * constants[110]) / constants[115]
    ) - (algebraic[169] * constants[111]) / constants[115]
    rates[28] = algebraic[186]
    algebraic[180] = (states[7] - states[4]) / 0.200000
    algebraic[182] = (
        (-algebraic[57] + 2.00000 * algebraic[131])
        * (constants[105] / ((2.00000 * constants[118]) * constants[0]))
        + algebraic[31] * (constants[110] / constants[118])
    ) - algebraic[180]
    algebraic[188] = (
        (
            ((1.00000 * 60.0000) * algebraic[182])
            * (
                1.00000
                + 1.00000 / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
            )
        )
        / (1.00000 + np.power(0.750000 / states[6], 8.00000))
        if algebraic[182] > 0.0
        else 0.00000
    )
    algebraic[42] = (
        (1.00000 * 20.0000)
        * (
            1.00000
            + 1.00000 / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
        )
    ) / (1.00000 + np.power(0.500000 / states[6], 8.00000))
    algebraic[44] = 0.00100000 if algebraic[42] < 0.00100000 else algebraic[42]
    rates[1] = (algebraic[188] - states[1]) / algebraic[44]
    algebraic[0] = (states[4] - states[5]) / constants[4]
    algebraic[183] = (
        algebraic[0] + states[2] * (constants[111] / constants[114])
    ) - algebraic[181] * (constants[117] / constants[114])
    algebraic[189] = (
        (
            1.00000
            * (
                (250.000 * algebraic[183])
                * (
                    1.00000
                    + 1.00000
                    / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
                )
            )
        )
        / (1.00000 + np.power(0.750000 / states[3], 8.00000))
        if algebraic[183] > 0.0
        else 0.00000
    )
    algebraic[45] = (
        50.0000
        * (
            1.00000
            + 1.00000 / (1.00000 + np.power(constants[1] / algebraic[38], 8.00000))
        )
    ) / (1.00000 + np.power(0.500000 / states[3], 8.00000))
    algebraic[46] = 0.00100000 if algebraic[45] < 0.00100000 else algebraic[45]
    rates[2] = (algebraic[189] - states[2]) / algebraic[46]
    algebraic[184] = 1.00000 / (
        (
            1.00000
            + (constants[66] * constants[68])
            / ((constants[68] + states[4]) * (constants[68] + states[4]))
        )
        + (constants[71] * constants[70])
        / ((constants[70] + states[4]) * (constants[70] + states[4]))
    )
    algebraic[190] = algebraic[184] * (
        (
            (
                (
                    -((algebraic[161] + algebraic[61]) - 2.00000 * algebraic[101])
                    * constants[105]
                )
                / (((2.00000 * 2.00000) * constants[0]) * constants[113])
                - (algebraic[179] * constants[116]) / constants[113]
            )
            + (algebraic[180] * constants[118]) / constants[113]
        )
        - algebraic[0]
    )
    rates[4] = algebraic[190]
    algebraic[185] = 1.00000 / (
        (
            1.00000
            + (constants[66] * constants[68])
            / ((constants[68] + states[5]) * (constants[68] + states[5]))
        )
        + (constants[71] * constants[70])
        / ((constants[70] + states[5]) * (constants[70] + states[5]))
    )
    algebraic[191] = algebraic[185] * (
        (
            states[2] * (constants[111] / constants[114])
            + (algebraic[0] * constants[114]) / constants[113]
        )
        - (algebraic[181] * constants[117]) / constants[114]
    )
    rates[5] = algebraic[191]
    algebraic[187] = 1.00000 / (
        (
            1.00000
            + (constants[63] * constants[65])
            / ((constants[65] + states[7]) * (constants[65] + states[7]))
        )
        + (constants[62] * constants[64])
        / ((constants[64] + states[7]) * (constants[64] + states[7]))
    )
    algebraic[192] = algebraic[187] * (
        (
            (
                -(algebraic[159] + algebraic[57] - 2.00000 * algebraic[131])
                * constants[105]
            )
            / (((2.00000 * 2.00000) * constants[0]) * constants[118])
            + (algebraic[31] * constants[110]) / constants[118]
        )
        - algebraic[180]
    )
    rates[7] = algebraic[192]
    return rates
