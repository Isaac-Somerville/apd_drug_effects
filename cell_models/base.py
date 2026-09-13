"""Base class for the single-cell ventricular action potential models.

Each species model is a system of ODEs describing the transmembrane potential
and the ionic concentrations and gating variables that drive it.  This class
holds everything the species have in common: integration, action-potential
feature extraction, feasibility checking, drug block, and saving/loading of
limit-cycle states.  Subclasses supply the model-specific parts by overriding
:meth:`initConsts`, :meth:`computeRates` and :meth:`computeAlgebraic`.

Key concepts
------------
**Limit cycle.** Paced at a fixed period, these models drift for thousands of
beats before settling into a repeating cycle.  Simulations therefore start from
a pre-computed *limit state* (see ``data/saved_states/``) rather than from the
published initial conditions.

**Time units.** Most models integrate in milliseconds; the guinea pig model
integrates in **seconds**.  ``time_unit`` handles the conversion for stimulus
period and duration, but note that AP features are reported in the model's own
units — guinea pig APDs come back in seconds and must be scaled by 1000 to be
comparable with the other species.

**Drug block.** :meth:`init_drug` attaches a :class:`~drug.Drug`
and stores per-channel multipliers in the last ``len(channel_list)`` entries of
``init_algebraic``; the subclass rate equations read those to scale the
corresponding maximal conductances.
"""

import json
import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

from drug import Drug
from paths import (
    AP_LIMITS_DIR,
    CONSTANT_BASE_VALUE_DIR,
    CONSTANT_IDX_DIR,
    LIMIT_CYCLE_TOLERANCES_DIR,
    MIN_STEPS_NEEDED,
    PARAMETER_CVS,
    SAVED_STATES,
)


class CardiacCellModel:
    def __init__(
        self,
        sizeStates,
        sizeAlgebraic,
        sizeConstants,
        species="",
        time_unit="ms",
        channel_list=["ICaL", "IK1", "IKs", "INa", "INaL", "Ito", "hERG"],
        default_t_max=300,
        state_idx_dict=None,
        rate_idx_dict=None,
        algebraic_idx_dict=None,
        ap_limits_dict=None,
        ode_hyperparameters={},
        num_cycles_feasible_ap=None,
        num_cycles_limit_state=None,
    ):
        """
        Args:
            sizeStates: Number of ODE state variables, plus one for drug
                concentration.
            sizeAlgebraic: Number of algebraic (derived) variables, plus one per
                channel for the drug multipliers.
            sizeConstants: Number of model constants.
            species: Species name. Also the key used to locate this model's
                metadata under ``data/model_constants/`` and
                ``data/ap_features/``, so it must match those filenames.
            time_unit: Unit the model integrates in — ``"ms"`` for every species
                except the guinea pig, which uses ``"s"``.
            channel_list: Channels this model represents, a subset of the seven
                in the CiPA dataset. Determines which drug blocks apply.
            default_t_max: Default simulation duration, in ``time_unit``.
            state_idx_dict: Maps names (``"cai"``, ``"nai"``, ``"ki"``,
                ``"cli"``) to indices into the state vector. ``"v"`` is always
                index 0. Concentrations absent from a model are simply omitted.
            rate_idx_dict: Maps names to indices into the rate vector;
                ``"dvdt"`` is always index 0.
            algebraic_idx_dict: Maps names to indices into the algebraic vector.
            ap_limits_dict: Feasibility bounds on AP features. If None, loaded
                from ``data/ap_features/limits/`` for this species and cycle
                count.
            ode_hyperparameters: Overrides passed to the SciPy integrator, e.g.
                ``{"max_step": 0.001}`` for the guinea pig's smaller time unit.
            num_cycles_feasible_ap: Cycle count whose AP limits to load.
                Defaults to ``num_cycles_limit_state``.
            num_cycles_limit_state: Cycles needed for this species to reach its
                limit cycle. Fixed per species, e.g. 20,000 for human and 1,000
                for rabbit.
        """
        self.sizeStates = sizeStates
        self.sizeAlgebraic = sizeAlgebraic
        self.sizeConstants = sizeConstants
        self.species = species
        self.state_idx_dict = state_idx_dict
        self.channel_list = channel_list
        self.time_unit = time_unit
        self.default_t_max = default_t_max
        self.ode_hyperparameters = ode_hyperparameters
        self.num_cycles_feasible_ap = (
            num_cycles_feasible_ap
            if num_cycles_feasible_ap is not None
            else num_cycles_limit_state
        )
        self.num_cycles_limit_state = num_cycles_limit_state

        self.constants = np.zeros(sizeConstants, dtype=np.float64)
        self.init_states = np.zeros(sizeStates, dtype=np.float64)
        self.init_algebraic = np.ones(sizeAlgebraic, dtype=np.float64)

        self.legend_constants = np.full(sizeConstants, "")
        self.legend_states = np.full(sizeStates, "")
        self.legend_algebraic = np.full(sizeAlgebraic, "")
        self.legend_rates = np.full(sizeStates, "")
        self.legend_voi = ""
        self.legend_dict = {
            "constants": self.legend_constants,
            "states": self.legend_states,
            "algebraic": self.legend_algebraic,
        }
        self.time_unit_conversion = {"ms": 1, "s": 1000, "min": 60000, "h": 3600000}

        default_state_idx_dict = {"v": 0}

        if state_idx_dict is None:
            self.state_idx_dict = default_state_idx_dict.copy()
        else:
            # Override defaults with provided subset
            self.state_idx_dict = {**default_state_idx_dict, **state_idx_dict}
            # print(self.state_idx_dict)

        default_rate_idx_dict = {"dvdt": 0}
        if rate_idx_dict is None:
            self.rate_idx_dict = default_rate_idx_dict.copy()
        else:
            # Override defaults with provided subset
            self.rate_idx_dict = {**default_rate_idx_dict, **rate_idx_dict}

        default_algebraic_idx_dict = {}

        if algebraic_idx_dict is None:
            self.algebraic_idx_dict = default_algebraic_idx_dict.copy()
        else:
            # Override defaults with provided subset
            self.algebraic_idx_dict = {
                **default_algebraic_idx_dict,
                **algebraic_idx_dict,
            }

        # Per-species metadata is looked up by species name rather than passed
        # in, so subclasses never carry a filesystem path.
        default_constant_idx_dict = {"stim_period": 0, "stim_start": 1, "stim_end": 2}
        with open(CONSTANT_IDX_DIR / f"{self.species}.json", "r") as f:
            self.constant_idx_dict = {**default_constant_idx_dict, **json.load(f)}

        # Reference values of the conductances that vary between subjects.
        with open(CONSTANT_BASE_VALUE_DIR / f"{self.species}.json", "r") as f:
            self.constant_base_value_dict = json.load(f)

        # Coefficients of variation used to sample virtual subjects.
        with open(PARAMETER_CVS, "r") as f:
            self.parameter_CVs = json.load(f)

        default_ap_limits_dict = {
            "APD_40": [85, 320],
            "APD_50": [110, 350],
            "APD_90": [180, 440],
            "tri_90_40": [50, 150],
            "dvdt_max": [100, 1000],
            "v_peak": [10, 55],
            "RMP": [-95, -80],
            "CTD_50": [120, 420],
            "CTD_90": [220, 785],
        }
        if ap_limits_dict is None:
            limits_path = (
                AP_LIMITS_DIR
                / f"{self.species}_{self.num_cycles_feasible_ap}_cycles_AP_limits.json"
            )
            with open(limits_path, "r") as f:
                self.ap_limits_dict = json.load(f)
        else:
            # Override defaults with provided subset
            self.ap_limits_dict = {**default_ap_limits_dict, **ap_limits_dict}

        # Integration steps per cycle below which APD is not resolved accurately.
        with open(MIN_STEPS_NEEDED, "r") as f:
            self.min_steps_needed = json.load(f).get(self.species, 4000)

        # Per-feature tolerances defining convergence to the limit cycle. Absent
        # for cycle counts that were never characterised, in which case
        # convergence simply is not checked.
        tol_path = (
            LIMIT_CYCLE_TOLERANCES_DIR
            / f"{self.species}_{self.num_cycles_limit_state}"
            "_cycles_limit_cycle_tolerances.json"
        )
        try:
            with open(tol_path, "r") as f:
                self.limit_cycle_tol_dict = json.load(f)
        except FileNotFoundError:
            self.limit_cycle_tol_dict = {}

    def createLegends(self):
        """Set variable names for legends (to be overridden in subclasses)."""
        pass

    def initConsts(self):
        """Set default constants (to be overridden in subclasses)."""
        pass

    def computeRates(self, voi, states, constants, drug_multipliers=None):
        """Compute rate of change for each state variable."""
        pass

    def computeAlgebraic(self, voi, states, constants, drug_multipliers=None):
        """Compute algebraic variables based on states."""
        pass

    def custom_piecewise(self, cases):
        """Compute result of a piecewise function"""
        return np.select(cases[0::2], cases[1::2])

    def perturb_constant(self, reference_value, perturbation=None, cv=None):
        if perturbation is None or (
            reference_value == 0.0 and perturbation == "lognormal"
        ):
            return reference_value
        elif perturbation == "normal":
            scale = abs(reference_value) * cv
            if cv is None:
                raise ValueError("cv must be provided for normal perturbation!")
            return np.random.normal(loc=reference_value, scale=scale)
        elif perturbation == "lognormal":
            if cv is None:
                raise ValueError("cv must be provided for lognormal perturbation!")
            sign = -1 if reference_value < 0 else 1
            mu = np.log(
                reference_value**2
                / np.sqrt(cv**2 * reference_value**2 + reference_value**2)
            )
            sigma = np.sqrt(np.log(cv**2 + 1))
            return sign * np.random.lognormal(mean=mu, sigma=sigma)

    def solve(
        self,
        init_states=None,
        save_gradients=True,
        t_start=0,
        t_max=1000,
        num_steps=5000,
        solver="lsoda",
        method="bdf",
        atol=1e-06,
        rtol=1e-06,
        max_step=1,
        nsteps=500,
        stim_period=1000,
        num_ap_checks=10,
        stop_on_bad_AP=True,
        stop_on_good_AP=False,
        save_states=True,
        save_ap_measures=False,
        plot_apd=False,
        file_path=None,
        ap_measures_file_name=None,
        states_file_name=None,
    ):
        """Integrate the model, optionally checking AP feasibility as it goes.

        Args:
            init_states: Starting state vector. Defaults to ``self.init_states``,
                which for a converged run should have been loaded from a saved
                limit state.
            save_gradients: Record the rate vector at every step. Required for
                AP feature extraction, since APD is measured from max dV/dt.
            t_start, t_max: Simulation window, in **milliseconds** regardless of
                the model's own ``time_unit`` — both are converted internally.
            num_steps: Total integration steps. Should be at least
                ``self.min_steps_needed`` per cycle for accurate APD.
            solver, method, atol, rtol, max_step, nsteps: Passed to
                :class:`scipy.integrate.ode`. ``method`` applies to ``vode`` only.
            stim_period: Pacing period in milliseconds.
            num_ap_checks: How many times during the run to extract AP features
                and test feasibility. 0 disables checking, which is much faster.
            stop_on_bad_AP: Abort (returning all-None) at the first infeasible AP.
            stop_on_good_AP: Stop at the first feasible AP — used to find the
                earliest cycle at which a subject becomes physiological.
            save_states: Write the final state to ``states_file_name``.
            save_ap_measures: Write every checked AP's features to
                ``ap_measures_file_name``.
            plot_apd: Plot each checked AP. Interactive; for debugging only.
            file_path: Directory for the two output files. Defaults to
                ``data/saved_states/``.
            ap_measures_file_name, states_file_name: Filenames within
                ``file_path``. If None, that output is skipped.

        Returns:
            ``(voi, states, algebraic, rates)`` where ``voi`` is the time vector
            and the rest are arrays with one row per variable. Returns
            ``(None, None, None, None)`` if the solver fails, a non-finite value
            appears, or an infeasible AP is hit with ``stop_on_bad_AP``.
        """
        if file_path is None:
            file_path = f"{SAVED_STATES}/"

        time_unit_conversion = self.time_unit_conversion[self.time_unit]
        # print(f"Time unit conversion: {time_unit_conversion}")

        t_start = t_start / time_unit_conversion
        t_max = t_max / time_unit_conversion

        voi = np.linspace(t_start, t_max, num_steps)
        stim_period = stim_period / time_unit_conversion
        self.set_constants(
            constants=[stim_period],
            constant_idxs=[self.constant_idx_dict["stim_period"]],
        )
        # print(f"Stimulus period: {stim_period} {self.time_unit}")
        # print(f"Total number of steps: {num_steps}")
        # print(f"Start time: {t_start}{self.time_unit}")
        # print(f"End time: {t_max}{self.time_unit}\n")
        num_steps_per_cycle = int(stim_period * (num_steps / (t_max - t_start)))
        # print(f"Number of steps per cycle: {num_steps_per_cycle}")
        if num_ap_checks > 0:
            check_every_n_steps = int(len(voi) / num_ap_checks)
        else:
            check_every_n_steps = math.inf
        # print(f"Check every {check_every_n_steps} steps")
        num_cycles = int((t_max - t_start) / stim_period)
        # print(f"Number of cycles: {num_cycles}\n")

        all_ap_measures_dict = {
            "APD_40": [],
            "APD_50": [],
            "APD_90": [],
            "tri_90_40": [],
            "dvdt_max": [],
            "v_peak": [],
            "RMP": [],
            "CTD_50": [],
            "CTD_90": [],
        }

        if save_gradients:
            rates_over_time = np.array([[0.0] * len(voi)] * self.sizeStates)
        else:
            rates_over_time = None

        states_over_time = np.array([[0.0] * len(voi)] * self.sizeStates)
        if init_states is not None:
            states_over_time[:, 0] = init_states
        else:
            states_over_time[:, 0] = self.init_states
            init_states = self.init_states

        algebraic_over_time = np.array([[0.0] * len(voi)] * self.sizeAlgebraic)

        r = ode(self.computeRates)
        if solver == "vode":
            r.set_integrator(
                solver,
                method=method,
                atol=atol,
                rtol=rtol,
                max_step=max_step,
                nsteps=nsteps,
            )
        else:
            r.set_integrator(
                solver, atol=atol, rtol=rtol, max_step=max_step, nsteps=nsteps
            )
        r.set_initial_value(init_states, voi[0])
        r.set_f_params(np.array(self.constants, dtype=np.float64))

        for i, t in enumerate(voi):  # i from 0 to num_steps - 1
            if i == 0:  # skip first time point, just save gradient
                if save_gradients:
                    rates_over_time[:, i] = self.computeRates(
                        t,
                        np.array(states_over_time[:, i], dtype=np.float64),
                        np.array(self.constants, dtype=np.float64),
                    )
                continue
            if r.successful():
                try:
                    r.integrate(t)
                except ZeroDivisionError:
                    print(f"ZeroDivisionError at time {t}, stopping simulation.")
                    return None, None, None, None
                states_over_time[:, i] = r.y
                if not np.all(np.isfinite(states_over_time[:, i])):
                    print(f"Non-finite value at time {t}: {states_over_time[:, i]}")
                    return None, None, None, None
                if save_gradients:
                    rates_over_time[:, i] = self.computeRates(
                        t,
                        np.array(states_over_time[:, i], dtype=np.float64),
                        np.array(self.constants, dtype=np.float64),
                    )
                if num_ap_checks > 0 and (
                    (i + 1) % check_every_n_steps == 0 or i + 1 == len(voi)
                ):
                    # print(i - num_steps_per_cycle + 1)
                    # print(i + 1)
                    # print(voi[i - num_steps_per_cycle + 1 : i + 1])

                    algebraic_over_time[:, i + 1 - check_every_n_steps : i + 1] = (
                        self.computeAlgebraic(
                            voi[i + 1 - check_every_n_steps : i + 1],
                            states_over_time[:, i + 1 - check_every_n_steps : i + 1],
                            self.constants,
                        )
                    )

                    if plot_apd:
                        self.plot_apd(
                            voi=voi,
                            states=states_over_time,
                            rates=rates_over_time,
                            time_idx_range=[i - num_steps_per_cycle + 1, i + 1],
                        )
                        # self.plot_var(
                        #     voi = voi,
                        #     states = states_over_time,
                        #     algebraic= algebraic_over_time,
                        #     rates = rates_over_time,
                        #     rate_item = 'dvdt',
                        #     time_idx_range = [i - num_steps_per_cycle + 1, i + 1]
                        # )

                    print(f"Calculating AP features at time {t}{self.time_unit}")
                    ap_measures_dict = self.calculate_ap_features(
                        voi=voi,
                        states=states_over_time,
                        rates=rates_over_time,
                        algebraic=algebraic_over_time,
                        time_idx_range=[i - num_steps_per_cycle + 1, i + 1],
                    )

                    if save_ap_measures:
                        for key in all_ap_measures_dict.keys():
                            all_ap_measures_dict[key].append(ap_measures_dict[key])

                    print(
                        f"Checking feasibility of AP features at time {t}{self.time_unit}"
                    )
                    if not self.check_ap_feasibility(ap_measures_dict=ap_measures_dict):
                        # If AP features are not feasible, print a message and either break or continue
                        if stop_on_bad_AP:
                            print(
                                f"AP features not feasible on cycle {i // num_steps_per_cycle}, stopping simulation."
                            )
                            return None, None, None, None
                        else:
                            print(
                                f"AP features not feasible on cycle {i // num_steps_per_cycle}, continuing simulation."
                            )
                    else:
                        if stop_on_good_AP:
                            print(
                                f"AP features feasible on cycle {i // num_steps_per_cycle}, stopping simulation."
                            )
                            break
                        print(
                            f"AP features feasible on cycle {i // num_steps_per_cycle}, continuing simulation."
                        )
            # break if ODE solver is not successful
            else:
                print(f"ODE solver not successful at time {t}, stopping simulation.")
                return None, None, None, None

        if num_ap_checks == 0:
            algebraic_over_time = self.computeAlgebraic(
                voi, states_over_time, self.constants
            )

        if save_states:
            if states_file_name is None:
                print("No states_file_name provided, states will not be saved.")
            else:
                # Save the states to a JSON file
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                # states_file_name = f"{self.species}_{self.drug.name if self.drug is not None else 'baseline'}_limit_state_{num_cycles}_cycles.json"
                self.save_model(
                    voi[-1],
                    states_over_time[:, -1],
                    algebraic_over_time[:, -1],
                    self.constants,
                    file_path,
                    states_file_name,
                )

        if save_ap_measures:
            if ap_measures_file_name is None:
                print(
                    "No ap_measures_file_name provided, AP measures will not be saved."
                )
            else:
                # Save the AP measures to a JSON file
                # ap_measures_file_name = f"AP_Measures/{self.species}_{self.drug.name if self.drug is not None else 'baseline'}_AP_{num_cycles}_cycles.json"
                with open(file_path + ap_measures_file_name, "w") as f:
                    json.dump(all_ap_measures_dict, f, indent=4)

        return voi, states_over_time, algebraic_over_time, rates_over_time

    def detect_apd(self, time, potential, dvdt, percent_repolarisation=90):
        """Finds APD_XX for one beat starting at time apd_start_time"""
        dvdt = list(dvdt)[:-1]
        time = time[:-1]
        potential = potential[
            :-1
        ]  # remove last point since often has max dvdt from next beat

        max_dvdt_idx = dvdt.index(max(dvdt))
        apd_start_time = time[max_dvdt_idx]
        # print(max_dvdt_idx)
        # print(apd_start_time)

        potential = list(potential)
        repolarisation_frac = (
            max(potential)
            - (max(potential) - min(potential)) * percent_repolarisation / 100
        )
        max_potential_idx = potential.index(max(potential))
        # print(max_potential_idx)
        # print(time[max_potential_idx])
        # print("\n")
        if max_potential_idx < max_dvdt_idx:
            print("APD cannot be calculated, APD start time is after AP peak!")
            return (apd_start_time, None)
        # print(max_potential_idx)
        for i in range(max_potential_idx, len(potential)):
            if potential[i] <= repolarisation_frac:
                # print(apd_start_time)
                # print(time[i])
                return (apd_start_time, time[i] - apd_start_time)
        # print(f"{percent_repolarisation} percent repolarisation not reached!")
        return (apd_start_time, None)

    def detect_ctd(self, time, calcium, percent_decay=90):
        """Finds CTD_XX for one beat starting at time ctd_start_time"""
        # Normalise time to find stimulus start time (normed_time = 0)
        stim_start = (
            0
            if self.constant_idx_dict["stim_start"] is None
            else self.constants[self.constant_idx_dict["stim_start"]]
        )
        tmp = (time - stim_start) % self.constants[
            self.constant_idx_dict["stim_period"]
        ]
        normed_time = [int(i) for i in tmp]
        # print(normed_time[0], normed_time[-1])
        if 0 not in normed_time:
            ctd_start_idx = normed_time.index(min(normed_time))
        else:
            ctd_start_idx = normed_time.index(0)
        calcium = list(calcium)

        min_calcium = min(calcium)
        max_calcium = max(calcium)
        max_calcium_idx = calcium.index(max(calcium))

        resting_calcium_frac = (
            max_calcium - (max_calcium - min_calcium) * percent_decay / 100
        )
        if normed_time[max_calcium_idx] < 0:
            raise ValueError(
                "CTD cannot be calculated, CTD start time (stimulus time) is after calcium peak!"
            )
        for i in range(max_calcium_idx, len(calcium)):
            if calcium[i] <= resting_calcium_frac:
                return (time[ctd_start_idx], time[i] - time[ctd_start_idx])
        # print(f"{percent_decay} percent decay not reached!")
        return (time[ctd_start_idx], None)

    def plot_apd(
        self, voi, states, rates, percent_repolarisation=90, time_idx_range=None
    ):
        if time_idx_range is None:
            time_idx_range = [0, len(voi)]

        dvdt = list(rates[self.rate_idx_dict["dvdt"]])[
            time_idx_range[0] : time_idx_range[1]
        ]
        time = voi[time_idx_range[0] : time_idx_range[1]]
        potential = states[self.state_idx_dict["v"]][
            time_idx_range[0] : time_idx_range[1]
        ]

        apd_start_time, apd_frac = self.detect_apd(
            time, potential, dvdt, percent_repolarisation
        )
        print(f"APD{percent_repolarisation} = ", apd_frac)
        plt.plot(time, potential, label="Potential")
        if apd_frac is not None:
            plt.axvline(
                apd_frac + apd_start_time,
                ymin=min(potential),
                ymax=max(potential),
                color="red",
                label=f"APD_{percent_repolarisation} Endpoint",
            )
            plt.axvline(
                apd_start_time,
                ymin=min(potential),
                ymax=max(potential),
                color="red",
                linestyle="--",
                label="AP Startpoint",
            )
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Transmembrane Potential (mV)")
        plt.title(self.species + " Ventricular Transmembrane Potential")
        plt.legend()
        plt.show()

    def plot_ctd(
        self, voi, states=None, algebraic=None, percent_decay=90, time_idx_range=None
    ):
        if time_idx_range is None:
            time_idx_range = [0, len(voi)]

        time = voi[time_idx_range[0] : time_idx_range[1]]
        calcium = states[self.state_idx_dict["cai"]][
            time_idx_range[0] : time_idx_range[1]
        ]

        ctd_start_time, ctd_frac = self.detect_ctd(time, calcium, percent_decay)
        print(f"ctd{percent_decay} = ", ctd_frac)
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
        """Calculate AP features such as APD, CTD, and dvdt."""
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
        if "cai" in self.state_idx_dict:
            calcium = states[self.state_idx_dict["cai"]][
                time_idx_range[0] : time_idx_range[1]
            ]

        ap_measures_dict = {}
        for percent_repolarisation in [40, 50, 90]:
            ap_measures_dict[f"APD_{percent_repolarisation}"] = self.detect_apd(
                time, potential, dvdt, percent_repolarisation
            )[1]

        if (ap_measures_dict["APD_90"] is None) or (ap_measures_dict["APD_40"] is None):
            # print("APD_90 or APD_40 not reached, cannot calculate tri_90_40!")
            ap_measures_dict["tri_90_40"] = None
        else:
            ap_measures_dict["tri_90_40"] = (
                ap_measures_dict["APD_90"] - ap_measures_dict["APD_40"]
            )
        ap_measures_dict["dvdt_max"] = max(dvdt[:-1])
        ap_measures_dict["v_peak"] = max(potential)
        ap_measures_dict["RMP"] = min(potential)

        if "cai" in self.state_idx_dict:
            for percent_decay in [50, 90]:
                ap_measures_dict[f"CTD_{percent_decay}"] = self.detect_ctd(
                    time, calcium, percent_decay
                )[1]

        return ap_measures_dict

    def check_ap_feasibility(
        self,
        voi=None,
        states=None,
        rates=None,
        algebraic=None,
        ap_measures_dict=None,
        time_idx_range=None,
        return_dict=False,
    ):
        """Check if AP curve is feasible based on APD, potential derivative and calcium transient."""
        feasible_ap = True
        # print(ap_measures_dict is None)
        if ap_measures_dict is None:
            if voi is None and states is None:
                raise ValueError(
                    "Either (voi, states, rates, algebraic) or ap_measurement_dict must be supplied!"
                )
            ap_measures_dict = self.calculate_ap_features(
                voi, states, rates, algebraic, time_idx_range
            )

        # print(f"AP features dict: {ap_measures_dict}")
        # print(f"\nAP limits dict: {self.ap_limits_dict}")
        feasibility_dict = {}

        for key, value in ap_measures_dict.items():
            if value is None:
                # print(f"{key} not reached!")
                feasible_ap = False
                feasibility_dict[key] = False
            elif value < self.ap_limits_dict[key][0]:
                # print(f"{key} below lower limit!")
                feasible_ap = False
                feasibility_dict[key] = False
            elif value > self.ap_limits_dict[key][1]:
                # print(f"{key} above upper limit!")
                feasible_ap = False
                feasibility_dict[key] = False
            else:
                feasibility_dict[key] = True

        # print("AP Measures:", ap_measures_dict)
        if return_dict:
            return feasibility_dict
        else:
            return feasible_ap

    def init_drug(self, drug_name, drug_amount=None, drug_data=None):
        """Attach a drug at a fixed concentration.

        Stores the concentration in the last state and the per-channel
        multipliers in the last ``len(channel_list)`` algebraic variables, where
        the subclass rate equations read them.  Concentration is held constant
        for the whole simulation.

        Args:
            drug_name: Row label in the CiPA table.
            drug_amount: Concentration in nM. Defaults to the drug's free
                therapeutic concentration.
            drug_data: Optional replacement parameter table, passed through to
                :class:`~drug.Drug`.
        """
        self.drug = Drug(
            name=drug_name, drug_data=drug_data, channel_list=self.channel_list
        )

        if drug_amount is None:
            self.drug_amount = self.drug.c_therapeutic
            print("No drug_amount given, therapeutic concentration used as default!")
        else:
            self.drug_amount = drug_amount

        self.init_states[-1] = self.drug_amount
        num_channels = len(self.channel_list)
        self.init_algebraic[-num_channels:] = self.drug.calculate_all_inhibitions(
            concentrations=np.array([self.drug_amount]), return_multipliers=True
        ).reshape((num_channels,))

    def calculate_drug_multipliers(self, concentrations):
        """Per-channel conductance multipliers at each concentration.

        Returns an array of shape ``(n_channels, n_concentrations)``. With no
        drug attached, every multiplier is 1, which is only meaningful if the
        concentrations really are all zero.
        """
        if self.drug is None:
            if not np.all(concentrations == 0):
                raise ValueError(
                    "Drug must first be attached with init_drug unless "
                    "concentrations are all 0!"
                )
            return {
                channel: np.ones_like(concentrations)
                for channel in self.channel_list
            }
        return self.drug.calculate_all_inhibitions(
            concentrations=concentrations, return_multipliers=True
        )

    def plot_var(
        self,
        voi,
        states,
        algebraic,
        rates,
        state_item=None,
        algebraic_item=None,
        rate_item=None,
        time_idx_range=None,
        show_fig=True,
        save_fig=False,
        file_path=None,
        file_name=None,
    ):
        """Plot a variable against variable of integration"""
        if time_idx_range is None:
            time_idx_range = [0, len(voi)]
        time = voi[time_idx_range[0] : time_idx_range[1]]
        if state_item is not None:
            state_idx = self.state_idx_dict[state_item]
            # print(state_idxs)
            state_value = states[state_idx][time_idx_range[0] : time_idx_range[1]]
            # print(state_values)
            state_label = [self.legend_states[state_idx]]
            # print(state_labels)
            plt.plot(time, state_value, label=state_label)
        if algebraic_item is not None:
            algebraic_idx = self.algebraic_idx_dict[algebraic_item]
            # print(algebraic_idxs)
            algebraic_value = algebraic[algebraic_idx][
                time_idx_range[0] : time_idx_range[1]
            ]
            # print(algebraic_values)
            algebraic_label = [self.legend_algebraic[algebraic_idx]]
            # print(algebraic_labels)
            plt.plot(time, algebraic_value, label=algebraic_label)
        if rate_item is not None:
            rate_idx = self.rate_idx_dict[rate_item]
            # print(rate_idxs)
            rate_value = rates[rate_idx][time_idx_range[0] : time_idx_range[1]]
            # print(rate_values)
            rate_label = [self.legend_rates[rate_idx]]
            # print(rate_labels)
            plt.plot(time, rate_value, label=rate_label)
        plt.xlabel(self.legend_voi)
        plt.legend()
        plt.title(self.species + " Variables over Time")
        if save_fig:
            if file_path is None or file_name is None:
                raise ValueError(
                    "File path and file name must be provided to save figure!"
                )
            plt.savefig(file_path + file_name)
        if show_fig:
            plt.show()
        else:
            plt.close()

    def set_states(self, states, state_idxs=None):
        """Set initial states based on provided indices."""
        if state_idxs is None:
            state_idxs = list(range(len(states)))
        for i in range(len(state_idxs)):
            self.init_states[state_idxs[i]] = states[i]
        return

    def set_constants(self, constants, constant_idxs=None):
        """Set constants based on provided indices."""
        if constant_idxs is None:
            constant_idxs = list(range(len(constants)))
        for i in range(len(constant_idxs)):
            self.constants[constant_idxs[i]] = constants[i]
        return

    def set_algebraic(self, algebraic, algebraic_idxs=None):
        """Set initial algebraic variables based on provided indices."""
        if algebraic_idxs is None:
            algebraic_idxs = list(range(len(algebraic)))
        for i in range(len(algebraic_idxs)):
            self.init_algebraic[algebraic_idxs[i]] = algebraic[i]
        return

    def save_model(
        self, voi, states, algebraic, constants, file_path, file_name, seed=0
    ):
        """Save states, algebraic variables, and constants to a file."""

        def to_list(x):
            return x.tolist() if isinstance(x, np.ndarray) else x

        os.makedirs(file_path, exist_ok=True)

        with open(file_path + file_name, "w") as f:
            state_list = [to_list(states[i]) for i in range(len(states))]
            algebraic_list = [to_list(algebraic[i]) for i in range(len(algebraic))]
            constant_list = [to_list(constants[i]) for i in range(len(constants))]
            json.dump(
                {
                    "voi": to_list(voi),
                    "states": to_list(state_list),
                    "algebraic": to_list(algebraic_list),
                    "constants": to_list(constant_list),
                    "seed": seed,
                },
                f,
                indent=4,
            )

    def calculate_model_distance(self, model2=None, model2_constants=None):
        """Calculate the distance between two models' constants."""
        if model2 is None and model2_constants is None:
            raise ValueError("Either model2 or model2_constants must be provided!")
        non_zero_idxs = np.nonzero(self.constants)
        model1_constants = self.constants[non_zero_idxs]
        if model2 is not None:
            model2_constants = model2.constants[non_zero_idxs]
            return np.linalg.norm(
                (model1_constants - model2_constants) / model1_constants
            )
        else:
            model2_constants = np.array(model2_constants)[non_zero_idxs]
            return np.linalg.norm(
                (model1_constants - model2_constants) / model1_constants
            )

    def calculate_limit_state_features(
        self,
        voi,
        states_over_time,
        rates_over_time,
        algebraic_over_time,
        time_idx_range=None,
    ):
        """AP features plus the min and max of each ionic concentration.

        The concentration extrema are what limit-cycle convergence is judged on:
        they drift for far longer than APD does, so a model whose APD has
        stabilised may still be far from its limit cycle.
        """
        if time_idx_range is None:
            time_idx_range = [0, len(voi)]

        features_dict = self.calculate_ap_features(
            voi,
            states_over_time,
            rates_over_time,
            algebraic_over_time,
            time_idx_range=time_idx_range,
        )

        states = states_over_time[:, time_idx_range[0] : time_idx_range[1]]
        for state in self.state_idx_dict:
            if state != "v" and self.state_idx_dict[state] is not None:
                features_dict[f"{state}_min"] = min(states[self.state_idx_dict[state]])
                features_dict[f"{state}_max"] = max(states[self.state_idx_dict[state]])
        return features_dict
