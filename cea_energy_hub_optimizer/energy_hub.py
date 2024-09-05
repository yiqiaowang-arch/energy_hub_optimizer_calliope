import pandas as pd
import numpy as np
import calliope
from typing import Union, List, cast

from pandas.core.tools.datetimes import Scalar
from cea_energy_hub_optimizer.my_config import MyConfig
from cea_energy_hub_optimizer.timeseries import TimeSeries
from cea_energy_hub_optimizer.district import District

""" A class definition of a single-building energy hub optimization model.

set an energy hub, with the following attributes:
- name:                 str                             name of the building
- locator:              InputLocator                    locator object has multiple methods that helps with locating certain file paths
- yaml_path:            str                             path to the yaml file that contains the energy hub configuration
- config:               Configuration                   configuration object that contains the user's input in plugin.config
- emission_type:        str                             type of emission system, either 'HVAC_HEATING_AS1' or 'HVAC_HEATING_AS4'
- area:                 float                           area of the building
- location:             dict                            location of the building, with keys 'lat' and 'lon'
- district.tech_dict:   TechAttrDict                    calliope configuration object from the yaml file
- dict_timeseries_df:   dict: [str, pd.DataFrame]       dictionary of timeseries dataframes, with keys 'demand_electricity', 
                                                        'demand_space_heating', 'demand_hot_water', 'demand_space_cooling', 
                                                        'supply_PV', 'supply_PVT_e', 'supply_PVT_h', 'supply_SCFP', 'supply_SCET'
"""


class EnergyHub:
    def __init__(
        self,
        buildings: Union[str, List[str]],
        calliope_yaml_path: str,
    ):
        """__init__ initializes the EnergyHub class with the given parameters.

        This function initializes a single-building energy hub optimization model.

        Args:
            name (str): name of the building that follows CEA convention
            locator (InputLocator): the locator object that helps locate different files
            calliope_yaml_path (str): the path to the yaml file that contains the calliope energy hub configuration
            config (Configuration): the configuration object that contains user inputs in plugin.config
        """
        self.my_config = MyConfig()
        self.district = District(buildings, calliope_yaml_path)
        self.district.tech_dict.add_locations_from_district(self.district)
        self.district.tech_dict.set_temporal_resolution(
            self.my_config.temporal_resolution
        )
        self.district.tech_dict.set_solver(self.my_config.solver)
        self.district.tech_dict.set_wood_availaility(extra_area=400, energy_density=0.5)
        self.district.tech_dict.select_evaluated_demand()
        self.district.tech_dict.select_evaluated_solar_supply()
        if self.my_config.use_temperature_sensitive_cop:
            self.district.tech_dict.set_cop_timeseries()

        self.timeseries = TimeSeries(self.district)

        if self.my_config.flatten_spike:
            self.timeseries.demand.flatten_spikes(
                percentile=self.my_config.flatten_spike_percentile,
                is_positive=False,
            )

    def get_calliope_model(
        self,
        to_lp: bool = False,
        to_yaml: bool = False,
        obj: str = "cost",
        emission_constraint: Union[None, float] = None,
    ) -> calliope.Model:
        """
        Description:
        This function gets building parameters and read the scenario files to create a calliope model for the building.

        Input:
        building_status:            pd.Series, the status of the building, including is_new, is_rebuilt, already_GSHP, already_ASHP, is_disheat
        flatten_spikes:             bool, if True, flatten the demand spikes
        flatten_percentile:         float, the percentile to flatten the spikes
        to_lp:                      bool, if True, store the model in lp format
        to_yaml:                    bool, if True, store the model in yaml format
        obj:                        str, the objective function, either 'cost' or 'emission'
        emission_constraint:        float, the emission constraint

        Return:
        Model:                      calliope.Model, the optimized model
        """
        # if emission constraint is not None, add it to the self.district.tech_dict
        if emission_constraint is None:
            if bool(
                self.district.tech_dict.get_global_max_co2()
            ):  # if exists, delete it
                self.district.tech_dict.set_global_max_co2(None)
        else:
            self.district.tech_dict.set_global_max_co2(emission_constraint)

        self.district.tech_dict.set_objective(obj)
        model = calliope.Model(
            self.district.tech_dict,
            timeseries_dataframes=self.timeseries.timeseries_dict,
        )
        if to_lp:
            model.to_lp(
                f"{self.store_folder}/{self.district.buildings[0].name}.lp"  # TODO: think of better naming convention
            )
        if to_yaml:
            model.save_commented_model_yaml(
                f"{self.store_folder}/{self.district.buildings[0].name}.yaml"  # TODO: think of better naming convention
            )
        return model

    def get_pareto_front(
        self,
        epsilon: int,
        store_folder: str,
        approach_tip: bool = False,
        approach_percentile: float = 0.01,
        to_lp: bool = False,
        to_yaml: bool = False,
        to_nc: bool = False,
    ):
        """
        Finds the pareto front of one building regarding cost and emission.

        This function finds the pareto front of one building regarding cost and emission. It follows the steps below:

        1. Prepare cost data (monetary: HSLU database; emission: KBOB, limitation: different database).
        2. Input the cost data into the calliope configuration.
        3. Define available technology.
        4. Solve for the emission-optimal solution and store the cost and emission in df_pareto.
        5. Solve for the cost-optimal solution and store the cost and emission in df_pareto.
        6. Read the number of epsilon cuts and evenly distribute the emissions between the cost-optimal and emission-optimal solutions.
        7. For each epsilon, solve for the epsilon-optimal solution and store the cost and emission in df_pareto.
        8. Return df_pareto, which contains two columns: cost and emission, along with the index of the number of epsilon cuts. The first row represents the emission-optimal solution, and the last row represents the cost-optimal solution.
        9. Return df_tech_cap_pareto, which contains the technology capacities of each solution.

        Args:
            epsilon (int): The number of epsilon cuts between the cost-optimal and emission-optimal solutions.
            building_name (str): The name of the building.
            building_scenario_folder (str): The folder that contains the building's scenario files.
            yaml_path (str): The path to the yaml file that contains the energy hub configuration.
            store_folder (str): The folder that stores the results.
            building_status (pd.Series): The status of the building, including is_new, is_rebuilt, already_GSHP, already_ASHP, is_disheat.
            flatten_spikes (bool): If True, flatten the demand spikes.
            flatten_percentile (float): The percentile to flatten the spikes.
            to_lp (bool): If True, store the model in lp format.
            to_yaml (bool): If True, store the model in yaml format.

        Returns:
            df_pareto (pd.DataFrame): The pareto front of the building, with cost and emission as columns.

            df_tech_cap_pareto (pd.DataFrame): The technology capacities of each solution.

        """
        calliope.set_log_verbosity(
            verbosity="error", include_solver_output=False, capture_warnings=False
        )
        if epsilon < 1:
            raise ValueError("There must be at least one epsilon cut!")

        if approach_tip:
            # if approach_tip is True, then there are in total epsilon+4 points in the pareto front:
            # emission-optimal, close-to-emission-optimal, epsilon points, close-to-cost-optimal, cost-optimal
            n_solution = epsilon + 4
            print(
                f"""
                  Approaching tip of Pareto Front. 
                  Adding two more epsilon cuts close to the ends. 
                  Original solutions: {epsilon+2} , now: {epsilon+4}"""
            )
        else:
            n_solution = epsilon + 2

        self.initialize_pareto_df(n_solution)
        self.store_folder = store_folder
        model_emission = self.get_calliope_model(
            to_lp=to_lp, to_yaml=to_yaml, obj="emission"
        )
        model_emission.run()
        if to_nc:
            model_emission.to_netcdf(
                f"{self.store_folder}/{self.district.buildings[0].name}_emission.nc"  # TODO: think of better naming convention
            )
        print("optimization for emission is done")
        self.get_cap_from_model(model_emission, 0)

        # then get the cost-optimal solution
        model_cost = self.get_calliope_model(to_lp=to_lp, to_yaml=to_yaml, obj="cost")
        # run model cost, and find both cost and emission of this result
        model_cost.run()
        if to_nc:
            model_cost.to_netcdf(
                f"{self.store_folder}/{self.district.buildings[0].name}_cost.nc"  # TODO: think of better naming convention
            )
        print("optimization for cost is done")

        self.get_cap_from_model(model_cost, n_solution - 1)
        emission_max: float = (
            self.df_pareto.loc[(slice(None), n_solution - 1), "emission"]
            .sum()
            .astype(float)
        )
        emission_min: float = (
            self.df_pareto.loc[(slice(None), 0), "emission"].sum().astype(float)
        )
        if emission_max <= emission_min:
            print(
                f"cost-optimal and emission-optimal of building {self.district.buildings[0].name} have the same emission, no pareto front"  # TODO: think of better naming convention
            )
            print(self.df_pareto.iloc[:, 0:2])
        else:
            epsilon_list = self.get_co2_epsilon_cut(
                emission_min, emission_max, epsilon, approach_tip, approach_percentile
            )

            for i, emission_constraint in enumerate(epsilon_list):
                n_epsilon = i + 1
                print(
                    f"starting epsilon {n_epsilon}, life-time emission smaller or equal to {emission_constraint} kgCO2"
                )
                model_epsilon = self.get_calliope_model(
                    to_lp=to_lp,
                    to_yaml=to_yaml,
                    obj="cost",
                    emission_constraint=emission_constraint,
                )
                model_epsilon.run()
                if to_nc:
                    model_epsilon.to_netcdf(
                        path=self.store_folder
                        + "/"
                        + self.district.buildings[
                            0
                        ].name  # TODO: think of better naming convention
                        + f"_epsilon_{n_epsilon}.nc"
                    )
                print(f"optimization at epsilon {n_epsilon} is done")
                self.get_cap_from_model(model_epsilon, n_epsilon)

            # df_pareto = df_pareto.astype({"cost": float, "emission": float})
            print(
                "Pareto front for building "
                + self.district.buildings[
                    0
                ].name  # TODO: think of better naming convention
                + " is done. First row is emission-optimal, last row is cost-optimal."
            )

            print(self.df_pareto.iloc[:, 0:2])

    def initialize_pareto_df(self, n_solution: int) -> None:
        multi_index = pd.MultiIndex.from_product(
            [self.district.buildings_names, range(n_solution)],
            names=["building", "pareto_index"],
        )
        self.df_pareto = pd.DataFrame(
            data=0.0,
            index=multi_index,
            columns=["cost", "emission"] + self.district.tech_list,
        )
        self.df_pareto.astype(float)

    def get_cap_from_model(self, model: calliope.Model, i_solution: int) -> None:
        """
        From every model, two dfs are generated:
        df_cost: [locs, costs] a df containing two columns: "monetary" (CHF) and "co2" (kg) costs, aggregated from all techs;
        df_energy_cap: [(locs, techs), energy_cap] a double-indexed df containing energy_cap (kW) of all techs.
        Then, they are stored in the df_pareto at the i_solution-th row.

        Example:
        df_cost:

            ```
            costs     co2  monetary
            locs
            B162298  3347      2012
            B162299  5000      2000
            ```
        df_energy_cap:

            ```
            techs        ASHP  DHDC_large_heat  ...  hot_water_pipe  cold_water_pipe
            locs                                ...
            B162298  2.155492              0.0  ...             0.0              0.0
            B162299  0.000000              1.0  ...             0.0              0.0
            ```

        :param model: calliope model after model.run(), which saves the model.results as a xarray dataset
        :type model: calliope.Model
        :param i_solution: indicates which row of the df_pareto to store the results
        :type i_solution: int
        """
        # fmt: off
        df_cost: pd.DataFrame = model.get_formatted_array("cost").sum("techs").to_pandas().transpose() # type: ignore
        df_energy_cap: pd.DataFrame = model.get_formatted_array("energy_cap").to_pandas() # type: ignore
        # fmt: on

        missing_column = [
            col for col in self.district.tech_list if col not in df_energy_cap.columns
        ]
        df_energy_cap[missing_column] = 0.0
        # reindex two dfs to make sure building sequences align with the df_pareto
        df_cost = df_cost.reindex(self.district.buildings_names, fill_value=0.0)
        df_energy_cap = df_energy_cap.reindex(
            self.district.buildings_names, fill_value=0.0
        )
        # add the cost and emission to df_pareto
        self.df_pareto.loc[(slice(None), i_solution), ("cost", "emission")] = (
            df_cost.loc[:, ["monetary", "co2"]].values
        )
        self.df_pareto.loc[
            (slice(None), i_solution), tuple(self.district.tech_list)
        ] = df_energy_cap.values

    def get_co2_epsilon_cut(
        self,
        emission_min: float,
        emission_max: float,
        n_epsilon: int,
        approach_tip: bool,
        approach_percentile: float,
    ):
        emission_array = np.linspace(emission_min, emission_max, n_epsilon + 2)
        epsilon_list = list(emission_array[1:-1])
        if approach_tip:
            del_emission_begin = np.diff(emission_array)[0] * approach_percentile
            del_emission_end = np.diff(emission_array)[-1] * approach_percentile
            epsilon_list = (
                [emission_min + del_emission_begin]
                + epsilon_list
                + [emission_max - del_emission_end]
            )
            n_epsilon += 2  # add two more epsilon cuts close to the ends
        print(
            f"Maximal emission: {emission_max}, minimal emission: {emission_min}, number of epsilon cuts in between: {n_epsilon + 2}"
        )
        return epsilon_list
