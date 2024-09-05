import pandas as pd
import numpy as np
import calliope
from typing import Union, List
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

        # # fmt: off
        # demands = Demand(self.cea_config, self.locator, self.district)
        # PVs = PV(self.cea_config, self.locator, self.district)
        # PVTs = PVT(self.cea_config, self.locator, self.district)
        # SCETs = SC(self.cea_config, self.locator, "ET", self.district)
        # SCFPs = SC(self.cea_config, self.locator, "FP", self.district)
        # COPs = COP(self.cea_config, self.locator, self.district)
        # # fmt: on
        self.timeseries = TimeSeries(self.district)

        if self.my_config.flatten_spike:
            self.timeseries.demand.flatten_spikes(
                percentile=self.my_config.flatten_spike_percentile,
                is_positive=False,
            )

        # self.dict_timeseries_df: dict[str, pd.DataFrame] = {
        #     **demands.result_dict,
        #     **PVs.result_dict,
        #     **PVTs.result_dict,
        #     **SCETs.result_dict,
        #     **SCFPs.result_dict,
        #     **COPs.cop_dict,
        # }

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
            # check if the emission constraint is already in the config, if so, delete it

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
            # so we should locate cost-optimal at epsilon+3
            idx_cost = epsilon + 3
            print(
                f"""
                  Approaching tip of Pareto Front. 
                  Adding two more epsilon cuts close to the ends. 
                  Original: {epsilon} , now: {epsilon+2}"""
            )
        else:
            idx_cost = epsilon + 1

        self.store_folder = store_folder
        df_pareto = pd.DataFrame(
            columns=["cost", "emission"], index=range(idx_cost + 1)
        )
        # read yaml file and get the list of technologies
        tech_list: List[str] = list(self.district.tech_dict.techs.keys())
        # calliope does not define the type of the return value, so it's ignored
        df_tech_cap_pareto = pd.DataFrame(columns=tech_list, index=range(idx_cost + 1))
        df_tech_cap_pareto.fillna(0, inplace=True)
        # first get the emission-optimal solution
        model_emission = self.get_calliope_model(
            to_lp=to_lp, to_yaml=to_yaml, obj="emission"
        )
        model_emission.run()
        if to_nc:
            model_emission.to_netcdf(
                f"{self.store_folder}/{self.district.buildings[0].name}_emission.nc"  # TODO: think of better naming convention
            )
        print("optimization for emission is done")
        # store the cost and emission in df_pareto
        df_emission = (
            model_emission.get_formatted_array("cost")
            .sel(
                locs=self.district.buildings[0].name
            )  # TODO: think of better naming convention
            .to_pandas()
            .transpose()
            .sum(axis=0)
        )
        # add the cost and emission to df_pareto
        df_pareto.loc[0] = [df_emission["monetary"], df_emission["co2"]]
        # store the technology capacities in df_tech_cap_pareto
        df_tech_cap_pareto.loc[0] = (
            model_emission.get_formatted_array("energy_cap").to_pandas().iloc[0]
        )

        # then get the cost-optimal solution
        model_cost = self.get_calliope_model(to_lp=to_lp, to_yaml=to_yaml, obj="cost")
        # run model cost, and find both cost and emission of this result
        model_cost.run()
        if to_nc:
            model_cost.to_netcdf(
                f"{self.store_folder}/{self.district.buildings[0].name}_cost.nc"  # TODO: think of better naming convention
            )
        print("optimization for cost is done")
        # store the cost and emission in df_pareto
        # add epsilon name as row index, start with epsilon_0
        df_cost = (
            model_cost.get_formatted_array("cost")
            .sel(
                locs=self.district.buildings[0].name
            )  # TODO: think of better naming convention
            .to_pandas()
            .transpose()
            .sum(axis=0)
        )  # first column co2, second column monetary

        df_pareto.loc[idx_cost] = [df_cost["monetary"], df_cost["co2"]]
        df_tech_cap_pareto.loc[idx_cost] = (
            model_cost.get_formatted_array("energy_cap").to_pandas().iloc[0]
        )
        # then get the epsilon-optimal solution
        # first find out min and max emission, and epsilon emissions are evenly distributed between them
        # if cost and emission optimal have the same emission, then there's no pareto front
        if df_cost["co2"] <= df_emission["co2"]:
            print(
                f"cost-optimal and emission-optimal of building {self.district.buildings[0].name} have the same emission, no pareto front"  # TODO: think of better naming convention
            )
            self.df_pareto = df_pareto
            self.df_tech_cap_pareto = df_tech_cap_pareto
            print(df_pareto)
        else:
            emission_max = df_cost["co2"]
            emission_min = df_emission["co2"]
            emission_array = np.linspace(emission_min, emission_max, epsilon + 2)
            epsilon_list = list(emission_array[1:-1])
            # calculate the interval between two emissions
            # for each epsilon, get the cost-optimal solution under a maximal emission constraint
            if approach_tip:
                del_emission_begin = np.diff(emission_array)[0] * approach_percentile
                del_emission_end = np.diff(emission_array)[-1] * approach_percentile
                epsilon_list = (
                    [emission_min + del_emission_begin]
                    + epsilon_list
                    + [emission_max - del_emission_end]
                )
            print(
                f"Maximal emission: {emission_max}, minimal emission: {emission_min}, number of epsilon cuts: {idx_cost-1}"
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
                # store the cost and emission in df_pareto
                df_epsilon = (
                    model_epsilon.get_formatted_array("cost")
                    .sel(
                        locs=self.district.buildings[0].name
                    )  # TODO: think of better naming convention
                    .to_pandas()
                    .transpose()
                    .sum(axis=0)
                )
                # add the cost and emission to df_pareto
                df_pareto.loc[n_epsilon] = [df_epsilon["monetary"], df_epsilon["co2"]]
                # store the technology capacities in df_tech_cap_pareto
                df_tech_cap_pareto.loc[n_epsilon] = (
                    model_epsilon.get_formatted_array("energy_cap").to_pandas().iloc[0]
                )

            df_pareto = df_pareto.astype({"cost": float, "emission": float})
            print(
                "Pareto front for building "
                + self.district.buildings[
                    0
                ].name  # TODO: think of better naming convention
                + " is done. First row is emission-optimal, last row is cost-optimal."
            )
            # show the pareto front
            print(df_pareto)
            self.df_pareto = df_pareto
            self.df_tech_cap_pareto = df_tech_cap_pareto
