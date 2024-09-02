import geopandas as gpd
import pandas as pd
import numpy as np
import calliope
from cea.inputlocator import InputLocator
from cea.config import Configuration
from cea_energy_hub_optimizer.timeseries import Demand, PV, PVT, SC, COP

""" A class definition of a single-building energy hub optimization model.

set an energy hub, with the following attributes:
- name:                 str                             name of the building
- locator:              InputLocator                    locator object has multiple methods that helps with locating certain file paths
- yaml_path:            str                             path to the yaml file that contains the energy hub configuration
- config:               Configuration        configuration object that contains the user's input in plugin.config
- emission_type:        str                             type of emission system, either 'HVAC_HEATING_AS1' or 'HVAC_HEATING_AS4'
- area:                 float                           area of the building
- location:             dict                            location of the building, with keys 'lat' and 'lon'
- calliope_config:      calliope.AttrDict               calliope configuration object from the yaml file
- dict_timeseries_df:   dict: [str, pd.DataFrame]       dictionary of timeseries dataframes, with keys 'demand_electricity', 
                                                        'demand_space_heating', 'demand_hot_water', 'demand_space_cooling', 
                                                        'supply_PV', 'supply_PVT_e', 'supply_PVT_h', 'supply_SCFP', 'supply_SCET'
"""


class EnergyHub:
    def __init__(
        self,
        name: str,
        locator: InputLocator,
        calliope_yaml_path: str,
        config: Configuration,
    ):
        """__init__ initializes the EnergyHub class with the given parameters.

        This function initializes a single-building energy hub optimization model.

        Args:
            name (str): name of the building that follows CEA convention
            locator (InputLocator): the locator object that helps locate different files
            calliope_yaml_path (str): the path to the yaml file that contains the calliope energy hub configuration
            config (Configuration): the configuration object that contains user inputs in plugin.config
        """

        self.name: str = name
        self.names = [
            name
        ]  # for compatibility between the single building model and the district model
        self.locator = locator
        # locator.scenario returns a str of the scenario path, which includes /inputs and /outputs
        self.calliope_config: calliope.AttrDict = calliope.AttrDict.from_yaml(
            calliope_yaml_path
        )
        self.config = config
        calliope.set_log_verbosity(
            verbosity="error", include_solver_output=False, capture_warnings=False
        )

        # get type of emission system
        # emission_dict = {'HVAC_HEATING_AS1': 80, # radiator, needs high supply temperature
        #                  'HVAC_HEATING_AS4': 45  # floor heating, needs low supply temperature
        #                  } # output temperature of the heating emission system
        air_conditioning_df: pd.DataFrame = gpd.read_file(
            self.locator.get_building_air_conditioning(), ignore_geometry=True
        )
        air_conditioning_df.set_index(keys="Name", inplace=True)
        self.emission_type: str = str(air_conditioning_df.loc[self.name, "type_hs"])
        # self.emission_temp: int = emission_dict[self.emission_type]

        # get building area
        zone: gpd.GeoDataFrame = gpd.read_file(self.locator.get_zone_geometry())
        zone.index = zone["Name"]
        self.area: float = zone.loc[self.name, "geometry"].area
        self.location: dict[str, float] = {
            "lat": float(zone.loc[self.name, "geometry"].centroid.y),
            "lon": float(zone.loc[self.name, "geometry"].centroid.x),
        }

        # set temporal resolution
        self.calliope_config.set_key(
            key="model.time.function_options.resolution",
            value=self.config.energy_hub_optimizer.temporal_resolution,
        )

        building_sub_dict_temp: calliope.AttrDict = self.calliope_config[
            "locations"
        ].pop("Building")
        self.calliope_config["locations"][self.name] = building_sub_dict_temp
        del building_sub_dict_temp

        # set solver
        self.calliope_config.set_key(
            key="run.solver", value=self.config.energy_hub_optimizer.solver
        )
        # constarin wood supply to 0.5kWh/m2 of the building area + 400m2 surroundings
        self.calliope_config.set_key(
            key=f"locations.{self.name}.techs.wood_supply.constraints.energy_cap_max",
            value=(self.area + 400) * 0.5 * 0.001,
        )

        demands = Demand(self.config, self.locator, self.names)
        PVs = PV(self.config, self.locator, self.names, {self.name: self.area})
        PVTs = PVT(self.config, self.locator, self.names, {self.name: self.area})
        SCETs = SC(self.config, self.locator, self.names, "ET", {self.name: self.area})
        SCFPs = SC(self.config, self.locator, self.names, "FP", {self.name: self.area})
        COPs = COP(self.config, self.locator, self.names)
        # divide the supplies with self.area

        self.dict_timeseries_df: dict[str, pd.DataFrame] = {
            **demands.result_dict,
            **PVs.result_dict,
            **PVTs.result_dict,
            **SCETs.result_dict,
            **SCFPs.result_dict,
            **COPs.cop_dict,
        }
        if self.config.energy_hub_optimizer.flatten_spike:
            for key in [
                "demand_electricity",
                "demand_space_heating",
                "demand_hot_water",
                "demand_space_cooling",
            ]:
                self.dict_timeseries_df[key] = self.flattenSpikes(
                    df=self.dict_timeseries_df[key],
                    percentile=self.config.energy_hub_optimizer.flatten_spike_percentile,
                )

        if self.config.energy_hub_optimizer.temperature_sensitive_cop:
            print(
                "temperature sensitive COP is enabled. Getting COP timeseries from outdoor air temperature."
            )
            self.calliope_config.set_key(
                key="techs.ASHP.constraints.carrier_ratios.carrier_out.DHW",
                value="df=cop_dhw",
            )
            self.calliope_config.set_key(
                key="techs.ASHP.constraints.carrier_ratios.carrier_out.cooling",
                value="df=cop_sc",
            )

    def getBuildingModel(
        self, to_lp=False, to_yaml=False, obj="cost", emission_constraint=None
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

        # modify the self.calliope_config to match the building's status
        self.calliope_config.set_key(
            key=f"locations.{self.name}.available_area", value=self.area
        )
        print(
            "the area of building "
            + self.name
            + " is "
            + str(round(self.area, 1))
            + " m2"
        )

        # if emission constraint is not None, add it to the self.calliope_config
        if emission_constraint is not None:
            self.calliope_config.set_key(
                key="group_constraints.systemwide_co2_cap.cost_max.co2",
                value=emission_constraint,
            )
        else:
            # check if the emission constraint is already in the config, if so, delete it
            if bool(
                self.calliope_config.get_key(
                    "group_constraints.systemwide_co2_cap.cost_max.co2"
                )
            ):
                self.calliope_config.set_key(
                    key="group_constraints.systemwide_co2_cap.cost_max.co2", value=None
                )

        # if obj is cost, set the objective to be cost; if obj is emission, set the objective to be emission
        if obj == "cost":
            self.calliope_config.set_key(
                key="run.objective_options.cost_class.monetary", value=1
            )
            self.calliope_config.set_key(
                key="run.objective_options.cost_class.co2", value=0
            )
        elif obj == "emission":
            self.calliope_config.set_key(
                key="run.objective_options.cost_class.monetary", value=0
            )
            self.calliope_config.set_key(
                key="run.objective_options.cost_class.co2", value=1
            )
        else:
            raise ValueError("obj must be either cost or emission")

        print(self.calliope_config.get_key("run.objective_options.cost_class"))
        model = calliope.Model(
            self.calliope_config, timeseries_dataframes=self.dict_timeseries_df
        )
        if to_lp:
            model.to_lp(self.store_folder + "/" + self.name + ".lp")
        if to_yaml:
            model.save_commented_model_yaml(
                self.store_folder + "/" + self.name + ".yaml"
            )
        return model

    def getParetoFront(
        self,
        epsilon: int,
        store_folder: str,
        approach_tip=False,
        approach_percentile=0.01,
        to_lp=False,
        to_yaml=False,
        to_nc=False,
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
        tech_list = self.calliope_config.get_key(f"locations.{self.name}.techs").keys()  # type: ignore
        # calliope does not define the type of the return value, so it's ignored
        df_tech_cap_pareto = pd.DataFrame(columns=tech_list, index=range(idx_cost + 1))
        df_tech_cap_pareto.fillna(0, inplace=True)
        # first get the emission-optimal solution
        model_emission = self.getBuildingModel(
            to_lp=to_lp, to_yaml=to_yaml, obj="emission"
        )
        model_emission.run()
        if to_nc:
            model_emission.to_netcdf(
                path=self.store_folder + "/" + self.name + "_emission.nc"
            )
        print("optimization for emission is done")
        # store the cost and emission in df_pareto
        df_emission = (
            model_emission.get_formatted_array("cost")
            .sel(locs=self.name)
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
        model_cost = self.getBuildingModel(to_lp=to_lp, to_yaml=to_yaml, obj="cost")
        # run model cost, and find both cost and emission of this result
        model_cost.run()
        if to_nc:
            model_cost.to_netcdf(path=self.store_folder + "/" + self.name + "_cost.nc")
        print("optimization for cost is done")
        # store the cost and emission in df_pareto
        # add epsilon name as row index, start with epsilon_0
        df_cost = (
            model_cost.get_formatted_array("cost")
            .sel(locs=self.name)
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
                f"cost-optimal and emission-optimal of building {self.name} have the same emission, no pareto front"
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
                model_epsilon = self.getBuildingModel(
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
                        + self.name
                        + f"_epsilon_{n_epsilon}.nc"
                    )
                print(f"optimization at epsilon {n_epsilon} is done")
                # store the cost and emission in df_pareto
                df_epsilon = (
                    model_epsilon.get_formatted_array("cost")
                    .sel(locs=self.name)
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
                + self.name
                + " is done. First row is emission-optimal, last row is cost-optimal."
            )
            # show the pareto front
            print(df_pareto)
            self.df_pareto = df_pareto
            self.df_tech_cap_pareto = df_tech_cap_pareto

    def getCurrentCostEmission(self):
        """
        Description:
            This function reads the current technology setup of the building, delete all irrelevant technologies,
            then "optimize" the building (with the only feasible choice) to get the current cost and emission.
            Finally, it returns the cost and emission in a pd.Series, and add it to the self.df_pareto with index 999.

            the current tech setup is stored in self.building_status. For example, it looks like:
            DisHeat                       2025
            Rebuild                        0.0
            Renovation                     0.0
            type_hs         SUPPLY_HEATING_AS7
            is_disheat                    True
            is_rebuilt                   False
            is_renovated                 False
            already_GSHP                 False
            already_ASHP                 False
            no_heat                      False

        Meaning of type_hs (SUPPLY_HEATING_ASX):
            0. no heating
            1. oil boiler
            2. coal boiler
            3. gas boiler
            4. electric boiler
            5. wood boiler
            6. GSHP (ground source heat pump)
            7. ASHP (air source heat pump)
        """

        # first, delete DHDC_small_heat, DHDC_medium_heat, DHDC_large_heat,
        # PV, SCFP, battery, DHW_storage, heaat_storage. These do not exist in the current setup
        unrealistic_tech_list = [
            "DHDC_small_heat",
            "DHDC_medium_heat",
            "DHDC_large_heat",
            "PV",
            "SCFP",
            "battery",
            "DHW_storage",
            "heat_storage",
        ]

        if self.building_status["no_heat"]:
            # in this case, should raise error because there's no heating system
            raise ValueError("no heating system in the building")
        elif self.building_status["already_GSHP"]:
            # in this case, delete ASHP
            unrealistic_tech_list += ["ASHP", "wood_boiler"]
        elif self.building_status["already_ASHP"]:
            # in this case, delete GSHP
            unrealistic_tech_list += ["GSHP", "wood_boiler"]
        else:
            # if the building is originally using gas or oil boiler, by 2050 it should have changed to ASHP
            unrealistic_tech_list += ["GSHP", "wood_boiler"]

        for tech in unrealistic_tech_list:
            # first check if the tech exists in the building config
            if tech in self.df_tech_cap_pareto.columns:
                self.calliope_config.del_key(f"locations.{self.name}.techs.{tech}")

        model_current = self.getBuildingModel(
            flatten_spikes=False,
            flatten_percentile=0.98,
            to_lp=False,
            to_yaml=False,
            obj="cost",
        )
        print(f"calculating current cost and emission for building {self.name}")
        model_current.run()
        print(f"current cost and emission for building {self.name} is done")
        sr_cost_current: pd.Series = (
            model_current.get_formatted_array("cost")
            .sel(locs=self.name)
            .to_pandas()
            .transpose()
            .sum(axis=0)
        )
        # add the cost and emission to df_pareto
        self.df_pareto.loc[999] = [sr_cost_current["monetary"], sr_cost_current["co2"]]
        # store the technology capacities in df_tech_cap_pareto
        self.df_tech_cap_pareto.loc[999] = (
            model_current.get_formatted_array("energy_cap").to_pandas().iloc[0]
        )
        self.df_tech_cap_pareto.fillna(0, inplace=True)

    @staticmethod
    def flattenSpikes(
        df: pd.DataFrame,
        percentile: float = 0.98,
        is_positive: bool = False,
    ) -> pd.DataFrame:
        """
        This function removes extreme values in the DataFrame by setting them to a lower percentile value.

        - Currently, this will change the integral of the original timeseries.
        - Also, the function cannot handle negative values when is_positive is True, or data with both positive and negative values.

        Args:
            df (pd.DataFrame): dataframe that contain only numbers.
            percentile (float, optional): The part of non-zero values that are preserved in flattening (the rest is flattened). Defaults to 0.98.
            is_positive (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: if not all values in the DataFrame are numbers
            ValueError: if all columns in the DataFrame don't have at least one non-zero value
            ValueError: if columns have both positive and negative values

        Returns:
            df (pd.DataFrame): the DataFrame with the extreme values flattened
        """
        # Check if all values in the DataFrame are numbers
        if not df.applymap(lambda x: isinstance(x, (int, float))).all().all():
            raise ValueError("All values in the DataFrame must be numbers")

        # check if columns don't have both positive and negative values
        if is_positive:
            if not df.applymap(lambda x: x >= 0).all().all():
                raise ValueError(
                    "All columns in the DataFrame must have only non-negative values"
                )
        else:
            if not df.applymap(lambda x: x <= 0).all().all():
                raise ValueError(
                    "All columns in the DataFrame must have only non-positivve values"
                )

        for column_name in df.columns:
            if not is_positive:
                df[column_name] = -df[column_name]

            nonzero_subset = df[df[column_name] != 0]
            percentile_value = nonzero_subset[column_name].quantile(1 - percentile)
            df.loc[df[column_name] > percentile_value, column_name] = percentile_value

            if not is_positive:
                df[column_name] = -df[column_name]

        return df
