import pandas as pd
import geopandas as gpd
from typing import List, Union
from calliope import AttrDict
from cea_energy_hub_optimizer.my_config import MyConfig


class Node:
    # TODO: implement the network (read from CEA network optimization) to the energy hub
    pass


class Building(Node):
    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self.locator = MyConfig().locator
        self.get_geometry()

    def __str__(self):
        return self.name

    def get_geometry(self):
        zone: gpd.GeoDataFrame = gpd.read_file(self.locator.get_zone_geometry())
        zone.set_index("Name", inplace=True)
        try:
            self.area = float(zone.loc[self.name, "geometry"].area)  # type: ignore
            self.lon = float(zone.loc[self.name, "geometry"].centroid.x)  # type: ignore
            self.lat = float(zone.loc[self.name, "geometry"].centroid.y)  # type: ignore
        except KeyError:
            print(
                f"Building {self.name} not found in the zone geometry file, and probably not inside scenario."
            )

    @property
    def emission(self):
        """
        get the heating emission system of the building (e.g., radiator, floor heating, etc.)
        """
        air_conditioning_df: pd.DataFrame = gpd.read_file(
            self.locator.get_building_air_conditioning(), ignore_geometry=True
        )
        air_conditioning_df.set_index("Name", inplace=True)
        self._emission = str(air_conditioning_df.loc[self.name, "type_hs"])
        return self._emission


class District:
    def __init__(
        self,
        building_names: Union[str, List[str]],
        # yml_path: str,
    ):
        if isinstance(building_names, str):
            building_names = [building_names]

        self.locator = MyConfig().locator
        self._get_input_buildings(building_names)
        self._get_cea_input_files()
        # self._get_techs_from_yaml(yml_path)

    def _get_input_buildings(self, building_names: List[str]):
        self.buildings: List[Building] = []
        for building_name in building_names:
            building = Building(name=building_name)
            self.buildings.append(building)

    def _get_cea_input_files(self):
        zone: gpd.GeoDataFrame = gpd.read_file(self.locator.get_zone_geometry())
        zone.set_index("Name", inplace=True)
        air_conditioning: pd.DataFrame = gpd.read_file(
            self.locator.get_building_air_conditioning(), ignore_geometry=True
        )
        air_conditioning.set_index("Name", inplace=True)
        self.zone = zone.loc[self.buildings_names]
        self.air_conditioning = air_conditioning.loc[self.buildings_names]

    # def _get_techs_from_yaml(self, yml_path: str):
    #     self.tech_dict = TechAttrDict(yml_path=yml_path)
    #     self.tech_dict.add_locations_from_district(self)

    def add_building_from_name(self, building_name: str):
        building = Building(name=building_name)
        self.buildings.append(building)
        self.tech_dict._add_locations_from_building(building)

    def add_building(self, building: Building):
        self.buildings.append(building)
        self.tech_dict._add_locations_from_building(building)

    @property
    def buildings_names(self) -> List[str]:
        return [building.name for building in self.buildings]

    # @property
    # def tech_list(self) -> List[str]:
    #     return list(self.tech_dict.techs.keys())

    @property
    def name(self) -> str:
        # if only one building, take the building's name;
        # if multiple buildings, call it district
        if len(self.buildings) == 1:
            return self.buildings[0].name
        else:
            return "district"


class TechAttrDict(AttrDict):
    def __init__(self, yml_path: str):
        super().__init__()
        yaml_data = AttrDict.from_yaml(yml_path)
        self.update(yaml_data)
        self.my_config = MyConfig()

    @property
    def tech_list(self) -> List[str]:
        return list(self.techs.keys())

    def _add_locations_from_building(self, buildings: Union[Building, List[Building]]):
        tech_name_dict = {key: None for key in self.techs.keys()}
        if isinstance(buildings, Building):
            buildings = [buildings]
        for building in buildings:
            location_dict = {"techs": tech_name_dict, "available_area": building.area}
            self.set_key(key=f"locations.{building.name}", value=location_dict)

    def add_locations_from_district(self, district: District):
        for building in district.buildings:
            self._add_locations_from_building(building)

    def set_temporal_resolution(self, temporal_resolution: str):
        self.set_key("model.time.function_options.resolution", temporal_resolution)

    def set_solver(self, solver: str):
        self.set_key("run.solver", solver)

    def set_cop_timeseries(self):
        self.set_key(
            key="techs.ASHP_35_small.constraints.energy_eff",
            value="df=cop_heating_35",
        )
        self.set_key(
            key="techs.ASHP_35_large.constraints.energy_eff",
            value="df=cop_heating_35",
        )
        self.set_key(
            key="techs.ASHP_60_small.constraints.energy_eff",
            value="df=cop_heating_60",
        )
        self.set_key(
            key="techs.ASHP_60_large.constraints.energy_eff",
            value="df=cop_heating_60",
        )
        self.set_key(
            key="techs.ASHP_85_small.constraints.energy_eff",
            value="df=cop_heating_85",
        )
        self.set_key(
            key="techs.ASHP_85_large.constraints.energy_eff",
            value="df=cop_heating_85",
        )
        print(
            "temperature sensitive COP is enabled. Getting COP timeseries from outdoor air temperature."
        )

    def select_evaluated_demand(self):
        # demand techs starts with demand_ and is key of self.techs
        # demand_techs = [key for key in self.techs.keys() if key.startswith("demand_")]
        # for tech in demand_techs:
        #     if tech not in self.my_config.evaluated_demand:
        #         for building in self.locations.keys():
        #             self.del_key(f"locations.{building}.techs.{tech}")
        # TODO: correctly consider demand tech with different temperature
        for building in self.locations.keys():
            self.del_key(f"locations.{building}.techs.demand_space_cooling")
            print(f"demand_space_cooling is disabled for {building} ...")

    def select_evaluated_solar_supply(self):
        solar_supply_techs = [
            "PV_small",
            "PV_middle",
            "PV_large",
            "PV_extra_large",
            "PVT",
            "SCET",
            "SCFP",
        ]

        for tech in solar_supply_techs:
            tech_type = tech.split("_")[0]
            if tech_type not in self.my_config.evaluated_solar_supply:
                for building in self.locations.keys():
                    if tech in self.locations[building].techs:
                        self.del_key(f"locations.{building}.techs.{tech}")

    # def set_electricity_tariff(self):
    #     ls_var_elec = [
    #         "electricity_pronatur",
    #         "electricity_natur",
    #         "electricity_econatur",
    #     ]
    #     for tech in ls_var_elec:
    #         if tech in self.tech_list:
    #             self.set_key(
    #                 key=f"techs.{tech}.costs.monetary.om_con",
    #                 value=f"df={tech}_tariff",
    #             )

    # def set_feedin_tariff(self):
    #     ls_var_feed = [
    #         "PV_small",
    #         "PV_middle",
    #         "PV_large",
    #         "PV_extra_large",
    #         "gas_micro_CHP",
    #     ]
    #     for tech in ls_var_feed:
    #         if tech in self.tech_list:
    #             self.set_key(
    #                 key=f"techs.{tech}.costs.monetary.export",
    #                 value=f"df={tech}_feedin_tariff",
    #             )

    def set_global_max_co2(self, max_co2: Union[float, None]):
        self.set_key(
            key="group_constraints.systemwide_co2_cap.cost_max.co2",
            value=max_co2,
        )

    def get_global_max_co2(self):
        return self.get_key("group_constraints.systemwide_co2_cap.cost_max.co2")

    def set_objective(self, objective: str):
        if objective == "cost":
            self.set_key(key="run.objective_options.cost_class.monetary", value=1)
            self.set_key(key="run.objective_options.cost_class.co2", value=0)
        elif objective == "emission":
            self.set_key(key="run.objective_options.cost_class.monetary", value=0)
            self.set_key(key="run.objective_options.cost_class.co2", value=1)
        else:
            raise ValueError("objective must be either cost or emission")
        print(f"Objective set to {objective} ...")

    def set_emission_temperature(self, district: District):
        """
        check the emission system of the district, set the carrier of demand_space_heating accordingly.
        the following emission systems are available from inputs/technology/assemblies/HVAC.xlsx:
        ```
        Emission system 	            Carrier
        None	                    HVAC_HEATING_AS0
        Radiator (90/70) 	            HVAC_HEATING_AS1
        Radiator (70/55) 	            HVAC_HEATING_AS2
        central AC (40/20) 	            HVAC_HEATING_AS3
        Floor heating (40/35) 	    HVAC_HEATING_AS4
        ```
        """
        carrier_dict = {
            "HVAC_HEATING_AS0": None,
            "HVAC_HEATING_AS1": "demand_space_heating_85",
            "HVAC_HEATING_AS2": "demand_space_heating_60",
            "HVAC_HEATING_AS3": "demand_space_heating_35",
            "HVAC_HEATING_AS4": "demand_space_heating_35",
        }
        heating_demand_techs = [
            "demand_space_heating_35",
            "demand_space_heating_60",
            "demand_space_heating_85",
        ]
        for building in district.buildings:
            carrier = carrier_dict[building.emission]
            # in locations.{building_name}.techs, delete all heating demand techs except the one that is needed
            for tech in heating_demand_techs:
                if tech != carrier:
                    self.del_key(key=f"locations.{building.name}.techs.{tech}")

            # TODO: wait until calliope 0.7.0 and restore back to the following code

            # self.set_key(
            #     key=f"locations.{building.name}.techs.demand_space_heating.essentials.carrier",
            #     value=carrier,
            # )
            # print(
            #     f"Building {building} has emission system {building.emission}, set space heating carrier to {carrier} ..."
            # )
