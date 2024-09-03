import pandas as pd
import geopandas as gpd
from os import PathLike
from typing import List, Union, Optional
from calliope import AttrDict
from cea.inputlocator import InputLocator
from cea.config import Configuration


class District:
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        buildings: Union[str, List[str]],
    ):
        if isinstance(buildings, str):
            buildings = [buildings]
        zone: gpd.GeoDataFrame = gpd.read_file(locator.get_zone_geometry())
        zone.set_index("Name", inplace=True)
        air_conditioning: pd.DataFrame = gpd.read_file(
            locator.get_building_air_conditioning(), ignore_geometry=True
        )
        air_conditioning.set_index("Name", inplace=True)
        self.zone: gpd.GeoDataFrame = zone.loc[buildings]
        self.air_conditioning: pd.DataFrame = air_conditioning.loc[buildings]
        self.cea_config = cea_config
        self.locator = locator
        self.get_buildings()

    def get_buildings(self):
        self.buildings: List[Building] = []
        for index, row in self.zone.iterrows():
            building = Building(name=row.name, locator=self.locator, zone=self.zone)
            building.get_emission_system(
                locator=self.locator, air_conditioning_df=self.air_conditioning
            )
            self.buildings.append(building)


class Node:
    pass


class Building(Node):
    def __init__(
        self,
        name: str,
        locator: InputLocator,
        zone: Optional[gpd.GeoDataFrame] = None,
    ):
        self.name = name
        self.locator = locator
        self.get_geometry(zone)

    def get_geometry(self, zone: Optional[gpd.GeoDataFrame]):
        if zone is None:
            zone: gpd.GeoDataFrame = gpd.read_file(self.locator.get_zone_geometry())
            zone.set_index("Name", inplace=True)
        self.area = float(zone.loc[self.name, "geometry"].area)
        self.lon = zone.loc[self.name, "geometry"].centroid.x
        self.lat = zone.loc[self.name, "geometry"].centroid.y

    def get_emission_system(
        self, locator: InputLocator, air_conditioning_df: Optional[pd.DataFrame] = None
    ):
        if air_conditioning_df is None:
            air_conditioning_df: pd.DataFrame = gpd.read_file(
                locator.get_building_air_conditioning(), ignore_geometry=True
            )
        self.emission = str(air_conditioning_df.loc[self.name, "type_hs"])


class CalliopeConfig(AttrDict):
    def __init__(
        self, cea_config: Configuration, locator: InputLocator, yml_path: PathLike
    ):
        # Initialize the AttrDict part
        super().__init__()

        # Manually load the YAML data and update the AttrDict
        yaml_data = AttrDict.from_yaml(yml_path)
        self.update(yaml_data)

        # Initialize the rest of the CalliopeConfig
        self.cea_config = cea_config
        self.locator = locator

    def add_techs_from_building(self, buildings: Union[Building, List[Building]]):
        tech_name_dict = {key: None for key in self.techs.keys()}
        if isinstance(buildings, Building):
            buildings = [buildings]
        for building in buildings:
            location_dict = {"techs": tech_name_dict, "available_area": building.area}
            self.set_key(key=f"locations.{building.name}", value=location_dict)

    def add_techs_from_district(self, district: District):
        for building in district.buildings:
            self.add_techs_from_building(building)
        self.district = district

    def set_temporal_resolution(self, temporal_resolution: str):
        self.set_key("model.time.function_options.resolution", temporal_resolution)

    def set_solver(self, solver: str):
        self.set_key("run.solver", solver)

    def set_wood_availaility(self, extra_area: float, energy_density: float):
        for building in self.district.buildings:
            self.set_key(
                key=f"locations.{building.name}.techs.wood_supply.constraints.energy_cap_max",
                value=(building.area + extra_area) * energy_density * 0.001,
            )

    def set_cop_timeseries(self):
        self.set_key(
            key="techs.ASHP.constraints.carrier_ratios.carrier_out.DHW",
            value="df=cop_dhw",
        )
        self.set_key(
            key="techs.ASHP.constraints.carrier_ratios.carrier_out.cooling",
            value="df=cop_sc",
        )

    def select_evaluated_demand(self):
        # demand techs starts with demand_ and is key of self.techs
        demand_techs = [key for key in self.techs.keys() if key.startswith("demand_")]
        for tech in demand_techs:
            if tech not in self.cea_config.energy_hub_optimizer.evaluated_demand:
                for building in self.locations.keys():
                    self.del_key(f"locations.{building}.techs.{tech}")

    def select_evaluated_solar_supply(self):
        solar_supply_techs = ["PV", "PVT", "SCET", "SCFP"]
        for tech in solar_supply_techs:
            if tech not in self.cea_config.energy_hub_optimizer.evaluated_solar_supply:
                for building in self.locations.keys():
                    self.del_key(f"locations.{building}.techs.{tech}")

    def set_global_max_co2(self, max_c02: float):
        self.set_key(
            key="group_constraints.systemwide_co2_cap.cost_max.co2",
            value=max_c02,
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
