import pandas as pd
import geopandas as gpd
from typing import List, Union
from calliope import AttrDict
from cea_energy_hub_optimizer.my_config import MyConfig


class Node:
    pass


class Building(Node):
    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self.locator = MyConfig().locator
        self.get_geometry()

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

    def get_emission_system(self):
        air_conditioning_df: pd.DataFrame = gpd.read_file(
            self.locator.get_building_air_conditioning(), ignore_geometry=True
        )
        air_conditioning_df.set_index("Name", inplace=True)
        self.emission = str(air_conditioning_df.loc[self.name, "type_hs"])


class District:
    def __init__(
        self,
        building_names: Union[str, List[str]],
        yml_path: str,
    ):
        if isinstance(building_names, str):
            building_names = [building_names]

        self.locator = MyConfig().locator
        self._get_input_buildings(building_names)
        self._get_cea_input_files()
        self._get_techs_from_yaml(yml_path)

    def _get_input_buildings(self, building_names: List[str]):
        self.buildings: List[Building] = []
        for building_name in building_names:
            building = Building(name=building_name)
            building.get_emission_system()
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

    def _get_techs_from_yaml(self, yml_path: str):
        self.tech_dict = TechAttrDict(yml_path=yml_path)
        self.tech_dict.add_locations_from_district(self)

    def add_building_from_name(self, building_name: str):
        building = Building(name=building_name)
        building.get_emission_system()
        self.buildings.append(building)
        self.tech_dict._add_locations_from_building(building)

    def add_building(self, building: Building):
        self.buildings.append(building)
        self.tech_dict._add_locations_from_building(building)

    @property
    def buildings_names(self) -> List[str]:
        return [building.name for building in self.buildings]

    @property
    def tech_list(self) -> List[str]:
        return list(self.tech_dict.techs.keys())


class TechAttrDict(AttrDict):
    def __init__(self, yml_path: str):
        super().__init__()
        yaml_data = AttrDict.from_yaml(yml_path)
        self.update(yaml_data)
        self.my_config = MyConfig()

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
        print(
            "temperature sensitive COP is enabled. Getting COP timeseries from outdoor air temperature."
        )

    def select_evaluated_demand(self):
        # demand techs starts with demand_ and is key of self.techs
        demand_techs = [key for key in self.techs.keys() if key.startswith("demand_")]
        for tech in demand_techs:
            if tech not in self.my_config.evaluated_demand:
                for building in self.locations.keys():
                    self.del_key(f"locations.{building}.techs.{tech}")

    def select_evaluated_solar_supply(self):
        solar_supply_techs = ["PV", "PVT", "SCET", "SCFP"]
        for tech in solar_supply_techs:
            if tech not in self.my_config.evaluated_solar_supply:
                for building in self.locations.keys():
                    self.del_key(f"locations.{building}.techs.{tech}")

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
