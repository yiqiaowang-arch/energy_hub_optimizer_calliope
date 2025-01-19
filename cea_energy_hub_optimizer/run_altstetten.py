import gc
from cea.config import Configuration
from cea_energy_hub_optimizer import my_config
from cea_energy_hub_optimizer.my_config import MyConfig
from cea_energy_hub_optimizer.energy_hub import EnergyHub
from cea_energy_hub_optimizer.energy_hub_optimizer import check_solar_technology
import geopandas as gpd
import pandas as pd
import os
import warnings

"""
This script is used to run the energy hub optimization for the Altstetten district in Zurich. 
It will read every building in the zone.shp file and optimize them one by one.
Also, based on the availability of district heating, some buildings might not be able to use it (so disable district heating for them).
The buildings listed in `D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\DH_availability.csv`
have access to the district heating, so we need to disable it for the rest of the buildings.

There are also some buildings that are part of a district energy network, which need to be optimized as a whole, 
but here we only care about their electricity demand.

These networks are:
149 (waste heat and shallow geothermal):
"B302062896", "B302062895", "B302062776", "B302062775", "B302060244", "B302060243", "B302060242", "B302030224", "B302030223", "B161046", 
"B161045", "B161044", "B161043", "B161042", "B161041", "B161040", "B161039", "B161038", "B161037", "B161034", "B161033", "B161032", 
"B161031", "B161030", "B161029", "B161027", "B161026", "B161025", "B161023", "B161022", "B161021", "B161020", "B161019", "B161018", 
"B161017", "B161016", "B161015", "B161014", "B161013", "B161012", "B161011", "B161010", "B161009", "B161004", "B161001", "B161000", 
"B160999", "B160998", "B160997", "B160996", "B160995", "B160993", "B160992", "B160991"

115 (waste heat and shallow geothermal):
"B302061170", "B302061169", "B302061168", "B302061167", "B302061166", "B302061165", "B302061164", "B302061163", "B302061162", 
"B302061161", "B302061160", "B302061159", "B302061158", "B302061157", "B302061156", "B302061155", "B302061154", "B302061153", 
"B302061149", "B302061148", "B302061147", "B302061146", "B302061145", "B302061144", "B302061143", "B302061142", "B302061141", 
"B302061140", "B302061139", "B302061138"

141 (waste heat):
"B302064232", "B302061319", "B302061318", "B302061317", "B302061316", "B161420", "B161419", "B161329", "B161328"

153 (waste heat and shallow geothermal):
"B302034725", "B302034726", "B302034727"

135 (waste heat and shallow geothermal):
"B302063438", "B302063435", "B302063434", "B302063432", "B302063428", "B163433", "B163432", "B163431", "B163430", 
"B163429", "B163428", "B163427", "B163426", "B163425", "B163424", "B163423", "B163422", "B163421", "B163420", 
"B163419", "B163418", "B163417", "B163416", "B163415", "B163400", "B163398", "B163397", "B163375", "B163374"

"""


def remove_district_heating_technologies(energy_hub: EnergyHub) -> None:
    """
    by setting exists to False, we remove district heating supply from the technology list.

    :param energy_hub: energy hub object that is being optimized
    :type energy_hub: EnergyHub
    """
    energy_hub.tech_dict.set_key(key="techs.district_heating.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.DH_15_50.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.DH_50_200.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.DH_200_500.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.DH_500_2000.exists", value=False)


def remove_oil_technologies(energy_hub: EnergyHub) -> None:
    energy_hub.tech_dict.set_key(key="techs.oil.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.oil_boiler_large.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.oil_boiler_middle.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.oil_boiler_small.exists", value=False)


def remove_gas_technologies(energy_hub: EnergyHub) -> None:
    energy_hub.tech_dict.set_key(key="techs.gas_standard.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.gas_boiler_large.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.gas_boiler_middle.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.gas_boiler_small.exists", value=False)


def remove_pallet_technologies(energy_hub: EnergyHub) -> None:
    energy_hub.tech_dict.set_key(key="techs.pallet.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.pallet_boiler_large.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.pallet_boiler_middle.exists", value=False)
    energy_hub.tech_dict.set_key(key="techs.pallet_boiler_small.exists", value=False)


warnings.filterwarnings("ignore")
# Load the buildings
my_config = MyConfig(Configuration())
# fix some parameters from my_config
# fmt: off
my_config.number_of_epsilon_cut = 5
my_config.approach_but_not_land_on_tip = False
my_config.temporal_resolution = "1D"
my_config.solver = "cplex"
my_config.use_temperature_sensitive_cop = True
my_config.exergy_efficiency = 0.52
my_config.flatten_spike = True
my_config.flatten_spike_percentile = 0.02
my_config.evaluated_demand = [ "demand_electricity", "demand_hot_water", "demand_space_heating"]
my_config.evaluated_solar_supply = ["PV", "SCET", "SCFP"]

locator = my_config.locator
check_solar_technology()
# locate zone.shp, just to get all the names of the buildings in "Name" column
zone: pd.DataFrame = gpd.read_file(locator.get_zone_geometry(), ignore_geometry=True)
# get all the building names
buildings = zone["Name"].tolist()
result_folder = os.path.join(locator.get_optimization_results_folder(), "calliope_energy_hub", "sbe_test_HP_price")
scenario_folder = locator.scenario
# read DH_availability.csv and get the list of buildings that have access to district heating
DH_availability = pd.read_csv(os.path.join(scenario_folder, "DH_availability.csv"))
ls_DH_available = DH_availability["Name"].tolist()
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
config_path = (r"cea_energy_hub_optimizer\data\energy_hub_config.yml")

# define buildings that belongs to a district energy network
network_149 = [ "B302062896", "B302062895", "B302062776", "B302062775", "B302060244", "B302060243", "B302060242", "B302030224", "B302030223", "B161046", 
                "B161045", "B161044", "B161043", "B161042", "B161041", "B161040", "B161039", "B161038", "B161037", "B161034", "B161033", "B161032", 
                "B161031", "B161030", "B161029", "B161027", "B161026", "B161025", "B161023", "B161022", "B161021", "B161020", "B161019", "B161018", 
                "B161017", "B161016", "B161015", "B161014", "B161013", "B161012", "B161011", "B161010", "B161009", "B161004", "B161001", "B161000", 
                "B160999", "B160998", "B160997", "B160996", "B160995", "B160993", "B160992", "B160991",
                ]

network_115 = [ "B302061170", "B302061169", "B302061168", "B302061167", "B302061166", "B302061165", "B302061164", "B302061163", "B302061162", 
                "B302061161", "B302061160", "B302061159", "B302061158", "B302061157", "B302061156", "B302061155", "B302061154", "B302061153", 
                "B302061149", "B302061148", "B302061147", "B302061146", "B302061145", "B302061144", "B302061143", "B302061142", "B302061141", 
                "B302061140", "B302061139", "B302061138"
                ]

network_141 = [ "B302064232", "B302061319", "B302061318", "B302061317", "B302061316", "B161420", "B161419", "B161329", "B161328"]

network_153 = [ "B302034725", "B302034726", "B302034727"]

network_135 = [ "B302063438", "B302063435", "B302063434", "B302063432", "B302063428", "B163433", "B163432", "B163431", "B163430",
                "B163429", "B163428", "B163427", "B163426", "B163425", "B163424", "B163423", "B163422", "B163421", "B163420",
                "B163419", "B163418", "B163417", "B163416", "B163415", "B163400", "B163398", "B163397", "B163375", "B163374"
                ]
# fmt: on

for building_name in buildings:
    # first, check if result is already in the result folder
    if (building_name + "_pareto.csv") in os.listdir(result_folder):
        print(building_name + " is already done, skipping...")
        continue

    # check if the building is part of a district energy network, if so, skip it
    if (
        building_name
        in network_149
        + network_115
        + network_141
        + network_153
        + network_135
        + ["B162979", "B163158", "B163482"]
    ):
        print(building_name + " is part of a district energy network, skipping...")
        continue

    energy_hub = EnergyHub(building_name, config_path)
    if building_name not in ls_DH_available:
        remove_district_heating_technologies(energy_hub)
    # for now we keep oil technologies so the following line is commented
    # remove_oil_technologies(energy_hub)
    # remove_gas_technologies(energy_hub)
    # remove_pallet_technologies(energy_hub)
    energy_hub.get_pareto_front(store_folder=result_folder)
    energy_hub.df_pareto.to_csv(
        result_folder + "/" + building_name + "_pareto.csv", index=True
    )
    energy_hub.df_cost_per_tech.to_csv(
        result_folder + "/" + building_name + "_cost_per_tech.csv", index=True
    )
    print(building_name + " is optimized! Results saved in " + result_folder)
    del energy_hub
    # gc.collect()
