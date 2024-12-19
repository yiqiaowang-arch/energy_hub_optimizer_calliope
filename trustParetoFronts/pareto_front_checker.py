import pandas as pd
import numpy as np
from trustParetoFronts.pareto_analysis import ParetoFront
import os
from cea.config import Configuration
from cea_energy_hub_optimizer.my_config import MyConfig
from cea_energy_hub_optimizer.energy_hub import EnergyHub

# load data
folder_path = r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\batch_after_presentation"
# pareto_files = [pareto_file for pareto_file in os.listdir(folder_path) if pareto_file.endswith("_pareto.csv")]
carrier_dict = {
    "HVAC_HEATING_AS0": None,
    "HVAC_HEATING_AS1": "demand_space_heating_85",
    "HVAC_HEATING_AS2": "demand_space_heating_60",
    "HVAC_HEATING_AS3": "demand_space_heating_35",
    "HVAC_HEATING_AS4": "demand_space_heating_35",
}
i = 0
config = MyConfig(Configuration())
error_files = []
for pareto_file in os.listdir(folder_path):
    if pareto_file.endswith("_pareto.csv"):
        df = pd.read_csv(os.path.join(folder_path, pareto_file))
        xs = df["emission"].values.round(2)
        ys = df["cost"].values.round(2)
        try:
            print(f"Checking {pareto_file}...")
            pf = ParetoFront(xs, ys)
            # print(f"{pareto_file} is a valid Pareto front.")
        except ValueError as e:
            print(f"{pareto_file} is not a valid Pareto front: {e}")
            i += 1
            error_files.append(pareto_file)
            # move the invalid pareto front file, along with the corresponding cost_per_tech file to a new folder called "invalid_pareto_fronts"
            # check if the folder exists, if not, create it
            if not os.path.exists(os.path.join(folder_path, "invalid_pareto_fronts")):
                os.makedirs(os.path.join(folder_path, "invalid_pareto_fronts"))
            os.rename(
                os.path.join(folder_path, pareto_file),
                os.path.join(folder_path, "invalid_pareto_fronts", pareto_file),
            )
            os.rename(
                os.path.join(
                    folder_path, pareto_file.replace("_pareto", "_cost_per_tech")
                ),
                os.path.join(
                    folder_path,
                    "invalid_pareto_fronts",
                    pareto_file.replace("_pareto", "_cost_per_tech"),
                ),
            )

print(f"Total number of invalid Pareto fronts: {i}")

# for error_file in error_files:
#     building_name = error_file.split("_")[0]
#     energy_hub = EnergyHub(
#         building_name, r"cea_energy_hub_optimizer\data\energy_hub_config.yml"
#     )
#     for building in energy_hub.district.buildings:
#         emission_system = carrier_dict[building.emission]
#         if emission_system != "demand_space_heating_35":
#             df = pd.read_csv(os.path.join(folder_path, error_file))
#             wrong_GSHP_activation = (df["GSHP_35"] > 0).sum()
#             wrong_ASHP_activation = (df["ASHP_35"] > 0).sum()
#             if wrong_ASHP_activation > 0 or wrong_GSHP_activation > 0:
#                 print(
#                     f"Building {building_name} has wrong GSHP or ASHP activation in {error_file}. GSHP: {wrong_GSHP_activation}, ASHP: {wrong_ASHP_activation}"
#                 )
