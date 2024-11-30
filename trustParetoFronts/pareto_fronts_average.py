import pandas as pd
import geopandas as gpd
import os
from cea.config import Configuration
from cea_energy_hub_optimizer.my_config import MyConfig

folder = r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub"
result_folder = os.path.join(folder, "batch_no_oil_renewable_gas_no_pallet")
config = MyConfig(Configuration())
zone: gpd.GeoDataFrame = gpd.read_file(config.locator.get_zone_geometry()).set_index(
    "Name"
)
typology = gpd.read_file(
    config.locator.get_building_typology(), ignore_geometry=True
).set_index("Name")
zone = zone.join(typology, on="Name")

# zone.set_index("Name", inplace=True)
zone["area"] = zone["geometry"].area * zone["floors_ag"]  # m2

# load data
pareto_files = [
    pareto_file
    for pareto_file in os.listdir(result_folder)
    if pareto_file.endswith("_pareto.csv")
]
dfs_list = []
for pareto_file in pareto_files:
    building_name = pareto_file.split("_")[0]
    df = pd.read_csv(os.path.join(result_folder, pareto_file), index_col=[0, 1])
    df["area"] = zone.loc[building_name, "area"]
    dfs_list.append(df)
    print(f"Loaded {pareto_file}.")

df_pareto_all = pd.concat(dfs_list)
average_emission = (
    (df_pareto_all["emission"] / df_pareto_all["area"]).groupby(level=1).mean()
)
average_cost = (df_pareto_all["cost"] / df_pareto_all["area"]).groupby(level=1).mean()
df_average = pd.DataFrame({"emission": average_emission, "cost": average_cost})
print(df_average)
