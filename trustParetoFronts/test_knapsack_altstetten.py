import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from cea.config import Configuration
from cea_energy_hub_optimizer.my_config import MyConfig
from trustParetoFronts.pareto_analysis import ParetoFront
import trustParetoFronts.geometry_analysis as ga
from trustParetoFronts.maximal_emission_reduction_dp_ortools import (
    maximal_emission_reduction_dp,
)

config = MyConfig(Configuration())

# Plot settings
# plt.rcParams["font.family"] = "Roboto"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["legend.loc"] = "lower center"
plt.rcParams["legend.frameon"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titley"] = 1.03
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["xtick.major.width"] = 2  # setting the x-axis tick width globally
plt.rcParams["ytick.major.width"] = 2  # setting the y-axis tick width globally
# Set fig size
plt.rcParams["figure.figsize"] = (12, 8)

with_oil = False
if with_oil:
    pareto_fronts_path = os.path.join(
        config.locator.get_optimization_results_folder(),
        "calliope_energy_hub",
        "batch_with_oil",
    )
else:
    pareto_fronts_path = os.path.join(
        config.locator.get_optimization_results_folder(),
        "calliope_energy_hub",
        "batch_without_oil",
    )


def cluster_by_shadow_price_and_reduction_potential(x, y):
    if x < 0.2:
        return 0
    elif y < -100 / 3 * x + 120:
        return 1
    else:
        return 2


def get_geometry_values(gdf: gpd.GeoDataFrame):
    gdf["elongation"] = gdf["geometry"].apply(ga.calculate_elongation)
    gdf["shape_factor"] = gdf.apply(
        lambda row: ga.calculate_shape_factor(row["geometry"], row["height_ag"]), axis=1
    )
    gdf["concavity"] = gdf["geometry"].apply(ga.calculate_concavity)
    gdf["compactness"] = gdf.apply(
        lambda row: ga.calculate_compactness(row["geometry"], row["height_ag"]), axis=1
    )
    gdf["direction"] = gdf["geometry"].apply(ga.calculate_building_direction)
    return gdf


total_demand = pd.read_csv(config.locator.get_total_demand(), index_col=0)
# prepare data
zone_path = config.locator.get_zone_geometry()
typology_path = config.locator.get_building_typology()
emission_systems_path = config.locator.get_building_air_conditioning()

zone_gdf: gpd.GeoDataFrame = gpd.read_file(zone_path)
zone_gdf["floor_area"] = zone_gdf.geometry.area
zone_gdf = get_geometry_values(zone_gdf)
zone_df = pd.DataFrame(zone_gdf).drop(columns="geometry").set_index("Name")
typology_df: pd.DataFrame = gpd.read_file(
    typology_path, ignore_geometry=True
).set_index("Name")
emission_system_df: pd.DataFrame = gpd.read_file(
    emission_systems_path, ignore_geometry=True
).set_index("Name")
zone_df = pd.concat([zone_df, typology_df, emission_system_df], axis=1, join="inner")
zone_df["area"] = zone_df["floor_area"] * zone_df["floors_ag"]
zone_df.drop(
    columns=[
        "floors_bg",
        "height_bg",
        "type_cs",
        "type_dhw",
        "heat_starts",
        "heat_ends",
        "cool_starts",
        "cool_ends",
        # "1ST_USE",
        "1ST_USE_R",
        "2ND_USE",
        "2ND_USE_R",
        "3RD_USE",
        "3RD_USE_R",
    ],
    inplace=True,
)
del zone_gdf, typology_df, emission_system_df

df_pareto_dict = {}
cost_per_tech_dict = {}
for building_name, row in zone_df.iterrows():
    area = float(row["area"])
    csv_name = os.path.join(pareto_fronts_path, f"{building_name}_pareto.csv")
    if not os.path.exists(csv_name):
        zone_df.drop(index=building_name, inplace=True)
        continue
    df_pareto = pd.read_csv(
        csv_name,
        index_col=[0, 1],
    )
    df_pareto_dict[building_name] = df_pareto
    emissions: np.array = df_pareto.loc[building_name, "emission"].values / area
    costs: np.array = df_pareto.loc[building_name, "cost"].values / area
    pf = ParetoFront(np.round(emissions, 3), np.round(costs, 3))
    zone_df.loc[building_name, "shadow_price"] = 1 / pf.slope()
    zone_df.loc[building_name, "pf_curvature"] = pf.curvature()
    zone_df.loc[building_name, "emission_range_abs"] = pf.x_range()
    zone_df.loc[building_name, "cost_range"] = pf.y_range()
    zone_df.loc[building_name, "emission_range_rel"] = pf.x_range(rel=True)
    # sh, ee, ww = get_demand_intensity(building_name, area)
    sh = total_demand.loc[building_name, "Qhs_sys_MWhyr"] * 1000 / area
    ee = total_demand.loc[building_name, "E_sys_MWhyr"] * 1000 / area
    ww = total_demand.loc[building_name, "Qww_sys_MWhyr"] * 1000 / area
    if row["type_hs"] == "HVAC_HEATING_AS1":
        sh_peak = df_pareto.loc[building_name, "demand_space_heating_85"].values[0]
    else:
        sh_peak = df_pareto.loc[building_name, "demand_space_heating_35"].values[0]

    # ww_peak = df_pareto.loc[building_name, "demand_hot_water"].values[0]
    # ee_peak = df_pareto.loc[building_name, "demand_electricity"].values[0]
    # people_intensity = total_demand.loc[building_name, "people0"] / area
    zone_df.loc[building_name, "demand_space_heating_intensity"] = float(
        sh
    )  # in kWh/m2yr
    zone_df.loc[building_name, "demand_electricity_intensity"] = float(
        ee
    )  # in kWh/m2yr
    zone_df.loc[building_name, "demand_hot_water_intensity"] = float(ww)  # in kWh/m2yr
    # zone_df.loc[building_name, "people_intensity"] = float(people_intensity)  # in p/m2
    zone_df.loc[building_name, "peak_space_heating_demand"] = float(sh_peak)  # in kW
    # zone_df.loc[building_name, "peak_electricity_demand"] = float(ee_peak)  # in kW
    # zone_df.loc[building_name, "peak_hot_water_demand"] = float(ww_peak)  # in kW
    cluster = cluster_by_shadow_price_and_reduction_potential(
        1 / pf.slope(), pf.x_range(rel=True) * 100
    )
    zone_df.loc[building_name, "cluster"] = cluster

    cost_per_tech_path = os.path.join(
        pareto_fronts_path, f"{building_name}_cost_per_tech.csv"
    )
    if not os.path.exists(cost_per_tech_path):
        continue
    df_cost_per_tech = (
        pd.read_csv(cost_per_tech_path, index_col=[0, 1, 2])
        / zone_df.loc[building_name, "area"]
    )
    cost_per_tech_dict[building_name] = df_cost_per_tech

df_pareto_all = pd.concat(df_pareto_dict.values())
df_cost_per_tech_all = pd.concat(cost_per_tech_dict.values())

demand_color_dict = {
    "demand_space_heating_35": "gold",
    "demand_space_heating_85": "tab:red",
    "demand_hot_water": "tab:orange",
    "demand_electricity": "tab:green",
}

ls_supply_name = df_pareto_all.columns.difference(
    list(demand_color_dict.keys())
    + ["emission", "cost", "demand_space_heating_60", "demand_space_cooling"]
)
if not with_oil:
    ls_supply_name = ls_supply_name.difference(
        ["oil", "oil_boiler_large", "oil_boiler_middle", "oil_boiler_small"]
    )
df_pareto_all = df_pareto_all.merge(zone_df, left_on="building", right_index=True)
df_cost_per_tech_all = df_cost_per_tech_all[ls_supply_name]


minimal_cost = df_pareto_all.groupby("building")["cost"].min().sum()
print(f"Minimal cost: {minimal_cost}")
maximal_cost = df_pareto_all.groupby("building")["cost"].max().sum()
print(f"Maximal cost: {maximal_cost}")

costs = []
emission_reductions = []
dfs = []
import warnings

warnings.filterwarnings("ignore")
for cost in np.linspace(minimal_cost + 10000, maximal_cost - 10000, 100):
    df, emission_reduction, actual_cost = maximal_emission_reduction_dp(
        df_pareto_all, cost, precision=-2
    )
    costs.append(actual_cost)
    emission_reductions.append(emission_reduction)
    print(f"Cost: {actual_cost}, Emission reduction: {emission_reduction}")
    dfs.append(df)

i = 0
for df in dfs:
    df.to_csv(f"df_{i}.csv")
    i += 1

plt.plot(costs / 1e6, emission_reductions / 1e3)
plt.xlabel("Additional Cost [MCHF]")
plt.ylabel("Emission Reduction [tCO2eq]")
plt.title("Additional Investment vs. Emission Reduction from Cost-Optimal Solutions")
plt.show()
