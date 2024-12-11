import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from cea.config import Configuration
from cea_energy_hub_optimizer.my_config import MyConfig
from trustParetoFronts.pareto_analysis import ParetoFront
import trustParetoFronts.geometry_analysis as ga
from trustParetoFronts.maximal_emission_reduction_dp import (
    maximal_emission_reduction_dp,
)
import warnings

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
        "batch_after_presentation",
    )

pareto_df_list = []
for file in os.listdir(pareto_fronts_path):
    if file.endswith("_pareto.csv"):
        decision_df = pd.read_csv(
            os.path.join(pareto_fronts_path, file), index_col=[0, 1]
        )
        pareto_df_list.append(decision_df)

df_pareto_all = pd.concat(pareto_df_list, axis=0)

minimal_cost = df_pareto_all.groupby("building")["cost"].min().sum()
print(f"Minimal cost: {minimal_cost}")
maximal_cost = df_pareto_all.groupby("building")["cost"].max().sum()
print(f"Maximal cost: {maximal_cost}")

costs = []
emission_reductions = []
decision_dfs = []


warnings.filterwarnings("ignore")
i = 0
for cost in np.linspace(minimal_cost + 3100, maximal_cost, 3):
    decision_df, emission_reduction, actual_cost = maximal_emission_reduction_dp(
        df_pareto_all, cost, precision=-2
    )
    decision_df.columns = [i]
    costs.append(actual_cost)
    emission_reductions.append(emission_reduction)
    print(f"Cost: {actual_cost}, Emission reduction: {emission_reduction}")
    decision_dfs.append(decision_df)
    decision_df.to_csv(f"df_{i}.csv")
    i += 1

"""
df example:
building    pareto_index
A           0
B           1
C           2
D           0
"""
i = 0
for decision_df in decision_dfs:
    # change df's only column name to i
    # df.columns = [i]

    i += 1

# merge all dfs into one df, each run is a column
decision_df_all = pd.concat(decision_dfs, axis=1)
decision_df_all.to_csv("knapsack_decision_after_presentation.csv")

result_df_all = pd.DataFrame({"cost": costs, "emission_reduction": emission_reductions})
result_df_all.to_csv("knapsack_result_after_presentation.csv")

# plt.plot(costs, emission_reductions)
# plt.xlabel("Additional Cost [MCHF]")
# plt.ylabel("Emission Reduction [tCO2eq]")
# plt.title(
#     "Additional Investment vs. Emission Reduction from Cost-Optimal Solutions (self-built knapsack)"
# )
# plt.show()
