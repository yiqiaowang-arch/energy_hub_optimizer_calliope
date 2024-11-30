from typing import Tuple
import pandas as pd
from ortools.algorithms.python import knapsack_solver

"""
Warning! This script requires pandas 2.0.3 or higher! Calliope requires 1.5.4 or lower.
  WARNING: Failed to remove contents in a temporary directory 'C:\Users\yiqwang\AppData\Local\Temp\pip-uninstall-51z_dg4e'.
  You can safely remove it manually.
  WARNING: Failed to remove contents in a temporary directory 'C:\Users\yiqwang\Documents\CityEnergyAnalyst\dependencies\micromamba\envs\cea\Lib\site-packages\pandas\_libs\~indow'.
  You can safely remove it manually.
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
calliope 0.6.10 requires numpy~=1.23.5, but you have numpy 1.24.4 which is incompatible.
calliope 0.6.10 requires pandas~=1.5.2, but you have pandas 2.0.3 which is incompatible.
Successfully installed absl-py-2.1.0 immutabledict-4.2.1 ortools-9.11.4210 pandas-2.0.3 protobuf-5.26.1
"""


def preprocess_and_deduplicate(df: pd.DataFrame, precision: int = 0) -> pd.DataFrame:
    """
    Preprocess and deduplicate Pareto solutions for DP:
    - Convert costs to integers.
    - Deduplicate solutions by keeping the one with the highest pareto_index.

    :param df: DataFrame with Pareto solutions, must contain "cost", "emission", and "pareto_index".
    :param precision: Number of decimal places to retain during scaling.
    :return: Deduplicated DataFrame with integer costs.
    """
    # Convert cost to integers
    scaling_factor = 10**precision
    df["int_cost"] = (df["cost"] * scaling_factor).round().astype(int)
    df["int_emission"] = (df["emission"] * scaling_factor).round().astype(int)

    # Sort values to prioritize higher pareto_index within each building and int_cost
    deduplicated_idx = (
        df.reset_index().groupby(["building", "int_cost"])["pareto_index"].idxmax()
    )

    # Deduplicate within each building and int_cost, keeping the first occurrence
    deduplicated_df = df.iloc[deduplicated_idx.values].copy()

    return deduplicated_df


def maximal_emission_reduction_dp(
    df: pd.DataFrame, cost_budget: float, precision: int = 2
) -> Tuple[pd.DataFrame, float, float]:
    """
    Optimized dynamic programming approach for maximizing emission reduction.
    """
    # Preprocess and deduplicate
    df = preprocess_and_deduplicate(df, precision=precision)
    scaling_factor = 10**precision

    # Minimal and maximal costs
    minimal_cost_solutions = df.loc[df.groupby(level="building")["int_cost"].idxmin()]
    minimal_cost = minimal_cost_solutions["int_cost"].sum()
    result_df = minimal_cost_solutions.reset_index()[
        ["building", "pareto_index"]
    ].set_index("building")

    # Compute `additional_cost` and store in DataFrame
    df["additional_cost"] = df["int_cost"] - df.index.map(
        lambda idx: minimal_cost_solutions.loc[idx[0], "int_cost"].values[0]
    )

    df["emission_reduction"] = -(
        df["int_emission"]
        - df.index.map(
            lambda idx: minimal_cost_solutions.loc[idx[0], "int_emission"].values[0]
        )
    )

    # sort df by its multiindex
    df = df.sort_index()
    # Feasibility Check
    if cost_budget < minimal_cost / scaling_factor:
        raise ValueError(
            f"The cost budget ({cost_budget}) is infeasible. Minimum required is {minimal_cost / scaling_factor}."
        )

    # Adjust budget
    additional_budget = int(cost_budget * scaling_factor) - minimal_cost

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, "emission"
    )

    solver.init(
        profits=list(df["emission_reduction"].values),
        weights=[list(df["additional_cost"].values)],
        capacities=[additional_budget],
    )
    maximal_reduction = solver.solve()

    additional_cost = 0
    # Trace back solutions
    for i in range(len(df)):
        if solver.best_solution_contains(i):
            result_df.loc[df.index[i][0], "pareto_index"] = df.index[i][1]
            additional_cost += df.loc[df.index[i], "additional_cost"]

    # Compute actual cost
    actual_cost = (minimal_cost + additional_cost) / scaling_factor

    return result_df, maximal_reduction / scaling_factor, actual_cost


if __name__ == "__main__":
    # Example DataFrame
    data = {
        # fmt: off
        "building":     ["A", "A", "A", "A",    "B", "B", "B", "B",     "C", "C", "C",  "D", "D", "D", "D"],
        "pareto_index": [0, 1, 2, 3,            0, 1, 2, 3,             0, 1, 2,        0, 1, 2, 3],
        "cost":         [5, 5, 8, 10,           4, 6, 6, 9,             3, 7, 12,       4, 6, 11, 11],
        "emission":     [10, 10, 7.0, 5,        15, 12, 12, 8,          20, 15, 10,     12, 10, 5, 5],
        # fmt: on
    }
    df_pareto = pd.DataFrame(data).set_index(["building", "pareto_index"])

    # Budget
    cost_budget = 25

    # Run DP
    result, max_reduction, actual_cost = maximal_emission_reduction_dp(
        df_pareto, cost_budget, 0
    )
    print(result, f"\nMax Reduction: {max_reduction}\nActual Cost: {actual_cost}")

    import timeit

    # benchmark the speed of the function
    print(
        timeit.timeit(
            "maximal_emission_reduction_dp(df_pareto, cost_budget, 0)",
            globals=globals(),
            number=1000,
        )
    )
