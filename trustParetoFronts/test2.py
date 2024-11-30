import pandas as pd
import numpy as np


def maximal_emission_reduction_dp(
    df: pd.DataFrame, cost_budget: float, precision: int = 2
) -> pd.DataFrame:
    """
    Dynamic programming approach for maximizing emission reduction with integer costs and float emissions.
    Handles cases where `building` is the first-level index.

    :param df: DataFrame with Pareto solutions, must contain "cost", "emission", and "pareto_index".
               The `building` should be the first-level index.
    :param cost_budget: The cost budget for the optimization.
    :param precision: Number of decimal places to retain during scaling.
    :return: A DataFrame with the optimal Pareto solution for each building.
    """
    # Step 1: Preprocess and deduplicate
    df = preprocess_and_deduplicate(df, precision=precision)
    scaling_factor = 10**precision

    # Step 2: Calculate minimal cost and emission solutions
    minimal_cost_solutions = df.loc[df.groupby(level="building")["int_cost"].idxmin()]
    minimal_cost = minimal_cost_solutions["int_cost"].sum()

    # Feasibility Check
    if cost_budget < minimal_cost / scaling_factor:
        raise ValueError(
            f"The cost budget ({cost_budget}) is infeasible. Minimum required is {minimal_cost / scaling_factor}."
        )

    # Adjust the budget for additional cost
    additional_budget = cost_budget - (minimal_cost / scaling_factor)
    max_budget = int(
        np.ceil(additional_budget * scaling_factor)
    )  # Convert budget to integer

    # Initialize DP table and auxiliary structures
    buildings = df.index.get_level_values(0).unique()  # Get unique buildings from index
    num_buildings = len(buildings)
    DP = np.zeros(
        (2, max_budget + 1)
    )  # Only two rows are needed (current and previous)
    selected_solution = np.full((num_buildings, max_budget + 1), None, dtype=object)

    # Step 3: Populate DP table
    for b, building in enumerate(buildings):
        # Calculate additional cost and emission reduction for the current building
        df.loc[building, "additional_cost"] = (
            df.loc[building, "int_cost"]
            - minimal_cost_solutions.loc[building, "int_cost"][0]
        ).values
        df.loc[building, "emission_reduction"] = (
            minimal_cost_solutions.loc[building, "emission"][0]
            - df.loc[building, "emission"]
        ).values
        building_df = df.loc[[building]]
        for w in range(max_budget + 1):
            # DP Formula:
            # DP[b][w] = max(DP[b-1][w], DP[b-1][w - additional_cost] + emission_reduction)
            DP[b % 2][w] = DP[(b - 1) % 2][w]
            selected_solution[b, w] = selected_solution[b - 1, w] if b > 0 else None

            for idx, row in building_df.iterrows():
                cost = int(row["additional_cost"])
                value = row["emission_reduction"]
                if cost <= w:
                    if DP[(b - 1) % 2][w - cost] + value > DP[b % 2][w]:
                        DP[b % 2][w] = DP[(b - 1) % 2][w - cost] + value
                        selected_solution[b, w] = idx

    # Step 4: Trace back to find selected solutions
    remaining_budget = max_budget
    selected_solutions = []
    for b in range(num_buildings - 1, -1, -1):
        if selected_solution[b, remaining_budget] is not None:
            solution = selected_solution[b, remaining_budget]
            selected_solutions.append(solution)
            remaining_budget -= int(df.loc[solution, "additional_cost"])

    # Convert the selected solutions to a DataFrame
    result_df = pd.DataFrame(selected_solutions)

    return result_df


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

    # Deduplicate solutions for each building
    deduplicated = []
    for building, group in df.groupby("building"):
        # Sort by cost (ascending) and pareto_index (descending)
        group = group.sort_values(
            by=["int_cost", "pareto_index"], ascending=[True, False]
        )
        # Keep only one solution per cost
        group = group.loc[group["int_cost"].drop_duplicates(keep="first").index]
        deduplicated.append(group)

    return pd.concat(deduplicated)


if __name__ == "__main__":
    # Example DataFrame
    # fmt: off
    data = {
        "building":     ["A",   "A",    "B",    "B",    "C",    "C"],
        "pareto_index": [0,     1,      0,      1,      0,      1],
        "cost":         [10.5,  20.3,   15.2,   25.7,   18.1,   30.5],
        "emission":     [100.2, 80.1,   120.0,  70.5,   90.0,   60.3],
    }
    # fmt: on
    df_pareto = pd.DataFrame(data)
    df_pareto.set_index(["building", "pareto_index"], inplace=True)

    # Budget
    cost_budget = 60

    # Run DP
    result = maximal_emission_reduction_dp(df_pareto, cost_budget)
    print(result)
