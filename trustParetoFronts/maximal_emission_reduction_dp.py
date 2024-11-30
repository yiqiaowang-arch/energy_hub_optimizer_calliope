from typing import Tuple
import pandas as pd
import numpy as np


def maximal_emission_reduction_dp(
    df: pd.DataFrame, cost_budget: float, precision: int = 2
) -> Tuple[pd.DataFrame, float, float]:
    """
    Dynamic programming approach for maximizing emission reduction with integer costs and float emissions.
    Handles cases where `building` is the first-level index.

    :param df: DataFrame with Pareto solutions, must contain "cost", "emission", and "pareto_index".
               The `building` should be the first-level index.
    :param cost_budget: The cost budget for the optimization.
    :param precision: Number of decimal places to retain during scaling.
    :return: A DataFrame with the optimal Pareto solution for each building.
    :return: The maximal emission reduction.
    :return: The actual cost used to achieve the maximal emission reduction under budget limit.
    """
    # Preprocess and deduplicate
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
        lambda idx: minimal_cost_solutions.loc[idx[0], "int_cost"]
    )

    # Feasibility Check
    if cost_budget < minimal_cost / scaling_factor:
        raise ValueError(
            f"The cost budget ({cost_budget}) is infeasible. Minimum required is {minimal_cost / scaling_factor}."
        )

    # Adjust budget
    additional_budget = cost_budget - (minimal_cost / scaling_factor)
    max_budget = int(np.ceil(additional_budget * scaling_factor))

    # DP Initialization
    buildings = df.index.get_level_values(0).unique()
    num_buildings = len(buildings)
    DP = np.zeros((2, max_budget + 1))
    selected_solution = np.full((num_buildings, max_budget + 1), None, dtype=object)

    # Pre-extract building data for efficiency
    building_data = {}
    for building in buildings:
        building_df = df.loc[[building]]
        costs = building_df["additional_cost"].values.astype(int)
        emissions = (
            minimal_cost_solutions.loc[building, "emission"].values[0]
            - building_df["emission"].values
        )
        building_data[building] = (costs, emissions, building_df.index.to_numpy())

    # DP Iteration
    for b, building in enumerate(buildings):
        costs, emissions, indices = building_data[building]

        # Vectorized DP updates for valid budgets
        for w in range(max_budget + 1):
            DP[b % 2][w] = DP[(b - 1) % 2][w]  # Default: Carry forward
            selected_solution[b, w] = selected_solution[b - 1, w] if b > 0 else None

            # Valid updates
            valid_mask = costs <= w
            valid_costs = costs[valid_mask]
            valid_emissions = emissions[valid_mask]
            valid_values = DP[(b - 1) % 2][w - valid_costs] + valid_emissions

            # Apply updates
            max_idx = valid_values.argmax()
            if valid_values[max_idx] > DP[b % 2][w]:
                DP[b % 2][w] = valid_values[max_idx]
                selected_solution[b, w] = indices[valid_mask][max_idx]

    # Trace Back
    remaining_budget = max_budget

    for b in range(num_buildings - 1, -1, -1):
        solution = selected_solution[b, remaining_budget]
        if solution:
            result_df.loc[solution[0], "pareto_index"] = solution[1]
            remaining_budget -= int(df.loc[solution, "additional_cost"])

    # Compute actual cost
    actual_cost = (
        minimal_cost / scaling_factor + (max_budget - remaining_budget) / scaling_factor
    )

    return result_df, float(DP[(num_buildings - 1) % 2][max_budget]), actual_cost


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


if __name__ == "__main__":
    # Example DataFrame
    # fmt: off
    data = {
        "building":     ["A", "A", "A", "B", "B", "B",  "C", "C", "C",  "D", "D", "D"],
        "pareto_index": [0, 1, 2,       0, 1, 2,        0, 1, 2,        0, 1, 2],
        "cost":         [5, 8, 10,      4, 6, 9,        3, 7, 12,       4, 6, 11],
        "emission":     [10, 7, 5,      15, 12, 8,      20, 15, 10,     12, 10, 5],
    }
    # fmt: on
    df_pareto = pd.DataFrame(data)
    df_pareto.set_index(["building", "pareto_index"], inplace=True)

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
