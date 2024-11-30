from typing import Tuple
import pandas as pd


def maximal_emission_reduction(
    df: pd.DataFrame, cost_budget: float
) -> Tuple[float, pd.DataFrame]:
    """
    This function receives an additional cost budget and find the combination of Pareto solutions that maximizes the emission reduction, compared to the cost-effective (most emission-intensive) solution.

    It is essentially a knapsack problem, where the weight is the cost of the solution and the value is the emission reduction.

    First, it should calculate the global minimal/maximal cost/emission. The maximal cost/minimal emission solution is just adding up all the most expensive individual solutions.
    The minimal cost/maximal emission solution is just adding up all the most emission-intensive individual solutions.

    Once we have the minimal cost, we can compare if the cost budget is bigger than minimal cost. If it is not, we return an error saying that the budget is simply infeasible.
    If the cost is higher than the minimal cost, we can start the knapsack algorithm.

    :param df: DataFrame with the Pareto solutions. It needs to have two indices: "building" and "pareto_index". It also needs to have the columns "cost" and "emission".
    :type df: pandas.DataFrame
    :param cost_budget: The cost budget for the optimization.
    :type cost_budget: float

    :return: The maximal emission reduction that can be achieved with the given cost budget. The reduction is represented in negative values. For example, -2000 is a cleaner solution than -1000.
    :rtype: float
    :return: DataFrame with the Pareto solutions that maximize the emission reduction, given the cost budget. It contains one index "building" and one column "pareto_index".
    :rtype: pandas.DataFrame
    """

    # calculate the minimal cost and maximal emission
    # idx_emission is the index of the most emission-friendly, cost-intensive solution
    # idx_cost is the index of the most cost-effective, emission-intensive solution
    df = df["cost", "emission"].copy()
    idx_emission = df.index.get_level_values("pareto_index").unique().min()
    idx_cost = df.index.get_level_values("pareto_index").unique().max()

    minimal_cost = float(
        df.loc[df.index.get_level_values("pareto_index") == idx_cost, "cost"].sum()
    )
    maximal_emission = float(
        df.loc[
            df.index.get_level_values("pareto_index") == idx_emission, "emission"
        ].sum()
    )

    if cost_budget < minimal_cost:
        raise ValueError(
            "The cost budget is infeasible for the given Pareto solutions."
        )

    # calculate the maximal possible emission reduction and the maximal cost
    minimal_emission = float(
        df.loc[
            df.index.get_level_values("pareto_index") == idx_emission, "emission"
        ].sum()
    )
    maximal_cost = float(
        df.loc[df.index.get_level_values("pareto_index") == idx_cost, "cost"].sum()
    )
    emission_reduction = minimal_emission - maximal_emission

    if cost_budget >= maximal_cost:
        # just return the most emission-friendly solution because we can afford it
        df_solution: pd.DataFrame = df.loc[
            df.index.get_level_values("pareto_index") == idx_emission
        ].copy()
        print(
            "The cost budget is higher than the maximal cost. Returning the most emission-friendly solution."
        )
        return emission_reduction, df_solution

    # initialize the knapsack algorithm
    df["additional_cost"] = df["cost"] - minimal_cost
    df["additional_emission"] = df["emission"] - maximal_emission

    # we will use a dynamic programming approach, where we will store the best solution for each cost
    # we will use the cost as the index of the DataFrame
    # the value will be the emission reduction

    # initialize the DataFrame with the cost as the index
    # the value will be the emission reduction
    df_knapsack = pd.DataFrame(
        index=range(int(minimal_cost), int(cost_budget) + 1),
        columns=["emission_reduction"],
    )
