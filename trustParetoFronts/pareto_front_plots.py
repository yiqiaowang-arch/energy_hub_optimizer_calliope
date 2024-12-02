from typing import Iterable, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def tech_size_boxplot(
    df_pareto_all: pd.DataFrame,
    ls_supply_name: Iterable[str],
    demand_color_dict: Dict[str, str],
    title_left: str = "",
    title_right: str = "",
    title_fig: str = "",
    figsize: Tuple[int, int] = (40, 10),
    width_ratios: Tuple[int, int] = (8, 1),
) -> plt.figure:
    """
    receives a dataframe with technology sizes for each pareto solution, and return a box plot to represent the range of sizes for each technology.
    The box plot is divided into two subplots, the left subplot is detailed for each pareto solution (therefore for each technology, `n_solutions` boxes are plotted),
    and the right subplot is aggregated for each technology (therefore for each tech, only one box is plotted).

    Normally, the left plot is used for observing the varation of supply/conversion technologies between different pareto solutions,
    and the right plot is to show the intensity of demand (because demand doesn't change between pareto solutions).


    :param df_pareto_all: a dataframe with technology sizes for each pareto solution. names of technologies are in the columns, and the index should be bi-level:
        the first level is the building name, and the second level is the pareto solution index for this building. The values should be in kW.
        The dataframe also needs to contain a column named "area" which is the area of the building. The area is used to normalize the technology sizes into W/m^2.
        An example of such a dataframe is:
            ```
                ________________tech1   tech2   tech3   area
                bName   idx
                b1001   0       100     200     300     1000
                        1       150     250     350     1000
                        2       200     300     400     1000
                b1002   0       100     200     300     2000
                        1       150     250     350     2000
                        2       200     300     400     2000
            ```

    :type df_pareto_all: pd.DataFrame
    :param ls_supply_name: a list of strings, the names of the supply/conversion technologies. This list must be a subset of `df_pareto_all.columns`.
    :type ls_supply_name: Iterable[str]
    :param demand_color_dict: a dictionary with demand names as keys and colors as values. The colors should be in the format that matplotlib accepts. The
        demand names must be a subset of `df_pareto_all.columns`.
    :type demand_color_dict: Dict[str, str]
    :param title: a string, the title of the plot
    :type title: str
    :param figsize: a tuple of two integers, the size of the figure: (width, height). Normally the width should be much larger than the height.
        The default value is (40, 10).
    :type figsize: Tuple[int, int]
    :param width_ratios: a tuple of two integers, the ratio of the width of the two subplots. The first element is the width of the left subplot, and the second
        element is the width of the right subplot. Depend on the number of technologies and demands, the width of the left subplot should be adjusted
        to fit both plots evenly in the figure. The default value is (8, 1).
    :type width_ratios: Tuple[int, int]
    :return: a matplotlib figure object, which contains both the box subplots.
    :rtype: plt.figure
    """

    # Check if 'area' column exists in the dataframe
    if "area" not in df_pareto_all.columns:
        raise ValueError("df_pareto_all must contain a column named 'area'.")

    # Check if ls_supply_name is a subset of df_pareto_all.columns
    if not set(ls_supply_name).issubset(df_pareto_all.columns):
        raise ValueError("ls_supply_name must be a subset of df_pareto_all.columns.")

    # Check if demand_color_dict keys are a subset of df_pareto_all.columns
    if not set(demand_color_dict.keys()).issubset(df_pareto_all.columns):
        raise ValueError(
            "demand_color_dict keys must be a subset of df_pareto_all.columns."
        )

    # Check if the dataframe index is a MultiIndex with two levels
    if (
        not isinstance(df_pareto_all.index, pd.MultiIndex)
        or df_pareto_all.index.nlevels != 2
    ):
        raise ValueError("df_pareto_all index must be a MultiIndex with two levels.")

    # function starts here
    ls_demand_name = demand_color_dict.keys()
    n_solutions = df_pareto_all.index.get_level_values(1).nunique()
    fig3, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": width_ratios}
    )
    cmap = plt.get_cmap("RdYlGn", n_solutions).reversed()  # type: ignore
    boxprops = dict(linestyle="-", linewidth=1, color="k")  # Custom box properties
    medianprops = dict(
        linestyle="-", linewidth=0.5, color="k"
    )  # Custom median properties

    for i, tech_name in enumerate(ls_supply_name):
        tech_values = (df_pareto_all[tech_name] / df_pareto_all["area"]).values.reshape(
            -1, n_solutions
        )
        bp = axes[0].boxplot(
            tech_values * 1000,
            positions=np.arange(start=i * n_solutions, stop=(i + 1) * n_solutions),
            patch_artist=True,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
        )
        for j, box in enumerate(bp["boxes"]):
            box.set(facecolor=cmap(j))

    for i, demand in enumerate(ls_demand_name):
        demand_values = (df_pareto_all[demand] / df_pareto_all["area"]).values
        # delete all zeros in the demand_values
        demand_values = demand_values[demand_values != 0]
        demand_color = demand_color_dict[demand]
        bp = axes[1].boxplot(
            demand_values * 1000,
            positions=[i],
            patch_artist=True,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
            widths=0.5,
        )
        for box in bp["boxes"]:
            box.set(facecolor=demand_color)

    axes[0].set_xticks(
        np.arange(n_solutions / 2, len(ls_supply_name) * n_solutions, n_solutions)
    )  # Set x-ticks to the middle box of each technology group
    axes[0].set_xticklabels(ls_supply_name, rotation=45)
    if title_left:
        axes[0].set_title(title_left, fontsize=18)
    axes[0].set_ylabel("Specific Sizing of Technologies [$W/m^2$]", fontsize=14)

    for i in range(n_solutions, len(ls_supply_name) * n_solutions, n_solutions):
        axes[0].axvline(
            x=i - 0.5, color="k", linestyle="--", linewidth=0.5
        )  # Subtract 0.5 from the x position
    # Create a custom legend for the epsilon cuts
    ls_epsilon_cut = (
        ["Emission Optimal"]
        + [f"Epsilon {i+1}" for i in range(n_solutions - 2)]
        + ["Cost Optimal"]
    )
    legend_patches = [
        mpatches.Patch(color=cmap(i), label=ls_epsilon_cut[i])
        for i in range(n_solutions)
    ]
    axes[0].legend(handles=legend_patches, loc="best")

    axes[1].set_xticks(
        range(len(ls_demand_name))
    )  # Set x-ticks to the middle box of each technology group
    axes[1].set_xticklabels(ls_demand_name, rotation=45)
    if title_right:
        axes[1].set_title(title_right, fontsize=18)
    axes[1].set_ylabel("Specific Sizing of Technologies [$W/m^2$]", fontsize=14)
    # axes[1].sharey(axes[0])
    # Align the x-axis labels to the right
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    if title_fig:
        fig3.suptitle(title_fig, fontsize=24)
    fig3.tight_layout()
    return fig3


def tech_cost_stackedbar(
    df_cost_per_tech: pd.DataFrame,
    color_dict: Dict[str, str] = {},
    title: str = "",
    figsize: Tuple[int, int] = (20, 10),
    relative: bool = True,
    monetary_lim: Tuple[int, int] = None,
    co2_lim: Tuple[int, int] = None,
) -> plt.figure:

    cost_types = ["monetary", "co2"]
    fig4, axes = plt.subplots(2, 1, figsize=figsize)
    for idx, ax in enumerate(axes):
        if relative:
            df_processed = (
                df_cost_per_tech.loc[:, :, cost_types[idx]]
                .groupby("pareto_index")
                .mean()
            )
        else:
            df_processed = (
                df_cost_per_tech.loc[:, :, cost_types[idx]]
                .groupby("pareto_index")
                .sum()
            )
        if color_dict:
            df_processed.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=[color_dict[col] for col in df_processed.columns],
            )
        else:
            df_processed.plot(kind="bar", stacked=True, ax=ax)
        # ax.legend(ncol=int(len(ls_supply_name) / 3), loc="upper right")
        # ax.set_ylabel(f"average {cost_types[idx]} cost per $m^2$")
        if relative:
            if idx == 0:
                ax.set_ylabel("average monetary cost [CHF/$m^2$]")
            else:
                ax.set_ylabel("average $CO_2$ cost [kg$CO_2eq/m^2$]")
        else:
            if idx == 0:
                ax.set_ylabel("monetary cost [CHF]")
            else:
                ax.set_ylabel("$CO_2$ cost [kg$CO_2eq$]")

        ax.set_xlabel("pareto front index")

    if title:
        fig4.suptitle(title, fontsize=24)

    handles, labels = axes[0].get_legend_handles_labels()
    fig4.legend(handles, labels, loc="center right", ncol=1, prop={"size": 9})
    if monetary_lim:
        axes[0].set_ylim(monetary_lim)
    if co2_lim:
        axes[1].set_ylim(co2_lim)

    for ax in axes:
        ax.get_legend().remove()
    fig4.tight_layout(rect=[0, 0, 0.87, 0.95])

    return fig4
