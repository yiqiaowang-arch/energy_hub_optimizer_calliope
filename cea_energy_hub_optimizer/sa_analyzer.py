import os
from matplotlib.font_manager import font_scalings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from SALib.analyze import sobol


def compute_correlations(df: pd.DataFrame, problem: dict) -> pd.DataFrame:
    """
    Compute sensitivity of each parameter to each result using Spearman rank correlation.

    :param df: DataFrame containing both parameter and result values.
    :param problem: Dictionary defining the problem for SALib.
    :return: DataFrame of sensitivities (correlation coefficients).
    """
    parameter_names = problem["names"]
    parameters: pd.DataFrame = df[parameter_names]
    results = df.drop(columns=(parameter_names + ["variation_id"]))

    sensitivities = pd.DataFrame(index=parameters.columns, columns=results.columns)
    for param in parameters.columns:
        for result in results.columns:
            corr = parameters[param].corr(
                results[result], method="spearman"
            )  # Spearman rank correlation
            sensitivities.loc[param, result] = corr
    return sensitivities.astype(float)


def compute_sobol_sensitivities(
    df: pd.DataFrame,
    problem: dict,
) -> dict:
    """
    Compute Sobol sensitivity indices for each output using SALib.

    :param df: DataFrame containing both parameter and result values.
    :param problem: Dictionary defining the problem for SALib.
    :return: Dictionary containing Sobol indices for each result.
    """
    parameter_names = problem["names"]
    results = df.drop(columns=parameter_names + ["variation_id"])

    sensitivities = {}

    for result_column in results.columns:
        Y = results[result_column].values
        Si = sobol.analyze(problem, Y, print_to_console=False)
        # Handle NaNs by replacing them with 0
        for key in ["S1", "S2", "ST"]:
            Si[key] = np.nan_to_num(Si[key])
        sensitivities[result_column] = Si
    return sensitivities


def plot_correlation_barcharts(correlations: pd.DataFrame, output_folder: str = ""):
    """
    Plot bar charts of parameter sensitivities for each result.

    :param correlations: DataFrame of sensitivities.
    :param output_folder: Folder to save the plots.
    """
    output_folder = os.path.join(output_folder, "correlation_barcharts")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for result in correlations.columns:
        if not correlations[result].abs().any():
            continue
        plt.figure(figsize=(12, 10))
        # Sort by absolute values but plot actual values
        sorted_params = correlations[result].abs().sort_values().index
        correlations[result].loc[sorted_params].plot.barh()
        plt.xlabel("Correlation Coefficient")
        plt.title(f"Correlation to {result}", loc="left")
        plt.tight_layout()
        if output_folder:
            plt.savefig(f"{output_folder}/correlation_{result}.png")
        plt.close()


def plot_sobol_indices(sensitivities: dict, output_folder: str = ""):
    """
    Plot bar charts of first-order and total-order Sobol indices for each output.

    :param sensitivities: Dictionary of Sobol indices.
    :param output_folder: Folder to save the plots.
    """
    output_folder = os.path.join(output_folder, "sobol_indices")
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for output_name, Si in sensitivities.items():
        params = Si.problem["names"]
        S1 = Si["S1"]
        ST = Si["ST"]

        if not np.any(S1) and not np.any(ST):
            continue

        # Create bar positions
        x = np.arange(len(params))
        width = 0.35  # Width of the bars

        fig, ax = plt.subplots(figsize=(15, 15))
        rects1 = ax.bar(x - width / 2, S1, width, label="First-order")
        rects2 = ax.bar(x + width / 2, ST, width, label="Total-order")

        # also plot a  dashed line at 0
        ax.axhline(0, color="black", lw=0.5, ls="--")

        # Add labels and title
        ax.set_ylabel("Sensitivity Index")
        ax.set_xlabel("Parameters")
        ax.set_title(f"Sobol Sensitivity Indices for {output_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        if output_folder:
            plt.savefig(f"{output_folder}/sobol_{output_name}.png")
        plt.close()


def plot_sobol_heatmap(sensitivities: dict, output_folder: str = ""):
    """
    Plot heatmaps of first-order and total-order Sobol indices for all outputs.

    :param sensitivities: Dictionary of Sobol indices.
    :param output_folder: Folder to save the plots.
    """
    output_folder = os.path.join(output_folder, "sobol_heatmaps")
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    S1_matrix = []
    ST_matrix = []
    param_names = None

    for output_name, Si in sensitivities.items():
        if param_names is None:
            param_names = Si.problem["names"]
        S1_matrix.append(Si["S1"])
        ST_matrix.append(Si["ST"])

    S1_matrix = np.array(S1_matrix).T
    ST_matrix = np.array(ST_matrix).T

    fig, ax = plt.subplots(2, 1, figsize=(24, 15))

    im1 = ax[0].imshow(S1_matrix, cmap="coolwarm", aspect="auto")
    ax[0].set_title("First-order Sobol Indices")
    ax[0].set_xticks(np.arange(len(sensitivities)))
    ax[0].set_xticklabels(sensitivities.keys(), rotation=45, ha="right")
    ax[0].set_yticks(np.arange(len(param_names)))
    ax[0].set_yticklabels(param_names)
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(ST_matrix, cmap="coolwarm", aspect="auto")
    ax[1].set_title("Total-order Sobol Indices")
    ax[1].set_xticks(np.arange(len(sensitivities)))
    ax[1].set_xticklabels(sensitivities.keys(), rotation=45, ha="right")
    ax[1].set_yticks(np.arange(len(param_names)))
    ax[1].set_yticklabels(param_names)
    fig.colorbar(im2, ax=ax[1])

    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/sobol_heatmap.png")
    plt.close()


def plot_correlation_matrix(correlations: pd.DataFrame, output_path: str = ""):
    """
    Plot a heatmap showing sensitivities of parameters to results using matplotlib imshow.

    :param sensitivities: DataFrame of sensitivities, with parameters as rows and results as columns.
    :param output_path: Path to save the plot. If not provided, the plot will be displayed.
    """
    plt.figure(figsize=(40, 10))
    correlation_values = correlations.fillna(0).values.astype(float)

    # Plot heatmap using imshow
    im = plt.imshow(correlation_values, cmap="coolwarm", aspect="auto")

    # Set ticks and labels
    plt.xticks(
        ticks=np.arange(len(correlations.columns)),
        labels=correlations.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(ticks=np.arange(len(correlations.index)), labels=correlations.index)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation Coefficient")

    # Annotate each cell with the sensitivity value
    for i in range(len(correlations.index)):
        for j in range(len(correlations.columns)):
            plt.text(
                j,
                i,
                f"{correlation_values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.xlabel("Results")
    plt.ylabel("Parameters")
    plt.title("Parameter Correlation Matrix")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()


def plot_scatter_matrix(
    sensitivity_csv_path: str,
    problem: dict,
    parameter_names: List[str] = [],
    result_names: List[str] = [],
    output_folder: str = "",
):
    """
    Plot scatter matrix of parameter sensitivities for each result.

    :param sensitivity_csv_path: Path to the CSV file containing the sensitivity data.
    :param problem: Dictionary defining the problem for SALib.
    :param parameter_names: List of parameter names to plot. If not provided, all parameters will be plotted.
    :param results: List of result names to plot. If not provided, all results will be plotted.
    :param output_folder: Folder to save the plots.
    """
    output_folder = os.path.join(output_folder, "scatter_matrix")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(sensitivity_csv_path)
    if not parameter_names:
        parameter_names = problem["names"]
    if not result_names:
        # results = df.drop(columns=parameter_names + ["variation_id"])
        result_names = df.columns.difference(parameter_names).tolist()
    # else:
    #     results = df[result_names]

    # create a scatter plot for each pair of parameter-result. Parameters on rows, results on columns
    nrow = len(parameter_names)
    ncol = len(result_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 20))
    # if only one row or column, axes will be a 1D array, so convert to 2D
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    else:
        if nrow == 1:
            axes = axes[np.newaxis, :]
        if ncol == 1:
            axes = axes[:, np.newaxis]
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].scatter(df[result_names[j]], df[parameter_names[i]])
            if i == range(nrow)[-1]:
                axes[i, j].set_xlabel(result_names[j])
            else:
                # don't show x-axis ticklabels for all but the last row
                plt.setp(axes[i, j].get_xticklabels(), visible=False)
            if j == 0:
                axes[i, j].set_ylabel(parameter_names[i])
            else:
                # don't show y-axis ticklabels for all but the first column
                plt.setp(axes[i, j].get_yticklabels(), visible=False)

    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/scatter_matrix.png")
    plt.close()


def load_problem_definition(csv_path: str) -> dict:
    """
    Load problem definition for SALib from a CSV file.

    :param csv_path: Path to the CSV file containing the problem definition.
    :return: Dictionary defining the problem for SALib.
    """
    df = pd.read_csv(csv_path)
    problem = {
        "num_vars": len(df),
        "names": df["name"].tolist(),
        "bounds": df[["min", "max"]].values.tolist(),
    }
    return problem


def plot_spider_plot(sensitivities: dict, output_folder: str = ""):
    """
    Plot spider plots of total-order Sobol indices for each output.

    :param sensitivities: Dictionary of Sobol indices.
    :param output_folder: Folder to save the plots.
    """
    output_folder = os.path.join(output_folder, "spider_plots")
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for output_name, Si in sensitivities.items():
        params = Si.problem["names"]
        ST = Si["ST"]

        if not np.any(ST):
            continue

        # Create a spider plot
        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        ST = np.concatenate((ST, [ST[0]]))
        angles += angles[:1]

        ax.fill(angles, ST, color="red", alpha=0.25)
        ax.plot(angles, ST, color="red", linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params, fontsize=10)

        plt.title(
            f"Total Influence of Parameters on {output_name}",
            size=15,
            color="red",
            y=1.1,
        )
        if output_folder:
            plt.savefig(f"{output_folder}/spider_{output_name}.png")
        plt.close()


def plot_pairwise_interaction_heatmap(sensitivities: dict, output_folder: str = ""):
    """
    Plot heatmaps of pairwise interactions (S2 values) for each output.

    :param sensitivities: Dictionary of Sobol indices.
    :param output_folder: Folder to save the plots.
    """
    output_folder = os.path.join(output_folder, "pairwise_interactions")
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for output_name, Si in sensitivities.items():
        params = Si.problem["names"]
        S2 = Si["S2"]

        if not np.any(S2):
            continue

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(S2, cmap="coolwarm", aspect="auto")

        ax.set_xticks(np.arange(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels(params)

        cbar = plt.colorbar(im)
        cbar.set_label("Pairwise Interaction (S2)")
        max_val = np.max(np.abs(S2))
        im.set_clim(-max_val, max_val)

        plt.title(f"Pairwise Interaction Heatmap for {output_name}")
        plt.tight_layout()
        if output_folder:
            plt.savefig(f"{output_folder}/pairwise_interaction_{output_name}.png")
        plt.close()


if __name__ == "__main__":
    # Specify the path to the CSV file
    sensitivity_folder = r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_emission"
    variation_folder = os.path.join(sensitivity_folder, "variation")
    result_folder = os.path.join(sensitivity_folder, "result")
    plot_folder = os.path.join(sensitivity_folder, "plots")
    sensitivity_csv_path = os.path.join(
        sensitivity_folder, result_folder, "sensitivity_analysis_data.csv"
    )
    problem_csv_path = os.path.join(sensitivity_folder, "problem.csv")

    # Load the problem definition for SALib
    problem = load_problem_definition(problem_csv_path)

    # Load data
    result_df = pd.read_csv(sensitivity_csv_path)

    # Compute Spearman rank correlation sensitivities
    correlations = compute_correlations(result_df, problem)

    # Compute Sobol sensitivity indices
    sobol_indices = compute_sobol_sensitivities(result_df, problem)

    # plotting
    # Plot Spearman rank correlation sensitivities
    plot_correlation_barcharts(correlations, output_folder=plot_folder)

    # Plot Spearman rank correlation sensitivity matrix
    plot_correlation_matrix(
        correlations,
        output_path=os.path.join(plot_folder, "correlation_matrix.png"),
    )

    # Plot Sobol sensitivity indices
    plot_sobol_indices(sobol_indices, output_folder=plot_folder)

    # Plot Sobol sensitivity heatmap
    plot_sobol_heatmap(sobol_indices, output_folder=plot_folder)

    # Plot spider plots for total influence
    plot_spider_plot(sobol_indices, output_folder=plot_folder)

    # Plot pairwise interaction heatmaps
    plot_pairwise_interaction_heatmap(sobol_indices, output_folder=plot_folder)

    # Plot scatter matrix
    plot_scatter_matrix(
        sensitivity_csv_path,
        problem,
        [
            # "techs.district_heating.costs.monetary.om_con",
            "tech_groups.GSHP.costs.co2.energy_cap",
            # "techs.pallet.costs.monetary.om_con",
        ],
        ["activated_techs_std"],
        output_folder=plot_folder,
    )