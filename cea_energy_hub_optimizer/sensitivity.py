import gc
import warnings
from cea_energy_hub_optimizer.energy_hub import EnergyHub
from calliope import AttrDict
from cea_energy_hub_optimizer.my_config import MyConfig
import yaml
import os
import pandas as pd
import numpy as np
from typing import Union
from cea.config import Configuration
from SALib.sample import sobol_sequence
from SALib.analyze import sobol
from scipy.spatial.distance import pdist, squareform
from SALib.sample import sobol  # Add this import


def generate_variations(
    csv_path: os.PathLike,
    original_yaml_path: os.PathLike,
    output_folder: os.PathLike,
    num_samples: int,
    method: str = "sobol",
):
    """
    Generate variations of the original YAML file based on the given sampling method for sensitivity analysis.

    Sobol Sampling:
        - Sobol sequence is a quasi-random low-discrepancy sequence used for generating samples in high-dimensional space.
        - It ensures a more uniform and comprehensive coverage of the parameter space, suitable for global sensitivity analysis.


    Screening Method:
        - (Local Sensitivity)
        - The screening method varies one parameter at a time between its minimum and maximum values while keeping others at their base values.
        - It is used for local sensitivity analysis to identify parameters that have significant impact on the output.

    :param csv_path: Path of the CSV file containing the parameter names, min, max, and base values.
    :type csv_path: os.PathLike
    :param original_yaml_path: Original YAML file path that can be run by the energy hub model.
    :type original_yaml_path: os.PathLike
    :param output_folder: The folder where all variations will be stored.
    :type output_folder: os.PathLike
    :param num_samples: Number of samples to generate, in case of Sobol sequence.
    :type num_samples: int
    :param method: Sampling method, can either be "sobol" or "screening", defaults to "sobol".
    :type method: str, optional
    :raises ValueError: If the method is not "sobol" or "screening", an error is raised.
    """
    df = pd.read_csv(csv_path)
    problem = {
        "num_vars": len(df),
        "names": df["name"].tolist(),
        "bounds": df[["min", "max"]].values.tolist(),
    }

    variations_records = []
    base_values = df["base"].values.tolist()  # Assumed to have a 'base' column

    if method == "sobol":
        samples = sobol.sample(problem, N=num_samples, calc_second_order=False)
        print("number of samples: ", len(samples))
    elif method == "screening":
        samples = []
        for i in range(problem["num_vars"]):
            min_val, max_val = problem["bounds"][i]
            values = np.linspace(min_val, max_val, num_samples)
            for val in values:
                sample = base_values.copy()
                sample[i] = val
                samples.append(sample)
    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, sample in enumerate(samples):
        # Create modifications dict with only changed parameters
        modifications = {
            name: value
            for name, value, base in zip(problem["names"], sample, base_values)
            if value != base
        }

        if method == "screening":
            # print what is being modified
            print(f"Variation {i}: {modifications}")
        elif method == "sobol":
            # print onlz the variation number
            print(f"Variation {i} is generated.")

        new_yaml_path = os.path.join(output_folder, f"variation_{i}_{method}.yml")
        modify_yaml(original_yaml_path, modifications, new_yaml_path)

        # Record the variation
        record = {"variation_id": f"variation_{i}"}
        record.update(modifications)
        variations_records.append(record)

    # Save the variations record to a CSV file
    variations_df = pd.DataFrame(variations_records)
    variations_df.to_csv(
        os.path.join(output_folder, f"variations_record_{method}.csv"), index=False
    )

    return problem  # Return the problem dictionary


def modify_yaml(original_yaml_path, modifications, new_yaml_path):
    """
    Modify the original YAML file with the given modifications and save it as a new YAML file.
    Helper function for generating variations.
    """
    with open(original_yaml_path, "r") as file:
        # yaml_content = yaml.safe_load(file)
        yaml_content = AttrDict(yaml.safe_load(file))

    for key, value in modifications.items():
        #
        yaml_content.set_key(key, round(float(value), 3))

    with open(new_yaml_path, "w") as file:
        yaml.dump(yaml_content.as_dict(), file)


def execute_energy_hub_models(
    config: MyConfig,
    variations_folder: os.PathLike,
    results_folder: os.PathLike,
):
    """Execute energy hub models based on the stored variations one by one."""
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for variation_file in os.listdir(variations_folder):
        # compare the content in variations_folder and results_folder
        # if {name}.csv in variations_folder and {name}_pareto.csv in results_folder, skip
        # put the rest in a list and execute them one by one
        # if variation_file.endswith(".yml"):

        if variation_file.endswith(".yml"):
            variation_filename = variation_file.split(".")[0]
            if f"{variation_filename}_pareto.csv" in os.listdir(results_folder):
                print(f"Skipping {variation_file}...")
                continue

            print(f"Executing {variation_file}...")
            variation_path = os.path.join(variations_folder, variation_file)
            energy_hub = EnergyHub(config.buildings, variation_path)
            energy_hub.get_pareto_front(store_folder=results_folder)
            energy_hub.df_pareto.to_csv(
                os.path.join(results_folder, f"{variation_filename}_pareto.csv"),
                index=True,
            )
            energy_hub.df_cost_per_tech.to_csv(
                os.path.join(results_folder, f"{variation_filename}_cost_per_tech.csv"),
                index=True,
            )
            del energy_hub
            gc.collect()


def extract_sensitivity_values(results_folder, problem, threshold=1e-3):
    """
    Extract sensitivity values from the batch of results.

    Computes the number of effective Pareto points for each variation by removing
    points that are closer than the specified L2 distance threshold.

    :param results_folder: Folder containing the result CSV files.
    :param problem: Dictionary containing the problem definition for sensitivity analysis.
    :param threshold: L2 distance threshold for filtering near-duplicate points.
    :return: Sensitivity indices (Si) computed from the number of effective Pareto points.
    """
    counts = []
    variation_indices = []
    for result_file in os.listdir(results_folder):
        if result_file.endswith("_pareto.csv"):
            df = pd.read_csv(os.path.join(results_folder, result_file))
            pareto_points = df[["cost", "emission"]].values

            # Compute pairwise distances
            dist_matrix = squareform(pdist(pareto_points, "euclidean"))

            # Initialize list to keep track of points to keep
            num_points = len(pareto_points)
            keep_indices = set(range(num_points))

            for i in range(num_points):
                if i in keep_indices:
                    # Find indices of points that are within the threshold distance
                    close_indices = set(np.where(dist_matrix[i] < threshold)[0])
                    close_indices.discard(i)  # Exclude the point itself
                    # Remove close points from consideration
                    keep_indices -= close_indices

            effective_points = pareto_points[list(keep_indices)]
            counts.append(len(effective_points))
            variation_indices.append(result_file.split("_")[1])

    counts = np.array(counts)
    # export the index of variation and the number of effective points to a csv file
    df = pd.DataFrame({"variation": variation_indices, "effective_points": counts})
    df.to_csv(os.path.join(results_folder, "effective_points.csv"), index=False)

    # Si = sobol.analyze(problem, counts)
    # return Si


# Example usage
if __name__ == "__main__":
    config = MyConfig(Configuration())
    original_yaml_path = r"cea_energy_hub_optimizer\data\energy_hub_config.yml"
    sensitivity_setting_csv_path = r"C:\Users\yiqwang\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_large\problem.csv"
    if os.getlogin() == "yiqwang":
        # public computer, specify the first part of path
        path_first_part = r"C:\Users\yiqwang"
    elif os.getlogin() == "wangy":
        # own computer, specify the first part of path
        path_first_part = r"C:\Users\wangy"
    else:
        raise ValueError("Unknown user, please specify the first part of the path.")

    # variations_folder = r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\variation_global_supply"
    # results_folder = r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\result_global_supply"

    variations_folder = os.path.join(
        path_first_part,
        r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_large\variation",
    )
    results_folder = os.path.join(
        path_first_part,
        r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_large\result",
    )
    num_samples = 64  # better be power of 2
    method = "sobol"  # Set to "sobol" or "screening"

    if not os.path.exists(variations_folder):
        os.makedirs(variations_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # problem = generate_variations(
    #     sensitivity_setting_csv_path,
    #     original_yaml_path,
    #     variations_folder,
    #     num_samples,
    #     method,
    # )
    warnings.filterwarnings("ignore")
    execute_energy_hub_models(config, variations_folder, results_folder)
    # threshold = 1e-3  # Set the L2 distance threshold

    # df = pd.read_csv(sensitivity_setting_csv_path)

    # problem = {
    #     "num_vars": len(df),
    #     "names": df["name"].tolist(),
    #     "bounds": df[["min", "max"]].values.tolist(),
    # }

    # # Si = extract_sensitivity_values(results_folder, problem, threshold)
    # # print(Si)
    # extract_sensitivity_values(results_folder, problem, threshold)
