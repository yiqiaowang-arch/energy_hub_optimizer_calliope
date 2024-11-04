from cea_energy_hub_optimizer.energy_hub import EnergyHub
from calliope import AttrDict
from cea_energy_hub_optimizer.my_config import MyConfig
import yaml
import os
import pandas as pd
import numpy as np
from SALib.sample import sobol_sequence
from SALib.analyze import sobol


def modify_yaml(original_yaml_path, modifications, new_yaml_path):
    """Modify the original YAML file with the given modifications and save it as a new YAML file."""
    with open(original_yaml_path, "r") as file:
        # yaml_content = yaml.safe_load(file)
        yaml_content = AttrDict(yaml.safe_load(file))

    for key, value in modifications.items():
        #
        yaml_content.set_key(key, value)

    with open(new_yaml_path, "w") as file:
        yaml.safe_dump(yaml_content, file)


def generate_variations_from_sobol(
    csv_path, original_yaml_path, output_folder, num_samples
):
    """Generate different technology definition variations based on Sobol sampling method."""
    df = pd.read_csv(csv_path)
    problem = {
        "num_vars": len(df),
        "names": df["name"].tolist(),
        "bounds": df[["min", "max"]].values.tolist(),
    }

    samples = sobol_sequence.sample(num_samples, problem["num_vars"])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, sample in enumerate(samples):
        modifications = {name: value for name, value in zip(problem["names"], sample)}
        new_yaml_path = os.path.join(output_folder, f"variation_{i}.yml")
        modify_yaml(original_yaml_path, modifications, new_yaml_path)


def execute_energy_hub_models(config, variations_folder, results_folder):
    """Execute energy hub models based on the stored variations one by one."""
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for variation_file in os.listdir(variations_folder):
        if variation_file.endswith(".yml"):
            variation_path = os.path.join(variations_folder, variation_file)
            energy_hub = EnergyHub(config.buildings, variation_path)
            energy_hub.get_pareto_front(store_folder=results_folder)
            energy_hub.df_pareto.to_csv(
                os.path.join(results_folder, f"{variation_file}_pareto.csv"), index=True
            )


def extract_sensitivity_values(results_folder, problem):
    """Extract sensitivity values from the batch of results."""
    results = []
    for result_file in os.listdir(results_folder):
        if result_file.endswith("_pareto.csv"):
            df = pd.read_csv(os.path.join(results_folder, result_file))
            results.append(df["objective_value"].values)

    results = np.array(results)
    Si = sobol.analyze(problem, results)
    return Si


# Example usage
if __name__ == "__main__":
    config = MyConfig()
    original_yaml_path = "path/to/original/energy_hub_config.yml"
    csv_path = "path/to/sobol_parameters.csv"
    variations_folder = "path/to/sensitivity/variations"
    results_folder = "path/to/sensitivity/results"
    num_samples = 100

    generate_variations_from_sobol(
        csv_path, original_yaml_path, variations_folder, num_samples
    )
    execute_energy_hub_models(config, variations_folder, results_folder)
    problem = {
        "num_vars": len(pd.read_csv(csv_path)),
        "names": pd.read_csv(csv_path)["name"].tolist(),
        "bounds": pd.read_csv(csv_path)[["min", "max"]].values.tolist(),
    }
    Si = extract_sensitivity_values(results_folder, problem)
    print(Si)
