from calendar import c
import gc
from unittest import result
import warnings
from cea_energy_hub_optimizer.energy_hub import EnergyHub
from calliope import AttrDict
from cea_energy_hub_optimizer.my_config import MyConfig
import yaml
import os
import pandas as pd
import numpy as np
from typing import Tuple
from cea.config import Configuration
from SALib.sample import sobol_sequence
from SALib.analyze import sobol
from scipy.spatial.distance import pdist, squareform
from SALib.sample import sobol
from typing import Union


class SensitivityAnalysis:
    def __init__(
        self,
        config: MyConfig,
        original_yaml_path: os.PathLike,
        sensitivity_setting_csv_path: os.PathLike,
        variations_folder: os.PathLike,
        results_folder: os.PathLike,
        method: str = "sobol",
    ):
        """
        Initialize the SensitivityAnalysis object with file paths, configuration, and method.

        :param config: Configuration object.
        :param original_yaml_path: Path to the original YAML file.
        :param sensitivity_setting_csv_path: Path to the CSV file with sensitivity settings.
        :param variations_folder: Folder to store variations.
        :param results_folder: Folder to store results.
        :param method: Sampling method used ("sobol" or "screening").
        """
        self.config = config
        self.original_yaml_path = original_yaml_path
        self.sensitivity_setting_csv_path = sensitivity_setting_csv_path
        self.variations_folder = variations_folder
        self.results_folder = results_folder
        self.method = method  # Store method as an instance attribute

        self.problem = None  # Will be set after generating variations

        # Ensure folders exist
        if not os.path.exists(self.variations_folder):
            os.makedirs(self.variations_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def generate_variations(self, num_samples: int = 8):
        """
        Generate variations of the original YAML file based on the sampling method.

        :param num_samples: Number of samples for generating variations.
        """
        df = pd.read_csv(self.sensitivity_setting_csv_path)
        self.problem = {
            "num_vars": len(df),
            "names": df["name"].tolist(),
            "bounds": df[["min", "max"]].values.tolist(),
        }

        variations_records = []
        base_values = df["base"].values.tolist()

        method = self.method  # Use the instance attribute

        if method == "sobol":
            samples = sobol.sample(self.problem, N=num_samples)
        elif method == "screening":
            samples = []
            for i in range(self.problem["num_vars"]):
                min_val, max_val = self.problem["bounds"][i]
                values = np.linspace(min_val, max_val, num_samples)
                for val in values:
                    sample = base_values.copy()
                    sample[i] = val
                    samples.append(sample)
        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        for i, sample in enumerate(samples):
            modifications = {
                name: value
                for name, value, base in zip(self.problem["names"], sample, base_values)
                if value != base
            }

            if method == "screening":
                print(f"Variation {i}: {modifications}")
            elif method == "sobol":
                print(f"Variation {i} is generated.")

            new_yaml_path = os.path.join(
                self.variations_folder, f"variation_{i}_{method}.yml"
            )
            self.modify_yaml(self.original_yaml_path, modifications, new_yaml_path)

            record = {"variation_id": i}
            record.update(modifications)
            variations_records.append(record)

        variations_df = pd.DataFrame(variations_records)
        variations_df.to_csv(
            os.path.join(self.variations_folder, f"variations_record_{method}.csv"),
            index=False,
        )

    @staticmethod
    def modify_yaml(original_yaml_path, modifications, new_yaml_path):
        """
        Modify the original YAML file with the given modifications and save it as a new YAML file.
        """
        with open(original_yaml_path, "r") as file:
            yaml_content = AttrDict(yaml.safe_load(file))

        for key, value in modifications.items():
            yaml_content.set_key(key, round(float(value), 3))

        with open(new_yaml_path, "w") as file:
            yaml.dump(yaml_content.as_dict(), file)

    def execute_energy_hub_models(self):
        """
        Execute energy hub models based on the stored variations one by one.
        """
        for variation_file in os.listdir(self.variations_folder):
            if variation_file.endswith(".yml"):
                variation_filename = variation_file.split(".")[0]
                if f"{variation_filename}_pareto.csv" in os.listdir(
                    self.results_folder
                ):
                    print(f"Skipping {variation_file}...")
                    continue

                print(f"Executing {variation_file}...")
                variation_path = os.path.join(self.variations_folder, variation_file)
                energy_hub = EnergyHub(self.config.buildings, variation_path)
                energy_hub.get_pareto_front(store_folder=self.results_folder)
                energy_hub.df_pareto.to_csv(
                    os.path.join(
                        self.results_folder, f"{variation_filename}_pareto.csv"
                    ),
                    index=True,
                )
                energy_hub.df_cost_per_tech.to_csv(
                    os.path.join(
                        self.results_folder, f"{variation_filename}_cost_per_tech.csv"
                    ),
                    index=True,
                )
                del energy_hub
                gc.collect()

    @staticmethod
    def extract_sensitivity_values(
        result_file: os.PathLike, threshold: float = 1e-3
    ) -> int:
        """
        Extract the number of effective Pareto points from a result file.
        """
        df = pd.read_csv(result_file)
        pareto_points = df[["cost", "emission"]].values

        dist_matrix = squareform(pdist(pareto_points, "euclidean"))
        num_points = len(pareto_points)
        keep_indices = set(range(num_points))

        for i in range(num_points):
            if i in keep_indices:
                close_indices = set(np.where(dist_matrix[i] < threshold)[0])
                close_indices.discard(i)
                keep_indices -= close_indices

        effective_points = pareto_points[list(keep_indices)]
        return len(effective_points)

    def count_active_technologies(
        self, result_file: os.PathLike
    ) -> Tuple[float, float]:
        """
        Count the number of activated technologies in each Pareto solution.

        :param result_file: Path to the result file.
        :return: A tuple (average, standard deviation) of activated technologies.
        """
        df = pd.read_csv(result_file)
        # Assuming technology columns start from the fifth column
        tech_columns = df.columns[4:]
        counts = (df[tech_columns] > 0).sum(axis=1)
        avg_count = counts.mean()
        std_count = counts.std()
        return avg_count, std_count

    def analyze_results(self, threshold: float = 1e-3, to_file: bool = False):
        """
        Analyze results by reading the variations record and compiling a DataFrame with parameters
        for sensitivity analysis.

        :param threshold: Threshold for filtering near-duplicate Pareto points.
        """
        method = self.method  # Use the instance attribute
        variations_df = self.get_variation_df()
        data_records = {}

        for idx, row in variations_df.iterrows():
            variation_id = int(row["variation_id"])
            # Build the expected result file name
            result_file = f"variation_{variation_id}_{method}_pareto.csv"
            cost_file = f"variation_{variation_id}_{method}_cost_per_tech.csv"
            result_file_path = os.path.join(self.results_folder, result_file)
            cost_file_path = os.path.join(self.results_folder, cost_file)

            if os.path.exists(result_file_path) and os.path.exists(cost_file_path):
                effective_points_count = self.extract_sensitivity_values(
                    result_file_path, threshold
                )
                avg_activated_techs, std_activated_techs = (
                    self.count_active_technologies(result_file_path)
                )
                print(
                    f"Effective points for variation {variation_id}: {effective_points_count}"
                )
                # active_techs_count = self.count_active_techs(result_file_path)
                # Collect parameters and outputs
                parameters = row.drop("variation_id").to_dict()
                record = parameters.copy()
                record.update(
                    {
                        "effective_points": effective_points_count,
                        "avg_activated_techs": avg_activated_techs,
                        "std_activated_techs": std_activated_techs,
                        # Additional analysis can be added here
                    }
                )
                data_records[variation_id] = record
            else:
                print(f"Warning: Result file not found for variation {variation_id}")

        if data_records and to_file:
            df = pd.DataFrame.from_dict(data_records, orient="index")
            df.index.name = "variation_id"
            df.to_csv(
                os.path.join(self.results_folder, "sensitivity_analysis_data.csv"),
                index=True,
            )
        else:
            print("No data records to save.")

    def get_variation_df(self) -> pd.DataFrame:
        """read the variations_record_{method}.csv and return a dataframe of that file

        :return: a dataframe of the variations record file, typically looks like this:
        :rtype: pd.DataFrame
        ```
            variation_id  name_1    name_2  name_3
            0             0.21      0.1     0.2
            1             0.22      0.2     0.3
            2             0.23      0.3     0.4
        ```

        :raises FileNotFoundError: if there's no variations record file found
        """
        variations_record_path = os.path.join(
            self.variations_folder, f"variations_record_{self.method}.csv"
        )
        if not os.path.exists(variations_record_path):
            raise FileNotFoundError(
                f"Variations record file not found: {variations_record_path}"
            )
        variations_df = pd.read_csv(variations_record_path)
        return variations_df


# Example usage
if __name__ == "__main__":
    config = MyConfig(Configuration())
    original_yaml_path = (
        r"cea_energy_hub_optimizer\data\energy_hub_config_conversion_sensitivity.yml"
    )
    sensitivity_setting_csv_path = (
        r"cea_energy_hub_optimizer\data\sobol_parameters_conversion_emission.csv"
    )

    path_first_part = os.path.join(r"C:\Users", os.getlogin())

    variations_folder = os.path.join(
        path_first_part,
        r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential",
        r"outputs\data\optimization\calliope_energy_hub\variation_global_emission_only",
    )
    results_folder = os.path.join(
        path_first_part,
        r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential",
        r"outputs\data\optimization\calliope_energy_hub\result_global_emission_only",
    )

    sa = SensitivityAnalysis(
        config=config,
        original_yaml_path=original_yaml_path,
        sensitivity_setting_csv_path=sensitivity_setting_csv_path,
        variations_folder=variations_folder,
        results_folder=results_folder,
        method="sobol",  # Set the method here
    )

    # Call methods individually for transparency
    # Generate variations
    # sa.generate_variations(num_samples=8)

    # Execute energy hub models
    warnings.filterwarnings("ignore")
    sa.execute_energy_hub_models()

    # Analyze results
    # sa.analyze_results(threshold=1e-3)
