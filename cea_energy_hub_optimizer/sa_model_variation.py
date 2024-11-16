from calendar import c
import gc
import re
from unittest import result
import warnings
from cea_energy_hub_optimizer.energy_hub import EnergyHub
from calliope import AttrDict
from cea_energy_hub_optimizer.my_config import MyConfig
from cea_energy_hub_optimizer.sa_cost_estimation import power
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
from typing import Union, List


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

    def generate_variations(self, num_samples: int = 8, calc_second_order=False):
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
            samples = sobol.sample(
                self.problem, N=num_samples, calc_second_order=calc_second_order
            )
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
            if key == "DH.a":
                # do this because
                k1, b1 = power(a=float(value), b=0.5512, min_value=15.0, max_value=50.0)
                yaml_content.set_key("tech.DH_15_50.costs.monetary.energy_cap", k1)
                yaml_content.set_key("tech.DH_15_50.costs.monetary.purchase", b1)
                k2, b2 = power(
                    a=float(value), b=0.5512, min_value=50.0, max_value=200.0
                )
                yaml_content.set_key("tech.DH_50_200.costs.monetary.energy_cap", k2)
                yaml_content.set_key("tech.DH_50_200.costs.monetary.purchase", b2)
                k3, b3 = power(
                    a=float(value), b=0.5512, min_value=200.0, max_value=500.0
                )
                yaml_content.set_key("tech.DH_200_500.costs.monetary.energy_cap", k3)
                yaml_content.set_key("tech.DH_200_500.costs.monetary.purchase", b3)
                k4, b4 = power(
                    a=float(value), b=0.5512, min_value=500.0, max_value=2000.0
                )
                yaml_content.set_key("tech.DH_500_2000.costs.monetary.energy_cap", k4)
                yaml_content.set_key("tech.DH_500_2000.costs.monetary.purchase", b4)
            else:
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
    def get_effective_pareto_points(
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

    @staticmethod
    def count_active_technologies(result_file: os.PathLike) -> Tuple[float, float]:
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

    @staticmethod
    def count_specific_technology_activation(
        result_file: os.PathLike, tech_name: str
    ) -> float:
        """
        Analyze activation of a specific technology across Pareto solutions.

        :param result_file: Path to the result file.
        :param tech_name: Name of the technology to analyze.
        :return: A tuple (average_activation, activation_rate_change).
        """
        df = pd.read_csv(result_file)
        if tech_name not in df.columns:
            raise ValueError(f"Technology '{tech_name}' not found in the result file.")

        # Check activation of the specific technology
        # tech_activation = df[tech_name][df[tech_name] > 0]
        tech_activation = (df[tech_name] > 0).sum() / len(df)

        # # Calculate the average activation (fraction of Pareto fronts where tech is active)
        # activation_avg = tech_activation.mean()

        # # Calculate the standard deviation
        # activation_std = tech_activation.std()

        # return activation_avg, activation_std
        return tech_activation

    @staticmethod
    def get_average_cost(result_file: os.PathLike) -> Tuple[float, float]:
        """
        Get the average cost of each Pareto solution.

        :param result_file: Path to the result file.
        :return: A tuple (average, standard deviation) of (cost, emission).
        """
        df = pd.read_csv(result_file)
        avg_cost = float(df["cost"].mean())
        avg_emission = float(df["emission"].mean())
        return avg_cost, avg_emission

    @staticmethod
    def get_slope(result_file: os.PathLike, ignore_ends=False) -> float:
        """
        Get the slope of the Pareto front.

        :param result_file: Path to the result file.
        :param ignore_ends: Ignore the first and last points.
        :return: The slope (cost over emission, normally negative) of the Pareto front.
        """
        df = pd.read_csv(result_file)
        if ignore_ends:
            df = df.iloc[1:-1]
        cost = df["cost"].values
        emission = df["emission"].values
        slope = float((cost[-1] - cost[0]) / (emission[-1] - emission[0]))
        return slope

    def analyze_results(
        self,
        threshold: float = 1e-3,
        to_file: bool = False,
        tech_specific: bool = False,
    ) -> pd.DataFrame:
        """
        Analyze results by reading the variations record and compiling a DataFrame with parameters
        for sensitivity analysis.

        :param threshold: Threshold for filtering near-duplicate Pareto points.
        :param record_all_technologies: If True, analyze all technologies in the result files.

        :return: A DataFrame with sensitivity analysis data.
        :rtype: pd.DataFrame
        """
        method = self.method  # Use the instance attribute
        variations_df = self.get_variation_df()
        data_records = {}

        for idx, row in variations_df.iterrows():
            # if row["variation_id"] is not a number, split the string and get the number
            variation_id = row["variation_id"]
            if not isinstance(variation_id, int):
                variation_id = int(variation_id.split("_")[-1])
            else:
                variation_id = int(variation_id)
            # Build the expected result file name
            result_file = f"variation_{variation_id}_{method}_pareto.csv"
            cost_file = f"variation_{variation_id}_{method}_cost_per_tech.csv"
            result_file_path = os.path.join(self.results_folder, result_file)
            cost_file_path = os.path.join(self.results_folder, cost_file)

            if os.path.exists(result_file_path) and os.path.exists(cost_file_path):
                effective_points_count = self.get_effective_pareto_points(
                    result_file_path, threshold
                )
                print(
                    f"effective points count is done for variation {variation_id}, count = {effective_points_count}"
                )
                avg_activated_techs, std_activated_techs = (
                    self.count_active_technologies(result_file_path)
                )
                # active_techs_count = self.count_active_techs(result_file_path)
                # Collect parameters and outputs
                parameters = row.drop("variation_id").to_dict()
                record = parameters.copy()
                record.update(
                    {
                        "effective_points": effective_points_count,
                        "activated_techs_avg": avg_activated_techs,
                        "activated_techs_std": std_activated_techs,
                        "average_cost": self.get_average_cost(result_file_path)[0],
                        "average_emission": self.get_average_cost(result_file_path)[1],
                        "slope": abs(self.get_slope(result_file_path)),
                        # Additional analysis can be added here
                    }
                )
                if tech_specific:
                    df_result = pd.read_csv(result_file_path)
                    # Assuming technology columns start from the fifth column
                    tech_columns = df_result.columns[4:]
                    for tech_name in tech_columns:
                        avg_activation = self.count_specific_technology_activation(
                            result_file_path, tech_name
                        )
                        record.update(
                            {
                                f"{tech_name}_activation_avg": avg_activation,
                                # f"{tech_name}_activation_std": activation_rate_change,
                            }
                        )
                data_records[variation_id] = record
            else:
                print(f"Warning: Result file not found for variation {variation_id}")

            print("statistical analysis is done for variation ", variation_id)

        df = pd.DataFrame.from_dict(data_records, orient="index")
        df.index.name = "variation_id"
        if data_records and to_file:
            df.to_csv(
                os.path.join(self.results_folder, "sensitivity_analysis_data.csv"),
                index=True,
            )
            print(
                f"Data records are saved as {os.path.join(self.results_folder, 'sensitivity_analysis_data.csv')}"
            )
        else:
            print("No data records to save.")

        return df

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
    sensitivity_setting_csv_path = r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_large\problem.csv"

    path_first_part = os.path.join(r"C:\Users", os.getlogin())

    variations_folder = os.path.join(
        path_first_part,
        r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_large",
        r"variation",
    )
    results_folder = os.path.join(
        path_first_part,
        r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_large",
        r"result",
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
    # warnings.filterwarnings("ignore")
    # sa.execute_energy_hub_models()

    df = sa.analyze_results(threshold=1e-3, to_file=True, tech_specific=True)
