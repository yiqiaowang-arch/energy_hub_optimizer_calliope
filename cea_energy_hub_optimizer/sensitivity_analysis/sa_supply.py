from cea_energy_hub_optimizer.sa_model_variation import SensitivityAnalysis
from cea_energy_hub_optimizer.my_config import MyConfig
from cea.config import Configuration
import os
import warnings


config = MyConfig(Configuration())
path_first_part = os.path.join(
    r"C:\Users",
    os.getlogin(),
    r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_supply_B162582",
)
base_yaml_path = os.path.join(
    path_first_part,
    "energy_hub_config.yml",
)
sensitivity_csv_path = os.path.join(
    path_first_part,
    "sobol_parameters_supply.csv",
)
variations_folder = os.path.join(
    path_first_part,
    "variation",
)
results_folder = os.path.join(
    path_first_part,
    "result",
)

sa = SensitivityAnalysis(
    config,
    r"cea_energy_hub_optimizer\data\energy_hub_config_supply_sensitivity.yml",
    r"cea_energy_hub_optimizer\data\sobol_parameters_supply.csv",
    variations_folder,
    results_folder,
    "sobol",
)
sa.generate_variations(num_samples=32, calc_second_order=False)
warnings.filterwarnings("ignore")
sa.execute_energy_hub_models()
df = sa.analyze_results(threshold=0.001, to_file=True, tech_specific=True)
