from cea_energy_hub_optimizer.sa_model_variation import SensitivityAnalysis
from cea_energy_hub_optimizer.my_config import MyConfig
from cea.config import Configuration
import os
import warnings


config = MyConfig(Configuration())
config.buildings = ["B162298"]
config.number_of_epsilon_cut = 5
config.approach_but_not_land_on_tip = False
config.temporal_resolution = "1D"
config.solver = "cplex"
config.use_temperature_sensitive_cop = True
config.exergy_efficiency = 0.52
config.flatten_spike = True
config.flatten_spike_percentile = 0.02
config.evaluated_demand = [
    "demand_electricity",
    "demand_hot_water",
    "demand_space_heating",
]
config.evaluated_solar_supply = ["PV", "SCET", "SCFP"]
path_first_part = os.path.join(
    r"C:\Users",
    os.getlogin(),
    r"OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\global_conversion_B162298",
)
base_yaml_path = os.path.join(
    path_first_part,
    "energy_hub_config_conversion_sensitivity.yml",
)
sensitivity_csv_path = os.path.join(
    path_first_part,
    "sobol_parameters_emission.csv",
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
    r"cea_energy_hub_optimizer\data\energy_hub_config_conversion_sensitivity.yml",  # with only one size per boiler
    r"cea_energy_hub_optimizer\data\sobol_parameters_conversion.csv",
    variations_folder,
    results_folder,
    "sobol",
)
# sa.generate_variations(num_samples=32, calc_second_order=False)
# warnings.filterwarnings("ignore")
sa.execute_energy_hub_models()
df = sa.analyze_results(threshold=0.001, to_file=True, tech_specific=True)
