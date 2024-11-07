"""
Creates an optimization plugin for building energy hub using Calliope for the City Energy Analyst.
"""

from __future__ import division
from __future__ import print_function
from cea_energy_hub_optimizer.energy_hub import EnergyHub
from cea_energy_hub_optimizer.my_config import MyConfig
import warnings
import cea.config
import cea.inputlocator
import cea.plugin
import os
import calliope


__author__ = "Yiqiao Wang"
__copyright__ = "Copyright 2024, Yiqiao Wang"
__credits__ = ["Yiqiao Wang"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Yiqiao Wang"
__email__ = "yiqwang@ethz.ch / wangyiqiao97@gmail.com"
__status__ = "Production"


class EnergyHubOptimizer(cea.plugin.CeaPlugin):
    """
    Define the plugin class - unless you want to customize the behavior, you only really need to declare the class. The
    rest of the information will be picked up from ``default.config``, ``schemas.yml`` and ``scripts.yml`` by default.
    """

    pass


def main(config: cea.config.Configuration) -> None:
    """main main function for the energy hub optimizer.

    following the config, this function initializes each building as an energy hub and optimizes for a pareto front of this building.
    The results are saved in the subfolder 'calliope_energy_hub' in the optimization results folder, as a csv file.

    Args:
        config (cea.config.Configuration): this is the configuration object that is passed by the CEA scripts.
            User can modify their config using the CEA GUI.
    """
    # initialize the singleton class which contains all the config attributes that are needed in the script
    my_config = MyConfig(config)
    check_solar_technology()
    warnings.filterwarnings("ignore")
    locator = cea.inputlocator.InputLocator(config.scenario)
    buildings: list[str] = ["B162298"]
    yaml_path = my_config.technology_definition_file
    if yaml_path == "":
        yaml_path = os.path.join(
            os.path.dirname(__file__), "data", "energy_hub_config.yml"
        )
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            "Technology definition file not found. Please define technology data and build your .yml file using Calliope Config Constructor first."
        )

    if yaml_path.split(".")[-1] not in ["yaml", "yml"]:
        raise ValueError(
            "The technology definition file must be a .yaml or .yml file. Please provide a valid .yaml or .yml file."
        )
    store_folder: str = locator._ensure_folder(
        locator.get_optimization_results_folder(), "calliope_energy_hub"
    )
    calliope.set_log_verbosity(
        verbosity="error", include_solver_output=False, capture_warnings=False
    )
    if my_config.optimize_as_district:
        print(
            "Co-optimization is enabled, all buildings will be optimized in one shot."
        )
        energy_hub = EnergyHub(buildings, yaml_path)
        energy_hub.get_pareto_front(store_folder=store_folder)
        energy_hub.df_pareto.to_csv(store_folder + "/global_pareto.csv", index=True)
    else:
        print("Co-optimization is disabled, buildings will be optimized one by one.")
        for building in buildings:
            building_name = str(building)
            if (building_name + "_pareto.csv" in os.listdir(store_folder)) and (
                my_config.skip_optimized_building is True
            ):
                # in case the user has done some buildings and don't want to redo them all over again
                print(building_name + " is already done, skipping...")
                continue

            energy_hub = EnergyHub(building_name, yaml_path)
            energy_hub.get_pareto_front(store_folder=store_folder)
            # print(energy_hub.df_pareto.to_string())
            energy_hub.df_pareto.to_csv(
                store_folder + "/" + building_name + "_pareto.csv",
                index=True,
            )
            print(building_name + " is optimized! Results saved in " + store_folder)
            del energy_hub


def check_solar_technology() -> None:
    """check_solar_technology ensures that all building that are to be optimized have the necessary solar technology results.

    This function replaced the input check in the script.yml file, because there one needs to specify the technologies
    beforehand. In our script, we have the freedom to include the technologies we want to evaluate, so we need to read
    the choices first and check them more dynamically.

    Raises:
        FileNotFoundError: if not all buildings have the necessary solar technology results, the script is aborted.

    Args:
        config (cea.config.Configuration): this is the configuration object that is passed by the CEA scripts.
            User can modify their config using the CEA GUI.
    """
    config = MyConfig()
    tech_list = config.evaluated_solar_supply
    print(
        f"""
          Checking if solar technology already has been pre-evaluated by CEA...
          Evaluated solar supply: {tech_list}
          """
    )
    locator = config.locator
    errors = []

    for building in config.buildings:
        if "PV" in tech_list:
            if not os.path.exists(locator.PV_results(building)):
                errors.append(
                    f"{building} does not have PV results in the scenario folder."
                )

        if "PVT" in tech_list:
            if not os.path.exists(locator.PVT_results(building)):
                errors.append(
                    f"{building} does not have PVT results in the scenario folder."
                )

        if "SCET" in tech_list:
            if not os.path.exists(locator.SC_results(building, "ET")):
                errors.append(
                    f"{building} does not have SCET results in the scenario folder."
                )

        if "SCFP" in tech_list:
            if not os.path.exists(locator.SC_results(building, "FP")):
                errors.append(
                    f"{building} does not have SCFP results in the scenario folder."
                )

    if len(errors) > 0:
        for error in errors:
            print(error)
        print(
            "Not all buildings have results for the solar technologies used in optimization. Script aborted."
        )
        raise FileNotFoundError(
            "Not all buildings have results for the solar technologies used in optimization. Script aborted."
        )

    print(
        "All buildings have results for the solar technologies. Continue with the optimization."
    )


if __name__ == "__main__":
    main(cea.config.Configuration())
