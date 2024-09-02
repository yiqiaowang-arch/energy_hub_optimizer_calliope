"""
Creates an optimization plugin for building energy hub using Calliope for the City Energy Analyst.
"""

from __future__ import division
from __future__ import print_function
from cea_energy_hub_optimizer.energy_hub import EnergyHub
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
    check_solar_technology(config)
    warnings.filterwarnings("ignore")
    locator = cea.inputlocator.InputLocator(config.scenario)
    buildings: list[str] = config.energy_hub_optimizer.buildings
    yaml_path = os.path.join(os.path.dirname(__file__), "data", "techs_energy_hub.yml")
    store_folder: str = locator._ensure_folder(
        locator.get_optimization_results_folder(), "calliope_energy_hub"
    )
    calliope.set_log_verbosity(
        verbosity="error", include_solver_output=False, capture_warnings=False
    )

    for building in buildings:
        building_name = str(building)
        if (building_name + "_pareto.csv" in os.listdir(store_folder)) and (
            config.energy_hub_optimizer.skip_optimized_building is True
        ):
            # in case the user has done some buildings and don't want to redo them all over again
            print(building_name + " is already done, skipping...")
            continue

        energy_hub = EnergyHub(
            name=building, locator=locator, calliope_yaml_path=yaml_path, config=config
        )

        energy_hub.getParetoFront(
            epsilon=config.energy_hub_optimizer.number_of_epsilon_cut,
            store_folder=store_folder,
            approach_tip=config.energy_hub_optimizer.approach_but_not_land_on_tip,
            approach_percentile=config.energy_hub_optimizer.approach_percentile,
            to_lp=config.energy_hub_optimizer.save_constraint_to_lp,
            to_yaml=config.energy_hub_optimizer.save_energy_hub_to_yaml,
            to_nc=config.energy_hub_optimizer.save_result_to_nc,
        )

        if config.energy_hub_optimizer.get_current_solution:
            energy_hub.getCurrentCostEmission()
        df_pareto_aug = energy_hub.df_pareto.merge(
            energy_hub.df_tech_cap_pareto, left_index=True, right_index=True
        )
        df_pareto_aug.to_csv(
            store_folder + "/" + building_name + "_pareto.csv",
            index=True,
            index_label="index",
        )
        print(building_name + " is optimized! Results saved in " + store_folder)
        del energy_hub


def check_solar_technology(config: cea.config.Configuration) -> None:
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
    tech_list = config.energy_hub_optimizer.evaluated_solar_supply
    print(
        f"""
          Checking if solar technology already has been pre-evaluated by CEA...
          Evaluated solar supply: {tech_list}
          """
    )
    locator = cea.inputlocator.InputLocator(config.scenario)
    buildings: list[str] = config.energy_hub_optimizer.buildings
    errors = []

    for building in buildings:
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
