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


def main(config: cea.config.Configuration):
    """
    This is the main entry point to your script. Any parameters used by your script must be present in the ``config``
    parameter. The CLI will call this ``main`` function passing in a ``config`` object after adjusting the configuration
    to reflect parameters passed on the command line / user interface

    :param cea.config.Configuration config: The configuration for this script, restricted to the scripts parameters.
    :return: None
    """
    check_solar_technology(config)
    warnings.filterwarnings('ignore')
    locator = cea.inputlocator.InputLocator(config.scenario)
    buildings: list[str] = config.energy_hub_optimizer.buildings
    yaml_path = os.path.join(os.path.dirname(__file__), 'data', 'techs_energy_hub.yml')
    store_folder: str = locator._ensure_folder(locator.get_optimization_results_folder(), 'calliope_energy_hub')
    calliope.set_log_verbosity(verbosity='error', include_solver_output=False, capture_warnings=False)
    # comments on _ensure_folder: 
    # Return the *components joined together as a path to a folder and ensure that that folder exists on disc. 
    # If it doesn't exist yet, attempt to make it with os.makedirs.
    for building in buildings:
        building_name = str(building)
        if (building_name+'_pareto.csv' in os.listdir(store_folder)) and (config.energy_hub_optimizer.skip_optimized_building == True):
            # in case the user has done some buildings and don't want to redo them all over again
            print(building_name+' is already done, skipping...')
            continue

        energy_hub = EnergyHub(name=building, locator=locator, 
                               calliope_yaml_path=yaml_path, 
                               config=config)

        energy_hub.get_pareto_front(epsilon=config.energy_hub_optimizer.number_of_epsilon_cut, 
                                    store_folder=store_folder,
                                    approach_tip=config.energy_hub_optimizer.approach_but_not_land_on_tip,
                                    approach_percentile=config.energy_hub_optimizer.approach_percentile,
                                    to_lp=config.energy_hub_optimizer.save_constraint_to_lp, 
                                    to_yaml=config.energy_hub_optimizer.save_energy_hub_to_yaml,
                                    to_nc=config.energy_hub_optimizer.save_result_to_nc)
        
        if config.energy_hub_optimizer.get_current_solution:
            energy_hub.get_current_cost_emission()
        df_pareto_aug = energy_hub.df_pareto.merge(energy_hub.df_tech_cap_pareto, left_index=True, right_index=True)
        df_pareto_aug.to_csv(store_folder+'/'+building_name+'_pareto.csv', index=True, index_label='index')
        print(building_name+' is optimized! Results saved in ' + store_folder)
        del energy_hub


def check_solar_technology(config: cea.config.Configuration):
    """
    Check if the solar technology has been pre-evaluated by CEA. If not, raise an error.
    This function replaced the input check in the script.yml file, because there one needs to specify the technologies 
    beforehand. In our script, we have the freedom to include the technologies we want to evaluate, so we need to read
    the choices first and check them more dynamically.

    On the other hand, the demand of each building must be available anyway, so we don't need to check that in the function.
    Instead, scripts.yml will check if each building has a demand profile csv available.

    :param cea.config.Configuration config: The configuration for this script, restricted to the scripts parameters.
    """
    tech_list = config.energy_hub_optimizer.evaluated_solar_supply
    print(f"""
          Checking if solar technology already has been pre-evaluated by CEA...
          Evaluated solar supply: {tech_list}
          """)
    locator = cea.inputlocator.InputLocator(config.scenario)
    buildings: list[str] = config.energy_hub_optimizer.buildings
    errors = []

    for building in buildings:
        if "PV" in tech_list:
            if not os.path.exists(locator.PV_results(building)):
                errors.append(f"{building} does not have PV results in the scenario folder.")
                
        if "PVT" in tech_list:
            if not os.path.exists(locator.PVT_results(building)):
                errors.append(f"{building} does not have PVT results in the scenario folder.")
                
        if "SCET" in tech_list:
            if not os.path.exists(locator.SC_results(building, 'ET')):
                errors.append(f"{building} does not have SCET results in the scenario folder.")
                
        if "SCFP" in tech_list:
            if not os.path.exists(locator.SC_results(building, 'FP')):
                errors.append(f"{building} does not have SCFP results in the scenario folder.")
                
    if len(errors) > 0:
        for error in errors:
            print(error)
        print("Not all buildings have results for the solar technologies used in optimization. Script aborted.")
        raise FileNotFoundError("Not all buildings have results for the solar technologies used in optimization. Script aborted.")
    
    print("All buildings have results for the solar technologies. Continue with the optimization.")


if __name__ == '__main__':
    main(cea.config.Configuration())
