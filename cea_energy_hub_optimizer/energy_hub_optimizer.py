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
        if (building_name+'_pareto.csv' in os.listdir(store_folder)) and (config.energy_hub_optimizer.skip_done_building == True):
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

if __name__ == '__main__':
    main(cea.config.Configuration())
