"""
Creates an optimization plugin for building energy hub using Calliope for the City Energy Analyst.
"""
from __future__ import division
from __future__ import print_function
from energy_hub import EnergyHub
import cea.config
import cea.inputlocator
import cea.plugin
import os


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


def summarize(total_demand_df, fudge_factor):
    """
    Return only the following fields from the Total_demand.csv file:
    - Name (Unique building ID)
    - GFA_m2 (Gross floor area)
    - QC_sys_MWhyr (Total system cooling demand)
    - QH_sys_MWhyr (Total building heating demand)
    """
    result_df = total_demand_df[["Name", "GFA_m2", "QC_sys_MWhyr", "QH_sys_MWhyr"]].copy()
    result_df["QC_sys_MWhyr"] *= fudge_factor
    result_df["QH_sys_MWhyr"] *= fudge_factor
    return result_df


def main(config: cea.config.Configuration):
    """
    This is the main entry point to your script. Any parameters used by your script must be present in the ``config``
    parameter. The CLI will call this ``main`` function passing in a ``config`` object after adjusting the configuration
    to reflect parameters passed on the command line / user interface

    :param cea.config.Configuration config: The configuration for this script, restricted to the scripts parameters.
    :return: None
    """
    locator = cea.inputlocator.InputLocator(config.scenario)
    buildings: list[str] = config.energy_hub_optimizer.buildings
    yaml_path = './techs_energy_hub.yml'
    store_folder: str = locator._ensure_folder(locator.get_optimization_results_folder(), 'calliope_energy_hub')
    # comments on _ensure_folder:
    # Return the *components joined together as a path to a folder and ensure that that folder exists on disc. 
    # If it doesn't exist yet, attempt to make it with os.makedirs.
    max_retries = config.energy_hub_optimizer.max_retry
    for building in buildings:
        retry_count =0
        building_name = str(building)
        if (building_name+'_pareto.csv' in os.listdir(store_folder)) and (config.energy_hub_optimizer.skip_done_building == True):
            # in case the user has done some buildings and don't want to redo them all over again
            print(building_name+' is already done, skipping...')
            continue
        while retry_count < max_retries:
            try:
                energy_hub = EnergyHub(name=building, locator=locator, calliope_yaml_path=yaml_path)
                # energy_hub.set_building_specific_config()
                if energy_hub.building_status['no_heat'] == True: # if the building has no heating system, not worthy to optimize because it's just a pavilion
                    continue

                energy_hub.get_pareto_front(epsilon=config.energy_hub_optimizer.number_of_epsilon_cuts, 
                                            store_folder=store_folder,
                                            flatten_spikes=config.energy_hub_optimizer.flatten_spike, 
                                            flatten_percentile=config.energy_hub_optimizer.flatten_spike_percentile, 
                                            approach_tip=config.energy_hub_optimizer.approach_but_not_land_on_tip,
                                            to_lp=config.energy_hub_optimizer.save_constraint_to_lp, 
                                            to_yaml=config.energy_hub_optimizer.save_energy_hub_to_yaml,
                                            to_nc=config.energy_hub_optimizer.save_result_to_nc)
                
                if config.energy_hub_optimizer.get_current_solution:
                    energy_hub.get_current_cost_emission()
                df_pareto_aug = energy_hub.df_pareto.merge(energy_hub.df_tech_cap_pareto, left_index=True, right_index=True)
                df_pareto_aug.to_csv(store_folder+'/'+building_name+'_pareto.csv')
                print(building_name+' is optimized! Results saved in ' + store_folder)
                del energy_hub
                break
            except OSError:
                retry_count += 1
                print(f'OSError ignored! Retry {retry_count}/{max_retries}.')
                if retry_count >= max_retries:
                    print(f'Max retries reached. Stopping optimization for {building_name}.')
                    break

if __name__ == '__main__':
    main(cea.config.Configuration())
