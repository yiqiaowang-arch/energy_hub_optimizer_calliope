import calliope
from building import Building
import geopandas as gpd
import os
import warnings
import gc


calliope.set_log_verbosity(verbosity='error', include_solver_output=False, capture_warnings=False)
# disable the warnings
warnings.filterwarnings('ignore')

scenario_folder = r'C:\Users\wangy\OneDrive\ETHY2FW\IDP_Personal\CEA\2050 w3'
# yaml_path = './data/technology/techs_plot8_24h.yml'
yaml_path = './data/technology/techs_plot8_simplified.yml'
store_folder = './pareto_no_wood/pareto_no_wood_test'
zone_gdf = gpd.read_file(scenario_folder+r'\inputs\building-geometry\zone.shp')


# ------------------- Main ------------------- #
zone_gdf.index = zone_gdf['Name']
for index in zone_gdf.index:
    building_name = str(index)
    if building_name+'_pareto.csv' in os.listdir(store_folder):
        print(building_name+' is already done')
        continue

    # if building_name not in ['B162298', 'B162372', 'B162382', 'B162376', 'B162393', 'B162462', 
    #                          'B162465', 'B162467', 'B162379', 'B162378', 'B162381', 'B162394', 'B162396',
    #                          'B302030808', 'B302030809', 'B302030810', 'B302030807', 'B302030811', 'B302030812', 'B302030813']: # test config, to be deleted
    #     continue
    # if building_name not in ['B162376']:
    #     continue
    try:
        building = Building(name=building_name, scenario_path=scenario_folder, calliope_yaml_path=yaml_path)
        building.set_building_specific_config()
        if building.building_status['no_heat']: # if the building has no heating system, not worthy to optimize because it's just a pavilion
            continue

        # constarin wood supply to 0.5kWh/m2 of the building area + 400m2 surroundings
        building.calliope_config.set_key(key=f'locations.{building.name}.techs.wood_supply.constraints.energy_cap_max', 
                                        value=(building.area+400)*0.5*0.001)
        # update geothermal storage max capacity
        # building.calliope_config.set_key(key=f'locations.{building.name}.techs.geothermal_boreholes.constraints.energy_cap_max',
        #                                  value=(building.area+400)*0.1) # assume 100W/m2 max yield
        
        building.get_pareto_front(epsilon=5, store_folder=store_folder,
                                flatten_spikes=True, flatten_percentile=0.98, 
                                to_lp=False, to_yaml=False)
        building.get_current_cost_emission()
        df_pareto_aug = building.df_pareto.merge(building.df_tech_cap_pareto, left_index=True, right_index=True)
        df_pareto_aug.to_csv(store_folder+'/'+building_name+'_pareto.csv')
        print(building_name+' is done!')
        del building
        gc.collect()
    except OSError:
        print('OSError ignored! Check this later.')
        continue

    