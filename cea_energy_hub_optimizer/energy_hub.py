import geopandas as gpd
import pandas as pd
import numpy as np
import calliope
import cea.inputlocator
import cea.config
import os

"""
set an energy hub, with the following attributes:
- name:                 str                             name of the building
- locator:              cea.inputlocator.InputLocator   locator object has multiple methods that helps with locating certain file paths
- yaml_path:            str                             path to the yaml file that contains the energy hub configuration
- config:               cea.config.Configuration        configuration object that contains the user's input in plugin.config
- emission_type:        str                             type of emission system, either 'HVAC_HEATING_AS1' or 'HVAC_HEATING_AS4'
- area:                 float                           area of the building
- location:             dict                            location of the building, with keys 'lat' and 'lon'
- calliope_config:      calliope.AttrDict               calliope configuration object from the yaml file
- dict_timeseries_df:   dict: [str, pd.DataFrame]       dictionary of timeseries dataframes, with keys 'demand_el', 'demand_sh', 'demand_dhw', 'demand_sc', 'supply_PV', 'supply_PVT_e', 'supply_PVT_h', 'supply_SCFP', 'supply_SCET'




the following are in an array of 8760 elements, each element is a real number:
appliance and demand, heating demand, cooling demand, hot water demand,
PV generation, PVT generation, solar collector generation
I want to specify only the name of building and path of the folder that contains the data
and the class will automatically read the data and store them in the class
the data is stored in multiple subfolders of the path, some are for demand, some are for generation
the data is stored in csv files, each file contains the data of one building
the file name is the building id, or builidng id + PV, or building id + PVT, or building id + sc
"""



class EnergyHub:
    def __init__(self, name: str, 
                 locator: cea.inputlocator.InputLocator, 
                 calliope_yaml_path: str, 
                 config: cea.config.Configuration):
        """
        Description:
        This function initializes the building object, which contains the building's information.

        Inputs:
        - name:                     str, the name of the building
        - locator:                  cea.inputlocator.InputLocator, the locator object has multiple methods that helps with locating certain file paths
        - calliope_yaml_path:       str, the path to the yaml file that contains the energy hub configuration
        - config:                   cea.config.Configuration, the configuration object that contains the user's input in plugin.config
        """

        self.name: str = name
        self.locator = locator
        # locator.scenario returns a str of the scenario path, which includes /inputs and /outputs
        self.yaml_path: str = calliope_yaml_path
        self.config = config
        calliope.set_log_verbosity(verbosity='error', include_solver_output=False, capture_warnings=False)
        
        # get type of emission system
        # emission_dict = {'HVAC_HEATING_AS1': 80, # radiator, needs high supply temperature
        #                  'HVAC_HEATING_AS4': 45  # floor heating, needs low supply temperature
        #                  } # output temperature of the heating emission system
        air_conditioning_df: pd.DataFrame = gpd.read_file(self.locator.get_building_air_conditioning(), ignore_geometry=True)
        air_conditioning_df.set_index(keys='Name', inplace=True)
        self.emission_type: str = str(air_conditioning_df.loc[self.name, 'type_hs'])
        # self.emission_temp: int = emission_dict[self.emission_type]

        # get building area
        zone: gpd.GeoDataFrame = gpd.read_file(self.locator.get_zone_geometry())
        zone.index = zone['Name']
        self.area: float = zone.loc[self.name, 'geometry'].area
        self.location: dict[str, float] = {'lat': float(zone.loc[self.name, 'geometry'].centroid.y),
                                           'lon': float(zone.loc[self.name, 'geometry'].centroid.x)}
        
        self.calliope_config: calliope.AttrDict = calliope.AttrDict.from_yaml(self.yaml_path)
        building_sub_dict_temp = self.calliope_config['locations'].pop('Building')
        self.calliope_config['locations'][self.name] = building_sub_dict_temp
        del building_sub_dict_temp

        # set temporal resolution
        self.calliope_config.set_key(key='model.time.function_options.resolution', value=self.config.energy_hub_optimizer.temporal_resolution)

        # set solver
        self.calliope_config.set_key(key='run.solver', value=self.config.energy_hub_optimizer.solver)
        # constarin wood supply to 0.5kWh/m2 of the building area + 400m2 surroundings
        self.calliope_config.set_key(key=f'locations.{self.name}.techs.wood_supply.constraints.energy_cap_max', 
                                            value=(self.area+400)*0.5*0.001)

        self.get_demand_supply()
        if self.config.energy_hub_optimizer.flatten_spike:
            for key in ['demand_el', 'demand_sh', 'demand_dhw', 'demand_sc']:
                self.dict_timeseries_df[key] = self.flatten_spikes(df=self.dict_timeseries_df[key], 
                                                                   column_name=self.name, 
                                                                   percentile=self.config.energy_hub_optimizer.flatten_spike_percentile)


    def get_demand_supply(self):
        """
        Description:
        This method reads the input scenario_path, following the CEA result file structure, finds the pre-computed 
        result csvs for demand (electricity (E), heating (Qhs), cooling (Qcs) and hot water (Qww)),
        along with supply from PV, PVT and flat-panel solar collectors (SC_FP). 
        Currently, each timeseries is an independent dataframe. TODO: merge all dataframes into one!
        """
        get_df = EnergyHub.get_timeseries_df # rename for simplicity
        demand_df = get_df(path=self.locator.get_demand_results_file(building=self.name, format='csv'))

        # time series data
        # read demand data
        # demand_df = demand_df[['E_sys_kWh', 'Qhs_sys_kWh', 'Qcs_sys_kWh', 'Qww_sys_kWh']]
        app: pd.DataFrame = - demand_df[['E_sys_kWh']].astype('float64').rename(columns={'E_sys_kWh': self.name})
        sh: pd.DataFrame = - demand_df[['Qhs_sys_kWh']].astype('float64').rename(columns={'Qhs_sys_kWh': self.name})
        sc: pd.DataFrame = - demand_df[['Qcs_sys_kWh']].astype('float64').rename(columns={'Qcs_sys_kWh': self.name})
        dhw: pd.DataFrame = - demand_df[['Qww_sys_kWh']].astype('float64').rename(columns={'Qww_sys_kWh': self.name})

        # if demand not included in config.energy_hub_optimizer.evaluated_demand, set to 0
        for demand_type in ['electricity', 'space-heating', 'hot-water', 'space-cooling']:
            if demand_type not in self.config.energy_hub_optimizer.evaluated_demand:
                if demand_type == 'electricity':
                    app[self.name] = 0
                elif demand_type == 'space-heating':
                    sh[self.name] = 0
                elif demand_type == 'hot-water':
                    dhw[self.name] = 0
                elif demand_type == 'space-cooling':
                    sc[self.name] = 0


        # read supply data
        if 'PV' in self.config.energy_hub_optimizer.evaluated_solar_supply:
            pv_path = self.locator.PV_results(building=self.name)
            if not os.path.exists(pv_path):
                raise FileNotFoundError(f'PV result file for building {self.name} not found at {pv_path}! Consider running the PV simulation first.')
            pv_df = get_df(path=pv_path)
            pv: pd.DataFrame = pv_df[['E_PV_gen_kWh']].astype('float64').rename(columns={'E_PV_gen_kWh': self.name})
            # prepare intensity data, because calliope can only have one area for PV, PVT, SC to compete with. 
            # For example, if building's area is 100m2, then the intensity is the generation divided by 100.
            # Then, from the perspective of calliope, we might have 50m2 of PV, 30m2 of PVT, 20m2 of SC.
            # This actually means that by carefully laying out the panels on the realistic building's facade and rooftop,
            # we can achieve 50% of the maximal PV generation, 30% of the maximal PVT generation, and 20% of the maximal SC generation.
            pv_intensity: pd.DataFrame = pv.astype('float64') / self.area
        else:
            pv_intensity = pd.DataFrame(0, index=app.index, columns=[self.name])

        if 'PVT' in self.config.energy_hub_optimizer.evaluated_solar_supply:
            pvt_path = self.locator.PVT_results(building=self.name)
            if not os.path.exists(pvt_path):
                raise FileNotFoundError(f'PVT result file for building {self.name} not found at {pvt_path}! Consider running the PVT simulation first.')
            pvt_df = get_df(path=pvt_path)
            pvt_e: pd.DataFrame = pvt_df[['E_PVT_gen_kWh']].astype('float64').rename(columns={'E_PVT_gen_kWh': self.name})
            pvt_h: pd.DataFrame = pvt_df[['Q_PVT_gen_kWh']].astype('float64').rename(columns={'Q_PVT_gen_kWh': self.name})
            pvt_e_intensity: pd.DataFrame = pvt_e.astype('float64') / self.area
            pvt_h_intensity: pd.DataFrame = pvt_h.astype('float64') / self.area
            # because in PVT, heat comes with electricity and we can't control the ratio of heat to electricity,
            # the heat production is set to be a scaled version of the electricity production.
            # and this scaling factor is pvt_h_relative_intensity
            # devide pvt_h with pvt_e element-wise to get relative intensity, which is still a dataframe.
            # replace NaN and inf with 0s
            df_pvt_h_relative_intensity = pvt_h_intensity.divide(pvt_e_intensity[self.name], axis=0).fillna(0)
            df_pvt_h_relative_intensity.replace(np.inf, 0, inplace=True)
            pvt_h_relative_intensity: pd.DataFrame = df_pvt_h_relative_intensity.astype('float64')
        else:
            pvt_e_intensity = pd.DataFrame(0, index=app.index, columns=[self.name])
            pvt_h_relative_intensity = pd.DataFrame(0, index=app.index, columns=[self.name])

        if 'SCFP' in self.config.energy_hub_optimizer.evaluated_solar_supply:
            scfp_path = self.locator.SC_results(building=self.name, panel_type='FP')
            if not os.path.exists(scfp_path):
                raise FileNotFoundError(f'Flat-panel SC result file for building {self.name} not found at {scfp_path}! Consider running the SC simulation first.')
            scfp_df = get_df(path=scfp_path) # flat panel solar collector
            scfp: pd.DataFrame = scfp_df[['Q_SC_gen_kWh']].astype('float64').rename(columns={'Q_SC_gen_kWh': self.name})
            scfp_intensity: pd.DataFrame = scfp.astype('float64') / self.area
        else:
            scfp_intensity = pd.DataFrame(0, index=app.index, columns=[self.name])

        if 'SCET' in self.config.energy_hub_optimizer.evaluated_solar_supply:
            scet_path = self.locator.SC_results(building=self.name, panel_type='ET')
            if not os.path.exists(scet_path):
                raise FileNotFoundError(f'Evacuated tube SC result file for building {self.name} not found at {scet_path}! Consider running the SC simulation first.')
            scet_df = get_df(path=scet_path) # evacuated tube solar collector
            scet: pd.DataFrame = scet_df[['Q_SC_gen_kWh']].astype('float64').rename(columns={'Q_SC_gen_kWh': self.name})
            scet_intensity: pd.DataFrame = scet.astype('float64') / self.area
        else:
            scet_intensity = pd.DataFrame(0, index=app.index, columns=[self.name])

        # scet_intensity: pd.DataFrame = scet.astype('float64') / self.area
        self.dict_timeseries_df: dict[str, pd.DataFrame] = {'demand_el':     app, # kW
                                                            'demand_sh':     sh, # kW
                                                            'demand_dhw':    dhw, # kW
                                                            'demand_sc':     sc, # kW
                                                            'supply_PV':     pv_intensity, # kW/m2
                                                            'supply_PVT_e':  pvt_e_intensity, # kW/m2
                                                            'supply_PVT_h':  pvt_h_relative_intensity, # dimensionless
                                                            'supply_SCFP':   scfp_intensity, # kW/m2
                                                            'supply_SCET':   scet_intensity, # kW/m2
                                                            }
        
    @staticmethod
    def get_timeseries_df(path: str) -> pd.DataFrame:
        """
        Description:
        This function reads the timeseries csv files from the path, and returns a dictionary of dataframes.
        Due to calliope requirements, the index of the dataframe is set to be datetime, and the column name is set to be the building name.

        """
        # if path ends with .csv, then read the csv file and return the dataframe
        # if ends with .dbf, read the dbf file and return the dataframe

        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.dbf'):
            df = gpd.read_file(path, ignore_geometry=True)
        else:
            raise ValueError('path must end with .csv or .dbf')
        
        df.fillna(0, inplace=True)
        col_mapping = {col.lower(): col for col in df.columns}
        # this is because the date column could be 'DATE' or 'Date'
        if 'date' in col_mapping:
            date_col = col_mapping['date']
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.index = df.index.tz_localize(None)
            df.index.rename('t', inplace=True)
        else:
            raise ValueError("The dataframe does not contain a 'DATE' column.")
        
        return df


    def set_building_specific_config(self):
        """
        Description:
        This function sets the building specific configuration for the building model.
        - If the building is not in the district heating area, delete the district heating technologies keys.
        - If the building is only renovated (not rebuilt), set the original GSHP and ASHP costs to 0
            because they are already installed.
            Also, if they want to change to another technology, set the costs higher because 
            changing heating system in an existing building costs more.
        - If the building is rebuilt, set the costs to normal.
        - This function assumes that no ASHP is used for Dhw in any original buildings.

        Inputs:
        - self:                     Building object
        - building_specific_config: calliope.AttrDict
        - building_status:          pd.Series

        Outputs:
        - building_specific_config: calliope.AttrDict (modified)
        """
        name = self.name

        building_status_dfs_list = [pd.read_csv(self.scenario_path+r'\inputs\is_disheat.csv', index_col=0), # when the building is included in district heating zones
                                    pd.read_csv(self.scenario_path+r'\inputs\Rebuild.csv', index_col=0).fillna(0), # when will the building be rebuilt
                                    pd.read_csv(self.scenario_path+r'\inputs\Renovation.csv', index_col=0).fillna(0), # when will the building be renovated
                                    gpd.read_file(self.scenario_path+r'\inputs\building-properties\supply_systems.dbf', 
                                                  ignore_geometry=True).set_index("Name")[['type_hs']] # building's current heating system type
                                    ]
        
        # these dataframes all have index as building names
        # we need to create a pd.Series for each building across all dataframes
        building_status_df = pd.concat(building_status_dfs_list, axis=1).fillna(0)
        building_status: object = building_status_df.loc[name]
        building_status.fillna(False, inplace=True)
        building_status['is_disheat'] = building_status['DisHeat'].astype(bool)
        building_status['is_rebuilt'] = building_status['Rebuild'].astype(bool)
        building_status['is_renovated'] = building_status['Renovation'].astype(bool)
        building_status['already_GSHP'] = building_status['type_hs'] == 'HVAC_HEATING_AS6'
        building_status['already_ASHP'] = building_status['type_hs'] == 'HVAC_HEATING_AS7'
        building_status['no_heat'] = building_status['type_hs'] == 'HVAC_HEATING_AS0'
        self.building_status = building_status

        # if building is not in district heating area, delete the district heating technologies keys
        if not building_status['is_disheat']:
            self.calliope_config.del_key(f'locations.{self.name}.techs.DHDC_small_heat')
            self.calliope_config.del_key(f'locations.{self.name}.techs.DHDC_medium_heat')
            self.calliope_config.del_key(f'locations.{self.name}.techs.DHDC_large_heat')
            
        # if building is not rebuilt, set GSHP and ASHP costs higher
        if building_status['is_rebuilt']: # rebuilt, so everything is possible and price is normal
            pass
        elif building_status['is_renovated']: # renovated, can do GSHP and ASHP but price higher
            if building_status['already_GSHP']: # already has GSHP, only need to set ASHP price higher, and GSHP price to 0
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.purchase', 0)
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.energy_cap', 0)
                
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.purchase', 18086)
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.energy_cap', 1360)
            elif building_status['already_ASHP']: # ASHP for heating no cost; but ASHP for DHW higher; also GSHP higher
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.purchase', 39934)
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.energy_cap', 1316)
                
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.purchase', 0)
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.energy_cap', 0)
            else: # no GSHP and no ASHP, set both to higher price
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.purchase', 39934)
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.energy_cap', 1316)
                
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.purchase', 18086)
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.energy_cap', 1360)
        else: # not new, so no new GSHP but new ASHP allowed; however if they are already with GSHP or ASHP, then no corresponding cost is applied
            if building_status['already_GSHP']:
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.purchase', 0)
                self.calliope_config.set_key(f'locations.{self.name}.techs.GSHP.costs.monetary.energy_cap', 0)
                
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.purchase', 18086)
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.energy_cap', 1360)
            elif building_status['already_ASHP']: # no previous GSHP, so delete GSHP keys; 
                self.calliope_config.del_key(f'locations.{self.name}.techs.GSHP')
                self.calliope_config.del_key(f'locations.{self.name}.techs.geothermal_boreholes')
                
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.purchase', 0)
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.energy_cap', 0)
            else: # no previous GSHP and no previous ASHP, so delete GSHP keys and higher ASHP keys
                self.calliope_config.del_key(f'locations.{self.name}.techs.GSHP')
                
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.purchase', 18086)
                self.calliope_config.set_key(f'locations.{self.name}.techs.ASHP.costs.monetary.energy_cap', 1360)

        del building_status_dfs_list, building_status, name


    def get_building_model(self,
                           to_lp=False, to_yaml=False,
                           obj='cost',
                           emission_constraint=None) -> calliope.Model:
        """
        Description:
        This function gets building parameters and read the scenario files to create a calliope model for the building.

        Input:
        building_status:            pd.Series, the status of the building, including is_new, is_rebuilt, already_GSHP, already_ASHP, is_disheat
        flatten_spikes:             bool, if True, flatten the demand spikes
        flatten_percentile:         float, the percentile to flatten the spikes
        to_lp:                      bool, if True, store the model in lp format
        to_yaml:                    bool, if True, store the model in yaml format
        obj:                        str, the objective function, either 'cost' or 'emission'
        emission_constraint:        float, the emission constraint

        Return:
        Model:                      calliope.Model, the optimized model
        """
        
        # modify the self.calliope_config to match the building's status
        self.calliope_config.set_key(key=f'locations.{self.name}.available_area', value=self.area)
        print('the area of building '+self.name+' is '+str(round(self.area, 1))+' m2')
        
        # if emission constraint is not None, add it to the self.calliope_config
        if emission_constraint is not None:
            self.calliope_config.set_key(key='group_constraints.systemwide_co2_cap.cost_max.co2', value=emission_constraint)
        else:
            # check if the emission constraint is already in the config, if so, delete it
            if bool(self.calliope_config.get_key('group_constraints.systemwide_co2_cap.cost_max.co2')):
                self.calliope_config.set_key(key='group_constraints.systemwide_co2_cap.cost_max.co2', value=None)
        
        # if obj is cost, set the objective to be cost; if obj is emission, set the objective to be emission
        if obj == 'cost':
            self.calliope_config.set_key(key='run.objective_options.cost_class.monetary', value=1)
            self.calliope_config.set_key(key='run.objective_options.cost_class.co2', value=0)
        elif obj == 'emission':
            self.calliope_config.set_key(key='run.objective_options.cost_class.monetary', value=0)
            self.calliope_config.set_key(key='run.objective_options.cost_class.co2', value=1)
        else:
            raise ValueError('obj must be either cost or emission')

        print(self.calliope_config.get_key('run.objective_options.cost_class'))
        model = calliope.Model(self.calliope_config, timeseries_dataframes=self.dict_timeseries_df)
        if to_lp:
            model.to_lp(self.store_folder+'/'+self.name+'.lp')
        if to_yaml:
            model.save_commented_model_yaml(self.store_folder+'/'+self.name+'.yaml')
        return model
    

    def get_pareto_front(self, epsilon:int, 
                         store_folder: str,
                         approach_tip=False, approach_percentile=0.01,
                         to_lp=False, to_yaml=False, to_nc=False):
        """
        Description:
        This function finds the pareto front of one building regarding cost and emission.
        - First, it finds the emission-optimal solution and store the cost and emission in df_pareto.
        - Then, it finds the cost-optimal solution and store the cost and emission in df_pareto.
        - Then it reads the number of epsilon cuts 
            and evenly distribute the emissions between the cost-optimal and emission-optimal solutions.
        - For each epsilon, it finds the epsilon-optimal solution and store the cost and emission in df_pareto.
        - Finally, it returns the df_pareto, which contains two columns: first cost, second emission. 
            Along with index of number of epsilon cut. 0: emission-optimal, epsilon+1: cost-optimal.
        - It also returns the df_tech_cap_pareto, which contains the technology capacities of each solution.

        Steps to achieve multi-objective optimization (epsilon-cut):
        1. prepare cost data (monetary: HSLU database; emission: KBOB, limitation: different database);
        2. Input into calliope configuration;
        3. Define available technology;
        4. Solve for min-cost $(C_L, E_L)$ and min-emission $(C_R, E_R)$;
        5. define amount of cuts (n), and primary objective (normally Cost)
        6. Divide emission range $[E_L, E_R]$ into n parts, $E_0 = E_L, E_1, ..., E_i, ..., E_n-1, E_n=E_R$;
        7. optimize for C, with constriaint of $E\\leq E_i$;
        8. get n+1 points: $(C_0, E_0) = (C_L, E_L), (C_1, E_1), ..., (C_i, E_i), ..., (C_n-1, E_n-1), (C_n, E_n) = (C_R, E_R)$
        9. link these points in a coordinate plane to form the pareto front.

        Inputs:
        - epsilon:                  int, the number of epsilon cuts between cost-optimal and emission-optimal solutions
        - building_name:            str, the name of the building
        - building_scenario_folder: str, the folder that contains the building's scenario files
        - yaml_path:                str, the path to the yaml file that contains the energy hub configuration
        - store_folder:             str, the folder that stores the results
        - building_status:          pd.Series, the status of the building, including is_new, is_rebuilt, already_GSHP, already_ASHP, is_disheat
        - flatten_spikes:           bool, if True, flatten the demand spikes
        - flatten_percentile:       float, the percentile to flatten the spikes
        - to_lp:                    bool, if True, store the model in lp format
        - to_yaml:                  bool, if True, store the model in yaml format
        
        Outputs:
        - df_pareto:                pd.DataFrame, the pareto front of the building, with cost and emission as columns
        - df_tech_cap_pareto:       pd.DataFrame, the technology capacities of each solution
        """
        if epsilon < 1:
            raise ValueError('There must be at least one epsilon cut!')
        
        if approach_tip:
            # if approach_tip is True, then there are in total epsilon+4 points in the pareto front: 
            # emission-optimal, close-to-emission-optimal, epsilon points, close-to-cost-optimal, cost-optimal
            # so we should locate cost-optimal at epsilon+3
            idx_cost = epsilon+3
            print(f'''
                  Approaching tip of Pareto Front. 
                  Adding two more epsilon cuts close to the ends. 
                  Original: {epsilon} , now: {epsilon+2}''')
        else:
            idx_cost = epsilon+1
            
        self.store_folder = store_folder
        df_pareto = pd.DataFrame(columns=['cost', 'emission'], index=range(idx_cost+1))
        # read yaml file and get the list of technologies
        tech_list = self.calliope_config.get_key(f'locations.{self.name}.techs').keys() # type: ignore
        # calliope does not define the type of the return value, so it's ignored
        df_tech_cap_pareto = pd.DataFrame(columns=tech_list, index=range(idx_cost+1))
        df_tech_cap_pareto.fillna(0, inplace=True)
        # first get the emission-optimal solution
        model_emission = self.get_building_model(to_lp=to_lp, to_yaml=to_yaml, obj='emission')
        model_emission.run()
        if to_nc:
            model_emission.to_netcdf(path=self.store_folder + '/' + self.name+'_emission.nc')
        print('optimization for emission is done')
        # store the cost and emission in df_pareto
        df_emission = model_emission.get_formatted_array('cost').sel(locs=self.name).to_pandas().transpose().sum(axis=0)
        # add the cost and emission to df_pareto
        df_pareto.loc[0] = [df_emission['monetary'], df_emission['co2']]
        # store the technology capacities in df_tech_cap_pareto
        df_tech_cap_pareto.loc[0] = model_emission.get_formatted_array('energy_cap').to_pandas().iloc[0]
        
        # then get the cost-optimal solution
        model_cost = self.get_building_model(to_lp=to_lp, to_yaml=to_yaml, obj='cost')
        # run model cost, and find both cost and emission of this result
        model_cost.run()
        if to_nc:
            model_cost.to_netcdf(path=self.store_folder  + '/' + self.name+'_cost.nc')
        print('optimization for cost is done')
        # store the cost and emission in df_pareto
        # add epsilon name as row index, start with epsilon_0
        df_cost = model_cost.get_formatted_array('cost').sel(locs=self.name).to_pandas().transpose().sum(axis=0) # first column co2, second column monetary

        df_pareto.loc[idx_cost] = [df_cost['monetary'], df_cost['co2']]
        df_tech_cap_pareto.loc[idx_cost] = model_cost.get_formatted_array('energy_cap').to_pandas().iloc[0]
        # then get the epsilon-optimal solution
        # first find out min and max emission, and epsilon emissions are evenly distributed between them
        # if cost and emission optimal have the same emission, then there's no pareto front
        if df_cost['co2'] <= df_emission['co2']:
            print(f'cost-optimal and emission-optimal of building {self.name} have the same emission, no pareto front')
            self.df_pareto = df_pareto
        else:
            emission_max =df_cost['co2']
            emission_min =df_emission['co2']
            emission_array = np.linspace(emission_min, emission_max, epsilon+2)
            epsilon_list = list(emission_array[1:-1])
            # calculate the interval between two emissions
            # for each epsilon, get the cost-optimal solution under a maximal emission constraint
            if approach_tip:
                del_emission_begin = np.diff(emission_array)[0]*approach_percentile
                del_emission_end = np.diff(emission_array)[-1]*approach_percentile
                epsilon_list = [emission_min+del_emission_begin] + epsilon_list + [emission_max-del_emission_end]
            print(f"Maximal emission: {emission_max}, minimal emission: {emission_min}, number of epsilon cuts: {idx_cost-1}")
            for i, emission_constraint in enumerate(epsilon_list):
                n_epsilon = i+1
                print(f'starting epsilon {n_epsilon}, life-time emission smaller or equal to {emission_constraint} kgCO2')
                model_epsilon = self.get_building_model(to_lp=to_lp, to_yaml=to_yaml, obj='cost', 
                                                        emission_constraint=emission_constraint)
                model_epsilon.run()
                if to_nc:
                    model_epsilon.to_netcdf(path=self.store_folder  + '/' + self.name + f'_epsilon_{n_epsilon}.nc')
                print(f'optimization at epsilon {n_epsilon} is done')
                # store the cost and emission in df_pareto
                df_epsilon = model_epsilon.get_formatted_array('cost').sel(locs=self.name).to_pandas().transpose().sum(axis=0)
                # add the cost and emission to df_pareto
                df_pareto.loc[n_epsilon] = [df_epsilon['monetary'], df_epsilon['co2']]
                # store the technology capacities in df_tech_cap_pareto
                df_tech_cap_pareto.loc[n_epsilon] = model_epsilon.get_formatted_array('energy_cap').to_pandas().iloc[0]
                
            df_pareto = df_pareto.astype({'cost': float, 'emission': float})
            print("Pareto front for building "+self.name+" is done. First row is emission-optimal, last row is cost-optimal.")
            # show the pareto front
            print(df_pareto)
            self.df_pareto = df_pareto
            self.df_tech_cap_pareto = df_tech_cap_pareto

    
    def get_current_cost_emission(self):
        """
        Description:
        This function reads the current technology setup of the building, delete all irrelevant technologies,
        then "optimize" the building (with the only feasible choice) to get the current cost and emission.
        Finally, it returns the cost and emission in a pd.Series, and add it to the self.df_pareto with index 999.

        the current tech setup is stored in self.building_status. For example, it looks like:
        DisHeat                       2025
        Rebuild                        0.0
        Renovation                     0.0
        type_hs         SUPPLY_HEATING_AS7
        is_disheat                    True
        is_rebuilt                   False
        is_renovated                 False
        already_GSHP                 False
        already_ASHP                 False
        no_heat                      False
        Name: B162298, dtype: object

        Meaning of type_hs (SUPPLY_HEATING_ASX):
        0. no heating
        1. oil boiler
        2. coal boiler
        3. gas boiler
        4. electric boiler
        5. wood boiler
        6. GSHP (ground source heat pump)
        7. ASHP (air source heat pump)
        """

        # first, delete DHDC_small_heat, DHDC_medium_heat, DHDC_large_heat, 
        # PV, SCFP, battery, DHW_storage, heaat_storage. These do not exist in the current setup
        unrealistic_tech_list = ['DHDC_small_heat', 'DHDC_medium_heat', 'DHDC_large_heat',
                                    'PV', 'SCFP', 'battery', 'DHW_storage', 'heat_storage']
        
        if self.building_status['no_heat']:
            # in this case, should raise error because there's no heating system
            raise ValueError('no heating system in the building')
        elif self.building_status['already_GSHP']:
            # in this case, delete ASHP
            unrealistic_tech_list += ['ASHP', 'wood_boiler']
        elif self.building_status['already_ASHP']:
            # in this case, delete GSHP
            unrealistic_tech_list += ['GSHP', 'wood_boiler']
        else:
            # if the building is originally using gas or oil boiler, by 2050 it should have changed to ASHP
            unrealistic_tech_list += ['GSHP', 'wood_boiler']

        for tech in unrealistic_tech_list:
            # first check if the tech exists in the building config
            if tech in self.df_tech_cap_pareto.columns:
                self.calliope_config.del_key(f'locations.{self.name}.techs.{tech}')
        
        model_current = self.get_building_model(flatten_spikes=False, 
                                                flatten_percentile=0.98, to_lp=False, to_yaml=False, 
                                                obj='cost')
        print(f'calculating current cost and emission for building {self.name}')
        model_current.run()
        print(f'current cost and emission for building {self.name} is done')
        sr_cost_current: pd.Series = model_current.get_formatted_array('cost').sel(locs=self.name).to_pandas().transpose().sum(axis=0)
        # add the cost and emission to df_pareto
        self.df_pareto.loc[999] = [sr_cost_current['monetary'], sr_cost_current['co2']]
        # store the technology capacities in df_tech_cap_pareto
        self.df_tech_cap_pareto.loc[999] = model_current.get_formatted_array('energy_cap').to_pandas().iloc[0]
        self.df_tech_cap_pareto.fillna(0, inplace=True)

    @staticmethod
    def flatten_spikes(df: pd.DataFrame, column_name, percentile: float = 0.98, is_positive: bool = False):
        # first fine non-zero values of the given column of the given dataframe
        # then calculate the 98th percentile of the non-zero values
        # then find the index of the values that are greater than the 98th percentile
        # then set the values of the index to the 98th percentile
        # then return the dataframe
        # the input dataframe should have a datetime index
        if not is_positive:
            df = - df

        nonzero_subset = df[df[column_name] != 0]
        percentile_value = nonzero_subset[column_name].quantile(1-percentile)
        df.loc[df[column_name] > percentile_value, column_name] = percentile_value

        if not is_positive:
            df = - df

        return df
