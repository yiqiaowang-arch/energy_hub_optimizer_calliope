from cea.config import Configuration
from cea.inputlocator import InputLocator
from typing import Optional, List


class MyConfig:
    _instance: Optional["MyConfig"] = None
    cea_config: Optional[Configuration] = None

    def __new__(cls, cea_config: Optional[Configuration] = None):
        if cls._instance is None:
            if cea_config is None:
                raise ValueError(
                    "cea_config must be provided when creating the first instance of MyConfig"
                )
            cls._instance = super(MyConfig, cls).__new__(cls)
            cls._instance.cea_config = cea_config
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Since cea config content cannot be checked by static type checkers, we need to manually define the types of the attributes here.
        """
        self.locator: InputLocator = InputLocator(self.cea_config.scenario)  # type: ignore
        # fmt: off
        # energy_hub_optimizer.py/main
        self.buildings:                     List[str]   = self.cea_config.energy_hub_optimizer.buildings                        # type: ignore
        self.technology_definition_file:    str         = self.cea_config.energy_hub_optimizer.technology_definition_file       # type: ignore
        self.optimize_as_district:          bool        = self.cea_config.energy_hub_optimizer.optimize_as_district             # type: ignore
        self.skip_optimized_building:       bool        = self.cea_config.energy_hub_optimizer.skip_optimized_building          # type: ignore
        # energy_hub.py
        self.number_of_epsilon_cut:         int         = self.cea_config.energy_hub_optimizer.number_of_epsilon_cut            # type: ignore
        self.approach_but_not_land_on_tip:  bool        = self.cea_config.energy_hub_optimizer.approach_but_not_land_on_tip     # type: ignore
        self.approach_percentile:           float       = self.cea_config.energy_hub_optimizer.approach_percentile              # type: ignore
        self.save_constraint_to_lp:         bool        = self.cea_config.energy_hub_optimizer.save_constraint_to_lp            # type: ignore
        self.save_energy_hub_to_yaml:       bool        = self.cea_config.energy_hub_optimizer.save_energy_hub_to_yaml          # type: ignore
        self.save_result_to_nc:             bool        = self.cea_config.energy_hub_optimizer.save_result_to_nc                # type: ignore
        self.get_current_solution:          bool        = self.cea_config.energy_hub_optimizer.get_current_solution             # type: ignore
        self.temporal_resolution:           str         = self.cea_config.energy_hub_optimizer.temporal_resolution              # type: ignore
        self.solver:                        str         = self.cea_config.energy_hub_optimizer.solver                           # type: ignore
        # self.use_temperature_sensitive_cop: bool        = self.cea_config.energy_hub_optimizer.use_temperature_sensitive_cop    # type: ignore
        self.flatten_spike:                 bool        = self.cea_config.energy_hub_optimizer.flatten_spike                    # type: ignore
        self.flatten_spike_percentile:      float       = self.cea_config.energy_hub_optimizer.flatten_spike_percentile         # type: ignore
        # timeseries.py
        self.evaluated_demand:              List[str]   = self.cea_config.energy_hub_optimizer.evaluated_demand                 # type: ignore
        self.evaluated_solar_supply:        List[str]   = self.cea_config.energy_hub_optimizer.evaluated_solar_supply           # type: ignore
        self.exergy_efficiency:             float       = self.cea_config.energy_hub_optimizer.exergy_efficiency                # type: ignore
        # fmt: on
