# energy_hub_optimizer
A multi-objective optimization plugin for energy hubs using Calliope, tailored for the City Energy Analyst (CEA) to enhance energy system decision-making at both building and district levels. By incorporating mixed-integer linear programming (MILP), this plugin enables CEA to achieve globally optimal solutions for energy system design challenges. Designed for seamless integration, the plugin automatically handles CEA inputs and outputs, streamlining the entire optimization process into a single software solution—eliminating the need for multiple tools.

The plugin offers flexibility to optimize all buildings in a district either individually or collectively, based on user preferences. Equipped with an up-to-date technology database for Zurich, Switzerland, it includes a user-friendly, Excel-based technology definition file. This feature allows users to easily customize default technology definitions or create new variations effortlessly.

For advanced users, the plugin provides APIs for standalone operation, enabling detailed analysis of energy hub models without relying on a graphical user interface (GUI). Optimization results, including technology sizing and associated costs (both monetary and emissions), are stored in .csv format in the CEA outputs folder for easy access and further analysis.

Example visualization of optimization results:
![Pareto Front Examples](https://github.com/user-attachments/assets/e6f20ca6-73d6-497d-804c-cf3c73bede6a)

![technology variation](https://github.com/user-attachments/assets/4a04dcaf-5c1c-4a24-9a9c-7403cc20d7ed)

![cost composition](https://github.com/user-attachments/assets/b2ee6f91-b635-4762-a0b1-23a061549746)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Download
To install, clone this repo to a desired path (you would need to have `git` installed to run this command. Alternatively you can also run this command in the CEA console, which
comes with `git` pre-installed):

```git clone https://github.com/yiqiaowang-arch/energy_hub_optimizer_calliope.git DESIRED_PATH```

## Installation of the plugin
Open CEA console and enter the following command to install the plugin to CEA. In this command, -e means editable, which allows the plugin to be updated without re-installing. You can also check out the repository using your text editor and make any local changes you like, in order to alter the functionality of this plugin.

```pip install -e PATH_OF_PLUGIN_FOLDER```
For example:
    
    ```pip install -e "C:/path/to/the/repository/energy_hub_optimizer_calliope"```

(NOTE: PATH_OF_PLUGIN_FOLDER would be the DESIRED_PATH + 'cea_energy_hub_optimizer')

Since this plugin requires an additional python library named calliope, we need to install calliope via conda:

```
micromamba install calliope -c conda-forge
```

Currently, this line of code will automatically downgrade pandas to 1.5.3, along with other downgrades, because calliope hasn't been updated in the recent years. **The influence of this downgrade to other CEA functionalities is unknown. Please consider this before installing the plugin.**

Then, one need to register this plugin in cea-config. In the CEA console, enter the following command to enable the plugin in CEA:
```python
cea-config write --general:plugins cea_energy_hub_optimizer.energy_hub_optimizer.EnergyHubOptimizer
```
Note that this line will overwrite other existing plugins. If you have other plugins installed, you can add them to the list by separating them with commas. For example:
```python
cea-config write --general:plugins cea_energy_hub_optimizer.energy_hub_optimizer.EnergyHubOptimizer, --general:plugins other_plugin, --general:plugins another_plugin
```
You should include **ALL the plugins** in that command, otherwise you may lose already installed plugins.

Now we need to install solvers. 
- The default solver for calliope is glpk, which is already pre-installed along with calliope conda package.
- For Gurobi, one need to refer to [Using Gurobi in Calliope](https://calliope.readthedocs.io/en/stable/user/installation.html#gurobi) to install Gurobi Optimizer on your local computer. Note that Admin right and restart is needed for this installation. After installed, one should be able to directly use Gurobi in calliope.
- For CPLEX, one need to first download and install CPLEX, then install its python interface by running `conda install docplex` in CEA console. Note that to download CPLEX, one need to first have Java installed; to install cplex, one need to have admin right.


In the CEA console, enter the following command to enable the plugin in CEA:
```cea energy_hub_optimizer```
Or in the CEA Dashboard, click on the left Tools - Energy Hub Optimizer - Energy Hub Optimizer to activate the GUI of plugin.

## Usage
To set up a new energy hub optimization, one need to first configure the technology definitions in `cea_energy_hub_optimizer/data/techs_energy_hub.yml`. 

The reason that we do not use CEA's native technology definitions is that the energy carriers are pre-defined and fixed in CEA's code. In case one needs to optimize for different water temperature or new energy carriers, it should be defined in the `carrier` key of the technology definitions. For detailed instructions on how to configure the technology definitions, please refer to the comments in the file and the Calliope documentation on [building a model](https://calliope.readthedocs.io/en/stable/user/building.html). Additionally, there are example files from Calliope in [Tutorials](https://calliope.readthedocs.io/en/stable/user/tutorials.html).

Each technology can have multiple keys for detailed constraint configuration. For example, battery could have `charge_rate` for its charging speed; heat pumps could have `energy_cap_min` and `energy_cap_max` for its capacity constraints. Detailed configuration keys can be found in [Configuration and defaults](https://calliope.readthedocs.io/en/stable/user/config_defaults.html#run-configuration).

In Calliope, technologies are defined in several categories, which are called [Abstract base technology groups](https://calliope.readthedocs.io/en/stable/user/config_defaults.html#abstract-base-technology-groups). Each technology must belong to such a group, which is specified in the `(tech_name).essentials.parent` key. This group limits the configuration keys that can be used for the technology, for example, a `supply` technology may not use a `storage_max` key.

In addition to globally availble technology definitions in `techs` key of the yaml file, each building could also have its own technology list, which is a subset of global technology list. This is specified in the `location.Building` of the yaml file, where each value is the name of technologies defined in `techs`. Note that during the execution of the optimization, the plugin will automatically change the `location.Building` to `location.actual_building_name`, like `location.B1001` (see energy_hub.py/64-67). This is to enable future development of multi-building optimization.

After configuring the technology definitions, one can run the optimization by clicking the "Run Script" button in the GUI. The results will be stored in the `(scenario)/outputs/data/optimization/calliope_energy_hub` folder, in .csv format. The results include the number of epsilon cuts, the lifetime cost and CO2 emission of each non-dominated solution, and the installed capacity of each technology in each solution.

## Tricks of Technology Modelling
### Modelling Unlinearity
In the techs_energy_hub.yml file, one can see that there are three different kinds of district heating technologies, labelled with small, medium and large. This is because that the cost of installing district heating to the building is not linearly related to the capacity. The larger the capacity, the lower the cost per unit capacity. ![CAPEX_district_heating_Zurich](https://github.com/user-attachments/assets/d2373849-3b2c-42c3-a637-518b4a58ec33)

Therefore, three different hypothetical linear DH technologies are set, as shown in the image above. The small one has `energy_cap_min=50` and `energy_cap_max=100`, middle `energy_cap_min=100` and `energy_cap_max=700` and large `energy_cap_min=700` and `energy_cap_max=2000`. The slope of this line is set as `energy_cap` in CHF/kW, and the intercept is set as `purchase` in CHF. Note that each technology must have the same CAPEX on their interceptions (100, 700 and 2000kW) to make sure that both technologies are indifferent to this capacity.

## Development
To enable auto-completion for both CEA's native methods and the venv that CEA uses, one need to do two things in VSCode:
1. Add the following lines to the settings.json file (change the path to the path of your CEA **repository**):
```json
{
    "python.analysis.extraPaths": [
        "C:/path/to/the/repository/CityEnergyAnalyst/CityEnergyAnalyst"
    ]
}
```
2. Set the python interpreter to the one in the CEA venv, which is loacated in the **installation folder of CEA** (not the repository). For example, the path would be:
```
C:/path/to/the/installation/folder/CityEnergyAnalyst/Dependencies/micromamba/envs/cea/python.exe
```
## Licenses

This project is licensed under the MIT License. Additionally, it includes components licensed under the Apache License 2.0.

- [Calliope](https://github.com/calliope-project/calliope), licensed under Apache 2.0.
