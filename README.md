# energy_hub_optimizer
An optimization plugin for building energy hub using Calliope for the City Energy Analyst.

## Download
To install, clone this repo to a desired path (you would need to have `git` installed to run this command. Alternatively you can also run this command in the CEA console, which
comes with `git` pre-installed):

```git clone https://github.com/yiqiaowang-arch/energy_hub_optimizer_calliope.git DESIRED_PATH```

## Installation of the plugin
Open CEA console and enter the following command to install the plugin to CEA. In this command, -e means editable, which allows the plugin to be updated without re-installing. You can also check out the repository using your text editor and make any local changes you like, in order to alter the functionality of this plugin.

```pip install -e PATH_OF_PLUGIN_FOLDER```
For example:
    
    ```pip install -e "C:/Users/wangy/Documents/GitHub/energy_hub_optimizer_calliope"```

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
Note that this line will overwrite other existing plugins. If you have other plugins installed, you can add them to the list by separating them with commas. 

Now we need to install solvers. 
- The default solver for calliope is glpk, which is already pre-installed along with calliope conda package.
- For Gurobi, one need to refer to [Using Gurobi in Calliope](https://calliope.readthedocs.io/en/stable/user/installation.html#gurobi) to install Gurobi Optimizer on your local computer. Note that Admin right and restart is needed for this installation. After installed, one should be able to directly use Gurobi in calliope.
- For CPLEX, one need to first download and install CPLEX, then install its python interface by running `conda install docplex` in CEA console.


In the CEA console, enter the following command to enable the plugin in CEA:
```cea energy_hub_optimizer```
Or in the CEA Dashboard, click on the left Tools - Energy Hub Optimizer - Energy Hub Optimizer to activate the GUI of plugin.

## Development
To enable auto-completion for both CEA's native methods and the venv that CEA uses, one need to do two things in VSCode:
1. Add the following lines to the settings.json file (change the path to the path of your CEA **repository**):
```json
{
    "python.analysis.extraPaths": [
        "C:/Users/wangy/Documents/CityEnergyAnalyst/CityEnergyAnalyst"
    ]
}
```
2. Set the python interpreter to the one in the CEA venv, which is loacated in the **installation folder of CEA** (not the repository). For example, the path would be:
```
C:/Users/wangy/Documents/CityEnergyAnalyst/Dependencies/micromamba/envs/cea/python.exe
```