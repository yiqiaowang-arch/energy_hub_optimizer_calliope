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
    
    ```pip install -e "C:\Users\wangy\Documents\GitHub\energy_hub_optimizer_calliope"```

(NOTE: PATH_OF_PLUGIN_FOLDER would be the DESIRED_PATH + 'cea_energy_hub_optimizer')

Then, one need to register this plugin in cea-config. In the CEA console, enter the following command to enable the plugin in CEA:
```python
cea-config write --general:plugins cea_energy_hub_optimizer.energy_hub_optimizer.EnergyHubOptimizer
```
Note that this line will overwrite other existing plugins. If you have other plugins installed, you can add them to the list by separating them with commas. 

### Additional Installation
Since this plugin requires an additional python library named calliope, we need to install calliope via conda:

```conda install calliope```

This will automatically downgrade pandas to 1.5.3. 

Now we need to install solver. 
- The default solver for calliope is glpk, which is already pre-installed along with calliope conda package.
- For Gurobi, one need to refer to [Using Gurobi in Calliope](https://calliope.readthedocs.io/en/stable/user/installation.html#gurobi) to install Gurobi.
- For CPLEX, one need to first download and install CPLEX, then install its python interface by running `conda install docplex` in CEA console.


In the CEA console, enter the following command to enable the plugin in CEA:
```cea energy_hub_optimizer```
Or in the CEA Dashboard, click on the left Tools - Energy Hub Optimizer - Energy Hub Optimizer to activate the GUI of plugin.