# energy_hub_optimizer
An optimization plugin for building energy hub using Calliope for the City Energy Analyst.

To install, clone this repo to a desired path (you would need to have `git` installed to run this command. Alternatively you can also run this command in the CEA console, which
comes with `git` pre-installed):

```git clone https://github.com/yiqiaowang-arch/energy_hub_optimizer_calliope.git DESIRED_PATH```


Open CEA console and enter the following command to install the plugin to CEA. In this command, -e means editable, which allows the plugin to be updated without re-installing. You can also check out the repository using your text editor and make any local changes you like, in order to alter the functionality of this plugin.

```pip install -e PATH_OF_PLUGIN_FOLDER```
For example:
    
    ```pip install -e "C:\Users\wangy\Documents\GitHub\energy_hub_optimizer_calliope" --use-pep517```

(NOTE: PATH_OF_PLUGIN_FOLDER would be the DESIRED_PATH + 'cea_energy_hub_optimizer')

Since this plugin requires an additional python library named calliope, we need to install calliope via conda:

```conda install calliope```

Note that calliope currently only supports pandas 1.5.x, but CEA uses pandas 2.x. In order to use calliope normally, we need to manually downgrade pandas to 1.5.x:

```conda install pandas=1.5.3 --freeze-installed```

Now we need to install solver. 
- The default solver for calliope is glpk, which is already pre-installed along with calliope conda package.
- For Gurobi, one need to refer to [Using Gurobi in Calliope](https://calliope.readthedocs.io/en/stable/user/installation.html#gurobi) to install Gurobi.
- For CPLEX, one need to first download and install CPLEX, then install its python interface by running `pip install docplex` in CEA console.


In the CEA console, enter the following command to enable the plugin in CEA:

```python
cea-config write --general:plugins cea_energy_hub_optimizer.energy_hub_optimizer.EnergyHubOptimizer
```

Now you should be able to enter the following command to run the plugin:

```cea energy_hub_optimizer```

NOTE: When installing multiple plugins, add them as a comma separated list in the `cea-config write --general:plugins ...` command.
