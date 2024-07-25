# energy_hub_optimizer
An optimization plugin for building energy hub using Calliope for the City Energy Analyst.

To install, clone this repo to a desired path (you would need to have `git` installed to run this command. Alternatively you can also run this command in the CEA console, which
comes with `git` pre-installed):

```git clone https://github.com/yiqiaowang-arch/energy_hub_optimizer_calliope.git DESIRED_PATH```


Open CEA console and enter the following command to install the plugin to CEA:

```pip install -e PATH_OF_PLUGIN_FOLDER```

(NOTE: PATH_OF_PLUGIN_FOLDER would be the DESIRED_PATH + 'cea_energy_hub_optimizer')


In the CEA console, enter the following command to enable the plugin in CEA:

```python
cea-config write --general:plugins cea_energy_hub_optimizer.energy_hub_optimizer.EnergyHubOptimizer
```

Now you should be able to enter the following command to run the plugin:

```cea energy_hub_optimizer```

NOTE: When installing multiple plugins, add them as a comma separated list in the `cea-config write --general:plugins ...` command.
