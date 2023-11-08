# cea-plugin-template
A template repository for create CEA plugins

To install, clone this repo to a desired path (you would need to have `git` installed to run this command. Alternatively you can also run this command in the CEA console, which
comes with `git` pre-installed):

```git clone https://github.com/architecture-building-systems/cea-plugin-template.git DESIRED_PATH```


Open CEA console and enter the following command to install the plugin to CEA:

```pip install -e PATH_OF_PLUGIN_FOLDER```

(NOTE: PATH_OF_PLUGIN_FOLDER would be the DESIRED_PATH + 'cea-template-plugin')


In the CEA console, enter the following command to enable the plugin in CEA:

```cea-config write --general:plugins cea_plugin_template.demand_summary.DemandSummaryPlugin```

Now you should be able to enter the following command to run the plugin:

```cea demand-summary```

NOTE: When creating your own plugin based on this template, you'll have to adjust the installation instructions above to match.
NOTE: When installing multiple plugins, add them as a comma separated list in the `cea-config write --general:plugins ...` command.
