"""
construct the calliope config yaml file from the settings.yml file and the technology definition excel file.
"""

from __future__ import division
from __future__ import print_function
import cea.config
import cea.inputlocator
import cea.plugin
import os
import yaml
from cea_energy_hub_optimizer.tech_from_excel import *


__author__ = "Yiqiao Wang"
__copyright__ = "Copyright 2024, Yiqiao Wang"
__credits__ = ["Yiqiao Wang"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Yiqiao Wang"
__email__ = "yiqwang@ethz.ch / wangyiqiao97@gmail.com"
__status__ = "Production"


class CalliopeConfigConstructor(cea.plugin.CeaPlugin):
    """
    Define the plugin class - unless you want to customize the behavior, you only really need to declare the class. The
    rest of the information will be picked up from ``default.config``, ``schemas.yml`` and ``scripts.yml`` by default.
    """

    pass


def main(config: cea.config.Configuration) -> None:
    # read yaml file from /data/settings.yml and create a nested dictionary
    # my_config = MyConfig(config)
    with open(
        os.path.join(os.path.dirname(__file__), "data", "settings.yml"), "r"
    ) as file:
        calliope_config: dict = yaml.safe_load(file)
        if config.calliope_config_constructor.technology_excel_file == "":
            tech_excel_path = os.path.join(
                os.path.dirname(__file__), "data", "example_techDefinition.xlsx"
            )
        else:
            # check if the path belongs to a .xlsx file
            if config.calliope_config_constructor.technology_excel_file[-5:] != ".xlsx":
                raise ValueError(
                    "The technology definition file must be a .xlsx file. Please provide a valid .xlsx file."
                )
            tech_excel_path = config.calliope_config_constructor.technology_excel_file
        techs_subdict = read_tech_definition(filepath=tech_excel_path)
        calliope_config["techs"].update(techs_subdict)

        store_path = config.calliope_config_constructor.yaml_storage_path
        if store_path == "":
            store_path = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.isdir(store_path):
            raise ValueError("Please provide a valid folder path.")
        yaml_file_name = config.calliope_config_constructor.yaml_file_name
        if yaml_file_name == "":
            yaml_file_name = "energy_hub_config"
        yaml_file_name += ".yml"
        yaml_file_path = os.path.join(store_path, yaml_file_name)
        with open(yaml_file_path, "w") as file:
            yaml.dump(calliope_config, file)

        print(f"Yaml file is saved at {yaml_file_path}.")


if __name__ == "__main__":
    main(cea.config.Configuration())
