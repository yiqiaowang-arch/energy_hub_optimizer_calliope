"""
Creates an optimization plugin for building energy hub using Calliope for the City Energy Analyst.
"""
from __future__ import division
from __future__ import print_function

import cea.config
import cea.inputlocator
import cea.plugin

__author__ = "Yiqiao Wang"
__copyright__ = "Copyright 2024, Yiqiao Wang"
__credits__ = ["Yiqiao Wang"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Yiqiao Wang"
__email__ = "yiqwang@ethz.ch / wangyiqiao97@gmail.com"
__status__ = "Production"


class EnergyHubOptimizer(cea.plugin.CeaPlugin):
    """
    Define the plugin class - unless you want to customize the behavior, you only really need to declare the class. The
    rest of the information will be picked up from ``default.config``, ``schemas.yml`` and ``scripts.yml`` by default.
    """
    pass


def summarize(total_demand_df, fudge_factor):
    """
    Return only the following fields from the Total_demand.csv file:
    - Name (Unique building ID)
    - GFA_m2 (Gross floor area)
    - QC_sys_MWhyr (Total system cooling demand)
    - QH_sys_MWhyr (Total building heating demand)
    """
    result_df = total_demand_df[["Name", "GFA_m2", "QC_sys_MWhyr", "QH_sys_MWhyr"]].copy()
    result_df["QC_sys_MWhyr"] *= fudge_factor
    result_df["QH_sys_MWhyr"] *= fudge_factor
    return result_df


def main(config: cea.config.Configuration):
    """
    This is the main entry point to your script. Any parameters used by your script must be present in the ``config``
    parameter. The CLI will call this ``main`` function passing in a ``config`` object after adjusting the configuration
    to reflect parameters passed on the command line / user interface

    :param cea.config.Configuration config: The configuration for this script, restricted to the scripts parameters.
    :return: None
    """
    locator = cea.inputlocator.InputLocator(config.scenario, config.plugins)
    summary_df = summarize(locator.get_total_demand.read(), config.energy_hub_optimizer.fudge_factor)
    locator.energy_hub_optimizer.write(summary_df)


if __name__ == '__main__':
    main(cea.config.Configuration())
