from setuptools import setup, find_packages

__author__ = "Daren Thomas"
__copyright__ = "Copyright 2020, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

setup(name='cea_plugin_template',
      version=__version__,
      description="A template plugin for the City Energy Analyst",
      license='MIT',
      author='Architecture and Building Systems',
      author_email='cea@arch.ethz.ch',
      url='https://github.com/architecture-building-systems/cea-plugin-template',
      long_description="A template plugin for the City Energy Analyst - fork this and edit to create your own plugin.",
      py_modules=[''],
      packages=find_packages(),
      package_data={},
      include_package_data=True)
