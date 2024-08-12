from setuptools import setup, find_packages

__author__ = "Yiqiao Wang"
__copyright__ = "Copyright 2024, Yiqiao Wang"
__credits__ = ["Yiqiao Wang"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Yiqiao Wang"
__email__ = "yiqwang@ethz.ch / wangyiqiao97@gmail.com"
__status__ = "Production"

setup(
    name='energy_hub_optimizer',
    version=__version__,
    description="An optimization plugin for the City Energy Analyst",
    license='MIT',
    author='Yiqiao Wang',
    author_email='yiqwang@ethz.ch / wangyiqiao97@gmail.com',
    url='https://github.com/yiqiaowang-arch/energy_hub_optimizer_calliope',
    long_description="An optimization plugin for building energy hub using Calliope for the City Energy Analyst.",
    py_modules=[''],
    packages=find_packages(),
    package_data={},
    include_package_data=True,
)
