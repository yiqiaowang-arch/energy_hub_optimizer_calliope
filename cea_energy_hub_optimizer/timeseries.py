import pandas as pd
from cea.utilities.epwreader import epw_reader
from typing import Optional, Union, Iterable, Dict
from cea.config import Configuration
from cea.inputlocator import InputLocator
from cea_energy_hub_optimizer.district import District, Building


class EnergyIO:
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        mapping_dict: Dict[str, str],
        district: District,
    ):
        self.cea_config = cea_config
        self.locator = locator
        self.district = district
        self.mapping_dict = mapping_dict
        self.result_dict: dict[str, TimeSeriesDf] = {
            f"{key}": TimeSeriesDf(
                columns=self.district.buildings_names,
                locator=self.locator,
            )
            for key in self.mapping_dict.keys()
        }

        self.get_energy()

    def get_node_energy(self, building: "Building"):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def get_nodes_energy(self):
        for building in self.district.buildings:
            self.get_node_energy(building)

    def get_energy(self):
        self.get_nodes_energy()


class Demand(EnergyIO):
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        district: District,
    ):
        self.mapping_dict = {
            "demand_electricity": "E_sys_kWh",
            "demand_space_heating": "Qhs_sys_kWh",
            "demand_space_cooling": "Qcs_sys_kWh",
            "demand_hot_water": "Qww_sys_kWh",
        }
        super().__init__(cea_config, locator, self.mapping_dict, district)

    def get_node_energy(self, building: "Building"):
        demand_path = self.locator.get_demand_results_file(building=building.name)
        demand_df = pd.read_csv(demand_path, usecols=self.mapping_dict.values())
        for key in self.mapping_dict.keys():
            if building.name not in self.result_dict[key].columns:
                self.result_dict[key].add_columns(building.name)

            if key in self.cea_config.energy_hub_optimizer.evaluated_demand:
                self.result_dict[key][building.name] = -demand_df[
                    self.mapping_dict[key]
                ].to_numpy()


class SolarEnergy(EnergyIO):
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        mapping_dict: Dict[str, str],
        district: District,
    ):
        super().__init__(cea_config, locator, mapping_dict, district)

    def divide_by_area(self, result_key: str):
        for building in self.district.buildings:
            self.result_dict[result_key][building.name] /= building.area

    def get_node_energy(self, node: str):
        raise NotImplementedError("Subclasses should implement this method.")


class PV(SolarEnergy):
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        district: District,
    ):
        mapping_dict = {"supply_PV": "E_PV_gen_kWh"}
        super().__init__(cea_config, locator, mapping_dict, district)

    def get_node_energy(self, building: "Building"):
        if building.name not in self.result_dict["supply_PV"].columns:
            self.result_dict["supply_PV"].add_columns(building.name)

        if "PV" in self.cea_config.energy_hub_optimizer.evaluated_solar_supply:
            pv_path = self.locator.PV_results(building=building.name)
            pv_df = pd.read_csv(pv_path, usecols=["E_PV_gen_kWh"])
            self.result_dict["supply_PV"][building.name] = pv_df[
                "E_PV_gen_kWh"
            ].to_numpy()

        self.divide_by_area("supply_PV")


class PVT(SolarEnergy):
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        district: District,
    ):
        mapping_dict = {
            "supply_PVT_e": "E_PVT_gen_kWh",
            "supply_PVT_h": "Q_PVT_gen_kWh",
        }
        super().__init__(cea_config, locator, mapping_dict, district)

    def get_node_energy(self, building: "Building"):
        if building.name not in self.result_dict["supply_PVT_e"].columns:
            self.result_dict["supply_PVT_e"].add_columns(building.name)

        if "PVT" in self.cea_config.energy_hub_optimizer.evaluated_solar_supply:
            pvt_path = self.locator.PVT_results(building=building.name)
            pvt_df = pd.read_csv(pvt_path, usecols=self.mapping_dict.values())
            self.result_dict["supply_PVT_e"][building.name] = pvt_df[
                "E_PVT_gen_kWh"
            ].to_numpy()
            self.result_dict["supply_PVT_h"][building.name] = (
                pvt_df["Q_PVT_gen_kWh"].to_numpy() / pvt_df["E_PVT_gen_kWh"].to_numpy()
            )

        self.divide_by_area("supply_PVT_e")


class SC(SolarEnergy):
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        panel_type: str,
        district: District,
    ):
        mapping_dict = {f"supply_SC{panel_type}": "Q_SC_gen_kWh"}
        self.panel_type = panel_type
        super().__init__(cea_config, locator, mapping_dict, district)

    def get_node_energy(self, building: "Building"):
        result_key = f"supply_SC{self.panel_type}"
        if building.name not in self.result_dict[result_key].columns:
            self.result_dict[result_key].add_columns(building.name)

        if (
            f"SC{self.panel_type}"
            in self.cea_config.energy_hub_optimizer.evaluated_solar_supply
        ):
            sc_path = self.locator.SC_results(
                building=building.name, panel_type=self.panel_type
            )
            sc_df = pd.read_csv(sc_path, usecols=["Q_SC_gen_kWh"])
            self.result_dict[result_key][building.name] = sc_df[
                "Q_SC_gen_kWh"
            ].to_numpy()

        self.divide_by_area(result_key)


class COP:
    def __init__(
        self,
        cea_config: Configuration,
        locator: InputLocator,
        district: District,
    ):
        self.cea_config = cea_config
        self.cop_dict: Dict[str, TimeSeriesDf] = {
            "cop_dhw": TimeSeriesDf(columns=district.buildings_names, locator=locator),
            "cop_sc": TimeSeriesDf(columns=district.buildings_names, locator=locator),
        }

        epw_df = TimeSeriesDf._epw_data
        exergy_eff = cea_config.energy_hub_optimizer.nominal_cop / (
            (60 + 273.15) / (60 - 10)
        )

        cop_dhw_arr = (
            (60 + 273.15) / (60 - epw_df["drybulb_C"].to_numpy())
        ) * exergy_eff
        cop_sc_arr = (
            (10 + 273.15) / (30 - epw_df["drybulb_C"].to_numpy())
        ) * exergy_eff

        self.cop_dict["cop_dhw"].loc[:, :] = cop_dhw_arr[:, None]
        self.cop_dict["cop_sc"].loc[:, :] = cop_sc_arr[:, None]


# define a class which is a pd dataframe, having the index as epw's timestep and index's name as "t"
class TimeSeriesDf(pd.DataFrame):
    """
    A subclass of pd.Dataframe. The total length of the dataframe is restricted to be the same as the epw data length;
    the index is the epw's timestep (normally hourly) and the index's name is "t".
    """

    _epw_data: Optional[pd.DataFrame] = None

    def __init__(self, columns: Union[str, Iterable[str]], locator: InputLocator):
        """
        Initialize the TimeSeriesDf object using the epw data as the index and input column names as the columns.
        All values are initialized to 0.0.

        :param columns: A list or other iterables of column name strings. If only one string is passed, it will be converted to a list.
        :type columns: Union[str, Iterable[str]]

        Example:
            ```
            df = TimeSeriesDf(columns=["a", "b", "c"])
            print(df)
            ```
            Will output:
            ```
                                    a  b  c
            t
            2050-01-01 00:00:00     0  0  0
            2050-01-01 01:00:00     0  0  0
            2050-01-01 02:00:00     0  0  0
            2050-01-01 03:00:00     0  0  0
            2050-01-01 04:00:00     0  0  0
            ...                    .. .. ..
            2050-12-31 19:00:00     0  0  0
            2050-12-31 20:00:00     0  0  0
            2050-12-31 21:00:00     0  0  0
            2050-12-31 22:00:00     0  0  0
            2050-12-31 23:00:00     0  0  0
            ```
        """

        if TimeSeriesDf._epw_data is None:
            epw_path: str = locator.get_weather_file()
            TimeSeriesDf._epw_data = epw_reader(epw_path)

        if isinstance(columns, str):
            columns = [columns]

        super().__init__(
            data=0.0, index=TimeSeriesDf._epw_data["date"], columns=columns
        )
        self.index.set_names("t", inplace=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """
        An alternative constructor for the TimeSeriesDf class.

        In case there is already a dataframe with the same length as the epw data,
        this class method can change the dataframe to a TimeSeriesDf object,
        by changing its index to a pd.DatetimeIndex the same as the epw data.

        :param df: dataframe that has the same length as the epw data
            (normally 8760 rows, but in case it's a leap year, it might be 8784 rows)
        :type df: pd.DataFrame
        :raises ValueError: if the input dataframe doesn't have the same length as the epw data, then raise an error.
        :return: a converted TimeSeriesDf object with the same data as the input dataframe.
            However, there is one more column called "original_index" which is the original index of the input dataframe.
            The new index is the pd.DatetimeIndex from the epw data, starting hourly from the beginning of the year until the end.
        :rtype: TimeSeriesDf
        """
        if cls._epw_data is None:
            epw_path: str = InputLocator().get_weather_file()
            cls._epw_data = epw_reader(epw_path)

        if len(df) != len(cls._epw_data):
            raise ValueError(
                "Dataframe length does not match epw data length, unable to create TimeSeriesDf"
            )
        # save the original index because this will be deleted when resetting the index
        df["oiginal_index"] = df.index
        result = cls(df.columns)
        result[:] = df.to_numpy()
        return result

    def add_columns(self, columns: Union[str, Iterable[str]]):
        """
        checks if the input columns are already in the dataframe, if not, add them to the dataframe and initialize them to 0.0.

        :param columns: names of columns to be added to the dataframe. If only one string is passed, it will be converted to a list.
        :type columns: Union[str, Iterable[str]]
        :raises ValueError: If the column is already in the dataframe, one should not readd and reinitialize it with zero, so raise an error.
        """
        # check if columns are alraedy in the dataframe
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            if column not in self.columns:
                self[column] = 0.0
            else:
                raise ValueError(f"{column} is already in the dataframe")
