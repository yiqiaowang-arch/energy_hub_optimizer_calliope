import numpy as np
import pandas as pd
from cea.utilities.epwreader import epw_reader
from typing import Union, Iterable, Dict
from cea_energy_hub_optimizer.district import District, Building
from cea_energy_hub_optimizer.my_config import MyConfig


class TimeSeries:
    def __init__(self, district: District):
        self.demand = Demand(district)
        self.pv = PV(district)
        self.pvt = PVT(district)
        self.sc_et = SC("ET", district)
        self.sc_fp = SC("FP", district)
        self.cop = COP(district)

    @property
    def timeseries_dict(self):
        return {
            **self.demand.result_dict,
            **self.pv.result_dict,
            **self.pvt.result_dict,
            **self.sc_et.result_dict,
            **self.sc_fp.result_dict,
            **self.cop.cop_dict,
        }


class EnergyIO:
    def __init__(
        self,
        mapping_dict: Dict[str, str],
        district: District,
    ):
        self.my_config = MyConfig()
        self.locator = self.my_config.locator
        self.district = district
        self.mapping_dict = mapping_dict
        self.result_dict: dict[str, TimeSeriesDf] = {
            f"{key}": TimeSeriesDf(
                columns=self.district.buildings_names,
            )
            for key in self.mapping_dict.keys()
        }

        self.get_energy()

    def get_node_energy(self, building: "Building") -> None:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def get_nodes_energy(self):
        for building in self.district.buildings:
            self.get_node_energy(building)

    def get_energy(self):
        self.get_nodes_energy()

    def flatten_spikes(self, percentile: float = 0.98, is_positive: bool = False):
        for key in self.result_dict.keys():
            self.result_dict[key].flattenSpikes(percentile, is_positive)


class Demand(EnergyIO):
    def __init__(
        self,
        district: District,
    ):
        self.my_config = MyConfig()
        self.locator = self.my_config.locator
        self.mapping_dict = {
            "demand_electricity": "E_sys_kWh",
            "demand_space_heating": "Qhs_sys_kWh",
            "demand_space_cooling": "Qcs_sys_kWh",
            "demand_hot_water": "Qww_sys_kWh",
        }
        super().__init__(self.mapping_dict, district)

    def get_node_energy(self, building: "Building"):
        demand_path = self.locator.get_demand_results_file(building=building.name)
        demand_df = pd.read_csv(demand_path, usecols=list(self.mapping_dict.values()))
        for key in self.mapping_dict.keys():
            if building.name not in self.result_dict[key].columns:
                self.result_dict[key].add_columns(building.name)

            if key in self.my_config.evaluated_demand:
                self.result_dict[key][building.name] = -demand_df[
                    self.mapping_dict[key]
                ].to_numpy()


class SolarEnergy(EnergyIO):
    def __init__(
        self,
        mapping_dict: Dict[str, str],
        district: District,
    ):
        self.my_config = MyConfig()
        self.locator = self.my_config.locator
        super().__init__(mapping_dict, district)

    def divide_by_area(self, result_key: str):
        for building in self.district.buildings:
            self.result_dict[result_key][building.name] /= building.area


class PV(SolarEnergy):
    def __init__(
        self,
        district: District,
    ):
        self.my_config = MyConfig()
        self.locator = self.my_config.locator
        mapping_dict = {"supply_PV": "E_PV_gen_kWh"}
        super().__init__(mapping_dict, district)
        self.divide_by_area("supply_PV")

    def get_node_energy(self, building: "Building"):
        if building.name not in self.result_dict["supply_PV"].columns:
            self.result_dict["supply_PV"].add_columns(building.name)

        if "PV" in self.my_config.evaluated_solar_supply:
            pv_path = self.locator.PV_results(building=building.name)
            pv_df = pd.read_csv(pv_path, usecols=["E_PV_gen_kWh"])
            self.result_dict["supply_PV"][building.name] = pv_df[
                "E_PV_gen_kWh"
            ].to_numpy()


class PVT(SolarEnergy):
    def __init__(
        self,
        district: District,
    ):
        mapping_dict = {
            "supply_PVT_e": "E_PVT_gen_kWh",
            "supply_PVT_h": "Q_PVT_gen_kWh",
        }
        super().__init__(mapping_dict, district)
        self.divide_by_area("supply_PVT_e")

    def get_node_energy(self, building: "Building"):
        if building.name not in self.result_dict["supply_PVT_e"].columns:
            self.result_dict["supply_PVT_e"].add_columns(building.name)

        if "PVT" in self.my_config.evaluated_solar_supply:
            pvt_path = self.locator.PVT_results(building=building.name)
            pvt_df = pd.read_csv(pvt_path, usecols=list(self.mapping_dict.values()))
            self.result_dict["supply_PVT_e"][building.name] = pvt_df[
                "E_PVT_gen_kWh"
            ].to_numpy()
            self.result_dict["supply_PVT_h"][building.name] = (
                pvt_df["Q_PVT_gen_kWh"].to_numpy() / pvt_df["E_PVT_gen_kWh"].to_numpy()
            )


class SC(SolarEnergy):
    def __init__(
        self,
        panel_type: str,
        district: District,
    ):
        self.my_config = MyConfig()
        self.locator = self.my_config.locator
        mapping_dict = {f"supply_SC{panel_type}": "Q_SC_gen_kWh"}
        self.panel_type = panel_type
        super().__init__(mapping_dict, district)
        self.divide_by_area(f"supply_SC{self.panel_type}")

    def get_node_energy(self, building: "Building"):
        result_key = f"supply_SC{self.panel_type}"
        if building.name not in self.result_dict[result_key].columns:
            self.result_dict[result_key].add_columns(building.name)

        if f"SC{self.panel_type}" in self.my_config.evaluated_solar_supply:
            sc_path = self.locator.SC_results(
                building=building.name, panel_type=self.panel_type
            )
            sc_df = pd.read_csv(sc_path, usecols=["Q_SC_gen_kWh"])
            self.result_dict[result_key][building.name] = sc_df[
                "Q_SC_gen_kWh"
            ].to_numpy()


class COP:
    def __init__(
        self,
        district: District,
    ):
        self.my_config = MyConfig()
        self.locator = self.my_config.locator
        self.district = district
        self.cop_dict: Dict[str, TimeSeriesDf] = {}

        self.T_out = TimeSeriesDf._epw_data["drybulb_C"].to_numpy()
        exergy_eff = self.my_config.nominal_cop / ((60 + 273.15) / (60 - 10))
        self.add_cop_timeseries_from_temp(
            mode="heating",
            T_H=60,
            T_L=self.T_out,
            exergy_eff=exergy_eff,
            df_key="cop_dhw",
        )
        self.add_cop_timeseries_from_temp(
            mode="cooling",
            T_H=self.T_out,
            T_L=10,
            exergy_eff=exergy_eff,
            df_key="cop_sc",
        )

    def add_cop_timeseries_from_temp(
        self, mode: str, T_H, T_L, exergy_eff: float, df_key: str
    ):
        cop_arr = np.zeros_like(self.T_out)
        # check: if mode==heating, then T_H should be one value and T_L should be an array;
        # if mode==cooling, then T_L should be one value and T_H should be an array
        if mode == "heating":
            if isinstance(T_H, (int, float)) and isinstance(T_L, np.ndarray):
                cop_arr = (T_H + 273.15) / (T_H - T_L) * exergy_eff
            else:
                raise ValueError(
                    f"T_H must be a single value and T_L must be an array. Currently, T_H is a {type(T_H)} and T_L is a {type(T_L)}."
                )

        elif mode == "cooling":
            if isinstance(T_L, (int, float)) and isinstance(T_H, np.ndarray):
                cop_arr = (T_L + 273.15) / (T_H - T_L) * exergy_eff
            else:
                raise ValueError(
                    f"T_L must be a single value and T_H must be an array. Currently, T_L is a {type(T_L)} and T_H is a {type(T_H)}."
                )

        # limit the COP to be between 0 and 10
        cop_arr = np.clip(cop_arr, 0, 10)

        self.cop_dict[df_key] = TimeSeriesDf(columns=self.district.buildings_names)
        self.cop_dict[df_key].loc[:, :] = cop_arr[:, None]


# define a class which is a pd dataframe, having the index as epw's timestep and index's name as "t"
class TimeSeriesDf(pd.DataFrame):
    """
    A subclass of pd.Dataframe. The total length of the dataframe is restricted to be the same as the epw data length;
    the index is the epw's timestep (normally hourly) and the index's name is "t".
    """

    _epw_data = pd.DataFrame()

    def __init__(self, columns: Union[str, Iterable[str]]):
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

        if TimeSeriesDf._epw_data.empty:
            epw_path: str = MyConfig().locator.get_weather_file()
            TimeSeriesDf._epw_data = epw_reader(epw_path)

        if isinstance(columns, str):
            columns = [columns]

        super().__init__(
            data=0.0, index=TimeSeriesDf._epw_data["date"], columns=columns  # type: ignore
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
        if cls._epw_data.empty:
            epw_path: str = MyConfig().locator.get_weather_file()
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

    def flattenSpikes(
        self,
        percentile: float = 0.98,
        is_positive: bool = False,
    ):
        """
        This function removes extreme values in the DataFrame by setting them to a lower percentile value.

        - Currently, this will change the integral of the original timeseries.
        - Also, the function cannot handle negative values when is_positive is True, or data with both positive and negative values.

        Args:
            df (pd.DataFrame): dataframe that contain only numbers.
            percentile (float, optional): The part of non-zero values that are preserved in flattening (the rest is flattened). Defaults to 0.98.
            is_positive (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: if not all values in the DataFrame are numbers
            ValueError: if all columns in the DataFrame don't have at least one non-zero value
            ValueError: if columns have both positive and negative values

        Returns:
            df (pd.DataFrame): the DataFrame with the extreme values flattened
        """
        # Check if all values in the DataFrame are numbers
        assert isinstance(self, pd.DataFrame)
        if not self.applymap(lambda x: isinstance(x, (int, float))).all().all():  # type: ignore
            raise ValueError("All values in the DataFrame must be numbers")

        # check if columns don't have both positive and negative values
        if is_positive:
            if not self.applymap(lambda x: x >= 0).all().all():  # type: ignore
                raise ValueError(
                    "All columns in the DataFrame must have only non-negative values"
                )
        else:
            if not self.applymap(lambda x: x <= 0).all().all():  # type: ignore
                raise ValueError(
                    "All columns in the DataFrame must have only non-positivve values"
                )

        for column_name in self.columns:
            if not is_positive:
                self[column_name] = -self[column_name]

            nonzero_subset = self[self[column_name] != 0]
            percentile_value = nonzero_subset[column_name].quantile(1 - percentile)
            self.loc[self[column_name] > percentile_value, column_name] = (
                percentile_value
            )

            if not is_positive:
                self[column_name] = -self[column_name]
