import pandas as pd
from cea.utilities.epwreader import epw_reader
from typing import Optional, Union, Iterable, List
from cea.config import Configuration
from cea.inputlocator import InputLocator


class Demand:
    def __init__(self, config: Configuration, locator: InputLocator, nodes: List[str]):
        self.config = config
        self.locator = locator
        self.nodes = nodes
        self.mapping_dict = {
            "electricity": "E_sys_kWh",
            "space_heating": "Qhs_sys_kWh",
            "space_cooling": "Qcs_sys_kWh",
            "hot_water": "Qww_sys_kWh",
        }
        self.demand_dict: dict[str, TimeSeriesDf] = {
            f"demand_{cat}": TimeSeriesDf(columns=self.nodes, locator=self.locator)
            for cat in self.mapping_dict.keys()
        }
        self.get_demand()

    def get_node_demand(self, node: str):
        demand_path = self.locator.get_demand_results_file(node)
        demand_df = pd.read_csv(demand_path, usecols=self.mapping_dict.values())
        # first, check if the node is in the demand_dict value dataframes. If not, add the column to all dataframes
        for cat in self.mapping_dict.keys():
            if node not in self.demand_dict[f"demand_{cat}"].columns:
                self.demand_dict[cat].add_columns(node)

            if cat in self.config.energy_hub_optimizer.evaluated_demand:
                self.demand_dict[f"demand_{cat}"][node] = -demand_df[
                    self.mapping_dict[cat]
                ].to_numpy()
                # negative sign is used for convention, as energy demand is negative and supply is positive
                # to_numpy() is used to avoid index mismatch, as long as the length of the dataframe is the same

    def get_nodes_demand(self, nodes: List[str]):
        for node in nodes:
            self.get_node_demand(node)

    def get_demand(self):
        self.get_nodes_demand(self.nodes)


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
