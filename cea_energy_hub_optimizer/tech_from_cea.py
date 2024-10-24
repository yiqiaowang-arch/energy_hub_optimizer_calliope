from os import remove
import pandas as pd
import yaml
from typing import List, Union


def set_nested_dict_value(
    nested_dict: dict, keys: Union[str, List[str]], value: any, force_overwrite=False
) -> None:
    """
    set value in a nested dictionary, with a list of keys leading to the value

    :param nested_dict: a dictionary of dictionaries
    :type nested_dict: dict
    :param keys: a list of strings, representing the keys to the value.
    :type keys: Union[str, List[str]]
    :param value: the value to be assigned to the leaf of this tree-structured nested dictionary
    :type value: any
    :param force_overwrite: if True, previous value of the dict (if any) will be simply ignored;
        if False and there's already value, it will not be overwritten. defaults to False
    :type force_overwrite: bool, optional
    :return: nothing, since dict is mutable and will be modified.
    :rtype: None
    """
    if isinstance(keys, str):
        keys = [keys]
    current_subdict = nested_dict
    keys = remove_consecutive_duplicates(keys)

    for key in keys[:-1]:
        current_subdict = current_subdict.setdefault(key, {})
    if force_overwrite:
        current_subdict[keys[-1]] = value
    else:
        current_subdict.setdefault(keys[-1], value)


def remove_consecutive_duplicates(lst: list) -> list:
    """
    list items that are continuously duplicated will be simplified.
        For example, `[a, a, b, c, c]` will be simplified into `[a, b, c]`.

    :param lst: list of items, which might contain consecutively duplicated items.
    :type lst: list
    :raises ValueError: when keys is an empty list.
    :return: a cleaned list without duplication.
    :rtype: list
    """

    if not lst:
        raise ValueError("The input list cannot be empty")

    result = [lst[0]]  # Start with the first element
    for item in lst[1:]:
        if item != result[-1]:  # Add only if it's different from the last added element
            result.append(item)
    return result


def read_tech_definition(
    filepath: str, to_yaml: bool = False, yaml_path: str = None
) -> dict:
    """
    read calliope technology definition from an excel file, and return a dictionary.
    The dictionary can be written to a yaml file if `to_yaml` is set to True.
    If no `yaml_path` is specified, the yaml file will be saved in the same directory as the excel file, with the same name.

    The structure of the excel file closely follows the structure of the calliope technology definition
    (see link: https://calliope.readthedocs.io/en/stable/user/config_defaults.html#per-tech-constraints).
    A template of such an excel file can be found in the `data` folder of this repository. The data was prepared to resemble
    the techno-economical environment for building technology availability in Zurich, Switzerland.

    Note that all of the excel sheets must only contain structured technology data, and no other information.
    Otherwise the function will fail to construct the dictionary.

    :param filepath: path to the excel (`.xlsx`) file.
    :type filepath: str
    :param to_yaml: if True, a `.yaml` file resembling the nested dict will be stored, defaults to False
    :type to_yaml: bool, optional
    :param yaml_path: path to store the `.yaml` file. If not, it will be stored in the same path
        as the source `.xlsx` file, using the same name. defaults to None
    :type yaml_path: str, optional
    :return: a nested dict, which can be stored as a `.yaml` file, or transformed into a `calliope.AttrDict` object.
    :rtype: dict
    """

    xls = pd.ExcelFile(filepath)
    sheets_list = xls.sheet_names
    tech_dict = {}
    for sheet in sheets_list:
        df = pd.read_excel(filepath, sheet_name=sheet, header=None)
        if sheet == "conversion_plus":
            n_header_rows = 5
        else:
            n_header_rows = 3
        # extract header rows and create multi-row index
        header_rows = df.iloc[:n_header_rows]
        header_rows_filled = header_rows.ffill(axis=0).ffill(axis=1).T
        multi_row_index = pd.MultiIndex.from_frame(header_rows_filled)

        # remove header rows and reset index
        df = df.iloc[n_header_rows:].reset_index(drop=True)
        df.columns = multi_row_index

        # iterate through the rows of the dataframe
        for _, row in df.iterrows():
            # Get the technology name
            tech_name = row.loc[("essentials", slice(None), "name")].item()
            current_dict = tech_dict.setdefault(tech_name, {})

            # Iterate through the columns (which have a MultiIndex)
            for column_name_tuple in df.columns:
                value = row.loc[column_name_tuple]
                if pd.notna(value):
                    set_nested_dict_value(
                        current_dict, list(column_name_tuple), value=value
                    )

    if to_yaml:
        if not yaml_path:
            yaml_path = filepath.replace(".xlsx", ".yaml")

        print(f"Writing tech definitions to {yaml_path}")
        with open(yaml_path, "w") as file:
            yaml.dump(tech_dict, file)

    return tech_dict


filepath = (
    r"C:\Users\wangy\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\techDefinition.xlsx"
)
tech_dict = read_tech_definition(
    filepath, to_yaml=True, yaml_path="tech_definition.yaml"
)
