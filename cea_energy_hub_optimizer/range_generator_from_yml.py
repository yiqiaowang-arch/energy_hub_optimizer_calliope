import yaml
import pandas as pd
from calliope import AttrDict


def generate_csv_from_yaml(
    yaml_filepath: str, csv_filepath: str, keyword_not_allowed=None
) -> None:
    """
    Generate a CSV file from a YAML file with numerical values. The CSV file will have two columns: key and value.

    :param yaml_filepath: Path to the YAML file.
    :param csv_filepath: Path where the output CSV file will be saved.
    :param keyword_not_allowed: List of keywords to filter out from the keys. Default is None.
    :return: None
    """
    # Read the YAML file
    attrdict: AttrDict = AttrDict.from_yaml(yaml_filepath)
    flat_dict = attrdict.as_dict(flat=True)

    # Initialize an empty list to store the filtered data
    data = []

    # Filter out non-numerical values and keys containing any of the keywords
    for k, v in flat_dict.items():
        if isinstance(v, (int, float)):
            if keyword_not_allowed:
                contains_keyword = False
                for keyword in keyword_not_allowed:
                    if keyword in k:
                        contains_keyword = True
                        break
                if not contains_keyword:
                    data.append({"key": k, "value": v})
            else:
                data.append({"key": k, "value": v})

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filepath, index=False)
    print(f"CSV file has been generated at {csv_filepath}")


# Example usage
if __name__ == "__main__":
    yaml_filepath = r"cea_energy_hub_optimizer\data\energy_hub_config.yml"
    csv_filepath = r"cea_energy_hub_optimizer\data\sobol_parameters_conversion.csv"
    keyword_not_allowed = [
        "PV",
        "oil.",
        "gas.",
        "electricity_econatur",
        "pallet.",
        "district_heating.",
        "demand",
        "tank",
        "battery",
        "run",
        "model",
        "group",
        "interest",
        "energy_cap_max",
        "energy_cap_min",
        "energy_eff",
        "force_resource",
    ]
    # Call the function to generate the CSV file
    generate_csv_from_yaml(yaml_filepath, csv_filepath, keyword_not_allowed)
