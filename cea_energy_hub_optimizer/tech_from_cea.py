import pandas as pd
import yaml


filepath = (
    r"C:\Users\wangy\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\techDefinition.xlsx"
)

xls = pd.ExcelFile(filepath)
sheets_list = xls.sheet_names
tech_dict = {}
for sheet in sheets_list:
    df = pd.read_excel(filepath, sheet_name=sheet, header=None)

    # extract header rows and create multi-row index
    header_rows = df.iloc[:3]
    header_rows_filled = header_rows.ffill(axis=0).ffill(axis=1).T
    multi_row_index = pd.MultiIndex.from_frame(header_rows_filled)

    # remove header rows and reset index
    df = df.iloc[3:].reset_index(drop=True)
    df.columns = multi_row_index

    # create a new dictionary that holds the actual values

    # iterate through the rows of the dataframe
    for _, row in df.iterrows():
        # Get the technology name
        tech_name = row.loc[("essentials", slice(None), "name")].item()
        tech_dict[tech_name] = {}

        # Iterate through the columns (which have a MultiIndex)
        for column_name_tuple in df.columns:
            # Initialize references for creating the nested dictionary
            current_dict = tech_dict[tech_name]
            previous_level_name = None

            # Iterate through the levels of the column names (MultiIndex levels)
            for level_idx, level_name in enumerate(column_name_tuple):
                # Skip if this level is the same as the previous one (likely filled by pandas' ffill)
                if level_name == previous_level_name:
                    continue  # No need to update, skip to the next level

                previous_level_name = level_name

                # Check if this is the last level (where the value should be stored)
                if level_idx == len(column_name_tuple) - 1:
                    # Add the value from the row if it's not NaN
                    value = row.loc[column_name_tuple]
                    if pd.notna(value):
                        current_dict[level_name] = value
                    # if not, just don't create the key to leave calliope for assigning default values

                else:
                    # Create a new sub-dictionary if the key doesn't exist
                    if level_name not in current_dict:
                        current_dict[level_name] = {}

                    # Move deeper into the dictionary
                    current_dict = current_dict[level_name]

with open("tech_dict.yaml", "w") as file:
    yaml.dump(tech_dict, file)
