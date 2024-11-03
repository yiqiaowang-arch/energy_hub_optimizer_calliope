import pandas as pd


def generate_sobol_csv(excel_filepath: str, csv_filepath: str) -> None:
    """
    Generate a CSV file for Sobol sensitivity analysis with min value 0 and max value as the current parameter value,
    for all parameters in the Excel file. The parameter names are constructed as 'tech.<tech_name>.<parameter_path>'.

    :param excel_filepath: Path to the Excel file containing technology definitions.
    :param csv_filepath: Path where the output CSV file will be saved.
    :return: None
    """
    # Read the Excel file using pandas
    xls = pd.ExcelFile(excel_filepath)
    sheets_list = xls.sheet_names

    # List to store the parameter data
    data = []

    # Iterate through each sheet in the Excel file
    for sheet in sheets_list:
        # Read the sheet into a DataFrame
        df = pd.read_excel(excel_filepath, sheet_name=sheet, header=None)

        # Determine the number of header rows based on the sheet
        if sheet == "conversion_plus":
            n_header_rows = 5
        else:
            n_header_rows = 3

        # Extract header rows and create a MultiIndex for columns
        header_rows = df.iloc[:n_header_rows]
        # Forward fill missing values in headers
        header_rows_filled = header_rows.ffill(axis=0).ffill(axis=1).T
        # Create a MultiIndex from the header rows
        multi_row_index = pd.MultiIndex.from_frame(header_rows_filled)

        # Remove header rows from DataFrame and reset index
        df = df.iloc[n_header_rows:].reset_index(drop=True)
        df.columns = multi_row_index

        # Iterate through each row (technology) in the DataFrame
        for _, row in df.iterrows():
            # Get the technology name from the 'essentials' section
            tech_name = row.loc[("essentials", slice(None), "name")].values[0]
            # Iterate through all columns (parameters)
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    # Construct the full parameter name
                    col_keys = ["tech", tech_name] + list(col)
                    col_keys = remove_consecutive_duplicates(col_keys)
                    param_name = ".".join(col_keys)
                    # Append parameter data
                    data.append({"name": param_name, "min": 0, "max": value})

    # Create a DataFrame from the collected data
    sobol_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    sobol_df.to_csv(csv_filepath, index=False)
    print(f"Sobol CSV file has been generated at {csv_filepath}")


def remove_consecutive_duplicates(lst):
    """
    Remove consecutive duplicates from a list.

    :param lst: List from which to remove consecutive duplicates.
    :return: List without consecutive duplicates.
    """
    if not lst:
        return lst
    result = [lst[0]]
    for item in lst[1:]:
        if item != result[-1]:
            result.append(item)
    return result


# Example usage
if __name__ == "__main__":
    excel_filepath = r"C:\Users\wangy\Documents\GitHub\energy_hub_optimizer_calliope\cea_energy_hub_optimizer\data\example_techDefinition.xlsx"
    csv_filepath = r"C:\Users\wangy\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\sobol_parameters.csv"

    # Call the function to generate the Sobol CSV file
    generate_sobol_csv(excel_filepath, csv_filepath)
