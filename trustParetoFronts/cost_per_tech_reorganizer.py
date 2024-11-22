import pandas as pd
import os


def cost_per_tech_reorganizer(folder_path):
    swap_mapping = {"monetary": "co2", "co2": "monetary"}
    for file in os.listdir(folder_path):
        if file.endswith("cost_per_tech.csv"):
            df = pd.read_csv(os.path.join(folder_path, file), index_col=[0, 1, 2])
            building_name = file.split("_")[0]
            # df has three layers of index: building (a string), pareto_index (an integer starts from 0), and cost_type (either "monetary" or "co2")
            # we want to compare the loc[(building, 0, "monetary"), "electricity"] with loc[(building, 0, "co2"), "electricity"] and see which one is larger.
            # if the monetary cost is larger then there's no problem; if the co2 cost is larger then we need to swap monetary and co2 costs.
            print(f"checking building {building_name}...")
            if (
                df.loc[(building_name, 0, "monetary"), "electricity_econatur"]
                < df.loc[(building_name, 0, "co2"), "electricity_econatur"]
            ):
                # swap the last two index names
                df.index = df.index.set_levels(
                    df.index.levels[2].map(swap_mapping), level=2
                )

                # now we can save the file back using the same name
                df.to_csv(os.path.join(folder_path, file), index=True)
                print(f"Building {building_name} has been reorganized.")


if __name__ == "__main__":
    cost_per_tech_reorganizer(
        r"D:\OneDrive\ETHY3FW\semesterProjectYiqiaoWang\CEA\Altstetten\basecase_residential\outputs\data\optimization\calliope_energy_hub\batch_no_oil_with_DH"
    )
