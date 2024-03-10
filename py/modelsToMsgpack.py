import pandas as pd
import msgpack

csv_file_path = "data//scenariosSlowMerge//scenario"

drop_columns = ["Year", "USGDP%", "Inflation$", "10YrChange", "10YrRate", "Real Estate", "PE", "CAPE"]
scenarios = []  # List to store all scenarios

for model in range(1000):
    df = pd.read_csv(csv_file_path + str(model + 1) + ".csv", na_values="", header=0)

    # Calculate Real Returns
    columns = df.columns
    for name in columns:
        if name != "Inflation$" and name != "Year":
            df[name] = df[name] - df["Inflation$"]
            
    # Get rid of all columns that aren't asset types
    for name in drop_columns:
        df = df.drop(name, axis=1)

    # Data to serialize
    data_to_serialize = {
        "3Month": df["3Month"].to_list(),
        "10Yr": df["10Yr"].to_list(),
        "Baa": df["Baa Corporate"].to_list(),
        "SP500": df["SP500"].to_list(),
        "REIT": df["REIT"].to_list(),
        "Gold": df["Gold"].to_list()
    }

    scenarios.append(data_to_serialize)

# Serialize with MessagePack
with open(f"models//scenarios.msgpack", "wb") as outfile:
    msgpack_bytes = msgpack.packb(scenarios)
    outfile.write(msgpack_bytes)