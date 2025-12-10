# @title Load the imports

import pandas as pd
import plotly.express as px

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statements.")

# @title Load the datasets
species1_raw = pd.read_csv("landmarks_training_gpa/Conus_striolatus_landmarks_cropped_gpa_wide.csv")
species2_raw = pd.read_csv("landmarks_training_gpa/Conus_muriculatus_landmarks_cropped_gpa_wide.csv")

# @title Arrange datasets 
species1_raw["Class"] = "striolatus"
species2_raw["Class"] = "muriculatus"

shell_dataset_raw = pd.concat([species1_raw, species2_raw], ignore_index=True)

print(f"Total specimens: {len(shell_dataset_raw)}")

# @title Inspect structure of dataset
landmark_columns = [col for col in shell_dataset_raw.columns if col.startswith("x") or col.startswith("y")]

print(f"Number of landmarks: {len(landmark_columns)//2}")
print(f"Total features: {len(landmark_columns)}")

shell_dataset = shell_dataset_raw[landmark_columns + ["Class"]]

# summary stats
shell_dataset.describe()

# @title Create five 2D plots of landmarks against each other, color-coded by class

# @title Plot three features in 3D by entering their names and running this cell

x_axis_data = "x0"  # @param {type: "string"}
y_axis_data = "y0"  # @param {type: "string"}
z_axis_data = "x1"  # @param {type: "string"}

px.scatter_3d(
    shell_dataset,
    x=x_axis_data,
    y=y_axis_data,
    z=z_axis_data,
    color="Class",
).show()

#------------------------------------------

shell_dataset = shell_dataset_raw[landmark_columns + ["Class"]].copy()
shell_dataset.describe()

# x0, y0, x1 with color = y1
px.scatter_3d(
    shell_dataset,
    x="x0",
    y="y0",
    z="x1",
    color="y1",
    symbol="Class",
    symbol_map={
        "Conus_striolatus": "square",
        "Conus_muriculatus": "triangle"
    }
).show()

# x0 vs distance between y8 and y9
shell_dataset["ant_distance"] = (shell_dataset["y9"] - shell_dataset["y8"]).abs()

px.scatter(
    shell_dataset,
    x="x0",
    y="ant_distance",
    color="Class",
).show()

# x0 vs distance between y10 and y11

shell_dataset["apert_distance"] = (shell_dataset["y10"] - shell_dataset["y11"]).abs()

px.scatter(
    shell_dataset,
    x="x0",
    y="apert_distance",
    color="Class",
).show()