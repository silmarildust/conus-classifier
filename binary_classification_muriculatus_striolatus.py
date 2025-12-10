# @title Load the imports

import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statements.")

# @title Load the datasets
species1_raw = pd.read_csv("landmarks_training_gpa/Conus_striolatus_landmarks_cropped_gpa_wide.csv")
species2_raw = pd.read_csv("landmarks_training_gpa/Conus_muriculatus_landmarks_cropped_gpa_wide.csv")

# @title
# Add a 'Class' column to distinguish species.
species1_raw["Class"] = "striolatus"
species2_raw["Class"] = "muriculatus"

# Combine the datasets into one DataFrame.
shell_dataset_raw = pd.concat([species1_raw, species2_raw], ignore_index=True)

print(f"Total specimens: {len(shell_dataset_raw)}")

# @title Inspect structure of dataset
# Landmarks are stored as x0, y0, x1, y1, ..., so we will treat them as features.
landmark_columns = [col for col in shell_dataset_raw.columns if col.startswith("x") or col.startswith("y")]

print(f"Number of landmarks: {len(landmark_columns)//2}")
print(f"Total features: {len(landmark_columns)}")

# Keep only landmarks + class
shell_dataset = shell_dataset_raw[landmark_columns + ["Class"]]

# Show summary statistics for coordinates
shell_dataset.describe()

# @title Create five 2D plots of landmarks against each other, color-coded by class
for x_axis_data, y_axis_data in [
    ("x0", "y0"),
    ("x1", "y1"),
    ("x2", "y2"),
    ("x3", "y3"),
    ("x4", "y4"),
]:
  px.scatter(shell_dataset, x=x_axis_data, y=y_axis_data, color="Class").show()

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

# @title Normalize landmark coordinates (Z-scores)

feature_mean = shell_dataset[landmark_columns].mean()
feature_std = shell_dataset[landmark_columns].std()
normalized_dataset = (shell_dataset[landmark_columns] - feature_mean) / feature_std

# Copy the class to the new dataframe
normalized_dataset["Class"] = shell_dataset["Class"]

# Examine some of the values of the normalized dataset
normalized_dataset.head()

keras.utils.set_random_seed(42)

# Create a boolean label column: striolatus = 1, muriculatus = 0
normalized_dataset["Class_Bool"] = (
    normalized_dataset["Class"] == "striolatus"
).astype(int)
normalized_dataset.sample(10)

# @title Split into train, validation, and test sets (80/10/10)

number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

# Randomize order and split
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

# Show the first five rows of the last split
test_data.head()

label_columns = ["Class", "Class_Bool"]

train_features = train_data.drop(columns=label_columns)
train_labels = train_data["Class_Bool"].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data["Class_Bool"].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data["Class_Bool"].to_numpy()

# Name of the features we'll train our model on.
# (Here we include ALL landmark coordinates.)
input_features = landmark_columns

# @title Define the functions that create and train a model.


def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple classification model."""
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  # Use a Concatenate layer to assemble the different inputs into a single
  # tensor which will be given as input to the Dense layer.

  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  model_output = keras.layers.Dense(
      units=1, name="dense_layer", activation=keras.activations.sigmoid
  )(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  # Compile model for classification
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )
  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Feed a dataset into the model in order to train it."""

  # The x parameter of keras.Model.fit can be a list of arrays, where
  # each array contains the data for one feature.
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )


print("Defined the create_model and train_model functions.")

# @title Run the experiment
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name="accuracy", threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name="precision", thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name="recall", thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name="auc"),
]

# Establish the model's topography.
model = create_model(settings, metrics)

# Train the model on the training set.
experiment = train_model(
    "baseline", model, train_features, train_labels, settings
)

# Plot metrics vs. epochs
ml_edu.results.plot_experiment_metrics(experiment, ["accuracy", "precision", "recall"])
ml_edu.results.plot_experiment_metrics(experiment, ["auc"])

def compare_train_validation(experiment: ml_edu.experiment.Experiment, validation_metrics: dict[str, float]):
  print("Comparing metrics between train and validation:")
  for metric, validation_value in validation_metrics.items():
    print("------")
    print(f"Train {metric}: {experiment.get_final_metric_value(metric):.4f}")
    print(f"Validation {metric}:  {validation_value:.4f}")

# Evaluate validation metrics
validation_metrics = experiment.evaluate(validation_features, validation_labels)
compare_train_validation(experiment, validation_metrics)
