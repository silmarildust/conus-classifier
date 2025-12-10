import pandas as pd
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#SETUP OF DATASET
species1_raw = pd.read_csv("landmarks_training_gpa/Conus_striolatus_landmarks_cropped_gpa_wide.csv")
species2_raw = pd.read_csv("landmarks_training_gpa/Conus_muriculatus_landmarks_cropped_gpa_wide.csv")

species1_raw["Class"] = "striolatus"
species2_raw["Class"] = "muriculatus"

shell_dataset_raw = pd.concat([species1_raw, species2_raw], ignore_index=True)

landmark_columns = [col for col in shell_dataset_raw.columns if col.startswith("x") or col.startswith("y")]
shell_dataset = shell_dataset_raw[landmark_columns + ["Class"]].copy()

shell_dataset["ant_distance"] = (shell_dataset["y9"] - shell_dataset["y8"]).abs()

print("data ready check!!!!!!!!!!!!!!!!!!!!!!!")

#MODEL TRAINING
X = shell_dataset[["x0", "ant_distance"]]
y = shell_dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

#model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#RESULTS VISUALIZATION

results = X_test.copy()
results["True Class"] = y_test.values
results["Predicted Class"] = y_pred

px.scatter(
    results,
    x="x0",
    y="ant_distance",
    color="Predicted Class",
    symbol="True Class",
    title="KNN Classification Results (k=5)"
).show()
