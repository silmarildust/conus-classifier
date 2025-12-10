import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pca_features_for_svm.csv")

X = df[["PC1", "PC2"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", C=1, random_state=42)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

results = X_test.copy()
results["True Class"] = y_test.values
results["Predicted Class"] = y_pred

px.scatter(
    results,
    x="PC1",
    y="PC2",
    #z="PC3",
    color="Predicted Class",
    symbol="True Class",
    title="SVM Classification Results (2D PCA Space)"
).show()