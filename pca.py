import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df1 = pd.read_csv("landmarks_training_gpa/Conus_striolatus_landmarks_cropped_gpa_wide.csv")
df2 = pd.read_csv("landmarks_training_gpa/Conus_muriculatus_landmarks_cropped_gpa_wide.csv")

df1 = df1.iloc[:, 2:]
df2 = df2.iloc[:, 2:]

data1 = df1.apply(pd.to_numeric, errors='coerce').to_numpy()
data2 = df2.apply(pd.to_numeric, errors='coerce').to_numpy()

print(f"Species 1: {data1.shape[0]} specimens, {data1.shape[1]} coordinates")
print(f"Species 2: {data2.shape[0]} specimens, {data2.shape[1]} coordinates")

X = np.vstack([data1, data2])
y = np.array([0]*len(data1) + [1]*len(data2))  # 0 = striolatus, 1 = muriculatus

n_dims = 2
n_landmarks = X.shape[1] // n_dims
print(f"Detected {n_landmarks} landmarks (2D).")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
scores = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

print("\nExplained variance by first 5 PCs:")
for i, var in enumerate(explained_var[:5]):
    print(f"PC{i+1}: {var:.3f}")

plt.figure(figsize=(6, 5))
plt.scatter(scores[y==0, 0], scores[y==0, 1], c='cornflowerblue', edgecolors='k', label='Conus striolatus')
plt.scatter(scores[y==1, 0], scores[y==1, 1], c='tomato', edgecolors='k', label='Conus muriculatus')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
plt.title("PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mean_shape = np.mean(X, axis=0)
pc1_vector = pca.components_[0]
scale = 3 * np.sqrt(pca.explained_variance_[0])

shape_plus = mean_shape + pc1_vector * scale
shape_minus = mean_shape - pc1_vector * scale

mean_shape = mean_shape.reshape(n_landmarks, n_dims)
shape_plus = shape_plus.reshape(n_landmarks, n_dims)
shape_minus = shape_minus.reshape(n_landmarks, n_dims)

plt.figure(figsize=(5, 5))
plt.plot(mean_shape[:, 0], mean_shape[:, 1], 'ko-', label='Mean shape')
plt.plot(shape_plus[:, 0], shape_plus[:, 1], 'r--', label='+PC1')
plt.plot(shape_minus[:, 0], shape_minus[:, 1], 'b--', label='-PC1')
plt.axis('equal')
plt.legend()
plt.title("Shape Variation Along PC1")
plt.show()

pca_df = pd.DataFrame(scores[:, :10], columns=[f"PC{i+1}" for i in range(10)])  # first 10 PCs
pca_df["species"] = y
pca_df.to_csv("pca_features_for_svm.csv", index=False)

print("\nSaved PCA features to 'pca_features_for_svm.csv'")