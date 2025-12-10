# conus-classifier
The master folder for my geometric morphometric-based machine learning classifier of Conus shells. I am working on this project independently.

This does not include the data files (images, landmarks, results), since the project is still ongoing. However, the principal components for specific measurements of *Conus muriculatus* and *Conus striolatus* are included since this is the direct data used to achieve 100% accuracy using Support Vector Machine.

# pipeline

The pipeline for geometric morphometrics is:
1. Landmark images using ImageJ, ImgLab, or a similar tool
2. If multiple specimen per image, crop images to bounding boxes using `image_preprocessing.py`
3. Generalized procrustes analysis through `generalized_procrustes_analysis.R`
4. Use the output data to train and test a chosen machine learning model (KNN, Perceptron, SVM)
