# Classification Web App

This Streamlit web application allows users to explore different classifiers on three datasets: Iris, Breast Cancer, and Wine. It supports three classifiers: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest. The app provides insights into dataset characteristics, including PCA plots, confusion matrices, and classification reports to evaluate classifier performance.

#### Datasets
1. Iris Dataset
The Iris dataset contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers.

2. Breast Cancer Dataset
The Breast Cancer dataset includes features derived from digitized images of fine needle aspirates (FNA) of breast masses. It aims to predict whether a mass is malignant or benign.

3. Wine Dataset
The Wine dataset consists of chemical analysis results for wines, classified into three classes. It serves as a classification task for wine categories.

#### Classifier Options
1. K-Nearest Neighbors (KNN)
KNN is a simple and effective algorithm that classifies a data point based on the majority class among its k-nearest neighbors.

2. Support Vector Machine (SVM)
SVM is a powerful algorithm that constructs a hyperplane in a high-dimensional space to separate data into classes.

3. Random Forest
Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions for more accurate and robust results.

#### PCA Plot
The Principal Component Analysis (PCA) plot in the sidebar provides a visual representation of the dataset in two dimensions.

#### Classifier Parameters
Adjust the parameters of the selected classifier in the sidebar to explore their impact on classification results.

#### Confusion Matrix
The confusion matrix visualizes the performance of the classifier by displaying true positive, true negative, false positive, and false negative values.

#### Classification Report
The classification report provides precision, recall, F1-score, and support for each class, offering a comprehensive evaluation of classifier performance.

#### Accuracy
The overall accuracy of the selected classifier on the test dataset is displayed at the top of the page.


## Usage

- Choose a dataset from the sidebar using the 'Select Dataset' dropdown.
- Explore dataset characteristics in the sidebar.
- Select a classifier from the 'Select Classifier' dropdown.
- Adjust classifier parameters in the sidebar if necessary.
- View classifier results, confusion matrix, and classification report.

## About

This application supports three datasets: Iris, Breast Cancer, and Wine. The available classifiers are K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest. Principal Component Analysis (PCA) plots provide insights into the dataset, while confusion matrices and classification reports help evaluate classifier performance.

## Feedback
We appreciate your feedback! Please rate your experience using the star rating system below.

If you rated 1-3 stars, we value your feedback! Please let us know how we can improve by providing your comments in the "Enter your Feedback" text area.
If you rated 4-5 stars, thank you for your positive feedback! Enjoy exploring the classifiers!
Note: If you encounter any issues or have suggestions, feel free to submit feedback.

## How to Run
Install the required packages by running pip install -r requirements.txt.
Run the app using streamlit run your_app_file.py.

