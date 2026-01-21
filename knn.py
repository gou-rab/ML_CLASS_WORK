# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Iris dataset
cancer = load_breast_cancer()
X = cancer.data  # Features (sepal length, sepal width, petal length, petal width)
y = cancer.target  # Target classes (0: setosa, 1: versicolor, 2: virginica)

# Step 2: Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train KNN classifier (k=5 neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 4: Make predictions on test data
y_pred = knn.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
#print("\nConfusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

# Step 6: Predict a new sample (example)
#new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Features of a new flower
#prediction = knn.predict(new_sample)
#print(f"\nPrediction for new sample: {iris.target_names[prediction[0]]}")
