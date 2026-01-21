import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# 1. Load the Excel file
data = pd.read_excel("iris.xlsx")   # columns: sepal_length, sepal_width, petal_length, petal_width, class

# 2. Features (X) and labels (y)
X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = data["class"]

# 3. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# 4. Train SVM model
model = SVC(kernel="linear", C=1.0, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Metrics
accuracy = accuracy_score(y_test, y_pred)

# macro = average over classes, good for balanced datasets like iris
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", accuracy)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)

print("\nDetailed classification report:\n")
print(classification_report(y_test, y_pred))
