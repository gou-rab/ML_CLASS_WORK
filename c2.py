# CORE IMPORTS ONLY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Load Sales data from image (adjust path if needed; assumes Excel screenshot extracted or manual CSV save as 'Sales.csv')
df = pd.read_csv('Sales.csv')  # Or pd.read_excel('Sales.xlsx') if Excel file available
print("Dataset shape:", df.shape)

# Infer target - assuming binary classification on Salesperson (multiple unique, so pick binary or change to multi-class)
# From image: Salesperson has Pamela Blair, Ricardo Lee, Chelsea Gonzales - treat as multi-class or binarize e.g., 'Pamela Blair' vs others
# For binary, let's binarize Salesperson: 0 if 'Pamela Blair', 1 otherwise (adjust as needed)
df['Salesperson_binary'] = (df['Salesperson'] != 'Pamela Blair').astype(int)
print("Target distribution:\n", pd.Series(df['Salesperson_binary']).value_counts().to_string())
print("\nSample data:\n", df.head().to_string())

# Preprocess: Drop irrelevant cols like ID, encode categoricals, convert date to numeric feature
df['Date'] = pd.to_datetime(df['Date'])
df['Date_numeric'] = (df['Date'] - df['Date'].min()).dt.days  # Days since first sale

df_processed = df.drop(['Sale_ID', 'Salesperson', 'Date'], axis=1).copy()  # Drop ID, original Salesperson, Date; keep binary target
label_encoders = {}
cat_cols = ['Payment_Method']
for col in cat_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

X = df_processed.drop('Salesperson_binary', axis=1)  # Features: Car, Quantity, Price, Payment_Method (encoded), Date_numeric
y = df_processed['Salesperson_binary']  # Target: Binary Salesperson

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (stratified for balance)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Models (binary classification optimized)
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Decision Tree': DecisionTreeClassifier('random_state'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Train & evaluate
results = {}
preds_dict = {}
print("\n" + "="*50)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    preds_dict[name] = y_pred
    print(f"\n{name}: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Pamela Blair', 'Others']))

# Results table
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\n" + "="*50)
print("MODEL COMPARISON TABLE")
print("-" * 50)
print(results_df.to_string(index=False, float_format='%.4f'))

# Feature importance (Random Forest)
rf_model = models['Random Forest']
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTOP FEATURES (Random Forest)")
print(feat_importance.to_string(index=False))

# Best model confusion matrix
best_model_name = results_df.iloc[0]['Model']
best_preds = preds_dict[best_model_name]
cm = confusion_matrix(y_test, best_preds)
print(f"\nCONFUSION MATRIX - {best_model_name} (Acc: {results[best_model_name]:.4f})")
print("[[TN, FP], [FN, TP]] (Pamela=0, Others=1)")
print(cm)

# Plot
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Pamela Blair', 'Others'])
plt.yticks(tick_marks, ['Pamela Blair', 'Others'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print(f"\n🏆 WINNER: {best_model_name} - {results[best_model_name]:.4f} accuracy")
