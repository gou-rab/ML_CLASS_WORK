
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

# Load data
df = pd.read_csv('Cars.csv')
print("Dataset shape:", df.shape)
print("Status distribution:\n", df['Status'].value_counts().to_string())
print("\nSample data:\n", df.head().to_string())

# Preprocess
df_processed = df.drop('Car_ID', axis=1).copy()
label_encoders = {}
cat_cols = ['Brand', 'Model', 'Color', 'Engine_Type', 'Transmission', 'Status']
for col in cat_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

X = df_processed.drop('Status', axis=1)
y = df_processed['Status']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Models dict
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
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
    print(classification_report(y_test, y_pred, target_names=['Available', 'Reserved', 'Sold']))

# Results table (NO tabulate needed)
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\n" + "="*50)
print("MODEL COMPARISON TABLE")
print("-" * 50)
print(results_df.to_string(index=False, float_format='%.4f'))

# Feature importance
rf_model = models['Random Forest']
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTOP FEATURES (Random Forest)")
print(feat_importance.head(10).to_string(index=False))

# Best model confusion matrix
best_model_name = results_df.iloc[0]['Model']
best_preds = preds_dict[best_model_name]
cm = confusion_matrix(y_test, best_preds)
print(f"\nCONFUSION MATRIX - {best_model_name} (Acc: {results[best_model_name]:.4f})")
print(cm)

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Available', 'Reserved', 'Sold'])
plt.yticks(tick_marks, ['Available', 'Reserved', 'Sold'])
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print(f"\n🏆 WINNER: {best_model_name} - {results[best_model_name]:.4f} accuracy")

