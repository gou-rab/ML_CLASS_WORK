# A small demo: different ways to collect data for ML
# and train a simple model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1) Data from hard-coded list (manual collection)
manual_data = [
    # hours_studied, sleep_hours, passed
    [1, 5, 0],
    [2, 6, 0],
    [3, 6, 0],
    [4, 7, 1],
    [5, 7, 1],
    [6, 8, 1],
]
df_manual = pd.DataFrame(manual_data,
                         columns=["hours_studied", "sleep_hours", "passed"])

print("Manual data:")
print(df_manual, "\n")

# 2) Data from CSV file (simulating downloaded/collected dataset)
# If 'study_data.csv' does not exist yet, create it once.
try:
    df_csv = pd.read_csv("study_data.csv")
    print("Loaded data from study_data.csv:\n", df_csv, "\n")
except FileNotFoundError:
    df_csv = df_manual.copy()
    df_csv.to_csv("study_data.csv", index=False)
    print("study_data.csv not found, created using manual data.\n")

# 3) Data from user input (e.g., form / live data)
user_rows = []
print("Enter 3 new records (hours_studied, sleep_hours, passed 0/1):")
for i in range(3):
    h = float(input(f"Record {i+1} - Hours studied: "))
    s = float(input(f"Record {i+1} - Sleep hours: "))
    p = int(input(f"Record {i+1} - Passed (1) or Failed (0): "))
    user_rows.append([h, s, p])

df_user = pd.DataFrame(user_rows,
                       columns=["hours_studied", "sleep_hours", "passed"])
print("\nUser-entered data:")
print(df_user, "\n")

# Combine all three sources
df_all = pd.concat([df_manual, df_csv, df_user], ignore_index=True)
print("Combined dataset:")
print(df_all, "\n")

# Separate features and target
X = df_all[["hours_studied", "sleep_hours"]]
y = df_all["passed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Simple ML model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Predict for a new example
new_sample = [[4.5, 7.0]]
pred = model.predict(new_sample)[0]
print("Prediction for new sample (4.5 hours study, 7 hours sleep):",
      "Pass" if pred == 1 else "Fail")
