import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# 1. Take training data from user
# -------------------------
n_rows = 8
n_features = 2

X_list = []
y_list = []

print(f"Enter {n_rows} rows of data.")
print("For each row, enter: x1 x2 y")

for i in range(n_rows):
    row = list(map(float, input(f"Row {i+1} (x1 x2 y): ").split()))
    if len(row) != n_features + 1:
        raise ValueError(f"Expected {n_features + 1} values, got {len(row)}")
    x1, x2, y_val = row
    X_list.append([x1, x2])
    y_list.append(y_val)

X = np.array(X_list, dtype=float)    # shape (8, 2)
y = np.array(y_list, dtype=float)    # shape (8,)

# -------------------------
# 2. Train multiple linear regression
# -------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------
# 3. Get b0, b1, b2
# -------------------------
b0 = model.intercept_       # intercept
b1, b2 = model.coef_        # coefficients

print("b0 (intercept):", b0)
print("b1, b2 (coefficients):", b1, b2)

# -------------------------
# 4. Predict y for new user input
# -------------------------
x1_new, x2_new = map(float, input("Enter new x1 x2 to predict y: ").split())
X_new = np.array([[x1_new, x2_new]], dtype=float)

y_pred = model.predict(X_new)[0]
print("Predicted y:", y_pred)