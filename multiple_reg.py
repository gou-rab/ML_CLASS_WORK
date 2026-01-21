import numpy as np
from sklearn.linear_model import LinearRegression

# Q1 data from notebook
# X1, X2, Y
data = np.array([
    [3, 8, -3.7],
    [4, 5,  3.5],
    [5, 7,  2.5],
    [6, 3, 11.5],
    [2, 1,  5.7]
])

X = data[:, :2]   # columns X1, X2
y = data[:, 2]    # column Y

# create and fit model
model = LinearRegression()
model.fit(X, y)

# coefficients
b0 = model.intercept_
b1, b2 = model.coef_

print("b0 (intercept) =", b0)
print("b1 (for X1)    =", b1)
print("b2 (for X2)    =", b2)

# predict Y for X1=3, X2=2 (last row in your table)
x_new = np.array([[3, 2]])
y_pred = model.predict(x_new)[0]
print("Predicted Y for X1=3, X2=2:", y_pred)
