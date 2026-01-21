# 1. Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 2. Sample data (X = feature, y = target)
# Suppose we have study hours (X) and marks (y)
X = np.array([-2,-1,0,1,2]).reshape(-1, 1)  # shape (n_samples, n_features)
y = np.array([1,2,3,3,4])

# 3. Create and train the model
model = LinearRegression()
model.fit(X, y)

# 4. Check learned parameters
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# 5. Predict for new data
X_new = np.array([[9], [10]])    # predict marks for 7 and 8 hours
y_pred = model.predict(X_new)
print("Predictions :", y_pred)

# 6. Plot data and regression line
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.legend()
plt.show()
