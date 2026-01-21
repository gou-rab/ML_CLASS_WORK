import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5,6]).reshape(-1,1)
Y=np.array([42,47,50,52,57,60])

model = LinearRegression()
model.fit(X,Y)

print("slope:(coefficient):",model.coef_[0])
print("intercept:",model.intercept)

X_new = np.array([7],[8])
Y_pred = model.predict(X_new)
print("predictions for 7 and 8 hours:",Y_pred)

