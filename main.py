import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = {
    'Hours': [10, 9, 2, 15, 10, 16, 11, 16],
    'Risk_Score': [95, 80, 10, 50, 45, 98, 38, 93]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Linear regression
X_linear = df['Hours'].values.reshape(-1, 1)
y_linear = df['Risk_Score'].values.reshape(-1, 1)

linear_model = LinearRegression()
linear_model.fit(X_linear, y_linear)
linear_pred = linear_model.predict(X_linear)

# Get the slope and intercept of the line best fit
slope = linear_model.coef_[0][0]
intercept = linear_model.intercept_[0]

print("Slope for Linear :", slope)
print("Intercept for Linear :", intercept)

# Evaluate performance
r_squared = r2_score(y_linear, linear_pred)
print("R-squared for Linear :", r_squared)

# Plot the data and regression lines
plt.scatter(df['Hours'], df['Risk_Score'], color='blue', label='Actual data')
plt.plot(df['Hours'], linear_pred, color='red', label='Linear Regression')
plt.xlabel('Hours')
plt.ylabel('Risk Score')
plt.title('Linear vs Polynomial Regression')
plt.legend()



# Polynomial regression
X_poly = df['Hours'].values.reshape(-1, 1)
y_poly = df['Risk_Score'].values

poly_features = PolynomialFeatures(degree=9)
X_poly = poly_features.fit_transform(X_poly)

poly_model = LinearRegression()
poly_model.fit(X_poly, y_poly)
poly_pred = poly_model.predict(X_poly)

# Get the intercept and coefficients for the polynomial regression
intercept_poly = poly_model.intercept_
coefficients_poly = poly_model.coef_

print("Intercept for Polynomial Regression:", intercept_poly)
print("Coefficients for Polynomial Regression:", coefficients_poly)

# Evaluate performance
r_squared = r2_score(y_linear, poly_pred)
print("R-squared for Poly :", r_squared)

# Plot the data and regression lines
plt.plot(df['Hours'], poly_pred, color='green', label='Polynomial Regression')
plt.show()

