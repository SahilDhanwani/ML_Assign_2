ynomial regression
X_poly = df['Hours'].values.reshape(-1, 1)
y_poly = df['Risk_Score'].values.reshape(-1, 1)

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_poly)

poly_model = LinearRegression()
poly_model.fit(X_poly, y_poly)
poly_pred = poly_model.predict(X_poly)

# Evaluate performance
linear_rmse = np.sqrt(mean_squared_error(y_linear, linear_pred))
poly_rmse = np.sqrt(mean_squared_error(y_poly, poly_pred))

# Compare models
print("Linear Regression RMSE:", linear_rmse)
print("Polynomial Regression RMSE:", poly_rmse)

# Plot the data and regression lines
import matplotlib.pyplot as plt

plt.scatter(df['Hours'], df['Risk_Score'], color='blue', label='Actual data')
plt.plot(df['Hours'], linear_pred, color='red', label='Linear Regression')
plt.plot(df['Hours'], poly_pred, color='green', label='Polynomial Regression')

plt.xlabel('Hours')
plt.ylabel('Risk Score')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()