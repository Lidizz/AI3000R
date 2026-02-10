import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm

# Load the sales forecasting data
input_file = 'sales_forecast_data.txt'
data = np.loadtxt(input_file, delimiter=',')

# Split into features (X) and target (y)
X = data[:, :-1]  # All columns except last
y = data[:, -1]   # Last column (Weekly Sales)

print("Data loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Sales range: ${y.min():.2f} - ${y.max():.2f}")
print(f"Average sales: ${y.mean():.2f}")
print("\nFeature columns:")
print("0: Marketing Spend ($)")
print("1: Website Traffic (visitors)")
print("2: Social Media Engagement")

# ============================================
# TODO: Task 1 - Data Exploration
# ============================================

# Split data into training (80%) and testing (20%)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Calculate correlation coefficients
print("\n=== Correlation with Sales ===")
for i, feature_name in enumerate(['Marketing Spend', 'Website Traffic', 'Social Media Engagement']):
    correlation = np.corrcoef(X[:, i], y)[0, 1]
    print(f"{feature_name}: {correlation:.3f}")

# Plot Marketing Spend vs Sales
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, alpha=0.5)
plt.xlabel('Marketing Spend ($)')
plt.ylabel('Weekly Sales ($)')
plt.title('Marketing Spend vs Weekly Sales')
plt.tight_layout()
plt.savefig('marketing_vs_sales.png', dpi=100)
print("\nPlot saved as 'marketing_vs_sales.png'")
plt.close()


# ============================================
# TODO: Task 2 - Linear Regression
# ============================================

print("\n=== LINEAR REGRESSION ===")
# Your code here:
# linear_regressor = linear_model.LinearRegression()
# linear_regressor.fit(X_train, y_train)
# y_test_pred_linear = linear_regressor.predict(X_test)

# Calculate metrics
# print("Mean Absolute Error =", round(sm.mean_absolute_error(y_test, y_test_pred_linear), 2))
# print("Mean Squared Error =", round(sm.mean_squared_error(y_test, y_test_pred_linear), 2))
# print("R2 Score =", round(sm.r2_score(y_test, y_test_pred_linear), 2))
# print("Explained Variance Score =", round(sm.explained_variance_score(y_test, y_test_pred_linear), 2))


# ============================================
# TODO: Task 3 - Polynomial Regression
# ============================================

print("\n=== POLYNOMIAL REGRESSION ===")

# Store results for comparison
results = {'degree': [], 'MAE': [], 'MSE': [], 'R2': [], 'EVS': []}

for degree in [2, 3, 5]:
    print(f"\nDegree {degree}:")
    # Your code here:
    # polynomial = PolynomialFeatures(degree=degree)
    # X_train_poly = polynomial.fit_transform(X_train)
    # X_test_poly = polynomial.transform(X_test)
    
    # poly_model = linear_model.LinearRegression()
    # poly_model.fit(X_train_poly, y_train)
    # y_test_pred_poly = poly_model.predict(X_test_poly)
    
    # Calculate and store metrics
    # mae = round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2)
    # mse = round(sm.mean_squared_error(y_test, y_test_pred_poly), 2)
    # r2 = round(sm.r2_score(y_test, y_test_pred_poly), 2)
    # evs = round(sm.explained_variance_score(y_test, y_test_pred_poly), 2)
    
    # results['degree'].append(degree)
    # results['MAE'].append(mae)
    # results['MSE'].append(mse)
    # results['R2'].append(r2)
    # results['EVS'].append(evs)
    
    # print(f"  MAE: {mae}")
    # print(f"  MSE: {mse}")
    # print(f"  R2: {r2}")
    # print(f"  EVS: {evs}")
    pass

# Print comparison table
print("\n=== COMPARISON TABLE ===")
print("Degree | MAE     | MSE      | R2    | EVS")
print("-------+---------+----------+-------+-----")
# for i in range(len(results['degree'])):
#     print(f"  {results['degree'][i]}    | {results['MAE'][i]:7.2f} | {results['MSE'][i]:8.2f} | {results['R2'][i]:.3f} | {results['EVS'][i]:.3f}")


# ============================================
# TODO: Task 4 - Model Persistence & Single Variable
# ============================================

print("\n=== MODEL PERSISTENCE ===")
# Save the best model (choose based on your results)
# best_model = linear_regressor  # or one of your polynomial models
# output_model_file = 'sales_model.pkl'
# with open(output_model_file, 'wb') as f:
#     pickle.dump(best_model, f)
# print(f"Model saved to {output_model_file}")

# Load and test
# with open(output_model_file, 'rb') as f:
#     loaded_model = pickle.load(f)
# test_prediction = loaded_model.predict(X_test[:1])
# print(f"Test prediction with loaded model: ${test_prediction[0]:.2f}")


print("\n=== SINGLE VARIABLE REGRESSION ===")
# Use only Marketing Spend (column 0)
# X_train_single = X_train[:, 0:1]
# X_test_single = X_test[:, 0:1]

# single_regressor = linear_model.LinearRegression()
# single_regressor.fit(X_train_single, y_train)
# y_test_pred_single = single_regressor.predict(X_test_single)

# r2_single = sm.r2_score(y_test, y_test_pred_single)
# print(f"Single variable R2 score: {r2_single:.3f}")
# print(f"Multivariate R2 score: {sm.r2_score(y_test, y_test_pred_linear):.3f}")
# print(f"Improvement from additional features: {(sm.r2_score(y_test, y_test_pred_linear) - r2_single):.3f}")

# Plot single variable regression
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test_single, y_test, color='blue', alpha=0.5, label='Actual')
# plt.plot(X_test_single, y_test_pred_single, color='red', linewidth=2, label='Predicted')
# plt.xlabel('Marketing Spend ($)')
# plt.ylabel('Weekly Sales ($)')
# plt.title('Single Variable Regression: Marketing Spend vs Sales')
# plt.legend()
# plt.tight_layout()
# plt.savefig('single_variable_regression.png', dpi=100)
# print("Plot saved as 'single_variable_regression.png'")


# ============================================
# Final Recommendations
# ============================================
print("\n=== YOUR ANALYSIS ===")
print("1. Which model performs best?")
print("   [Your answer here]")
print("\n2. Is there evidence of overfitting?")
print("   [Your answer here]")
print("\n3. What would you recommend to the business?")
print("   [Your answer here]")
