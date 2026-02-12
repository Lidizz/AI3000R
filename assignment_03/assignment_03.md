## Assignment 3: Sales Forecasting with Regression (Regression)

### Scenario
You're a data analyst for an e-commerce company. Your task is to predict weekly sales based on marketing spend and website traffic. The company wants to optimize their marketing budget and understand the relationship between spending and sales.

### Dataset
Create a synthetic dataset with the following features:

- **Marketing Spend ($):** 1000-10000
- **Website Traffic (visitors):** 5000-50000
- **Social Media Engagement:** 100-1000
- **Weekly Sales ($):** Target variable (with some non-linear relationship to features)

Or use this: `sales_forecast_data.txt`

Use this code as a starter: `assignment_regression_starter.py`

### Tasks

#### Task 1: Data Generation and Exploration
1. Generate 200 samples with the following relationship:
```
   Sales = 500 + 0.8*Marketing + 0.3*Traffic + 0.5*Engagement + noise
   Add a non-linear component: + 0.00001*(Marketing^2)
```
2. Split data into 80% training and 20% testing
3. Create a scatter plot showing the relationship between Marketing Spend and Sales
4. Calculate and print the correlation coefficients between each feature and Sales

#### Task 2: Linear Regression
1. Build a Linear Regression model using all three features
2. Train the model on training data
3. Make predictions on the test set
4. Calculate and print the following metrics:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score
   - Explained Variance Score
5. Plot actual vs. predicted values for the test set

#### Task 3: Polynomial Regression
1. Create polynomial features with degrees 2, 3, and 5
2. For each degree:
   - Train a model on the transformed training data
   - Make predictions on the test set
   - Calculate the same metrics as Task 2
   - Store the results
3. Create a comparison table showing how metrics change with polynomial degree
4. Answer: Which degree gives the best performance? Is there evidence of overfitting?

#### Task 4: Model Persistence and Single Variable Analysis
1. Save your best model to a file using pickle
2. Load the model back and verify it works by making a prediction
3. Create a single-variable regression using only Marketing Spend:
   - Train the model
   - Plot the regression line over a scatter plot of the data
   - Compare the R² score to your multivariate model
4. Answer: How much predictive power do the other features add?