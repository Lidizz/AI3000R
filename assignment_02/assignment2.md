## Assignment 2: Customer Churn Prediction (Classification)

### Scenario
You work for a telecom company that wants to predict which customers are likely to cancel their service (churn). You have customer data with various features, and you need to build a classifier to identify at-risk customers.

### Dataset
Create a synthetic dataset with the following features:

- **Monthly Charges:** 20-100 (numerical)
- **Tenure (months):** 1-72 (numerical)
- **Contract Type:** Month-to-month, One year, Two year (categorical)
- **Internet Service:** DSL, Fiber optic, No (categorical)
- **Customer Service Calls:** 0-10 (numerical)
- **Churn:** Yes/No (target variable)

Or use this: `customer_churn_data.txt`

Here is also a starter for the code: `assignment_churn_starter.py`

### Tasks

#### Task 1: Data Preparation
1. Generate a synthetic dataset with 500 samples using numpy
2. Create a correlation where:
   - High monthly charges + short tenure + month-to-month contracts → higher churn
   - More customer service calls → higher churn probability
3. Use LabelEncoder to encode the categorical features (Contract Type, Internet Service, Churn)
4. Split the data into 80% training and 20% testing sets

#### Task 2: Build Three Classifiers
Train the following models on your training data:

1. **Logistic Regression** (use C=100)
2. **Naive Bayes** (GaussianNB)
3. **Linear SVM** (LinearSVC)

For each model:
- Train on the training set
- Make predictions on the test set
- Store the predictions for comparison

#### Task 3: Model Evaluation
1. Create a confusion matrix for each classifier
2. Generate a classification report showing precision, recall, and F1-score
3. Compare the three models and answer:
   - Which model has the highest accuracy?
   - Which model has the best balance between precision and recall?
   - Which model would you recommend for this business problem and why?

#### Task 4: Cross-Validation
1. Perform 5-fold cross-validation on your best performing model
2. Report the mean accuracy and standard deviation
3. Does this confirm your model choice?

---