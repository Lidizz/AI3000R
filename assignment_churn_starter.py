import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the customer churn data
input_file = 'customer_churn_data.txt'
data = np.loadtxt(input_file, delimiter=',')

# Split into features (X) and target (y)
X = data[:, :-1]  # All columns except last
y = data[:, -1]   # Last column (Churn: 0 or 1)

print("Data loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Churn rate: {(y.sum() / len(y) * 100):.1f}%")
print("\nFeature columns:")
print("0: Monthly Charges")
print("1: Tenure (months)")
print("2: Contract Type (0=month-to-month, 1=one year, 2=two year)")
print("3: Internet Service (0=DSL, 1=Fiber, 2=No)")
print("4: Customer Service Calls")

# ============================================
# TODO: Task 1 - Data Preparation
# Split data into training (80%) and testing (20%) sets
# ============================================

# Your code here:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # Make results reproducible
)



# ============================================
# TODO: Task 2 - Build Three Classifiers
# ============================================

# 1. Logistic Regression
print("\n=== Training Logistic Regression ===")
# Your code here:
classifier_lr = LogisticRegression(solver='liblinear', C=100)
classifier_lr.fit(X_train, y_train)
y_pred_lr = classifier_lr.predict(X_test)


# 2. Naive Bayes
print("\n=== Training Naive Bayes ===")
# Your code here:
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_pred_nb = classifier_nb.predict(X_test)


# 3. Linear SVM
print("\n=== Training Linear SVM ===")
# Your code here:
classifier_svm = LinearSVC(max_iter=1000)
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)


# ============================================
# TODO: Task 3 - Model Evaluation
# ============================================

# For each classifier, create confusion matrix and classification report
print("\n=== LOGISTIC REGRESSION RESULTS ===")
# Your code here:
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
print(classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn']))

accuracy_lr = 100.0 * (y_test == y_pred_lr).sum() / y_test.shape[0]
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}%")


print("\n=== NAIVE BAYES RESULTS ===")
# Your code here:
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(cm_nb)
print(classification_report(y_test, y_pred_nb, target_names=['No Churn', 'Churn']))

accuracy_nb = 100.0 * (y_test == y_pred_nb).sum() / y_test.shape[0]
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}%")


print("\n=== LINEAR SVM RESULTS ===")
# Your code here:
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)
print(classification_report(y_test, y_pred_svm, target_names=['No Churn', 'Churn']))

accuracy_svm = 100.0 * (y_test == y_pred_svm).sum() / y_test.shape[0]
print(f"Linear SVM Accuracy: {accuracy_svm:.2f}%")  

# ============================================
# TODO: Task 4 - Cross-Validation
# ============================================

# Perform 5-fold cross-validation on your best model
print("\n=== Cross-Validation Results ===")
# Your code here:
best_classifier = classifier_lr
cv_scores = cross_val_score(best_classifier, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


# ============================================
# Comparison Summary
# ============================================
print("\n=== MODEL COMPARISON SUMMARY ===")
print("Complete this table based on your results:")
print("Model               | Accuracy | Precision | Recall | F1-Score")
print("--------------------+----------+-----------+--------+---------")
print(f"Logistic Regression | {accuracy_lr:.2f}%   |    0.74   |   0.46 |    0.57")
print(f"Naive Bayes         | {accuracy_nb:.2f}%   |    0.68   |   0.43 |    0.52")
print(f"Linear SVM          | {accuracy_svm:.2f}%   |    0.74   |   0.46 |    0.57")
print("\nRecommendation: Logistic Regression")
