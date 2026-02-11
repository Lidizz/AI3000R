# AI and Machine Learning Assignments

## Assignment 1: What Is Intelligence?
**Exploring the Nature of AI, Mind, and Machine**

**Duration:** 1.5 hour  
**Format:** Group research, reflection, and presentation  
**Group Size:** 2 students per group (5 groups total)

### Overview
Before we dive into algorithms and code, we need to wrestle with a fundamental question: What is intelligence? This assignment asks you to explore this question from multiple angles—philosophical, biological, computational, and beyond. You'll research, reflect, and present your findings to spark a class-wide discussion.

---

### Group 1: The Materialist Perspective — "Is It All Just Atoms?"
**Central Question:** If the brain is just atoms following physical laws, what makes it "intelligent"? Could any sufficiently complex arrangement of matter become intelligent?

**Guiding Questions:**
- What is physicalism/materialism in philosophy of mind?
- If intelligence emerges from atoms interacting, what distinguishes a brain from a rock?
- Does complexity alone create intelligence, or is something else required?
- Could a computer made of different materials (silicon vs. carbon) be truly intelligent?

**Reflection Prompt:** Do you find it satisfying or troubling to think of your own mind as "just" atoms? Why?

---

### Group 2: The Turing Test — "If It Acts Intelligent, Is It?"
**Central Question:** Alan Turing proposed that if a machine can convincingly imitate human conversation, we should consider it intelligent. Is behavior enough to define intelligence?

**Guiding Questions:**
- What is the Turing Test and what was Turing trying to demonstrate?
- What are the main criticisms of the Turing Test (e.g., Searle's Chinese Room)?
- Have any systems "passed" the Turing Test? Does that matter?
- Is fooling humans the same as being intelligent?

**Reflection Prompt:** If you couldn't tell whether you were talking to a human or an AI, would that change how you think about the AI's "mind"?

---

### Group 3: Biological Intelligence — "What Can Nature Teach Us?"
**Central Question:** Intelligence exists in many forms in nature—from octopuses to slime molds to human brains. What do these diverse systems share, and what makes each unique?

**Guiding Questions:**
- How do different animals demonstrate intelligence (problem-solving, tool use, social learning)?
- Is human intelligence fundamentally different from animal intelligence, or just more complex?
- Can organisms without brains (plants, fungi, single cells) be considered intelligent?
- What does evolution suggest about the "purpose" of intelligence?

**Reflection Prompt:** Does learning about intelligence in other species change your view of what AI could or should be?

---

### Group 4: Alternative Intelligences — "Could There Be Minds We Can't Recognize?"
**Central Question:** Our concept of intelligence is shaped by human experience. Could there be forms of intelligence so different from ours that we wouldn't recognize them?

**Guiding Questions:**
- What is "anthropocentrism" in discussions of intelligence?
- How might an alien intelligence or a radically different AI think in ways we can't comprehend?
- Could collective systems (ant colonies, the internet, corporations) be considered intelligent?
- What are the limits of defining intelligence based on human criteria?

**Reflection Prompt:** If we encountered a truly alien intelligence, how would we know? What would convince you?

---

### Group 5: Intelligence vs. Consciousness — "Does Understanding Require Experience?"
**Central Question:** Is there a difference between processing information intelligently and actually understanding or experiencing something? Can AI be intelligent without being conscious?

**Guiding Questions:**
- What is the distinction between intelligence and consciousness?
- What is the "hard problem of consciousness"?
- Could a system be highly intelligent but have no inner experience (a "philosophical zombie")?
- Does it matter ethically whether AI systems have experiences?

**Reflection Prompt:** When you use an AI system, do you ever catch yourself wondering if "anyone is home"? What prompts that feeling?

---

### Presentation Requirements
Your 5-minute presentation should include:

1. **Brief overview of your perspective** (1 min)
2. **Key insights from your research** (2 min)
3. **Your group's reflection** — What do you think? Where do you agree or disagree with what you found? (1.5 min)
4. **One question to pose to the class for discussion** (0.5 min)

You may use slides, whiteboard, or simply speak—whatever helps you communicate effectively.

---

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