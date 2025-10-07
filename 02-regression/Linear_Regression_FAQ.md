# Linear Regression - Summary and FAQ

## Course Summary

This document provides a comprehensive summary and FAQ for **Module 2: Machine Learning for Regression** from the ML Zoomcamp course.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Key Topics Covered](#key-topics-covered)
3. [Core Concepts](#core-concepts)
4. [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
5. [Code Examples](#code-examples)
6. [Additional Resources](#additional-resources)

---

## Overview

**Linear Regression** is a supervised machine learning algorithm used to predict continuous numerical values. In this module, we work on a **Car Price Prediction Project** to understand how to:
- Prepare and explore data
- Build and train linear regression models
- Validate and evaluate model performance
- Handle categorical variables and feature engineering
- Apply regularization to prevent overfitting

---

## Key Topics Covered

### 2.1 Car Price Prediction Project
- Introduction to the regression problem
- Understanding the dataset and business objective

### 2.2 Data Preparation
- Loading and cleaning data
- Handling missing values
- Normalizing column names and data types

### 2.3 Exploratory Data Analysis (EDA)
- Understanding data distributions
- Identifying patterns and correlations
- Visualizing relationships between features and target

### 2.4 Validation Framework
- Train/Validation/Test split (e.g., 60%/20%/20%)
- Importance of data shuffling with random seeds
- Preventing data leakage

### 2.5 Linear Regression (Simple)
- Understanding the linear relationship: `y = w₀ + w₁x`
- Finding optimal weights using simple formulas

### 2.6 Linear Regression (Vector Form)
- Matrix representation: `y = Xw`
- Working with multiple features simultaneously

### 2.7 Training Linear Regression - Normal Equation
- Mathematical solution: `w = (XᵀX)⁻¹Xᵀy`
- Computing weights without iterative methods

### 2.8 Baseline Model
- Creating a simple baseline for comparison
- Understanding model performance benchmarks

### 2.9 Root Mean Squared Error (RMSE)
- Evaluation metric for regression: `RMSE = √(mean((y_true - y_pred)²))`
- Understanding prediction errors

### 2.10 Validation with RMSE
- Evaluating model on validation dataset
- Comparing training vs validation performance

### 2.11 Feature Engineering
- Creating new features from existing ones
- Transformations (log, polynomial, etc.)

### 2.12 Categorical Variables
- One-hot encoding for categorical features
- Handling non-numeric data

### 2.13 Regularization
- Ridge Regression: Adding penalty term `r·I` to prevent overfitting
- Formula: `w = (XᵀX + rI)⁻¹Xᵀy`
- Choosing regularization parameter `r`

### 2.14 Tuning the Model
- Hyperparameter selection
- Cross-validation strategies

### 2.15 Using the Model
- Making predictions on new data
- Model deployment considerations

### 2.16 Summary
- Recap of key concepts
- Best practices

### 2.17 Explore More
- Additional resources and advanced topics

---

## Core Concepts

### 1. **Linear Regression Equation**
```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```
Where:
- `y` = predicted value
- `w₀` = bias/intercept
- `w₁, w₂, ..., wₙ` = weights/coefficients
- `x₁, x₂, ..., xₙ` = features

### 2. **Normal Equation**
```python
w = (XᵀX)⁻¹Xᵀy
```
This provides the closed-form solution for linear regression.

### 3. **Ridge Regression (Regularized)**
```python
w = (XᵀX + rI)⁻¹Xᵀy
```
Where:
- `r` = regularization parameter
- `I` = identity matrix

### 4. **RMSE (Root Mean Squared Error)**
```python
RMSE = √(mean((y_true - y_pred)²))
```
Lower RMSE indicates better model performance.

---

## Frequently Asked Questions (FAQ)

### **Q1: What is the difference between simple and multiple linear regression?**

**A:** 
- **Simple Linear Regression**: Uses one feature (independent variable) to predict the target
  - Example: `price = w₀ + w₁ × mileage`
- **Multiple Linear Regression**: Uses multiple features
  - Example: `price = w₀ + w₁ × mileage + w₂ × age + w₃ × horsepower`

---

### **Q2: Why do we split data into train/validation/test sets?**

**A:** 
- **Training Set (60%)**: Used to train/fit the model
- **Validation Set (20%)**: Used to tune hyperparameters and select the best model
- **Test Set (20%)**: Used for final evaluation to assess real-world performance

This prevents **overfitting** and ensures the model generalizes well to unseen data.

---

### **Q3: Why do we shuffle data before splitting?**

**A:** 
- Ensures random distribution of samples across all sets
- Prevents bias from ordered data (e.g., data sorted by time or category)
- Using a fixed random seed (e.g., `random_state=42`) makes results reproducible

---

### **Q4: How do we handle missing values in features?**

**A:** Common strategies:
1. **Fill with 0**: Simple but may introduce bias
2. **Fill with mean**: Better preserves distribution (calculate mean from training set only!)
3. **Fill with median**: Robust to outliers
4. **Drop rows**: When missing data is minimal

**Important**: Always compute statistics (mean, median) from the training set and apply to validation/test sets to avoid data leakage.

---

### **Q5: What is the Normal Equation and when should we use it?**

**A:** 
The Normal Equation is a closed-form solution: `w = (XᵀX)⁻¹Xᵀy`

**Advantages**:
- Direct solution, no iterations needed
- No learning rate to tune
- Works well for small to medium datasets

**Disadvantages**:
- Computationally expensive for large datasets (matrix inversion is O(n³))
- Requires XᵀX to be invertible
- For large datasets, use gradient descent instead

---

### **Q6: What is regularization and why do we need it?**

**A:** 
Regularization adds a penalty term to prevent the model from overfitting by constraining the weights.

**Ridge Regression formula**: `w = (XᵀX + rI)⁻¹Xᵀy`

**Benefits**:
- Prevents overly complex models
- Reduces variance in predictions
- Helps when features are highly correlated
- Makes XᵀX invertible even when it's singular

**The parameter `r`**:
- `r = 0`: No regularization (standard linear regression)
- Small `r` (e.g., 0.01): Mild regularization
- Large `r` (e.g., 100): Strong regularization (may underfit)

---

### **Q7: How do we choose the best regularization parameter `r`?**

**A:** 
1. Try multiple values: [0, 0.01, 0.1, 1, 5, 10, 100]
2. Train models with each `r` value
3. Evaluate on validation set using RMSE
4. Select `r` with the lowest validation RMSE
5. If multiple values tie, choose the smallest `r`

---

### **Q8: What does RMSE tell us about model performance?**

**A:** 
RMSE (Root Mean Squared Error) measures average prediction error in the same units as the target variable.

- **Lower RMSE = Better model**
- RMSE = 0.52 means on average, predictions are off by ±0.52 units
- Compare RMSE across different models to select the best one
- Always round RMSE consistently (e.g., 2 decimal places) for comparison

---

### **Q9: What is the difference between training RMSE and validation RMSE?**

**A:** 
- **Training RMSE**: Error on data used to train the model (usually lower)
- **Validation RMSE**: Error on unseen data (more realistic)

If training RMSE << validation RMSE → **Overfitting** (model memorized training data)
If both are high → **Underfitting** (model too simple)
If both are similar and low → **Good fit**

---

### **Q10: How does the random seed affect model performance?**

**A:** 
The random seed controls how data is shuffled and split:
- Different seeds create different train/val/test splits
- This leads to slight variations in RMSE
- Standard deviation of RMSE across seeds measures model stability
- Lower std (e.g., 0.007) indicates consistent performance regardless of data split

---

### **Q11: Should we combine train and validation sets for the final model?**

**A:** 
**Yes**, for the final model:
1. Use train set to develop and tune the model
2. Use validation set to select hyperparameters
3. **Combine train + validation** to train the final model (more data = better performance)
4. Use test set only once for final evaluation

---

### **Q12: How do we handle categorical variables in linear regression?**

**A:** 
Use **One-Hot Encoding**:

Original data:
```
color: ['red', 'blue', 'red', 'green']
```

One-hot encoded:
```
color_red:   [1, 0, 1, 0]
color_blue:  [0, 1, 0, 0]
color_green: [0, 0, 0, 1]
```

**Important**: Drop one category to avoid multicollinearity (dummy variable trap)

---

### **Q13: What is feature engineering and why is it important?**

**A:** 
Feature engineering creates new features to improve model performance:

Examples:
- **Log transformation**: `log(price)` for skewed distributions
- **Polynomial features**: `age²`, `mileage × age`
- **Ratios**: `horsepower / weight`
- **Binning**: Converting continuous to categorical

Good features often improve performance more than complex algorithms!

---

### **Q14: What does it mean if a distribution has a "long tail"?**

**A:** 
A long tail means the distribution is **skewed**:

- **Right-skewed (positive skew)**: Tail extends to the right (skewness > 1)
- **Left-skewed (negative skew)**: Tail extends to the left (skewness < -1)
- **Symmetric**: No significant tail (-0.5 < skewness < 0.5)

Check using: `df['column'].skew()`

---

### **Q15: What's the purpose of a baseline model?**

**A:** 
A baseline provides a reference point:

Simple baselines:
- Predict mean of target variable for all samples
- Predict median
- Use a single feature

**Purpose**:
- Sets minimum performance expectation
- If your complex model doesn't beat the baseline, something is wrong
- Helps justify model complexity

---

### **Q16: How do we implement linear regression in Python without sklearn?**

**A:** 
```python
import numpy as np

def train_linear_regression(X, y):
    # Add bias term
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    # Normal equation
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]  # bias, weights

def predict(X, w0, w):
    return w0 + X.dot(w)
```

For regularized version:
```python
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]
```

---

### **Q17: Common pitfalls to avoid?**

**A:** 
1. **Data Leakage**: Never use validation/test data to compute statistics (mean, std)
2. **Not Shuffling**: Always shuffle before splitting
3. **Overfitting**: Use regularization and validation
4. **Wrong Evaluation**: Don't evaluate on training data
5. **Forgetting Bias Term**: Always include w₀ (intercept)
6. **Singular Matrix**: Use regularization if XᵀX is not invertible
7. **Not Scaling**: Consider normalizing features for better numerical stability

---

## Code Examples

### Complete Workflow Example

```python
import pandas as pd
import numpy as np

# 1. Load and prepare data
df = pd.read_csv('car_data.csv')
df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 
         'model_year', 'fuel_efficiency_mpg']]

# 2. Shuffle and split (60/20/20)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df_shuffled)
n_train = int(0.6 * n)
n_val = int(0.2 * n)

df_train = df_shuffled[:n_train]
df_val = df_shuffled[n_train:n_train + n_val]
df_test = df_shuffled[n_train + n_val:]

# 3. Handle missing values (using mean from training set)
mean_hp = df_train['horsepower'].mean()
df_train['horsepower'].fillna(mean_hp, inplace=True)
df_val['horsepower'].fillna(mean_hp, inplace=True)
df_test['horsepower'].fillna(mean_hp, inplace=True)

# 4. Prepare features and target
features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'

X_train = df_train[features].values
y_train = df_train[target].values
X_val = df_val[features].values
y_val = df_val[target].values

# 5. Train model
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

# 6. Make predictions
def predict(X, w0, w):
    return w0 + X.dot(w)

y_pred = predict(X_val, w0, w)

# 7. Evaluate
def rmse(y_true, y_pred):
    se = (y_true - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

error = rmse(y_val, y_pred)
print(f"RMSE: {round(error, 2)}")
```

---

## Additional Resources

### Official Course Materials
- [Course GitHub Repository](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/02-regression)
- [YouTube Video Lectures](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

### Community Notes
- [Notes from Alvaro Navas](https://github.com/ziritrion/ml-zoomcamp/blob/main/notes/02_linear_regression.md)
- [Notes from Kemal Dahha](https://github.com/kemaldahha/machine-learning-course/blob/main/week_2_notes.ipynb)
- [Notes from Oscar Garcia](https://github.com/ozkary/machine-learning-engineering/tree/main/02-regression)

### Further Reading
- **Mathematics**: Linear Algebra review (matrix operations, inverse)
- **Statistics**: Understanding variance, bias-variance tradeoff
- **Advanced Topics**: Gradient Descent, Lasso Regression, Elastic Net

---

## Quick Reference Formulas

| Concept | Formula |
|---------|---------|
| Linear Regression | `y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ` |
| Normal Equation | `w = (XᵀX)⁻¹Xᵀy` |
| Ridge Regression | `w = (XᵀX + rI)⁻¹Xᵀy` |
| RMSE | `√(mean((y_true - y_pred)²))` |
| MSE | `mean((y_true - y_pred)²)` |
| MAE | `mean(\|y_true - y_pred\|)` |
| R² Score | `1 - (SS_res / SS_tot)` |

---

## Summary Checklist

- [ ] Understand train/val/test split and why it's important
- [ ] Know how to handle missing values properly
- [ ] Implement linear regression from scratch using NumPy
- [ ] Understand the Normal Equation and when to use it
- [ ] Know what regularization is and how to apply it
- [ ] Be able to evaluate models using RMSE
- [ ] Understand how to tune hyperparameters using validation set
- [ ] Know how to handle categorical variables (one-hot encoding)
- [ ] Practice feature engineering techniques
- [ ] Combine train+val for final model before testing

---

**Last Updated**: October 2025  
**Course**: ML Zoomcamp - Module 2: Machine Learning for Regression  
**Instructor**: Alexey Grigorev

---

*Happy Learning! 🚀*
