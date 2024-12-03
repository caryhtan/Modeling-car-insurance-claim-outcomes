# Modeling Car Insurance Claim Outcomes
![Car Insurance](car.jpg)
Welcome to my project on predicting car insurance claim outcomes! Insurance has long been a data-driven industry, and machine learning is enhancing how companies assess risk and make predictions. In this analysis, I explored the **car_insurance.csv** dataset to uncover insights into customer attributes and identify the most predictive feature for logistic regression models. Using Python and Jupyter Notebook, I analyzed the dataset and trained logistic regression models to guide future efforts in claim prediction.

## View the Notebook
You can check out the full analysis here: [View Notebook](link-to-notebook)

## What I Did
I focused on answering the following key question:
- Which feature in the dataset is the most predictive for determining insurance claim outcomes using logistic regression?

### Data Source
I used the following dataset:
1. **data/car_insurance.csv**  
This is a CSV file containing data on client demographics, policy details, and claim outcomes.  
- `age`: Client's age category (e.g., 16-25, 26-39, etc.)  
- `credit_score`: Client's credit score (between 0 and 1)  
- `driving_experience`: Years of driving experience (e.g., 0-9, 10-19, etc.)  
- `annual_mileage`: Number of miles driven annually  
- `outcome`: Response variable indicating whether a claim was made (0 = No claim, 1 = Claim made)  

## What I Found
Here are some key insights I discovered:
- Best Predictive Feature: `driving_experience`  
- Accuracy: The logistic regression model using `driving_experience` achieved an accuracy of 77.71%.

## How I Did It
To answer this question, I:
1. Loaded the dataset using `pandas` and handled missing values by imputing the mean for numerical columns.
2. Trained logistic regression models for each feature using the `statsmodels` library.
3. Evaluated model performance by calculating accuracy for each feature.
4. Identified the feature with the highest accuracy as the most predictive.

### Example Code
Here’s an example of the code I used:
```python
from statsmodels.formula.api import logit
import pandas as pd

# Load the dataset
cars = pd.read_csv('data/car_insurance.csv')

# Fill missing values with the mean
cars["credit_score"].fillna(cars["credit_score"].mean(), inplace=True)
cars["annual_mileage"].fillna(cars["annual_mileage"].mean(), inplace=True)

# Train logistic regression models for each feature
features = cars.drop(columns=["id", "outcome"]).columns
models = [logit(f"outcome ~ {col}", data=cars).fit() for col in features]

# Calculate accuracies
accuracies = [
    (model.pred_table()[0, 0] + model.pred_table()[1, 1]) / len(cars)
    for model in models
]

# Identify the best feature
best_feature = features[accuracies.index(max(accuracies))]
print(f"Best Feature: {best_feature}, Accuracy: {max(accuracies):.4f}")
