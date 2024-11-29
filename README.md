# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Abdul Rasak N 
RegisterNumber: 24002896
*/
# Importing required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('datasets/Employee.csv')

# Data overview
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

# Encode categorical 'salary' column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
print(data.head())

# Define features (X) and target (y)
x = data[[
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company", "Work_accident",
    "promotion_last_5years", "salary"
]]
y = data["left"]

# Display feature data
print(x.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train a Decision Tree model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_pred = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Test prediction for a new sample
sample_prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print(f"Prediction for sample: {sample_prediction}")

# Visualize the Decision Tree
plt.figure(figsize=(8, 6))
plot_tree(dt, feature_names=x.columns, class_names=['Not Left', 'Left'], filled=True)
plt.show()

```

## Output:
```
  satisfaction_level  last_evaluation  ...  Departments   salary
0                0.38             0.53  ...         sales     low
1                0.80             0.86  ...         sales  medium
2                0.11             0.88  ...         sales  medium
3                0.72             0.87  ...         sales     low
4                0.37             0.52  ...         sales     low

[5 rows x 10 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   satisfaction_level     14999 non-null  float64
 1   last_evaluation        14999 non-null  float64
 2   number_project         14999 non-null  int64  
 3   average_montly_hours   14999 non-null  int64  
 4   time_spend_company     14999 non-null  int64  
 5   Work_accident          14999 non-null  int64  
 6   left                   14999 non-null  int64  
 7   promotion_last_5years  14999 non-null  int64  
 8   Departments            14999 non-null  object 
 9   salary                 14999 non-null  object 
dtypes: float64(2), int64(6), object(2)
memory usage: 1.1+ MB
None
satisfaction_level       0
last_evaluation          0
number_project           0
average_montly_hours     0
time_spend_company       0
Work_accident            0
left                     0
promotion_last_5years    0
Departments              0
salary                   0
dtype: int64
left
0    11428
1     3571
Name: count, dtype: int64
   satisfaction_level  last_evaluation  ...  Departments   salary
0                0.38             0.53  ...         sales       1
1                0.80             0.86  ...         sales       2
2                0.11             0.88  ...         sales       2
3                0.72             0.87  ...         sales       1
4                0.37             0.52  ...         sales       1

[5 rows x 10 columns]
   satisfaction_level  last_evaluation  ...  promotion_last_5years  salary
0                0.38             0.53  ...                      0       1
1                0.80             0.86  ...                      0       2
2                0.11             0.88  ...                      0       2
3                0.72             0.87  ...                      0       1
4                0.37             0.52  ...                      0       1

[5 rows x 8 columns]
Accuracy: 0.984

```
![image](https://github.com/user-attachments/assets/ad7bcbfd-c0ed-44d8-b97e-d092befa1480)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
