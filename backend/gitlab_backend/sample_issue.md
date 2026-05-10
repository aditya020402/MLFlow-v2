## Task Type
classification

## Target Variable
`species`

## Validation
yes

## Business Context
Predict the species of a flower based on petal and sepal measurements.
This model will be used in the field-sampling app to auto-classify observations.

## ML Requirements
- prefer recall over precision
- try XGBoost and RandomForest
- handle class imbalance if present

## Data Notes
- sepal_length and petal_length may be highly correlated
- no missing values expected

## Evaluation Criteria
primary metric: f1
threshold: 0.80

[iris.csv](/uploads/abc123/iris.csv)
