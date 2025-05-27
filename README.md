# Iris Flower Classification

This project uses machine learning to classify iris flowers into three species: Setosa, Versicolor, and Virginica, based on their measurements.

## Dataset

- Source: [UCI Machine Learning Repository - Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- Features: Sepal length, Sepal width, Petal length, Petal width
- Target: Species

## How to Run

1. Make sure you have Python and the required libraries installed:
   - pandas
   - matplotlib
   - seaborn
   - scikit-learn
   - joblib

2. Place `Iris.csv` in the project directory.

3. Run the script:
   ```
   python iris_classification.py
   ```

## Example Output

```
Training set shape: (120, 4)
Test set shape: (30, 4)
Accuracy: 0.97

Classification Report:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      0.90      0.95        10
 Iris-virginica       0.91      1.00      0.95        10

Confusion Matrix:
 [[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
Predicted species for sample [[5.1, 3.5, 1.4, 0.2]]: Iris-setosa
```

## Project Steps

- Data loading and exploration
- Data visualization
- Model training and evaluation
- Model saving and loading
- Making predictions on new data

---

