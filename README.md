# Decision Tree Regression Model

## Overview

This project demonstrates how to build and visualize a Decision Tree Regression model using Python's `scikit-learn` library. The model is designed to predict a target variable (`y`) based on a set of features (`x`).

## Dataset

- **Feature Data (`x`)**: 
  - A numpy array with 50 samples and 6 features. 
  - Example structure: 
    ```python
    array([[0.0, 0.0, 1.0, 165349.2, 136897.8, 471784.1],
           [1.0, 0.0, 0.0, 162597.7, 151377.59, 443898.53],
           ...
           [1.0, 0.0, 0.0, 0.0, 116983.8, 45173.06]], dtype=object)
    ```
  
- **Label Data (`y`)**:
  - A numpy array with 50 samples, representing the target values.
  - Example structure:
    ```python
    array([[192261.83],
           [191792.06],
           ...
           [14681.4 ]])
    ```

## Model

- The model used is a **Decision Tree Regressor** from `scikit-learn`.
- The model is instantiated and trained using the following code:
  ```python
  from sklearn.tree import DecisionTreeRegressor

  decision_tree = DecisionTreeRegressor(random_state=0)
  decision_tree.fit(x, y)
  ```

## Visualization

- The model is visualized using Matplotlib. The feature data (`x`) is plotted against the actual labels (`y`) and the predictions made by the Decision Tree model.
- Example code for visualization:
  ```python
  import matplotlib.pyplot as plt

  plt.scatter(x[:, 0], y, color='red')  # Scatter plot for actual data
  plt.plot(x[:, 0], decision_tree.predict(x), color='blue')  # Line plot for model prediction
  plt.title('Decision Tree Regressor')
  plt.xlabel('Feature')
  plt.ylabel('Target')
  plt.show()
  ```

## Issues

- Ensure that the feature data (`x`) and label data (`y`) have the same number of samples to avoid errors during model training and visualization.
- The visualization uses only the first feature of `x` for simplicity. If `x` contains multiple features, the visualization may need to be adjusted accordingly.

## Dependencies

- `numpy`
- `scikit-learn`
- `matplotlib`

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the necessary dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script to build the model and visualize the results:
   ```bash
   python decision_tree_regression.py
   ```

## Notes

- The current visualization assumes a single feature for simplicity. If you have multiple features, consider reducing the dimensionality for better visualization or plotting each feature individually.
