<p align="center">
  <img src="https://github.com/marnixkoops/timefold/blob/master/img/timefold-logo.png">
  <img src="https://github.com/marnixkoops/timefold/blob/master/img/timefold-methods.png">
</p>

## INFO
Cross-validation methods for time-series data in Python, `sklearn` compatible.

## METHODS
| Method     | Parameters                                          | Description                                                              |
|------------|-----------------------------------------------------|--------------------------------------------------------------------------|
| nested     | `folds`                                             | Growing train folds                                                      |
| window     | `folds`                                             | Moving train and test folds                                              |
| step       | `min_train_size`, `min_test_size`, `step_size` | Step ahead folds. Size of train, test and step can be set                |
| shrinking  | `folds`                                             | Constant test fold, shrinking train folds                                |
| stratified | `folds`, `target`                                             | To be implemented. Preserves a ratio such as class distribution per fold |

## USAGE
```python
from timefold import timefold
import numpy as np

# Simulate some example data
X = np.random.randint(5, size=(10, 2))
y = np.random.randint(2, size=10)
list(zip(X, y))

[(array([1, 4]), 1),
 (array([0, 0]), 1),
 (array([3, 2]), 0),
 (array([2, 0]), 0),
 (array([2, 2]), 0),
 (array([0, 1]), 0),
 (array([4, 3]), 1),
 (array([1, 2]), 0),
 (array([1, 2]), 0),
 (array([2, 4]), 1)]

# Create timefold object for nested folds
tf = timefold(folds=3, method='nested')

# Generate and print train-test pair indices
for train_index, test_index in tf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]

TRAIN: [0 1 2] TEST: [3 4 5]
TRAIN: [0 1 2 3 4 5] TEST: [6 7 8]
TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]

# Create timefold object for windowed folds
tf = timefold(folds=3, method='window')

# Generate and print train-test pair indices
for train_index, test_index in tf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    
TRAIN: [0 1 2] TEST: [3 4 5]
TRAIN: [3 4 5] TEST: [6 7 8]
TRAIN: [6 7] TEST: [8 9]

# Create timefold object for one step ahead folds
tf = timefold(method='step', test_size=1)

# Generate and print train-test pair indices
for train_index, test_index in tf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]

TRAIN: [0] TEST: [1]
TRAIN: [0 1] TEST: [2]
TRAIN: [0 1 2] TEST: [3]
TRAIN: [0 1 2 3] TEST: [4]
TRAIN: [0 1 2 3 4] TEST: [5]
TRAIN: [0 1 2 3 4 5] TEST: [6]
TRAIN: [0 1 2 3 4 5 6] TEST: [7]
TRAIN: [0 1 2 3 4 5 6 7] TEST: [8]
TRAIN: [0 1 2 3 4 5 6 7 8] TEST: [9]

# Create timefold object for 3 steps ahead folds with a minimum train and test fold size
tf = timefold(method='step', min_train_size=5, min_test_size=3, step_size=3)

# Generate and print train-test pair indices
for train_index, test_index in tf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    
TRAIN: [0 1 2 3 4] TEST: [5 6 7]
TRAIN: [0 1 2 3 4 5 6 7] TEST: [ 8  9 10]

# Create timefold object for shrinkage folds
tf = timefold(folds=3, method='shrink')

# Generate and print train-test pair indices
for train_index, test_index in tf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    
TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
TRAIN: [3 4 5 6 7] TEST: [8 9]
TRAIN: [6 7] TEST: [8 9]
```
