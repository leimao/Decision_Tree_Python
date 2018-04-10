
# Decision Tree

Lei Mao

University of Chicago

## Introduction

This implementation of decision tree classification algorithm from scratch builds decision tree for training data extremely fast.
It makes no assumptions for the training data as long as the data has features and target.
It splits categorical features to branches using each value of the feature, i.e., the categorical feature could have more than two values, and the split could have more than two branches.
The categorical feature will only be used once in the decision tree.
It splits numerical feature to two branches.
The numerical feature could be used multiple times in the decision tree.

## Algorithm

To find the best feature to split, we find the feature that has the maximum information gain (entropy reduction) after split in a greedy manner. 

To determine the value to split numerical features, we implemented a O(nlgn) algrithm in the "Introduction to Data Mining" book referenced below.


## Dependencies

Python 3.5

Numpy 1.14

Pandas 0.22

## Files

```
.
├── arrhythmia.data
├── decision_tree.py
├── demo.png
├── LICENSE.md
├── main.py
├── README.md
└── utils.py
```

## Usage

In Python, to build decision tree using the package in a Scikit-Learn style:

```python
from decision_tree import DecisionTree

model = DecisionTree(tree_max_depth)
model.fit(data = train_set, features = features, target = target, feature_types = feature_types)

training_predictioins = model.predict(test = train_set)
test_predictions = model.predict(test = test_set)
```

## Demo

To build decision tree on [Arrhythmia](http://archive.ics.uci.edu/ml/datasets/Arrhythmia) dataset, and do 3-fold cross validation:

```bash
$ python main.py
```

<p align="center">
    <img src = "./demo.png" width="80%">
</p>

It takes around 5 minutes to run on a single core of Intel Core i7-3770 CPU.

```
Tree Max Depth: 2
Training Error: 0.346233
Test Error: 0.393977
Tree Max Depth: 4
Training Error: 0.213503
Test Error: 0.336433
Tree Max Depth: 6
Training Error: 0.111788
Test Error: 0.347398
Tree Max Depth: 8
Training Error: 0.044283
Test Error: 0.343041
Tree Max Depth: 10
Training Error: 0.019941
Test Error: 0.358567
Tree Max Depth: 12
Training Error: 0.002215
Test Error: 0.365205
Tree Max Depth: 14
Training Error: 0.000000
Test Error: 0.365205
Tree Max Depth: 16
Training Error: 0.000000
Test Error: 0.365205
Tree Max Depth: 18
Training Error: 0.000000
Test Error: 0.365205
Tree Max Depth: 20
Training Error: 0.000000
Test Error: 0.365205
Time Elapsed: 00:05:26
```

The test error stop increasing as max depth increases because the tree built cannot reach the maximum depth designated due to that it ran out of the training data (The dataset only contains about 400 data points).


## Reference:

* "Introduction to Data Mining" by Tan, Steinbach, and Kumar, page 162
* "Artificial Intelligence: A Modern Approach", Rusell-Norvig, page 702
