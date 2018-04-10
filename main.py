from utils import calculate_accuracy, implicit_feature_type_inferrence, train_test_split, fill_na, cross_validation
from decision_tree import DecisionTree
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def plot_curve(max_depth, training_error, test_error):

    # Plot the a cross validation curve using Matplotlib

    fig = plt.figure(figsize=(12,8))
    plt.plot(max_depth, training_error, color = 'r', marker='o', clip_on=False, label = 'Training Error')
    plt.plot(max_depth, test_error, color = 'b', marker='o', clip_on=False, label = 'Test Error')
    plt.legend()
    plt.ylabel('Error Rate')
    plt.xlabel('Max Depth')
    plt.xlim(max_depth[0], max_depth[-1])
    #plt.ylim(0, 1)
    fig.savefig('validation.png', format = 'png', dpi = 600, bbox_inches = 'tight')
    plt.close()


def validation_curve():

    # Test decision tree using cross validation

    # Preprocess data
    data = pd.read_csv('./arrhythmia.data', header = None, na_values = '?')
    data = fill_na(data = data)

    features = data.columns.tolist()[:-1]
    target = data.columns.tolist()[-1]

    feature_types = implicit_feature_type_inferrence(data = data[features], num_unique_values = 3)

    train_set, test_set = train_test_split(data = data, train_fraction = 0.8, reindex = False, random_seed = 0)

    max_depth_cv = list()
    training_error_cv = list()
    test_error_cv = list()

    # Start cross-validation
    for i in range(2,21,2):
        tree_max_depth = i
        print("Tree Max Depth: %d" %tree_max_depth)
        max_depth_cv.append(tree_max_depth)
        tree = DecisionTree(tree_max_depth)

        training_error, test_error = cross_validation(data = data, features = features, target = target, feature_types = feature_types, model = tree, fold = 3, random_seed = 0)
        training_error_cv.append(training_error)
        test_error_cv.append(test_error)
        print("Training Error: %f" %training_error)
        print("Test Error: %f" %test_error)

    plot_curve(max_depth = max_depth_cv, training_error = training_error_cv, test_error = test_error_cv)


if __name__ == '__main__':

    start_time = time.time()
    validation_curve()
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Time Elapsed: %02d:%02d:%02d" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

