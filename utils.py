import numpy as np
import pandas as pd


def calculate_accuracy(predictions, labels):

    # Caculate the prediction accuracy
    # predictions: an array of prediction
    # labels: an array of labels
    # return: a real-valued scalar of prediction accuracy

    assert len(predictions) == len(labels), "Length of predictions and labels do not match."
    return np.sum(np.array(predictions) == np.array(labels))/len(predictions)

def implicit_feature_type_inferrence(data, num_unique_values = 3):

    # Implicitly infer the feature type of the data in each column
    # data: pandas dataframe
    # return: a list of feature types

    feature_types = list()
    for feature in data.columns:
        if len(data[feature].unique()) <= num_unique_values:
            feature_types.append('categorical')
        else:
            feature_types.append('numerical')
    return feature_types

def train_test_split(data, train_fraction = 0.8, reindex = False, random_seed = 0):

    # Split the data into training data and test data
    # data: pandas dataframe
    # return: training data and test data, both are pandas dataframes

    np.random.seed(random_seed)

    dataset_size = len(data)
    train_size = int(train_fraction * dataset_size)
    train_idx = np.random.choice(dataset_size, train_size, replace = False)
    test_idx = np.delete(list(range(dataset_size)), train_idx)

    train_set = data.iloc[train_idx,:]
    test_set = data.iloc[test_idx,:]

    if reindex:
        train_set.index = range(len(train_set))
        test_set.index = range(len(test_set))

    return train_set, test_set

def fill_na(data):

    # Replace each missing value with the mode
    # data: pandas dataframe
    # return: pandas dataframe

    # The preferred pandas function for finding missing values is isnull()
    for feature in data.columns:
        if data[feature].isnull().sum() > 0:
            mode = data[feature].mode()[0]
            if mode is np.nan:
                print("Warning: mode is nan.")
            data[feature].fillna(mode, inplace = True)
    return data


def cross_validation_split(data, fold = 3, random_seed = 0):

    # Cross validation split of dataset
    # fold: number of fold in cross validation
    # return: a list of index in data

    np.random.seed(random_seed)

    dataset_size = len(data)
    dataset_idx = np.arange(dataset_size)
    np.random.shuffle(dataset_idx)

    split_size = dataset_size // fold

    split_idx = list()
    i = 0
    while (i < fold - 1):
        split_idx.append(list(dataset_idx[i*split_size:(i+1)*split_size]))
        i += 1
    split_idx.append(list(dataset_idx[i*split_size:]))

    return split_idx


def cross_validation(data, features, target, feature_types, model, fold = 3, random_seed = 0):

    # Cross validation of model
    # data: pandas dataframe
    # features: feature columns in data
    # target: target column in data
    # feature_types: feature types corresponding to features
    # model: model used for cross validation test
    # fold: number of fold in cross validation
    # return: training error, test error
    
    split_idx = cross_validation_split(data = data, fold = fold, random_seed = random_seed)

    training_errors = list()
    test_errors = list()

    for i in range(fold):
        test_idx = split_idx[i]
        test_set = data.iloc[test_idx, :]
        train_idx = []
        for j in range(fold):
            if j != i:
                train_idx += split_idx[j]
        train_set = data.iloc[train_idx, :]

        model.fit(data = train_set, features = features, target = target, feature_types = feature_types)

        training_predictioins = model.predict(test = train_set)
        training_accuracy = calculate_accuracy(predictions = training_predictioins, labels = train_set[target].values)
        #print("Training Error: %f" %(1-training_accuracy))
        test_predictions = model.predict(test = test_set)
        test_accuracy = calculate_accuracy(predictions = test_predictions, labels = test_set[target].values)
        #print("Test Error: %f" %(1-test_accuracy))
        training_errors.append(1-training_accuracy)
        test_errors.append(1-test_accuracy)

    return np.mean(training_errors), np.mean(test_errors)
