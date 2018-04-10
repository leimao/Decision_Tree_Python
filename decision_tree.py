'''
Decision Tree
Lei Mao
University of Chicago
4/9/2017
dukeleimao@gmail.com

Introduction:
This implementation of decision tree classification algorithm builds decision tree for training data extremely fast.
It makes no assumptions for the training data as long as the data has features and target.
It splits categorical features to branches using each value of the feature.
i.e., the categorical feature could have more than two values, and the split could have more than two branches.
The categorical feature will only be used once in the decision tree.
It splits numerical feature to two branches.
The numerical feature could be used multiple times in the decision tree.

Reference:
"Introduction to Data Mining" by Tan, Steinbach, and Kumar, page 162
"Artificial Intelligence: A Modern Approach", Rusell-Norvig, page 702
'''


import numpy as np
import pandas as pd


class TreeNode(object):

    def __init__(self, node_idx, is_leaf = False, label = None, split_feature = None, split_feature_type = None, split_feature_value = None):

        # Initialization for TreeNode
        # node_idx: unique index of node
        # is_leaf: is leaf or not, bool
        # label: label for leaf node
        # split_feature: feature to split for internal node
        # split_feature_type: feature type for split feature ('numerical' or 'categorical')
        # split_feature_value: split value for numerical split_feature

        self.node_idx = node_idx
        self.is_leaf = is_leaf
        self.label = label
        self.split_feature = split_feature
        self.split_feature_type = split_feature_type
        self.split_feature_value = split_feature_value

        # Children tree nodes
        self.children = dict()

    def __repr__(self):

        # Print TreeNode

        if not self.is_leaf:
            if self.split_feature_type == 'numerical':
                message = self.node_idx + " Internal node split on numerical feature " + str(self.split_feature) + " at " + str(self.split_feature_value) + "."
            else:
                message = self.node_idx + " Internal node split on categorical feature " + str(self.split_feature) + "."
        else:
            message = self.node_idx + " Leaf node with label " + str(self.label) + "."

        return message
    
class DecisionTree(object):

    def __init__(self, max_depth = None):

        # Initialize DecisionTree
        # max_depth: max depth of the decision tree

        if (max_depth is not None and (max_depth != int(max_depth) or max_depth < 0)):
            raise Exception("Invalid max depth value.")
        self.max_depth = max_depth
        
    def fit(self, data, features, target, feature_types):

        # Fit a tree on data
        # data: pandas dataframe
        # features: feature columns in data
        # target: target column in data
        # feature_types: feature types corresponding to features

        if (not isinstance(data, pd.DataFrame) or target not in data.columns):
            raise Exception("Invalid input")

        self.features = features
        self.target = target
        self.feature_types = feature_types

        self.root = self.recursive_build_tree(data = data, depth = 0, features = features, target = target, feature_types = feature_types, node_idx = '0', parent_pluarlity_target_value = None)
            
    def recursive_build_tree(self, data, depth, features, target, feature_types, node_idx, parent_pluarlity_target_value):

        # data: pandas dataframe
        # depth: current depth of tree
        # features: feature columns in data
        # target: target column in data
        # feature_types: feature types corresponding to features
        # node_idx: unique node index for current node
        # parent_pluarlity_target_value: pluarlity target value in parent node
        # return: a tree node

        # No data for split
        if data.empty:
            leaf_value = parent_pluarlity_target_value
            leaf = TreeNode(node_idx = node_idx, is_leaf = True, label = leaf_value)
            return leaf

        # Pluarlity target value in current node
        pluarlity_target_value = self.plurality_target_value(data, target)

        # Maximum depth reached
        if (self.max_depth is not None) and (depth >= self.max_depth):
            leaf_value = pluarlity_target_value
            leaf = TreeNode(node_idx = node_idx, is_leaf = True, label = leaf_value)
            return leaf

        # No feature for split
        if len(features) == 0:
            leaf_value = pluarlity_target_value
            leaf = TreeNode(node_idx = node_idx, is_leaf = True, label = leaf_value)
            return leaf

        # All data have the same target_values
        if len(data[target].unique()) == 1:
            leaf_value = pluarlity_target_value
            leaf = TreeNode(node_idx = node_idx, is_leaf = True, label = leaf_value)
            return leaf

        # Find optimal feature to split
        splitting_feature_idx, splitting_feature, splitting_feature_type, splitting_feature_value = self.optimal_feature_splitting(data = data, features = features, target = target, feature_types = feature_types)

        # Create a internal node
        node = TreeNode(node_idx = node_idx, is_leaf = False, split_feature = splitting_feature, split_feature_type = splitting_feature_type, split_feature_value = splitting_feature_value)

        # Split
        # Categorical feature
        if splitting_feature_type == 'categorical':
            feature_values = data[splitting_feature].unique()
            feature_remaining = features[:splitting_feature_idx] + features[splitting_feature_idx+1:]
            feature_types_remaining = feature_types[:splitting_feature_idx] + feature_types[splitting_feature_idx+1:]

            for i, feature_value in enumerate(feature_values):
                node_idx_child = node_idx + '.' + str(i)
                data_splitted = data[data[splitting_feature] == feature_value]
                child_node = self.recursive_build_tree(data = data_splitted, depth = depth + 1, features = feature_remaining, target = target, feature_types = feature_types_remaining, node_idx = node_idx_child, parent_pluarlity_target_value = pluarlity_target_value)
                node.children[feature_value] = child_node

        # Numerical feature
        else:
            data_splitted_left = data[data[splitting_feature] <= splitting_feature_value]
            data_splitted_right = data[data[splitting_feature] > splitting_feature_value]

            node_idx_left = node_idx + '.' + '0'
            node_idx_right = node_idx + '.' + '1'

            child_node_left = self.recursive_build_tree(data = data_splitted_left, depth = depth + 1, features = features, target = target, feature_types = feature_types, node_idx = node_idx_left, parent_pluarlity_target_value = pluarlity_target_value)
            child_node_right = self.recursive_build_tree(data = data_splitted_right, depth = depth + 1, features = features, target = target, feature_types = feature_types, node_idx = node_idx_right, parent_pluarlity_target_value = pluarlity_target_value)
            
            node.children['left'] = child_node_left
            node.children['right'] = child_node_right

        return node

    def optimal_feature_splitting(self, data, features, target, feature_types):

        # Find optimal feature to split
        # data: pandas dataframe
        # depth: current depth of tree
        # features: feature columns in data
        # target: target column in data
        # feature_types: feature types corresponding to features
        # return: index of feature in features list, feature, feature type, splitting value

        assert len(features) == len(feature_types), "Length of features and feature_types does not match."
        
        impurity_after_split_min = float('inf')
        feature_index = None
        feature_splitting_value = None

        for idx in range(len(features)):
            feature = features[idx]
            feature_type = feature_types[idx]

            # If the feature is categorical
            if feature_type == 'categorical':
                impurity_after_split = self.categorical_feature_splitting(data = data, feature = feature, target = target)
                if impurity_after_split < impurity_after_split_min:
                    impurity_after_split_min = impurity_after_split
                    feature_index = idx
                    feature_splitting_value = None

            # Else the feature is numerical
            else:
                optimal_splitting_value, impurity_after_split = self.numerical_feature_splitting(data = data, feature = feature, target = target)
                if impurity_after_split < impurity_after_split_min:
                    impurity_after_split_min = impurity_after_split
                    feature_index = idx
                    feature_splitting_value = optimal_splitting_value

        return feature_index, features[feature_index], feature_types[feature_index], feature_splitting_value

    def categorical_feature_splitting(self, data, feature, target):

        # Calculate the impurity after splitting data on categorical feature
        # data: pandas dataframe
        # feature: the feature column name in data
        # target: the target column name in data
        # return: an real-valued scalar of the impurity after the split

        feature_vector = data[feature].values
        target_vector = data[target].values

        feature_values = np.unique(feature_vector)
        feature_values, feature_value_counts = np.unique(feature_vector, return_counts = True)

        feature_value_fractions = feature_value_counts/np.sum(feature_value_counts)

        # Total impurity of the data after splitting on feature
        impurity = 0
        for feature_value, feature_value_fraction in zip(feature_values, feature_value_fractions):

            target_vector_splitted = target_vector[feature_vector == feature_value]
            target_values, target_value_counts = np.unique(target_vector_splitted, return_counts = True)
            impurity_splitted = self.information_entropy_counts(counts = target_value_counts)
            impurity += feature_value_fraction * impurity_splitted

        return impurity

    '''
    # Slow implementation
    # Keep for archieve
    def categorical_feature_splitting(self, data, feature, target):
        # Calculate the impurity after splitting data on categorical feature
        # data: pandas dataframe
        # feature: the feature column name in data
        # target: the target column name in data
        # return: an real-valued scalar of the impurity after the split

        value_counts = data[feature].value_counts()
        # The values in the feature
        feature_values = value_counts.keys().tolist()
        # The count of each value in the feature
        feature_counts = value_counts.tolist()
        # The fraction of each value in the feature
        feature_fractions = (np.array(feature_counts)/np.sum(feature_counts)).tolist()

        # Total impurity of the data after splitting on feature
        impurity = 0
        for feature_value, feature_fraction in zip(feature_values, feature_fractions):
            data_splitted = data[data[feature] == feature_value]
            impurity_splitted = self.data_impurity(data = data_splitted, target = target)
            impurity += feature_fraction * impurity_splitted

        return impurity

    def data_impurity(self, data, target):
        # Calculate the impurity of data using target values
        # data: pandas dataframe
        # target: the target column name in data
        # return: a real-valued scalar of the impurity after the split

        # Slow
        # target_counts = data[target].value_counts().tolist()
        target_vector = data[target].values
        target_values, target_counts = np.unique(target_vector, return_counts = True)

        impurity = self.information_entropy_counts(counts = target_counts)
        return impurity
    '''

    def information_entropy_counts(self, counts):

        # Calculate the information entropy using counts
        # counts: a list of count representing the number of each class
        # return: a real-valued scalar of the entropy

        if np.sum(counts) == 0:
            return 0

        # counts list may contain zeros
        # Remove zeros
        counts_zero_removed = np.array([value for value in counts if value != 0])

        probabilities = np.array(counts_zero_removed)/np.sum(counts_zero_removed)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_entropy_probabilities(self, probabilities):

        # Calculate the information entropy using probabilities
        # probabilities: a list of probability representing the fraction of each class
        # return: a real-valued scalar of the entropy

        assert np.sum(probabilities) == 1, "Probabilities did not sum to 1."

        # probabilities list may contain zeros
        # Remove zeros
        probabilities_zero_removed = np.array([value for value in probabilities if value != 0])

        entropy = -np.sum(probabilities_zero_removed * np.log2(probabilities_zero_removed))
        return entropy

    def information_entropy_dictionary(self, dictionary):

        # Calculate the information entropy using dictionary
        # dictionary: a dictionary of count representing the number of each class
        # return: a real-valued scalar of the entropy

        counts = list(dictionary.values())
        entropy = self.information_entropy_counts(counts = counts)
        return entropy

    def numerical_feature_splitting(self, data, feature, target):

        # Find the optimal value to binarily split the numerical feature resulting maximum impurity reduction.
        # data: pandas dataframe
        # feature: the feature column name in data
        # target: the target column name in data
        # return: a real-valued scalar of optimal splitting value, and a real-valued scalar of the impurity after the split

        feature_vector = data[feature].values
        target_vector = data[target].values

        sorted_idx = np.argsort(feature_vector, kind = 'mergesort')
        feature_vector = feature_vector[sorted_idx]
        target_vector = target_vector[sorted_idx]

        # Numpy sort always sort in ascending order
        # Use mergesort worst case O(nlogn)
        feature_values = np.unique(feature_vector)
        feature_values = np.sort(feature_values, kind = 'mergesort')

        target_values, target_value_counts = np.unique(target_vector, return_counts = True)

        impurity_before_split = self.information_entropy_counts(counts = target_value_counts)

        if len(feature_values) == 1:
            # Only one feature values
            # Cannot split for this numerical feature
            return feature_values[0], impurity_before_split

        target_value_count_dict_left = dict()
        target_value_count_dict_right = dict()

        for value, count in zip(target_values, target_value_counts):
            target_value_count_dict_left[value] = 0
            target_value_count_dict_right[value] = count

        num_samples = len(feature_vector)
        branch_size_left = 0
        branch_size_right = num_samples

        # Split positions are the means of adjacent feature values
        split_positions = list()
        for i in range(len(feature_values)-1):
            split_positions.append((feature_values[i]+feature_values[i+1])/2)

        impurity_after_split_min = impurity_before_split
        optimal_split_value = feature_values[0]

        idx = 0

        for value in split_positions:
            while (feature_vector[idx] < value):
                target_value_count_dict_left[target_vector[idx]] += 1
                branch_size_left += 1
                target_value_count_dict_right[target_vector[idx]] -= 1
                branch_size_right -= 1
                idx += 1

            impurity_after_split = branch_size_left / num_samples * self.information_entropy_dictionary(dictionary = target_value_count_dict_left) + branch_size_right / num_samples * self.information_entropy_dictionary(dictionary = target_value_count_dict_right)

            if impurity_after_split < impurity_after_split_min:
                impurity_after_split_min = impurity_after_split
                optimal_split_value = value

        return optimal_split_value, impurity_after_split_min


    def plurality_target_value(self, data, target):

        # Return the plurality of the target values

        return data[target].mode()[0]

    def classify(self, treenode, datapoint):

        # Classify one data point using decision tree

        if treenode.is_leaf == True:
            return treenode.label
        else:
            # Categorical feature
            if treenode.split_feature_type == 'categorical':
                return self.classify(treenode = treenode.children[datapoint[treenode.split_feature]], datapoint = datapoint)
            # Numerical feature
            else:
                if datapoint[treenode.split_feature] <= treenode.split_feature_value:
                    return self.classify(treenode = treenode.children['left'], datapoint = datapoint)
                else:
                    return self.classify(treenode = treenode.children['right'], datapoint = datapoint)

    def predict(self, test):

        # Predict new samples
        # test: pandas dataframe
        # return: predicted values

        predictions = list()
        for i in range(len(test)):
            prediction = self.classify(treenode = self.root, datapoint = test.iloc[i,:])
            predictions.append(prediction)
        return np.array(predictions)

    def print(self):

        # Print the decision tree

        self.recursive_print(self.root)
    
    def recursive_print(self, node):

        # Recursively print tree from node
        # node: TreeNode

        print(node)
        for child in node.children:
            self.recursive_print(node.children[child])