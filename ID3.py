import math
import copy


class Node(object):
    """
    Decision Tree node structure
    """
    def __init__(self):
        self.last_feature_value = ""
        self.node_feature = ""
        self.child_node_list = []


def plurality_value(example_list):
    """
    Return the most frequent classification for examples in example_list
    """
    feature_count = {}
    """
    Count the occurrence of different classifications
    """
    for example in example_list:
        classification = example[-1]
        if classification in feature_count:
            feature_count[classification] += 1
        else:
            feature_count[classification] = 1
    """
    Find the most frequent classification
    """
    max_value = 0
    res = ""
    for feature in feature_count:
        if feature_count[feature] > max_value:
            max_value = feature_count[feature]
            res = feature
    return res


def calculate_entropy(example_list):
    """
    Calculate the entropy of a given data set
    """
    feature_count = {}
    """
    Count the occurrence of different classifications
    """
    for example in example_list:
        classification = example[-1]
        if classification in feature_count:
            feature_count[classification] += 1
        else:
            feature_count[classification] = 1
    entropy = 0.0
    total_num = 0
    """
    Calculate the entropy
    """
    for feature_val in feature_count:
        total_num += feature_count[feature_val]
    for feature_val in feature_count:
        prob = feature_count[feature_val] * 1.0 / total_num
        entropy -= prob * (math.log(prob) / math.log(2))
    return entropy


def importance(attribute_index, examples):
    """
    Calculate the information gain after splitting an attribute
    """
    total_entropy = calculate_entropy(examples)
    branches = {}
    for example in examples:
        branch_value = example[attribute_index]
        if branch_value in branches:
            branches[branch_value] += 1
        else:
            branches[branch_value] = 1
    total_num = len(examples)
    for branch in branches:
        example_list = []
        for example in examples:
            if example[attribute_index] == branch:
                example_list.append(example)
        total_entropy -= (branches[branch] * 1.0 / total_num) * calculate_entropy(example_list)
    return total_entropy


def check_all(example_list):
    """
    Check if all examples have the same classification
    """
    classification = example_list[0][-1]
    for example in example_list:
        if example[-1] != classification:
            return False
    return True


def remove_key(d, key):
    """
    Copy a dictionary and remove the element in the copied dictionary according to the given key
    """
    r = dict(d)
    del r[key]
    return r


feature_value = []


def decision_tree_learning(examples, attributes, parent_examples):
    """
    Learning the decision tree from given data
    """
    node = Node()
    if len(examples) == 0:
        node.node_feature = plurality_value(parent_examples)
        return node
    elif check_all(examples):
        node.node_feature = examples[0][-1]
        return node
    elif len(attributes) == 0:
        node.node_feature = plurality_value(examples)
        return node
    else:
        max_ig = 0.0
        for index in attributes:
            ig = importance(index, examples)
            if ig >= max_ig:
                max_ig = ig
                selected_index = index
        root = Node()
        root.node_feature = attributes[selected_index]
        for value in feature_value[selected_index]:
            exs = []
            for example in examples:
                if example[selected_index] == value:
                    exs.append(example)
            sub_tree = decision_tree_learning(exs, remove_key(attributes, selected_index), examples)
            sub_tree.last_feature_value = value
            root.child_node_list.append(sub_tree)
        return root


def decision_tree_classifier(root, feature_list, test_data_list):
    """
    Use a trained decision tree to do classification
    """
    if len(feature_list) != len(test_data_list):
        print("Test data doesn't match")
        return "Error"
    decision_tree = copy.copy(root)
    while decision_tree is not None:
        if len(decision_tree.child_node_list) == 0:
            return decision_tree.node_feature
        for i in range(len(feature_list) - 1):
            if feature_list[i] == decision_tree.node_feature:
                test_feature_value = test_data_list[i]
                next_node = Node()
                for node in decision_tree.child_node_list:
                    if node.last_feature_value == test_feature_value:
                        next_node = node
                decision_tree = next_node
