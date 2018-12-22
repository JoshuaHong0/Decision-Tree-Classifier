import ID3
import matplotlib.pyplot as plt

"""
Iris data training and testing
"""

"""
Read files
"""
my_examples = []
test_data_list = []
my_parent_example = []
my_attributes = {}
feature_list = ["sepal length", "sepal width", "petal length", "petal width", "class"]
ID3.feature_value = [list(["S", "MS", "L", "ML"]), list(["S", "MS", "L", "ML"]), list(["S", "MS", "L", "ML"]),
                     list(["S", "MS", "L", "ML"]), list(["Iris Setosa", "Iris Versicolour", "Iris Virginica"])]
"""
Get the total training set (105 records) and testing set (45 records)
"""
with open("iris.data.discrete.txt") as f:
    index = 0
    for line in f:
        if index == 50:
            index = 0
        if index < 35:
            my_examples.append(line.strip(" \n").split(","))
        else:
            test_data_list.append(line.strip(" \n").split(","))
        index += 1

index = 0
for feature in feature_list:
    my_attributes[index] = feature
    index += 1
my_attributes = ID3.remove_key(my_attributes, index-1)

training_data_size = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 20]
"""
Different size of training set
"""
xs = []
"""
Corresponding accuracy
"""
ys = []

for num in training_data_size:
    xs.append(3*num)
    training_set = []
    """
    Get training set
    """
    index = 0
    for item in my_examples:
        if index == 35:
            index = 0
        if index < num:
            training_set.append(item)
        index += 1

    """
    Learning the decision tree
    """
    decision_tree = ID3.decision_tree_learning(training_set, my_attributes, my_parent_example)

    """
    Accuracy evaluation & Record
    """
    total_test = len(test_data_list)
    correct_cnt = 0
    for test_data in test_data_list:
        ans = test_data[-1]
        classification = ID3.decision_tree_classifier(decision_tree, feature_list, test_data)
        if ans == classification:
            correct_cnt += 1
    ys.append(float(correct_cnt)/total_test)

"""
Draw the learning curve
"""
plt.plot(xs, ys)
plt.xlabel('Training set size')
plt.ylabel('Proportion correct on test set')
plt.title('Iris data set : Training set size - Accuracy')
plt.show()
