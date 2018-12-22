import ID3
import matplotlib.pyplot as plt

"""
Caesarian data training and testing
"""
data = []
train_data_list = []
test_data_list = []
my_parent_example = []
feature_list = []
my_attributes = {}

"""
Read file
"""
with open('caesarian.txt') as file:
    lines = (line.rstrip() for line in file)
    lines = (line for line in lines if line)
    data_begin = 0
    for line in lines:
        line = line.split("'")
        if line[0].strip(' ') == '@attribute':
            feature_list.append(line[1])
            value_list = line[2].strip("{ }").split(',')
            ID3.feature_value.append(value_list)
        if line[0] == '@data':
            data_begin = 1
            continue
        if data_begin == 1:
            record = line[0].split(',')
            data.append(record)
"""
Divide the data set into training set (60 records) and testing set (20 records)
"""
train_data_list = data[:60]
test_data_list = data[60:]

index = 0
for feature in feature_list:
    my_attributes[index] = feature
    index += 1
my_attributes = ID3.remove_key(my_attributes, index-1)

"""
Different size of training set
"""
xs = []
"""
Corresponding accuracy
"""
ys = []

training_data_size = [5, 10, 15, 20, 25, 30, 35, 45, 50]

for num in training_data_size:
    xs.append(num)
    train_data = train_data_list[:num]
    """
    Learning decision tree  
    """
    decision_tree = ID3.decision_tree_learning(train_data, my_attributes, my_parent_example)
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
plt.title('Caesarian data set : Training set size - Accuracy')
plt.show()