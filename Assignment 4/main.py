## Name: Aryan Satpathy
## Roll: 20EC10014

########################################## GUIDE TO CODE ##########################################
# My code is structured into the following parts:
# 1. Imports
# 2. Classes
# 3. Helper Functions
# 4. Experiments
# 5. Main / Driver
 
## Imports:
#### Numpy: Store data, matrix multiplcations, data manipulation, etc
#### Pandas: To read the dataset and beauty printing(Experiment 3)
#### Matplotlib: For all plots 
#### Typing: For typesetting, very helpful to understand code

## Classes:
#### Metrics: Contain static methods to calculate Accuracy, Precision, Recall and F1
#### Node: Contains the following methods:
###### predict: Predicts class given Decision Tree and Data
###### get_size: Returns number of nodes in a Decision Tree 
###### get_root: Returns root of the Decision Tree given any node
###### print_tree: Prints Decision Tree in readable format
###### get_entropy: Returns entropy at a given node
###### info_gain: Returns info gain upon splitting
###### split_attribute: Returns attribute to split on, in order to get max info_gain
###### Generate: Creates Decision Tree based on Training Data
#### LoggedNode: Inherits from Node, with following overloads:
###### Generate: Logs accuracy wrt branching and implements early stopping

## Helper Functions:
#### load_dataset: Loads the dataset
#### split: My implementation of spliting data into test, train and validation
#### process_dataset: Prepares dataset and returns numpy arrays

## Experiments:
#### Experiment1: Trains Decision Tree for different threshold values and plots Accuracy v/s
####              threshold and Size v/s threshold
#### Experiment2: Plots Model Accuracy v/s branching and implements Early Stopping
#### Experiment3: Prints the Decision Trees obtained with and without Early Stopping 
####              in readable format

## Main / Driver:
#### Simply loads the dataset, processes it, and then calls Experiments one by one.

###################################################################################################

## Imports
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple, Container

# For the sake of reproducibility, I'll set the seed of random module
np.random.seed(42)

## Classes
class Metrics:
    '''
    Class containing functions to calculate the various metrics.
    All the functions take two arguments as inputs and return a float value.

    1. Accuracy: It measures (TP + TN) / (TP + FP + TN + FN)
    2. Precision: It measures TP / (TP + FP)
    3. Recall: It measures TP / (TP + FN)
    4. F1: Its a combination of precision(p) and recall(r) (2 * p + r) / (p + r)
    '''
    @staticmethod
    def accuracy(out: np.array, y: np.array) -> float:
        # print(out, y)
        # exit()
        if len(out.shape) > 1:
            return np.sum(np.all(out == y, axis = 1)) / out.shape[0]                                                    # Finding indices of matching predictions and labels
        
        return np.sum(np.all(out[:, None] == y[:, None], axis = 1)) / out.shape[0]
    
    @staticmethod
    def precision(out: np.array, y: np.array) -> float:
        pos_indices = np.where(out == 1)[0]                                                                             # Finding indices of predictions that are positive
        
        _out = out[pos_indices]
        _y = y[pos_indices]

        return np.where(np.abs(_out - _y) <= 1e-5)[0].shape[0] / pos_indices.shape[0]                                       # Finding indices of matching predictions and labels
    
    @staticmethod
    def recall(out: np.array, y: np.array) -> float:
        pos_indices = np.where(y == 1)[0]                                                                               # Finding indices of labels that are positive
        
        _out = out[pos_indices]
        _y = y[pos_indices]

        return np.where(np.abs(_out - _y) <= 1e-5)[0].shape[0] / pos_indices.shape[0]                                       # Finding indices of matching predictions and labels
    
    @staticmethod
    def f1(out: np.array, y: np.array) -> float:
        p = Metrics.precision(out, y)
        r = Metrics.recall(out, y)

        if p == r == 0: return 0

        return (2 * p * r) / (p + r)

class Node:
    def __init__(self):
        self.attr_index = None
        self.class_name = None

        self.value = None

        self.parent = None

        self.children = []

    def predict(self, X: np.array):
        if self.attr_index is None:
            return self.class_name
        
        value = X[self.attr_index]

        for child in self.children:
            if child.value == value:
                return child.predict(X)
            
    def get_size(self, size: int = 0):
        size = size + 1

        for child in self.children:
            size = child.get_size(size)
        
        return size

    def get_root(self):
        if self.parent is None: return self

        return self.parent.get_root()
    
    def print_tree(self, col_names: list = [], level = 0): #displaying the tree formed hierarchically
        
        spaces = ' ' * level * 3
        prefix = spaces + "|--" if self.parent else "" #using indentation and |_ as representation of depth & parent-child relation
        
        if self.parent != None:
            if self.attr_index != None: print(prefix + " if value: " + str(self.value) + " => discriminant feature: " + col_names[self.attr_index])
            else: print(prefix + " if value: " + str(self.value) + " => class: "+ str(self.class_name))
        else:
            if self.attr_index != None: print(prefix + " discriminant feature: " + col_names[self.attr_index])
            else: print(prefix + " class: "+ str(self.class_name))

        for child in self.children:
            child.print_tree(col_names, level + 1) #recursively use the method for displaying the subtrees
        
        print(prefix + ' end')

    @staticmethod
    def get_entropy(labels: np.array):
        unique, count = np.unique(labels, return_counts = True)
        number_of_samples = labels.shape[0]

        prob = count / number_of_samples
    
        entropy = np.sum(-prob * np.log(prob)) # / np.log(4)
        # if unique.shape[0] > 1: entropy = entropy / np.log(unique.shape[0])

        class_name = unique[np.argmax(count)]

        return class_name, entropy
    
    @staticmethod
    def info_gain(Data: np.array, labels: np.array, attribute: int):
        unique = np.unique(Data[:, attribute])

        mean_entroy = 0
        for u in unique:
            indices = np.where(Data[:, attribute] == u)[0]
            c_id, entropy = Node.get_entropy(labels = labels[indices])

            mean_entroy += entropy * indices.shape[0] / labels.shape[0]

        return Node.get_entropy(labels = labels)[1] - mean_entroy

    @staticmethod
    def split_attribute(Data: np.array, labels: np.array, attributes: list):
        info_gains = np.array(list(map(lambda attr: Node.info_gain(Data, labels, attr), attributes)))
        
        index = np.argmax(info_gains)
        return attributes[index], info_gains[index]
    
    @staticmethod
    def Generate(X: np.array, Y: np.array, attributes: list, threshold: float):
        node = Node()

        class_name, entropy = Node.get_entropy(Y)

        if entropy < threshold or len(attributes) == 0:
            node.class_name = class_name
            
            return node

        attribute, info_gain = Node.split_attribute(X, Y, attributes)
        if info_gain <= 0:
            # node =  Node.Generate(X, Y, [element for element in attributes if element != attribute], threshold)
            
            # return node
            node.class_name = class_name
            
            return node
        
        node.attr_index = attribute
        
        uniques = np.unique(X[:, attribute])
        for value in uniques:
            indices = np.where(X[:, attribute] == value)[0]
            child = Node.Generate(X[indices, :], Y[indices], [element for element in attributes if element != attribute], threshold)

            child.value = value
            child.parent = node

            node.children.append(child)
        
        return node

class LoggedNode(Node):
    latest_tree = None
    best_val_accuracy = [0]
    best_train_accuracy = [0]

    early_stopping = False

    @staticmethod
    def Generate(X: np.array, Y: np.array, X_val: np.array, Y_val: np.array, attributes: list, threshold: float, index = None, parent = None, value = None):
        node = LoggedNode()
        node.parent = parent
        node.value = value
        # if parent is None: 
        #     root = node

        class_name, entropy = LoggedNode.get_entropy(Y)

        if entropy < threshold or len(attributes) == 0:
            node.class_name = class_name
            
            return node

        attribute, info_gain = LoggedNode.split_attribute(X, Y, attributes)
        if info_gain <= 0:
            node.class_name = class_name
            
            return node
        
        node.attr_index = attribute
        
        uniques = np.unique(X[:, attribute])
        
        for value in uniques:
            indices = np.where(X[:, attribute] == value)[0]
            child = LoggedNode.Generate(X[indices, :], Y[indices], X_val, Y_val, [], threshold, None, None, None)

            child.value = value
            child.parent = node

            node.children.append(child)
        
        # node.class_name = class_name
        node.parent = parent
        if LoggedNode.early_stopping and parent != None:
            leaf, parent.children[index] = parent.children[index], node

        root = node.get_root()
        preds = np.array(list(map(lambda i: root.predict(X_val[i, :]), range(X_val.shape[0]))))
        accuracy = Metrics.accuracy(preds, Y_val) * 100

        if LoggedNode.early_stopping and parent != None and accuracy < LoggedNode.best_val_accuracy[-1]:
            # This split needs to be undone
            parent.children[index] = leaf
            return None
        else:
            if LoggedNode.early_stopping:
                LoggedNode.best_val_accuracy.append(accuracy)
                preds = np.array(list(map(lambda i: root.predict(X_train[i, :]), range(X_train.shape[0]))))
                accuracy = Metrics.accuracy(preds, Y_train) * 100
                LoggedNode.best_train_accuracy.append(accuracy)
        
        for i in range(len(uniques)):
            value = uniques[i]
            indices = np.where(X[:, attribute] == value)[0]
            temp_node = LoggedNode.Generate(X[indices, :], Y[indices], X_val, Y_val, [element for element in attributes if element != attribute], threshold, i, node, value)
            
            if temp_node is None:
                continue

            temp_node.value = value
            temp_node.parent = node

            node.children[i] = temp_node

        if not LoggedNode.early_stopping:
            root = node.get_root()
            preds = np.array(list(map(lambda i: root.predict(X_val[i, :]), range(X_val.shape[0]))))
            accuracy = Metrics.accuracy(preds, Y_val) * 100

            LoggedNode.best_val_accuracy.append(accuracy)

            preds = np.array(list(map(lambda i: root.predict(X_train[i, :]), range(X_train.shape[0]))))
            accuracy = Metrics.accuracy(preds, Y_train) * 100

            LoggedNode.best_train_accuracy.append(accuracy)

        return node

## Helper Functions
def load_dataset() -> pd.DataFrame:
    import pathlib
    
    # Leave it as 'dataset.csv' if it is at the same level as main.py
    # My Directory Structure looks something like 
    # Assignment 4
    # |- data
    # |- |- car_evaluation.csv
    # |- main.py
    # |- Report.pdf
    relative_path = 'data/car_evaluation.csv'

    df = pd.read_csv(str(pathlib.Path(__file__).parent.resolve()) + '/' + relative_path)

    return df

def split(X: np.array, Y: np.array, ttv: Tuple[int]) -> Tuple[np.array]:
    num_samples = X.shape[0]
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train, test, val = ttv
    
    train = (train * num_samples) // 100
    test = (test * num_samples) // 100
    val = (val * num_samples) // 100

    train_indices, test_indices, val_indices = indices[: train], indices[train: train + test], indices[-val: ]

    X_train, Y_train = X[train_indices, :], Y[train_indices]
    X_test, Y_test = X[test_indices, :], Y[test_indices]
    X_val, Y_val = X[val_indices, :], Y[val_indices]

    return (X_train, Y_train, X_test, Y_test, X_val, Y_val)

def process_dataset(df: pd.DataFrame):
    np_df = df.to_numpy()

    X = np_df[:, : -1]
    Y = np_df[:, -1]

    X_train, Y_train, X_test, Y_test, X_val, Y_val = split(X, Y, (60, 20, 20))

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

## Experiments
def Experiment1(X_train, Y_train, X_validation, Y_validation, col_names):
    
    print('-' * 35 + f'Experiment 1' + '-' * 35)

    thresholds = [0, 0.25, 0.5, 0.75, 1]
    val_accuracies = []
    train_accuracies = []
    
    sizes = []

    for threshold in thresholds:
        root = Node.Generate(X_train, Y_train, list(range(X_train.shape[1])), threshold)

        preds = np.array(list(map(lambda i: root.predict(X_validation[i, :]), range(X_validation.shape[0]))))
        val_accuracies.append(Metrics.accuracy(preds, Y_validation) * 100)
        
        preds = np.array(list(map(lambda i: root.predict(X_train[i, :]), range(X_train.shape[0]))))
        train_accuracies.append(Metrics.accuracy(preds, Y_train) * 100)
        
        sizes.append(root.get_size())

    plt.figure()
    plt.plot(thresholds, val_accuracies)
    plt.plot(thresholds, train_accuracies)

    # plt.xscale('log')
    plt.xlabel('Threshold')
    plt.ylabel('% Model Accuracy')
    plt.title('Hyperparameter Tuning')
    plt.legend(['Validation', 'Training'])

    # plt.show()

    plt.figure()
    plt.plot(thresholds, sizes)

    # plt.xscale('log')
    plt.xlabel('Threshold')
    plt.ylabel('Size of the Tree(# of Nodes)')
    plt.title('Hyperparameter Tuning')

    # plt.show()

    # Its very difficult to automatically find the best threshold value for such a problem as 
    # one has to consider the tradeoff of Size and Accuracy. Hence, I am manually setting the value based on
    # my observations from the graph given my seed.
    # best_threhsold = thresholds[::-1][max(range(len(thresholds)), key = lambda i: accuracies[::-1][i])]
    best_threhsold = 0.5

    print(f'Best threshold value: {best_threhsold}')
    
    print('-' * 82)

    # root = Node.Generate(X_train, Y_train, list(range(X_train.shape[1])), best_threhsold)
    # root.print_tree(col_names = col_names)

    return best_threhsold

def Experiment2(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, X_val: np.array, Y_val: np.array, threshold: float, col_names):
    print('-' * 35 + f'Experiment 2' + '-' * 35)
    root = Node.Generate(X_train, Y_train, list(range(X_train.shape[1])), threshold)

    preds = np.array(list(map(lambda i: root.predict(X_train[i, :]), range(X_train.shape[0]))))
    train_accuracy = Metrics.accuracy(preds, Y_train) * 100

    preds = np.array(list(map(lambda i: root.predict(X_test[i, :]), range(X_test.shape[0]))))
    test_accuracy = Metrics.accuracy(preds, Y_test) * 100

    size = root.get_size()

    print(f'Accuracy on Train Dataset: {train_accuracy}')
    print(f'Accuracy on Test Dataset: {test_accuracy}')
    print(f'Size without early stopping: {size}')

    # root.print_tree(col_names = col_names)

    root = LoggedNode.Generate(X_train, Y_train, X_val, Y_val, list(range(X_train.shape[1])), threshold)
    val_accuracies = LoggedNode.best_val_accuracy
    train_accuracies = LoggedNode.best_train_accuracy

    preds = np.array(list(map(lambda i: root.predict(X_test[i, :]), range(X_test.shape[0]))))
    test_accuracy_wes = Metrics.accuracy(preds, Y_test) * 100
    
    plt.figure()
    plt.plot(val_accuracies)
    plt.plot(train_accuracies)

    # plt.xscale('log')
    plt.xlabel('Number of branchings')
    plt.ylabel('% Model Accuracy')
    plt.title('Percentage Accuracy v/s Branching')

    plt.legend(['Validation', 'Training'])

    # plt.show()
    
    LoggedNode.early_stopping = True
    LoggedNode.best_val_accuracy = []
    LoggedNode.best_train_accuracy = []
    root = LoggedNode.Generate(X_train, Y_train, X_val, Y_val, list(range(X_train.shape[1])), threshold)
    val_accuracies = LoggedNode.best_val_accuracy
    train_accuracies = LoggedNode.best_train_accuracy
    
    plt.figure()
    plt.plot(val_accuracies)
    plt.plot(train_accuracies)

    # plt.xscale('log')
    plt.xlabel('Number of branchings')
    plt.ylabel('% Model Accuracy')
    plt.title('Early Stopping')

    plt.legend(['Validation', 'Training'])

    # plt.show()
    
    size = root.get_size()

    preds = np.array(list(map(lambda i: root.predict(X_test[i, :]), range(X_test.shape[0]))))
    test_accuracy = Metrics.accuracy(preds, Y_test) * 100

    preds = np.array(list(map(lambda i: root.predict(X_train[i, :]), range(X_train.shape[0]))))
    train_accuracy = Metrics.accuracy(preds, Y_train) * 100

    print(f'Train accuracy with early stopping: {train_accuracy}')
    print(f'Test accuracy with early stopping: {test_accuracy}')
    print(f'Size with early stopping: {size}')
    
    # root.print_tree(col_names = col_names)

    print('-' * 82)

def Experiment3(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, X_val: np.array, Y_val: np.array, threshold: float, col_names):
    print('-' * 35 + f'Experiment 3' + '-' * 35)

    print('Decision Tree without early stopping')
    root = Node.Generate(X_train, Y_train, list(range(X_train.shape[1])), threshold)
    root.print_tree(col_names = col_names)

    print()

    print('Decision Tree with early stopping')
    root = LoggedNode.Generate(X_train, Y_train, X_val, Y_val, list(range(X_train.shape[1])), threshold)
    root.print_tree(col_names = col_names)    
    print('-' * 82)

## Main / Driver
if __name__ == '__main__':
    df = load_dataset()
    col_names = df.columns.to_list()[:-1]
    
    X_train, Y_train, X_test, Y_test, X_val, Y_val = process_dataset(df)

    threshold = Experiment1(X_train, Y_train, X_val, Y_val, col_names)

    Experiment2(X_train, Y_train, X_test, Y_test, X_val, Y_val, threshold, col_names)
    
    Experiment3(X_train, Y_train, X_test, Y_test, X_val, Y_val, threshold, col_names)

    plt.show()

