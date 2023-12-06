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
#### Model: Contains the following methods:
###### predict: Predicts class for an array of instances
###### _predict: Predicts class for a single instance

## Helper Functions:
#### load_dataset: Loads the dataset
#### split: My implementation of spliting data into test and train
#### process_dataset: Simply processes the data and returns
#### get_noise: Returns an array containing samples of gaussian noise with 0 mean and 2 standard deviation 
#### add_noise: Adds noise to some percentage of the training dataset.(Shuffling not required) 

## Experiments:
#### Experiment1: Displays plot of Model Accuracy vs K value and prints best K value.
#### Experiment2: Displays plot for Model Accuracies vs Noise level and prints them in a tabular form

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
np.random.seed(12)

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

class Model:
    def __init__(self, K: int, X: np.array, Y: np.array):
        self.X = X
        self.Y = Y

        self.K = K
    
    def predict(self, X: np.array):
        return np.array(list(map(lambda i: self._predict(X[i, :]), range(X.shape[0]))))

    def _predict(self, x: np.array):
        deltas = self.X - x

        distances = np.mean(np.square(deltas), axis = 1)

        K_nearest = np.argsort(distances)[:self.K]
        
        K_nearest_distances = distances[K_nearest]
        K_nearest_labels = self.Y[K_nearest]

        unique_labels = np.unique(K_nearest_labels)
        max_weight = 0
        prediction = None
        for label in unique_labels:
            label_weight = 1 / np.sum(K_nearest_distances[K_nearest_labels == label])

            if max_weight < label_weight:
                max_weight = label_weight
                prediction = label
        
        return prediction

## Helper Functions
def load_dataset() -> pd.DataFrame:
    import pathlib
    
    # Leave it as 'dataset.csv' if it is at the same level as main.py
    # My Directory Structure looks something like 
    # Assignment 3
    # |- data
    # |- |- Iris.csv
    # |- |- database.sqlite
    # |- main.py
    # |- Report.pdf
    relative_path = 'data/Iris.csv'

    df = pd.read_csv(str(pathlib.Path(__file__).parent.resolve()) + '/' + relative_path)

    return df

def split(X: np.array, Y: np.array, ttv: Tuple[int]) -> Tuple[np.array]:
    num_samples = X.shape[0]
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train, test = ttv
    
    train = (train * num_samples) // 100
    test = (test * num_samples) // 100

    train_indices, test_indices = indices[: train], indices[-test: ]

    X_train, Y_train = X[train_indices, :], Y[train_indices]
    X_test, Y_test = X[test_indices, :], Y[test_indices]

    return (X_train, Y_train, X_test, Y_test)

def get_noise(shape: tuple, mean = 0, std = 2.0) -> np.array:
    return np.random.normal(loc = mean, scale = std, size = shape)

def add_noise(X: np.array, Y: np.array, percent: float):
    num_samples = X.shape[0]
    
    indices = np.arange(num_samples)
    # np.random.shuffle(indices)

    percent = (percent * num_samples) // 100
    
    noisy_indices = indices[:percent]
    new_X = X.copy()
    new_X[noisy_indices, :] += get_noise(shape = X[noisy_indices, :].shape)
    return new_X, Y    

def process_dataset(df: pd.DataFrame):
    np_df = df.to_numpy()

    X = np_df[:, : -1].astype(np.float32)
    Y = np_df[:, -1]

    X_train, Y_train, X_test, Y_test = split(X, Y, (80, 20))

    return X_train, Y_train, X_test, Y_test

## Experiments
def Experiment1(X_train, Y_train, X_test, Y_test) -> int:
    print('-' * 35 + f'Experiment 1' + '-' * 35)

    k_values = [1, 3, 5, 10, 20]

    accuracies = []
    for k in k_values:
        model = Model(k, X_train, Y_train)

        preds = model.predict(X_test)
        accuracy = Metrics.accuracy(preds, Y_test)

        accuracies.append(accuracy * 100)
    
    plt.figure()
    plt.plot(k_values, accuracies)

    plt.xlabel('K Value')
    plt.ylabel('% Model Accuracy')
    plt.title('Hyperparameter Tuning')

    # plt.show()

    optimal_K = k_values[-np.argmax(accuracies[::-1]) -1]
    best_accuracy = accuracies[-np.argmax(accuracies[::-1]) -1]

    print(f'Optimal K value found: {optimal_K}')
    print(f'Model Accuracy of the associated K value: {np.round(best_accuracy, 2)}%')

    print('-' * 82)

    return optimal_K

def Experiment2(X_train, Y_train, X_test, Y_test, optimal_K: int):
    print('-' * 35 + f'Experiment 2' + '-' * 35)

    noise_levels = [i * 10 for i in range(1, 10)]

    noiseless_model = Model(K = optimal_K, X = X_train, Y = Y_train)
    preds = noiseless_model.predict(X_test)
    noiseless_accuracy = Metrics.accuracy(preds, Y_test) * 100

    noisy_accuracies = []
    for noise in noise_levels:
        noisy_X_train, _ = add_noise(X_train, Y_train, noise)
        model = Model(K = optimal_K, X = noisy_X_train, Y = Y_train)
        
        preds = model.predict(X_test)
        accuracy = Metrics.accuracy(preds, Y_test)

        noisy_accuracies.append(np.round(accuracy * 100, 2))

    df = pd.DataFrame(data = {'% of data noisy': noise_levels, r'% accuracy': noisy_accuracies})
    df.index = [''] * len(df)
    
    print(f'Accuracy without noise: {np.round(noiseless_accuracy, 2)}')

    print('Accuracy at different Noise values:')
    print(df)
    
    plt.figure()
    plt.plot(noise_levels, noisy_accuracies)

    plt.xlabel('% Noisy Dataset')
    plt.ylabel('% Model Accuracy')
    plt.title('Effect of noise on KNN')

    # plt.show()

    print('-' * 82)

## Main / Driver
if __name__ == '__main__':
    df = load_dataset()
    df.drop(columns = ['Id'], inplace = True)

    X_train, Y_train, X_test, Y_test = process_dataset(df)
    best_K = Experiment1(X_train, Y_train, X_test, Y_test)

    Experiment2(X_train, Y_train, X_test, Y_test, best_K)