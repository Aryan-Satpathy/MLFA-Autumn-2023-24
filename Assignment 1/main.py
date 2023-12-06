## Name: Aryan Satpathy
## Roll: 20EC10014

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

MAX_ITERATIONS = 2000
LEARNING_RATE = 1

# For the sake of reproducibility, I'll set the seed of random
np.random.seed(42)

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
        return np.where(np.abs(out - y) <= 1e-5)[0].shape[0] / out.shape[0]                                                 # Finding indices of matching predictions and labels
    
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

        return (2 * p + r) / (p + r)

class Model:
    '''
    Class representing Model. Contains Training Data, and few parameters(like Max Iteration, Learning Rate)
    Also contains methods for Learning(PLA), Prediction, Evaluation/Validation
    '''
    def __init__(self, x: np.array, y: np.array, MAX_IT: int = MAX_ITERATIONS, L_RATE: float = LEARNING_RATE) -> None:
        self.input = x
        self.label = y
        self.max_iterations = MAX_IT

        self.learning_rate = L_RATE

        self.weights = None

        self.log_info = {}
    
    def perceptron_learning_algorithm(self) -> None:
        x = self.input

        # Add a cloumn of ones in front of the data 
        x = np.concatenate((np.ones((1, x.shape[0])), x.T), axis= 0).T
        y = self.label

        if self.weights is None:
            # Initialize weights
            weights = np.random.uniform(-100, 100, (x.shape[-1], ))
        else:
            weights = self.weights

        # Logging
        mis_classification_vs_iterations = []

        # Training Iterations
        for i in range(self.max_iterations):
            error_indicator = y * (x @ weights).T
            
            indices = np.where(error_indicator <= 0)[0]

            # Logging
            mis_classification_vs_iterations.append(len(indices))
            
            # Check for convergence
            if len(indices) == 0:                                                                                               # No Mis Classification. Model has converged on the training data.
                break

            # Update weights
            # weights = weights + self.learning_rate * x[indices[0], :].T * y[indices[0]]                                       # One Instance Update
            weights = weights + self.learning_rate *  x[indices, :].T @ y[indices]                                          # Multi Instance Update(better)
    
        self.log_info['misclassification vs iterations'] = mis_classification_vs_iterations
        self.weights = weights                                                                  # Save learnt weights

    def predict(self, x: np.array) -> np.array:
        # Add a column of ones in front of data
        x = np.concatenate((np.ones((1, x.shape[0])), x.T), axis= 0).T

        # Find sgn(x * w)
        return np.where((x @ self.weights) > 0, np.ones((x.shape[0], )), -np.ones((x.shape[0], ))).astype(int)
    
    def evaluate(self, x: np.array, y: np.array):
        out = self.predict(x)
        y = y.astype(int)

        accuracy = Metrics.accuracy(out, y)
        precision = Metrics.precision(out, y)
        recall = Metrics.recall(out, y)
        f1 = Metrics.f1(out, y)

        return accuracy, precision, recall, f1

def retrieve_data(dataset: int):
    '''
    Simple helper function to load the data, given dataset number (1, 2 or 3).
    My dataset is stored in a folder named 'data' which is at the same level as the code.
    Assignment
    |-- main.py
    |-- data
    |-- |-- inputs_Dataset-1.npy
    |-- |-- outputs_Dataset-1.npy
    |-- |-- inputs_Dataset-2.npy
    |-- |-- outputs_Dataset-2.npy
    |-- |-- inputs_Dataset-3.npy
    |-- |-- outputs_Dataset-3.npy
    '''
    import pathlib
    
    directory_path = str(pathlib.Path(__file__).parent.resolve()) + '/data/'

    input_data = np.load(directory_path + f'inputs_Dataset-{dataset}.npy')
    label = np.load(directory_path + f'outputs_Dataset-{dataset}.npy')
    label = label * 2 - 1

    return input_data, label

def Question1():
    
    print('-' * 15 + 'Question 1' + '-' * 15)
    
    x, y = retrieve_data(1)

    # For different values of K
    for k in [3, 5, 8, 10]:

        print(f'K = {k}')
        print('Fold | Accuracy | Precision | Recall | F1')

        # K fold Cross Validation
        kf = KFold(n_splits=k)
        model_kfold = []

        metrics_kfold = np.ones((k, 4))

        fold = 0
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Model(X_train, y_train, MAX_ITERATIONS)

            # Train
            model.perceptron_learning_algorithm()

            # Get the metrics
            acc, prec, rec, f1 = model.evaluate(X_train, y_train)

            metrics_kfold[fold, :] = np.array([acc, prec, rec, f1])
        
            print(f'  {fold}  |    {np.round(acc, 2)}   |    {np.round(prec, 2)}    |  {np.round(rec, 2)}   | {np.round(f1, 2)}')

            fold += 1
        
        mean = np.mean(metrics_kfold, axis= 0)
        metrics_kfold = metrics_kfold - mean
        var = np.sqrt(np.mean(np.square(metrics_kfold), axis= 0))

        mean = np.round(mean, 2)
        var = np.round(var, 2)
        
        # Display Mean and Variance over all the folds
        print(f'Mean accuracy: {mean[0]}, Mean precision: {mean[1]}, Mean recall: {mean[2]}, Mean f1: {mean[3]}')
        print(f'Variance accuracy: {var[0]}, Variance precision: {var[1]}, Variance recall: {var[2]}, Variance f1: {var[3]}')

    # 80:20 Train Test Split without Shuffle
    num_instances = x.shape[0]
    train_instances = (num_instances * 4) //5

    X_train, X_test = x[: train_instances, :], x[train_instances :, :]
    y_train, y_test = y[: train_instances], y[train_instances :]

    model = Model(X_train, y_train, MAX_ITERATIONS)

    # Train
    model.perceptron_learning_algorithm()

    # Plot Mis Classifications vs Iteraitons
    plt.plot(list(range(len(model.log_info['misclassification vs iterations']))), model.log_info['misclassification vs iterations'])
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    plt.title('Question 1')

    plt.show()

    print('-' * 40)

def Question2and3(q: int):
    
    print('-' * 15 + f'Question {q}' + '-' * 15)
    
    x, y = retrieve_data(q)

    # 80:20 Train Test Split without shuffling
    num_instances = x.shape[0]
    train_instances = (num_instances * 4) //5

    X_train, X_test = x[: train_instances, :], x[train_instances :, :]
    y_train, y_test = y[: train_instances], y[train_instances :]

    model = Model(X_train, y_train, MAX_ITERATIONS)

    # Train
    model.perceptron_learning_algorithm()

    # Get the metrics
    acc, prec, rec, f1 = model.evaluate(X_test, y_test)

    print(f'Accuracy: {np.round(acc, 2)}, Precision: {np.round(prec, 2)}, Recall: {np.round(rec, 2)}, F1: {np.round(f1, 2)}')
        
    # Plot Misclassifications vs Iterations
    plt.plot(list(range(len(model.log_info['misclassification vs iterations']))), model.log_info['misclassification vs iterations'])
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    plt.title(f'Question {q}')

    plt.show()

    print('-' * 40)

def main():
    Question1()
    Question2and3(2)
    Question2and3(3)

if __name__ == '__main__':
    main()