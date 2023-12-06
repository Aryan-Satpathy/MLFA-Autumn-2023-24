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
###### grad_fit: Performs Mini Batch Gradient Based Logistic Regression and logs and prints metrics
###### Loss: Finds Categorical Cross Entropy Loss. To understand progress of training 
###### get_prob_prediction: Gets raw output of the model(probabilities for each class)
###### prediction: Finds prediction(takes maximum of the probability predictions for each class)
###### accuracy: Returns model Accuracy on Validation Set

## Helper Functions:
#### load_dataset: Loads the dataset
#### scale: Given meana and variance, scales the data
#### process_labels: Performs One Hot Encoding on Labels
#### split: My implementation of spliting data into test, train and validation
#### process_dataset: Performs scaling on data, One Hot Encoding on Labels and splits dataset
#### heatmap: My helper function that draws a heatmap for a given matrix using matplotlib
#### annotate_heatmap: My helper function that adds text on the heatmap generated

## Experiments:
#### Experiment1: Displays plot of Learning Rate vs Epochs and prints best learning rate.
#### Experiment2: Performs Training using lr found before and displays class-wise plots for average
####              probabilities predicted by the model.
#### Experiment3: Displays Confusion Matrix as a heatmap, prints it as a matrix, and prints a table
####              containing the metrics of the model, all on the test data.

## Main / Driver:
#### Simply loads the dataset, processes it, and then calls Experiments one by one.

###################################################################################################

## Imports
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple

# For the sake of reproducibility, I'll set the seed of random module
np.random.seed(42)

# Constants
EPOCHS = 50
BATCH_SIZE = 30

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
        return np.sum(np.all(out == y, axis = 1)) / out.shape[0]                                                 # Finding indices of matching predictions and labels
    
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
    '''
    Finds the Logistic Regression Model.
    '''

    def __init__(self, shape: tuple, log: bool = False, verbose: bool = False):
        self.weights = np.zeros(shape = shape)

        self.log = log
        self.log_info = {}

        self.verbose = verbose

    def grad_fit(self, X: np.array, Y: np.array, X_val: np.array, Y_val: np.array, learning_rate: float):
        self.log_info['Prob per epoch'] = []

        for i in range(EPOCHS):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for iteration in range(X.shape[0] // BATCH_SIZE):
                batch_indices = indices[BATCH_SIZE * iteration : BATCH_SIZE * (iteration + 1)]
                mini_X = X[batch_indices, :]
                mini_y = Y[batch_indices, :]

                _p = (mini_X @ self.weights)
                
                p = np.exp(_p)
                p = (p.T / np.sum(p, axis = 1)).T

                delta = mini_y - p

                grad = -1 / mini_X.shape[0] * mini_X.T @ delta 
                self.weights = self.weights - learning_rate * grad

            prob, predictions = self.prediction(X_val)
            
            accuracy = Metrics.accuracy(predictions, Y_val)
            loss = self.Loss(prob, Y_val)

            if (i + 1) % 10 == 0 and self.verbose:
                print(f'Epoch {i + 1}/{EPOCHS}:, Accuracy: {accuracy}, Loss: {loss}')

            if self.log:
                probs = self.get_prob_prediction(X)
                
                num_classes = Y.shape[1]
                class_indices = [np.where(Y[:, c] == 1)[0] for c in range(num_classes)]

                mean_probabilities = [np.mean(probs[indices, :], axis = 0) for indices in class_indices]
                
                self.log_info['Prob per epoch'].append(mean_probabilities)
    
    def Loss(self, probs: np.array, Y: np.array):
        '''
        Categorical Cross Entropy Loss
        '''
        probs = (1 - probs) * (1 - Y) + probs * Y
        probs = np.log(probs)

        return -np.mean(probs)

    def get_prob_prediction(self, X: np.array):
        _p = X @ self.weights
        
        p = np.exp(_p)
        p = (p.T / np.sum(p, axis = 1)).T

        return p

    def prediction(self, X: np.array) -> Tuple[np.array, np.array]:
        p = self.get_prob_prediction(X)
        
        preds = np.zeros_like(p)
        preds[np.arange(preds.shape[0]), p.argmax(axis = 1)] = 1

        return p, preds
    
    def accuracy(self, X_val, Y_val):
        '''
        Get accuracy of the model
        '''
        prob, predictions = self.prediction(X_val)
        accuracy = Metrics.accuracy(predictions, Y_val)

        return accuracy * 100

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

def scale(X: np.array, mean: np.array, variance: np.array) -> np.array:
    scaled_X = (X - np.repeat(mean[:, None], X.shape[0], axis = 1).T) / variance

    return scaled_X

def process_labels(Y: np.array) -> Tuple[np.array, np.array]:
    uniques=  np.unique(Y)
    num_classes = uniques.shape[0]

    U = np.repeat(uniques[:, None], Y.shape[0], axis = 1).T
    y = np.repeat(Y[:, None], num_classes, axis = 1)

    one_hot_Y = (U == y).astype(np.uint8)
    
    return one_hot_Y, uniques

def split(X: np.array, Y: np.array, ttv: Tuple[int]) -> Tuple[np.array]:
    num_samples = X.shape[0]
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train, test, val = ttv
    
    train = (train * num_samples) // 100
    test = (test * num_samples) // 100
    val = (val * num_samples) // 100

    train_indices, test_indices, val_indices = indices[: train], indices[train: train + test], indices[-val: ]

    X_train, Y_train = X[train_indices, :], Y[train_indices, :]
    X_test, Y_test = X[test_indices, :], Y[test_indices, :]
    X_val, Y_val = X[val_indices, :], Y[val_indices, :]

    return (X_train, Y_train, X_test, Y_test, X_val, Y_val)

def process_dataset(df: pd.DataFrame):
    np_df = df.to_numpy()

    X = np_df[:, : -1].astype(np.float32)
    Y = np_df[:, -1]

    one_hot_Y, uniques = process_labels(Y)

    X_train, Y_train, X_test, Y_test, X_val, Y_val = split(X, one_hot_Y, (60, 20, 20))

    mean_X = np.mean(X_train, axis = 0)
    var_X = np.var(X_train, axis = 0)

    X_train = scale(X_train, mean_X, var_X)
    X_test = scale(X_test, mean_X, var_X)
    X_val = scale(X_val, mean_X, var_X)

    X_train = np.concatenate((np.ones((1, X_train.shape[0])), X_train.T), axis= 0).T
    X_test = np.concatenate((np.ones((1, X_test.shape[0])), X_test.T), axis= 0).T
    X_val = np.concatenate((np.ones((1, X_val.shape[0])), X_val.T), axis= 0).T

    return X_train, Y_train, X_test, Y_test, X_val, Y_val, uniques

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

## Experiments
def Experiment1(X_val: np.array, Y_val: np.array) -> float:

    print('-' * 35 + f'Experiment 1' + '-' * 35)

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
    accuracies = []

    # Set this to false to hide accuracy and loss values while training
    SHOW_PROGRESS = True

    for learning_rate in learning_rates:
        print(f'Training for Learning Rate: {learning_rate}')
        
        model = Model(shape = (X_val.shape[-1], Y_val.shape[-1]), verbose = SHOW_PROGRESS)
        model.grad_fit(X_val, Y_val, X_val, Y_val, learning_rate = learning_rate)

        accuracy = model.accuracy(X_val, Y_val)
        accuracies.append(accuracy)
    
    plt.figure()
    plt.plot(learning_rates, accuracies)

    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('% Model Accuracy')
    plt.title('Hyperparameter Tuning')

    plt.show()

    best_learning_rate = learning_rates[max(range(6), key = lambda x: accuracies[x])]
    print(f'Best learning rate found: {best_learning_rate}')

    print('-' * 82)

    return best_learning_rate

def Experiment2(X_train: np.array, Y_train: np.array, X_val: np.array, Y_val: np.array, uniques, learning_rate: float) -> Model:
    
    print('-' * 35 + f'Experiment 2' + '-' * 35)

    model = Model(shape = (X_train.shape[-1], Y_train.shape[-1]), log = True, verbose = True)
    model.grad_fit(X_train, Y_train, X_val, Y_val, learning_rate)

    prob_vs_epoch = model.log_info['Prob per epoch']
    epochs = list(range(EPOCHS))

    for class_index in range(len(uniques)):
        plt.figure()

        class_prob_vs_epoch = [epoch[class_index] for epoch in prob_vs_epoch]
        for class_index_2 in range((len(uniques))):
            plt.plot(epochs, [epoch[class_index_2] for epoch in class_prob_vs_epoch])
        
        plt.legend(uniques)
        plt.xlabel("Epochs")
        plt.ylabel("Average Probability")
        plt.title(f"Probability vs Epoch for Class {uniques[class_index]}")

        plt.show()
    
    print('-' * 82)

    return model

def Experiment3(mdoel: Model, X_test: np.array, Y_test: np.array, uniques):
    print('-' * 35 + f'Experiment 3' + '-' * 35)

    _, predictions = model.prediction(X_test)
    
    confusion_matrix = Y_test.T @ predictions
    confusion_matrix = confusion_matrix.astype(np.uint8)

    print(f"Confusion Matrix : \n{confusion_matrix}")

    fig, ax = plt.subplots()

    im, cbar = heatmap(confusion_matrix, uniques, uniques, ax=ax,
                    cmap="YlGn", cbarlabel="Number of Samples")
    texts = annotate_heatmap(im, valfmt="{x}")

    fig.tight_layout()
            
    plt.xlabel("Prediction")
    plt.ylabel("Data")
    plt.title("Confusion Matrix")
    plt.show()

    precision, recall, f1 = [], [], []
    for c in range(uniques.shape[0]):
        y = Y_test[:, c]
        out = predictions[:, c]

        p, r, f = Metrics.precision(out, y), Metrics.recall(out, y), Metrics.f1(out, y)

        precision.append(p)
        recall.append(r)
        f1.append(f)
    
    df = pd.DataFrame(data = {'Class': uniques, 'Precision': precision, 'Recall': recall, 'F1': f1})
    df.index = [''] * len(df)
    print(df)

    print('-' * 82)

## Main / Driver
if __name__ == '__main__':
    df = load_dataset()
    df.drop(columns = ['Id'], inplace = True)

    X_train, Y_train, X_test, Y_test, X_val, Y_val, uniques = process_dataset(df)

    learning_rate = Experiment1(X_val, Y_val)
    model = Experiment2(X_train, Y_train, X_val, Y_val, uniques, learning_rate)

    Experiment3(model, X_test, Y_test, uniques)