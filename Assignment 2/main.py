## Name: Aryan Satpathy
## Roll: 20EC10014

# Imports
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MAX_ITERATIONS = 1000
BATCH_SIZE = 50

# Fill Mode
FILL_MEAN = 0
FILL_MEDIAN = 1
FILL_MODE = 2
FILL_FORWARD = 3
FILL_BACKWARD = 4

# For the sake of reproducibility, I'll set the seed of random
np.random.seed(42)

class Model:
    '''
    Finds the Model using Closed Form, Grad and Ridge Regression.
    '''

    def __init__(self, shape: tuple):
        self.weights = np.zeros(shape = shape)
    def closed_fit(self, X: np.array, Y: np.array):
        self.weights = np.linalg.pinv(X) @ Y.reshape((Y.shape[0], 1))

    def grad_fit(self, X: np.array, Y: np.array, learning_rate: float):
        for i in range(MAX_ITERATIONS):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            batch_indices = indices[:BATCH_SIZE]
            mini_X = X[batch_indices, :]
            mini_y = Y[batch_indices]

            delta = (mini_X @ self.weights) - mini_y.reshape((BATCH_SIZE, 1))
            grad = 1 / mini_X.shape[0] * mini_X.T @ delta 
            self.weights = self.weights - learning_rate * grad
    
    def ridge_fit(self, X: np.array, Y: np.array, learning_rate: float, alpha: float):
        for i in range(MAX_ITERATIONS):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            batch_indices = indices[:BATCH_SIZE]
            mini_X = X[batch_indices, :]
            mini_y = Y[batch_indices]

            delta = (mini_X @ self.weights) - mini_y.reshape((BATCH_SIZE, 1))
            grad = learning_rate / mini_X.shape[0] * mini_X.T @ delta + alpha * self.weights 
            self.weights = self.weights - grad

    def MSE(self, X: np.array, Y: np.array) -> float:
        delta = X @ self.weights - Y.reshape((Y.shape[0], 1))
        return np.average(np.square(delta))

def load_dataset() -> pd.DataFrame:
    import pathlib
    
    # Leave it as 'dataset.csv' if it is at the same level as main.py
    # My Directory Structure looks something like 
    # Assignment 2
    # |- data
    # |- |- dataset.csv
    # |- main.py
    # |- Report.pdf
    relative_path = 'data/dataset.csv'

    df = pd.read_csv(str(pathlib.Path(__file__).parent.resolve()) + '/' + relative_path)

    return df

def NaNfiller(df: pd.DataFrame, mode: int) -> pd.DataFrame:
    if mode is FILL_MEAN:
        Mean = df.mean(skipna = True)
        return df.fillna(Mean)
    if mode is FILL_MEDIAN:
        Median = df.median(skipna = True)
        return df.fillna(Median)
    if mode is FILL_MODE:
        Mode = df.mode()[0]
        return df.fillna(Mode)
    if mode is FILL_FORWARD:
        return df.fillna(method='ffill')
    if mode is FILL_BACKWARD:
        return df.fillna(method='bfill')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    num_rows = df.shape[0]

    for column in df.columns:
        num_NaN = sum(df[column].isna())
        if num_NaN <= num_rows // 2:
            out[column] = df[column]
            if num_NaN:
                out[column] = NaNfiller(out[column], FILL_MEDIAN if out[column].dtype in [int, float] else FILL_MODE)

    columns = [column for column in out.columns if out[column].dtype == object]
    
    for i, column in enumerate(columns):
        enc = OneHotEncoder()
        enc.fit(out[[column]])

        column_one_hot = enc.transform(out[[column]])
        out.drop(columns = [column], inplace = True)
        out[enc.categories_[0]] = column_one_hot.toarray()

    return out

def Experiment1(df: pd.DataFrame):
    import plotly
    import plotly.express as px
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    # For beautification purposes
    sns.set_style('darkgrid')
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['figure.figsize'] = (10, 6)
    matplotlib.rcParams['figure.facecolor'] = '#00000000'

    for column in df.columns:
        fig = px.histogram(df,
                        x=column,
                        marginal='box',
                        nbins=47,
                        title=f'Distribution of {column}')
        fig.update_layout(bargap=0.1)
        fig.show()
        
    df = preprocess(df)
    sns.heatmap(df.corr(), cmap='Reds', annot=True)
    plt.title('Correlation Matrix')
    plt.show()

def Experiment2(df: pd.DataFrame):
    df = preprocess(df)

    num_cols = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Purchase']
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    num_scaled = scaler.transform(df[num_cols])
    df[num_cols] = num_scaled
    
    model = Model((df.shape[1] - 1, 1))

    np_df = df.to_numpy()
    X = np_df[:, [x for x in np.arange(np_df.shape[1]) if x != 4]]
    y = np_df[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 42) # 0.25 x 0.8 = 0.2
    
    model.closed_fit(X_train, y_train)
    print(f'MSE for Closed Solution Model: {model.MSE(X_test, y_test)}')

def Experiment3(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    df = preprocess(df)

    num_cols = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Purchase']
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    num_scaled = scaler.transform(df[num_cols])
    df[num_cols] = num_scaled
    
    model = Model((df.shape[1] - 1, 1))

    np_df = df.to_numpy()
    X = np_df[:, [x for x in np.arange(np_df.shape[1]) if x != 4]]
    y = np_df[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 42) # 0.25 x 0.8 = 0.2

    Rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    MSEs = []
    for learning_rate in Rates:
        model.grad_fit(X_train, y_train, learning_rate)
        MSEs.append(model.MSE(X_test, y_test))

    plt.plot(Rates, MSEs)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Rate vs MSE')
    plt.show()

    print(f'Best Learning Rate: {Rates[MSEs.index(min(MSEs))]}')

def Experiment4(df: pd.DataFrame):
    learning_rate = 0.01

    import matplotlib.pyplot as plt

    df = preprocess(df)

    num_cols = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Purchase']
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    num_scaled = scaler.transform(df[num_cols])
    df[num_cols] = num_scaled
    
    model = Model((df.shape[1] - 1, 1))

    np_df = df.to_numpy()
    X = np_df[:, [x for x in np.arange(np_df.shape[1]) if x != 4]]
    y = np_df[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 42) # 0.25 x 0.8 = 0.2

    Alphas = [x / 10 for x in range(11)]
    MSEs = []
    for alpha in Alphas:
        model.ridge_fit(X_train, y_train, learning_rate, alpha)
        MSEs.append(model.MSE(X_val, y_val))

    plt.plot(Alphas, MSEs)
    plt.xlim(0, 1)
    plt.xlabel('Alpha Value')
    plt.ylabel('Mean Squared Error')
    plt.title('Alpha vs MSE')
    plt.show()

    print(f'Best Alpha Value: {Alphas[MSEs.index(min(MSEs))]}')

def Experiment5(df: pd.DataFrame):
    learning_rate = 0.01
    alpha = 0

    df = preprocess(df)

    num_cols = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Purchase']
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    num_scaled = scaler.transform(df[num_cols])
    df[num_cols] = num_scaled
    
    closed_model = Model((df.shape[1] - 1, 1))
    grad_model = Model((df.shape[1] - 1, 1))
    ridge_model = Model((df.shape[1] - 1, 1))

    np_df = df.to_numpy()
    X = np_df[:, [x for x in np.arange(np_df.shape[1]) if x != 4]]
    y = np_df[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 42) # 0.25 x 0.8 = 0.2

    closed_model.closed_fit(X_train, y_train)
    closed_mse = closed_model.MSE(X_test, y_test)

    grad_model.grad_fit(X_train, y_train, learning_rate)
    grad_mse = grad_model.MSE(X_test, y_test)
    
    ridge_model.ridge_fit(X_train, y_train, learning_rate, alpha)
    ridge_mse = ridge_model.MSE(X_test, y_test)

    print('Final MSEs')
    print(f'Closed Solution: {closed_mse}')
    print(f'Grad Solution: {grad_mse}')
    print(f'Ridge Solution: {ridge_mse}')
    
if __name__ == '__main__':
    df = load_dataset()
    df.drop(columns = ['Product_ID', 'User_ID'], inplace = True)

    Experiment1(df)
    Experiment2(df)
    Experiment3(df)
    Experiment4(df)
    Experiment5(df)