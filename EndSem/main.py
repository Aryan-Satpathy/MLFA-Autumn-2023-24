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
#### Torch: Models and Deep Learning

## Classes:
#### Metrics: Contain static methods to calculate Accuracy, Precision, Recall and F1
#### Model: Base class containing logging and other functionalities.
#### FeedForward: Model.
###### _train: Trains the model
###### Loss: Finds Categorical Cross Entropy Loss. To understand progress of training 
###### init_weights: Kaiming Initialization
#### KNearestNeighbour: Contains the following methods:
###### predict: Predicts class for an array of instances
###### _predict: Predicts class for a single instance

## Helper Functions:
#### load_set: Loads the MNIST Dataset
#### split: My implementation of spliting data into train and validation
#### load_dataset: Loads the MNIST Dataset Train and Test
#### save_model: Saves state dict of a model
#### load_model: Loads pre-trained model
#### preprocess_data: Calls load_dataset and split to generate dataset.

## Experiments:
#### Experiment1: Trains 3 models and shows learning curves
#### Experiment2: Adds noise to study its effects

## Main / Driver:
#### Simply loads the dataset, processes it, and then calls Experiments one by one.
###################################################################################################

## Imports
import numpy as np
import pandas as pd

import torch
from torch import nn as nn
from torch import functional as F
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple, Callable

# For the sake of reproducibility, I'll set the seed of random module
torch.random.manual_seed(42)

# Constants
EPOCHS = 25
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
## Helper Functions
to_np = lambda x: x.detach().cpu().numpy()

class Model(nn.Module):
    '''
    Base Class
    '''

    def __init__(self, log: bool = False, verbose: bool = False):
        super().__init__()
        
        self.dummy_param = nn.Parameter(torch.zeros((1,)))

        self.log = log
        self.log_info = {}

        self.verbose = verbose
    
    def device(self):
        return self.dummy_param.device

    def prediction(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.flatten(1)
        pred1, pred2, pred3 = self(x)
        return torch.argmax(pred1, dim = 1), torch.argmax(pred2, dim = 1), torch.argmax(pred3, dim = 1)

class FeedForward(Model):
    '''
    FeedForward Network.
    Set verbose = False if you do not wish to see the per epoch performance
    '''
    def __init__(self, num_layers: int, in_dim: int, hidden_dim: int, out_dim: int, log: bool = False, verbose: bool = False):
        super().__init__(log, verbose)

        self.n_layers = num_layers
        self.model = nn.ModuleList([nn.Linear([in_dim, hidden_dim][i > 0], [hidden_dim, out_dim][i == num_layers]) for i in range(num_layers + 1)])

        self.init_weights()

    def forward(self, x: torch.Tensor, noise_to_layer: int = None, training = True):
        '''
        Input x: minibatch of images.
        Shape: B, H * W
        Output: B, n_class
        '''

        activation = nn.Sigmoid()

        for idx, module in enumerate(self.model.modules()):        
            if idx:
                idx -= 1
            else:
                continue
            if noise_to_layer == idx:
                weight = torch.zeros_like(module.weight, device = self.device())
                nn.init.kaiming_normal_(weight)
                x = module(x) + torch.matmul(x, weight.T)
            else:
                x = module(x)
            if idx < self.n_layers:
                x = activation(x)
            
        return x
    
    def generate_embedding(self, x: torch.Tensor, noise_to_layer: int = None):
        '''
        Input x: minibatch of images.
        Shape: B, H * W
        Output: B, n_hidden
        '''

        activation = nn.Sigmoid()

        for idx, module in enumerate(self.model.modules()):        
            if idx:
                idx -= 1
            else:
                continue
            if idx < self.n_layers:
                if noise_to_layer == idx:
                    weight = torch.zeros_like(module.weight, device = self.device())
                    nn.init.kaiming_normal_(weight)
                    x = module(x) + torch.matmul(x, weight.T)
                else:
                    x = module(x)

                x = activation(x)
            
        return x
    
    def _train(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], epochs = EPOCHS):
        self.model.train()
        
        train_imgs, train_labels, val_imgs, val_labels = data
        
        train_imgs = train_imgs.to(self.device())
        train_labels = train_labels.to(self.device())
        val_imgs = val_imgs.to(self.device())
        val_labels = val_labels.to(self.device())

        num_batches = train_imgs.shape[0] // BATCH_SIZE
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        # opt = torch.optim.SGD(self.model1.parameters(), lr = 0.01, momentum= 0.9)
        
        for epoch in range(epochs):
            b_indices = torch.randperm(
                    train_imgs.shape[0], device=self.device()
                )
            for batch in range(num_batches):
                mb_indices = b_indices[batch * BATCH_SIZE: batch * BATCH_SIZE + BATCH_SIZE]

                imgs = train_imgs[mb_indices, :, :].flatten(1)
                labels = train_labels[mb_indices]

                pred = self(imgs)
                loss = criterion(pred, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if self.log:
                    self.log_info['Loss'] = self.log_info.get('Loss', []) + [loss.mean().item()]
                    with torch.no_grad():
                        pred = self(val_imgs.flatten(1), training = False)
                        acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(val_labels))
                        
                        self.log_info['Accuracy'] = self.log_info.get('Accuracy', []) + [acc]

            if self.verbose:
                # print(f'Epoch: {epoch}, Model1 loss: {loss1.mean().item():.2f}, Model2 loss: {loss2.mean().item():.2f}, Model3 loss: {loss3.mean().item():.2f}')
                print(f'Epoch: {epoch}, train loss: {loss.mean().item():.2f}, val accuracy: {acc * 100:.2f}')

        self.eval()

    def init_weights(self):
        for module in self.model.modules():
            if type(module) is nn.Linear:
                nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()
                # nn.init.kaiming_normal_(module.bias)    

class KNearestNeighbour:
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
            label_weight = np.sum(K_nearest_labels == label)

            if max_weight < label_weight:
                max_weight = label_weight
                prediction = label
        
        return prediction

## Helper Functions
to_np = lambda x: x.detach().cpu().numpy()

def load_set(prefix: str, filename: dict):
    data_types = {
        0x08: ('ubyte', 'B', 1),
        0x09: ('byte', 'b', 1),
        0x0B: ('>i2', 'h', 2),
        0x0C: ('>i4', 'i', 4),
        0x0D: ('>f4', 'f', 4),
        0x0E: ('>f8', 'd', 8)
    }

    imagesfile = open(prefix + filename['images'],'rb')
    labelsfile = open(prefix + filename['labels'],'rb')

    imagesfile.seek(0)
    labelsfile.seek(0)
    
    import struct as st
    images_magic = st.unpack('>4B', imagesfile.read(4))
    labels_magic = st.unpack('>4B', labelsfile.read(4))

    images_dataFormat = data_types[images_magic[2]][1]
    images_dataSize = data_types[images_magic[2]][2]
    labels_dataFormat = data_types[labels_magic[2]][1]
    labels_dataSize = data_types[labels_magic[2]][2]

    nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',imagesfile.read(4))[0] #num of column

    labelsfile.seek(8)

    images_array = np.zeros((nImg,nR,nC))
    labels_array = np.zeros((nImg, ))

    images_nBytesTotal = nImg * nR * nC * images_dataSize #since each pixel data is 1 byte
    labels_nBytesTotal = nImg * labels_dataSize #since each pixel data is 1 byte
    
    images_array = np.asarray(st.unpack('>' + images_dataFormat * images_nBytesTotal, imagesfile.read(images_nBytesTotal))).reshape((nImg,nR,nC))
    labels_array = np.asarray(st.unpack('>' + labels_dataFormat * labels_nBytesTotal, labelsfile.read(labels_nBytesTotal))).reshape((nImg, ))

    return images_array, labels_array

def split(train_imgs, train_labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.random.permutation(train_imgs.shape[0])

    t_indices = indices[:indices.shape[0] * 4 // 5]
    v_indices = indices[indices.shape[0] * 4 // 5:]

    val_imgs = train_imgs[v_indices, :, :]
    val_labels = train_labels[v_indices]

    train_imgs = train_imgs[t_indices, :, :]
    train_labels = train_labels[t_indices]

    return train_imgs, train_labels, val_imgs, val_labels

def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    import pathlib
    prefix = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'data/'
    train_filename = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
    test_filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
    
    train_imgs, train_labels = load_set(prefix, train_filename)
    test_imgs, test_labels = load_set(prefix, test_filename)

    return train_imgs, train_labels, test_imgs, test_labels

def save_model(model: FeedForward, suffix: str):
    import pathlib
    prefix = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'models/'

    torch.save(model.state_dict(), prefix + 'model_' + suffix + '.pt')

def load_model(model: FeedForward, suffix: str):
    import pathlib
    prefix = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'models/'

    model.load_state_dict(torch.load(prefix + 'model_' + suffix + '.pt'))

def preproces_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_imgs, train_labels, test_imgs, test_labels = load_dataset()
    train_imgs, train_labels, val_imgs, val_labels = split(train_imgs = train_imgs, train_labels = train_labels)

    return (
        torch.Tensor(train_imgs), torch.Tensor(train_labels).to(torch.long),
        torch.Tensor(val_imgs), torch.Tensor(val_labels).to(torch.long), 
        torch.Tensor(test_imgs), torch.Tensor(test_labels).to(torch.long)
        )

def main():
    train_imgs, train_labels, test_imgs, test_labels = load_dataset()

    plt.imshow(train_imgs[0])
    plt.show()
    print(f'Label: {train_labels[0]}')
    plt.imshow(test_imgs[0])
    plt.show()
    print(f'Label: {test_labels[0]}')

## Experiments
def Experiment1(Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):

    print('-' * 35 + f'Experiment 1' + '-' * 35)

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = Data

    in_dim = train_imgs.shape[1] * train_imgs.shape[2]

    # Set verbose = False if you do not wish to see training progress    
    model_1 = FeedForward(num_layers = 2, in_dim = in_dim, hidden_dim = 10, out_dim = 10, log = True, verbose = True)
    model_1 = model_1.to(DEVICE)

    model_2 = FeedForward(num_layers = 4, in_dim = in_dim, hidden_dim = 10, out_dim = 10, log = True, verbose = True)
    model_2 = model_2.to(DEVICE)

    print('2-Layer Model:')
    print(model_1)
    print('4-Layer Model:')
    print(model_2)

    print('Training Model with 2 hidden layers')
    model_1._train(data = (train_imgs, train_labels, val_imgs, val_labels), epochs = 10) #, optimizer = optimizer, scheduler = scheduler)
    
    print('Training Model with 4 hidden layers')
    model_2._train(data = (train_imgs, train_labels, val_imgs, val_labels), epochs = 20) #, optimizer = optimizer, scheduler = scheduler)

    test_imgs = test_imgs.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    
    plt.figure()
    plt.plot(model_1.log_info['Accuracy'])

    plt.xlabel('Iterations')
    plt.ylabel('Model Accuracy')
    plt.title('2 Hidden Layers')

    plt.figure()
    plt.plot(model_2.log_info['Accuracy'])

    plt.xlabel('Iterations')
    plt.ylabel('Model Accuracy')
    plt.title('4 Hidden Layers')

    with torch.no_grad():
        pred = model_1(test_imgs.flatten(1))
        acc_1 = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(test_labels)) * 100

        pred = model_2(test_imgs.flatten(1))
        acc_2 = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(test_labels)) * 100
    
    df = pd.DataFrame(data = {'Num Hidden': ['2 Layers', '4 Layers'], 'Acuuracy': [acc_1, acc_2]})
    df.index = [''] * len(df)
    print('Test Accuracy after 25 Epochs:')
    print(df)

    print('-' * 82)

    save_model(model_1, '1')
    save_model(model_2, '2')

def Experiment2(Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    print('-' * 35 + f'Experiment 2' + '-' * 35)

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = Data
    
    in_dim = train_imgs.shape[1] * train_imgs.shape[2]
    
    model_1 = FeedForward(num_layers = 2, in_dim = in_dim, hidden_dim = 10, out_dim = 10, log = True, verbose = True)
    model_1 = model_1.cuda()

    model_2 = FeedForward(num_layers = 4, in_dim = in_dim, hidden_dim = 10, out_dim = 10, log = True, verbose = True)
    model_2 = model_2.cuda()

    load_model(model_1, '1')
    load_model(model_2, '2')

    test_imgs = test_imgs.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    # Generate embeddings of training data 
    with torch.no_grad():
        v1 = to_np(model_1.generate_embedding(train_imgs.flatten(1).to(DEVICE)))
        v2 = to_np(model_2.generate_embedding(train_imgs.flatten(1).to(DEVICE)))

    # Generate embeddings of test data
    with torch.no_grad():
        v1_test = to_np(model_1.generate_embedding(test_imgs.flatten(1).to(DEVICE)))
        v2_test = to_np(model_2.generate_embedding(test_imgs.flatten(1).to(DEVICE)))

    knn1 = KNearestNeighbour(15, v1, to_np(train_labels))
    knn2 = KNearestNeighbour(15, v2, to_np(train_labels))

    pred1 = knn1.predict(v1_test)
    pred2 = knn2.predict(v2_test)

    acc_1 = Metrics.accuracy(pred1, to_np(test_labels)) * 100
    acc_2 = Metrics.accuracy(pred2, to_np(test_labels)) * 100

    df = pd.DataFrame(data = {'Num Hidden': ['2 Layers', '4 Layers'], 'Acuuracy': [acc_1, acc_2]})
    df.index = [''] * len(df)
    print('K Nearest Neighbour Accuracy:')
    print(df)

    print('-' * 82)

if __name__ == '__main__':
    Data = preproces_data()
    
    Experiment1(Data)
    Experiment2(Data)
