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
EPOCHS = 15
BATCH_SIZE = 30
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

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        stride: int = 1,
        norm_layer: Callable[..., nn.Module] = None,
        is_res: bool = False
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_dim, embed_dim, 3, stride, padding = 1)
        self.bn1 = norm_layer(embed_dim)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, stride, padding = 1)
        self.bn2 = norm_layer(embed_dim)
        self.stride = stride

        self.pool = nn.MaxPool2d(2)
        
        self.is_res = is_res
        # self.downsample = lambda x: x[:, :, 2:-2, 2:-2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.is_res: out += identity
        out = self.relu(out)

        out = self.pool(out)

        return out
    
class CNN_Vanilla(nn.Module):
    def __init__(self, in_channel, embed_channel) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels = in_channel, out_channels = embed_channel, kernel_size = 3, padding = 1)
        L = [BasicBlock(in_dim = embed_channel, embed_dim = embed_channel, is_res = False) for i in range(3)]
        self.model = nn.Sequential(*L)
        self.fc1 = nn.Linear(4 * 4 * embed_channel, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor):
        relu = nn.LeakyReLU()

        x = relu(self.init_conv(x))
        x = self.model(x)
        x = x.flatten(start_dim = 1)
        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class CNN_Resnet(nn.Module):
    def __init__(self, in_channel, embed_channel) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels = in_channel, out_channels = embed_channel, kernel_size = 3, padding = 1)
        L = [BasicBlock(in_dim = embed_channel, embed_dim = embed_channel, is_res = True) for i in range(3)]
        self.model = nn.Sequential(*L)
        self.fc1 = nn.Linear(4 * 4 * embed_channel, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor):
        relu = nn.LeakyReLU()

        x = relu(self.init_conv(x))
        x = self.model(x)
        x = x.flatten(start_dim = 1)
        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN_Resnet_mod1(nn.Module):
    def __init__(self, in_channel, embed_channel) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels = in_channel, out_channels = embed_channel, kernel_size = 3, padding = 1)
        self.model = nn.Sequential(*[BasicBlock(in_dim = embed_channel, embed_dim = embed_channel, is_res = True) for i in range(4)])
        self.fc1 = nn.Linear(2 * 2 * embed_channel, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor):
        relu = nn.ReLU()

        x = relu(self.init_conv(x))
        x = self.model(x)
        x = x.flatten(start_dim = 1)
        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN_Resnet_mod2(nn.Module):
    def __init__(self, in_channel, embed_channel) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels = in_channel, out_channels = embed_channel, kernel_size = 3, padding = 1)
        self.model = nn.Sequential(*[BasicBlock(in_dim = embed_channel, embed_dim = embed_channel, is_res = True) for i in range(3)])
        self.fc1 = nn.Linear(4 * 4 * embed_channel, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor):
        relu = nn.ReLU()

        x = relu(self.init_conv(x))
        x = self.model(x)
        x = x.flatten(start_dim = 1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)

        return x

## Helper Functions
to_np = lambda x: x.detach().cpu().numpy()

def load_dataset(normalize: bool = False) -> Tuple[Dataset, Dataset, Dataset]:
    
    import pathlib
    prefix = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'data/'

    L = [transforms.ToTensor(),
     ]
    
    if normalize:
        L.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    transform = transforms.Compose(L
    )
    
    train_dataset = torchvision.datasets.CIFAR10(root = prefix, train = True, transform = transform, download = True)
    test_dataset = torchvision.datasets.CIFAR10(root = prefix, train = False, transform = transform, download = True)

    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])

    return train_dataset, val_dataset, test_dataset

def save_model(model: nn.Module, suffix: str):
    import pathlib
    prefix = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'models/'

    torch.save(model.state_dict(), prefix + 'model_' + suffix + '.pt')

def load_model(model: nn.Module, suffix: str):
    import pathlib
    prefix = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'models/'

    model.load_state_dict(torch.load(prefix + 'model_' + suffix + '.pt'))

def preproces_data(normalize: bool = False, batch_size = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = load_dataset(normalize)

    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size, True), DataLoader(val_dataset, len(val_dataset), True), DataLoader(test_dataset, len(test_dataset), True)

    return (
        train_dataloader, val_dataloader, test_dataloader
        )

## Experiments
def Experiment1(Data: Tuple[DataLoader, DataLoader, DataLoader]):

    print('-' * 35 + f'Experiment 1' + '-' * 35)

    train_loader, val_loader, test_loader = Data
    
    v_cnn = CNN_Vanilla(3, 8).to(DEVICE)
    
    optimizer = torch.optim.NAdam(v_cnn.parameters(), lr = 0.005)
    criterion = nn.CrossEntropyLoss()

    print('Training Vanilla CNN:')
    
    accs = []
    for i in range(EPOCHS):
        v_cnn.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = v_cnn(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(v_cnn.parameters(), 1)
            optimizer.step()
        
        v_cnn.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = v_cnn(img)

                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc * 100)

    plt.figure()
    plt.plot(accs)
    plt.title('Vanilla CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    r_cnn = CNN_Resnet(3, 8).to(DEVICE)
    
    optimizer = torch.optim.NAdam(r_cnn.parameters(), lr = 0.005)
    criterion = nn.CrossEntropyLoss()

    print('Training Residual CNN:')
    
    accs = []
    for i in range(EPOCHS):
        r_cnn.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = r_cnn(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(r_cnn.parameters(), 1)
            optimizer.step()
        
        r_cnn.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn(img)
                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc * 100)

    plt.figure()
    plt.plot(accs)
    plt.title('Residual CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    count_params= lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Number of parameters in v_cnn: {count_params(v_cnn)}')
    print(f'Number of parameters in r_cnn: {count_params(r_cnn)}')

    print('-' * 82)

    save_model(v_cnn, 'v')
    save_model(r_cnn, 'r')

def Experiment2(Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    print('-' * 35 + f'Experiment 2' + '-' * 35)

    train_loader, val_loader, test_loader = Data

    r_cnn = CNN_Resnet(3, 8).to(DEVICE)
    
    optimizer = torch.optim.NAdam(r_cnn.parameters(), lr = 0.005)
    criterion = nn.CrossEntropyLoss()

    print('Training Residual CNN on Normalized Dataset:')
    
    accs = []
    for i in range(EPOCHS):
        r_cnn.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = r_cnn(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(r_cnn.parameters(), 1)
            optimizer.step()
        
        r_cnn.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn(img)
                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc * 100)

    plt.figure()
    plt.plot(accs)
    plt.title('Residual CNN with Data Normalization')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    print('-' * 82)

    save_model(r_cnn, 'r_normalized')

def Experiment3(Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    print('-' * 35 + f'Experiment 3' + '-' * 35)

    train_loader, val_loader, test_loader = Data

    r_cnn = CNN_Resnet(3, 8).to(DEVICE)
    
    # optimizer = torch.optim.NAdam(r_cnn.parameters(), lr = 0.005)
    optimizer =  torch.optim.SGD(r_cnn.parameters(), lr = 0.005)
    criterion = nn.CrossEntropyLoss()

    print('Training Residual CNN using SGD:')
    
    accs = []
    for i in range(10):
        r_cnn.train()
        for (img, label) in train_loader:
            batch_size = img.shape[0]
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            for j in range(batch_size):
                pred = r_cnn(img[j, :, :, :][None, :, :, :])

                loss = criterion(pred[0, :], label[j])
                
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(r_cnn.parameters(), 1)
                optimizer.step()
        
        r_cnn.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn(img)
                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc * 100)

    plt.figure()
    plt.plot(accs)
    plt.title('SGD')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    r_cnn = CNN_Resnet(3, 8).to(DEVICE)
    
    optimizer =  torch.optim.SGD(r_cnn.parameters(), lr = 0.005)
    print('Training Residual CNN using Minibatch GD without momentum:')
    
    accs = []
    for i in range(EPOCHS):
        r_cnn.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = r_cnn(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(r_cnn.parameters(), 1)
            optimizer.step()
        
        r_cnn.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn(img)
                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc)
    
    plt.figure()
    plt.plot(accs)
    plt.title('Mini Batch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    r_cnn = CNN_Resnet(3, 8).to(DEVICE)
    
    optimizer =  torch.optim.SGD(r_cnn.parameters(), lr = 0.005, momentum = 0.9)
    print('Training Residual CNN using Minibatch GD with momentum:')
    
    accs = []
    for i in range(EPOCHS):
        r_cnn.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = r_cnn(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(r_cnn.parameters(), 1)
            optimizer.step()
        
        r_cnn.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn(img)
                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc)

    plt.figure()
    plt.plot(accs)
    plt.title('Mini Batch with Momentum')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    print('-' * 82)

def Experiment4(Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    print('-' * 35 + f'Experiment 4' + '-' * 35)

    train_loader, val_loader, test_loader = Data
    
    r_cnn1 = CNN_Resnet_mod1(3, 8).to(DEVICE)
    
    optimizer = torch.optim.NAdam(r_cnn1.parameters(), lr = 0.005)
    criterion = nn.CrossEntropyLoss()

    print('Training R CNN (a):')
    
    accs = []
    for i in range(EPOCHS):
        r_cnn1.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = r_cnn1(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(v_cnn.parameters(), 1)
            optimizer.step()
        
        r_cnn1.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn1(img)

                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc * 100)

    plt.figure()
    plt.plot(accs)
    plt.title('Modification 1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    r_cnn2 = CNN_Resnet_mod2(3, 8).to(DEVICE)
    
    optimizer = torch.optim.NAdam(r_cnn2.parameters(), lr = 0.005)
    criterion = nn.CrossEntropyLoss()

    print('Training R CNN(b):')
    
    accs = []
    for i in range(EPOCHS):
        r_cnn2.train()
        for (img, label) in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = r_cnn2(img)

            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(r_cnn.parameters(), 1)
            optimizer.step()
        
        r_cnn2.eval()

        with torch.no_grad():
            for (img, label) in test_loader:
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                
                pred = r_cnn2(img)
                acc = Metrics.accuracy(to_np(torch.argmax(pred, dim = 1)), to_np(label))

        print(f'Epoch: {i}/{EPOCHS} Accuracy: {round(acc * 100, 2)}%')
        accs.append(acc)

    plt.figure()
    plt.plot(accs)
    plt.title('Modification 2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    count_params= lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Number of parameters in r_cnn(a): {count_params(r_cnn1)}')
    print(f'Number of parameters in r_cnn(b): {count_params(r_cnn2)}')

    print('-' * 82)

    save_model(r_cnn1, 'r1')
    save_model(r_cnn2, 'r2')

if __name__ == '__main__':
    Data = preproces_data()
    
    Experiment1(Data)

    Data = preproces_data(normalize = True)
    Experiment2(Data)

    Experiment3(Data)

    Experiment4(Data)

    plt.show()

