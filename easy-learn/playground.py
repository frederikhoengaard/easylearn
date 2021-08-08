from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


class ANN(nn.Module):
    def __init__(self):
       super().__init__()
       self.fc1 = nn.Linear(in_features=4, out_features=16)
       self.fc2 = nn.Linear(in_features=16, out_features=12)
       self.output = nn.Linear(in_features=12, out_features=3)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

    def fit(self):
        pass

    def predict(self):
        pass

def main():
    data = load_iris()
    features, labels = data.data, data.target
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print(features)

if __name__ == '__main__':
    main()