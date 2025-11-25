# Import Libraries and Prepare Data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

t = np.linspace(0, 100, 1000)
data = np.sin(t)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data, seq_length)

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

print(X.shape)        # (990, 10)
print(y.shape)        # (990,)
print(trainX.shape)   # torch.Size([990, 10, 1])
print(trainY.shape)   # torch.Size([990, 1])

# See the first input-target pair
print("first numpy array x:", X[0])        # 10 values: data[0:10]
print("first numpy array y:", y[0])        # the 11th value: data[10]
print("first tensor x:", trainX[0])   # 10 values: data[0:10]
print("first tensor y:", trainY[0])   # the 11th value: data[10]

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        


