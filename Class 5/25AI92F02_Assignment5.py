# %% [markdown]
# # Installs

# %%
# ! pip install -q ucimlrepo

# %% [markdown]
# # Imports

# %%
# Basic libraries
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Framework
import torch

# Data and its management related libraries
from torch.utils.data import Dataset, DataLoader, TensorDataset

# For creating neural networks
import torch.nn as nn
import torch.nn.functional as F

# For optimizing neural networks
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# For metrics and analysis
from sklearn.metrics import r2_score, mean_squared_error

# For dataset loading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from ucimlrepo import fetch_ucirepo

from tqdm.notebook import tqdm, trange

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device in Use:",device)

# %%
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# # Load dataset

# %%
df = fetch_ucirepo(id=464).data.original
print(f"Dataset shape: {df.shape}")

# %%
X = df.drop(columns=['critical_temp'])
y = df['critical_temp']

# %% [markdown]
# # Dataset splitting

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

len(X_train), len(X_test), len(X_val), len(X)

# %% [markdown]
# # Scale data

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # Dataloaders

# %%
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_torch = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_torch = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# %%
train_dataset = TensorDataset(X_train_torch, y_train_torch)
val_dataset = TensorDataset(X_val_torch, y_val_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)

# Define batch size
batch_size = 64

# Create DataLoaders for training, validation, and testing data
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# # Batch Normalization Implementation

# %%
class BatchNorm(nn.Module) :

    def __init__(self, num_features, epsilon=1e-5, momentum=0.1) :
        super().__init__()
        
        self.epsilon = epsilon
        self.momentum = momentum

        # Trainable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Non-trainable parameters
        self.register_buffer('running_avg', torch.zeros(num_features))
        self.register_buffer('running_variance', torch.ones(num_features))

    def forward(self, X) :

        # Compute batch statistics and running statistics during training
        # The running statistics will be used during evaluation
        if self.training :
            mean = X.mean(dim=0)
            var = X.var(dim=0)

            self.running_avg = (1 - self.momentum) * self.running_avg + self.momentum * mean.detach()
            self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * var.detach()

        else :
            mean = self.running_avg
            var = self.running_variance

        # Normalize the batch
        X_normalized = (X - mean) / torch.sqrt(var + self.epsilon)

        # Scale and shift
        y = self.gamma * X_normalized + self.beta

        return y

# %% [markdown]
# # Model

# %%
class NNwithScratchBatchNorm(nn.Module) :

    def __init__(self):
        super(NNwithScratchBatchNorm, self).__init__()

        self.fc1 = nn.Linear(X.shape[1], 256)
        self.bn1 = BatchNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = BatchNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = BatchNorm(64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x) :
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# %% [markdown]
# # Train loop

# %%
def train_model(model, train_dataset, val_dataset, test_dataset, batch_sizes) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    train_loss_history = {}
    val_loss_history = {}
    test_loss_history = {}
    preds_labels_history = {}

    for batch_size in batch_sizes :
        print(f"\nTraining with batch size: {batch_size}")

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []

        for epoch in trange(epochs) :
            model.train()
            running_loss = 0.0
            total = 0
            for inputs, labels in tqdm(trainloader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

            train_loss = running_loss / total
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(valloader, leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    total += inputs.size(0)

            val_loss /= total
            val_losses.append(val_loss)

            if (epoch + 1) % 20 == 0:
                tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(testloader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                r2 = r2_score(all_labels, all_preds)
                mse = mean_squared_error(all_labels, all_preds)

        train_loss_history[batch_size] = train_losses
        val_loss_history[batch_size] = val_losses
        test_loss_history[batch_size] = {'r2': r2, 'mse': mse}
        preds_labels_history[batch_size] = {'preds': all_preds, 'labels': all_labels}

    return train_loss_history, val_loss_history, test_loss_history, preds_labels_history

# %%
batch_sizes = [4, 8, 16, 32, 64, 128, 256]

model = NNwithScratchBatchNorm()
train_loss_history, val_loss_history, test_loss_history, preds_labels_history = train_model(model, train_dataset, val_dataset, test_dataset, batch_sizes)

# %% [markdown]
# # Plots

# %%
# Plot in a single graph, train and val loss for all batches
plt.figure(figsize=(12, 8))
for batch_size in batch_sizes:
    plt.plot(train_loss_history[batch_size], label=f'Train Loss (Batch {batch_size})')
    plt.plot(val_loss_history[batch_size], label=f'Val Loss (Batch {batch_size})')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss for Different Batch Sizes')
plt.legend()
plt.show()

# %%
# Tabulate test MSE and R2 for all batch sizes
test_results = pd.DataFrame(test_loss_history).T
test_results.columns = ['R2', 'MSE']
test_results.index.name = 'Batch Size'
test_results

# %%
# Plot R2 score against batch sizes
plt.figure(figsize=(10, 6))
plt.plot(test_results.index, test_results['R2'], marker='o')
plt.xlabel('Batch Size')
plt.ylabel('R2 Score')
plt.title('R2 Score for Different Batch Sizes')
plt.grid(True)
plt.show()

# %% [markdown]
# The R2 score increased as batch size increased as the average over a larger batch size leads to more generalisation over the whole dataset rather than a local group of points.
# 
# Thus, the lowest batch size of 4 got the lowest R2 score and the largest batch size of 256 had the best score

# %%
best_batch_size = test_results['R2'].idxmax()
print(f"Best Batch Size: {best_batch_size}, R2 Score: {test_results.loc[best_batch_size, 'R2']:.4f}")

# Scatter plot of predictions vs true labels for the best batch size
best_preds = preds_labels_history[best_batch_size]['preds']
best_labels = preds_labels_history[best_batch_size]['labels']

plt.figure(figsize=(8, 8))
plt.scatter(best_labels, best_preds, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.title(f'Predictions vs True Labels (Batch Size {best_batch_size})')
plt.grid(True)
plt.show()

# %% [markdown]
# # PyTorch implementation

# %%
class NNwithPyTorchBatchNorm(nn.Module) :

    def __init__(self):
        super(NNwithPyTorchBatchNorm, self).__init__()

        self.fc1 = nn.Linear(X.shape[1], 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x) :
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# %%
batch_sizes = [int(best_batch_size)]

model = NNwithPyTorchBatchNorm()
train_loss_history, val_loss_history, test_loss_history, preds_labels_history = train_model(model, train_dataset, val_dataset, test_dataset, batch_sizes)

# %%
print(f"PyTorch R2 Score: {test_loss_history[int(best_batch_size)]['r2']:.4f}")

# %%
# Scatter plot
best_preds = preds_labels_history[int(best_batch_size)]['preds']
best_labels = preds_labels_history[int(best_batch_size)]['labels']

plt.figure(figsize=(8, 8))
plt.scatter(best_labels, best_preds, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.title(f'Predictions vs True Labels')
plt.grid(True)
plt.show()


