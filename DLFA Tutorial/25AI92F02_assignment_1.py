# %% [markdown]
# # Installs

# %%
# ! pip install -q ucimlrepo fvcore torch-summary ptflops

# %% [markdown]
# # Imports

# %%
# Basic libraries
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Hyperparameter tuning
import optuna

# Framework
import torch
import torchvision

# Data and its management related libraries
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

# For creating neural networks
import torch.nn as nn
import torch.nn.functional as F
import torchview
from torchviz import make_dot

# For optimizing neural networks
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# For metrics and analysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torchsummary import summary

# For dataset loading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from ucimlrepo import fetch_ucirepo
from tqdm.notebook import tqdm, trange

from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device in Use:",device)

# %% [markdown]
# # Set Seed

# %%
torch.manual_seed(2026)
np.random.seed(2026)

# %% [markdown]
# # Load dataset

# %%
df = fetch_ucirepo(id=189).data.original
print(f"Dataset shape: {df.shape}")

# %% [markdown]
# # EDA

# %%
df.info()

# %% [markdown]
# Dropping `subject#` because that is just the subject ID

# %%
df.drop(columns=['subject#'], inplace=True)

# %% [markdown]
# Only checking for duplicates as from `df.info()` it is evident that there are no missing values.

# %%
print(f"Duplicate rows in the dataset: {df.duplicated().sum()}")
df = df.drop_duplicates()

# %% [markdown]
# One hot encoding sex as that is the only categorical field (albeit with integer values)

# %%
df = pd.get_dummies(df, columns=['sex'], dtype=int)
df.info()

# %% [markdown]
# # Dataset splitting

# %%
X = df.drop(columns=['motor_UPDRS', 'total_UPDRS'])
y = df[['motor_UPDRS', 'total_UPDRS']]

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

len(X_train), len(X_test), len(X_val), len(X)

# %% [markdown]
# ## Scale data

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Convert to tensors

# %%
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32)
X_val_torch = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_torch = torch.tensor(y_val.values, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32)

# %% [markdown]
# ## Create dataloaders

# %%
# Create TensorDatasets for training, validation, and testing data
train_dataset = TensorDataset(X_train_torch, y_train_torch)
val_dataset = TensorDataset(X_val_torch, y_val_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)

# Define batch size
batch_size = 512

# Create DataLoaders for training, validation, and testing data
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# # Define model

# %% [markdown]
# No activation function is used for any of the models in the output layer as the task is a regression task. ReLU possibly could have been used as the output is always positive but its better to not use any at all.

# %%
class ParkNet_1(nn.Module):
    def __init__(self):
        super(ParkNet_1, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ParkNet_2(nn.Module):
    def __init__(self):
        super(ParkNet_2, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class ParkNet_3(nn.Module):
    def __init__(self):
        super(ParkNet_3, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class ParkNet_4(nn.Module):
    def __init__(self):
        super(ParkNet_4, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# %% [markdown]
# # Training

# %% [markdown]
# ## Train Loop

# %%
def get_accuracy(output, target, batch_size):
    corrects = (output == target).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

# %%
def fit(model, device, learning_rate, trainloader, valloader, testloader, epochs=30, early_stopping_patience=10):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Lists for saving history
    train_loss_history = []
    val_loss_history = []

    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in trange(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, labels in tqdm(trainloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zeroing the Gradients
            outputs = model(inputs) # Forward Pass
            loss = criterion(outputs, labels) # Computing the Loss
            loss.backward() # Backward Pass
            optimizer.step() # Updating the Model Parameters
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        train_loss = running_loss / total
        train_loss_history.append(train_loss)

        # Validation phase
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
        val_loss_history.append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                tqdm.write("Early stopping triggered.")
                break

    return train_loss_history, val_loss_history

# %% [markdown]
# ## Hyperparameter Tuning

# %%
model_1 = ParkNet_1().to(device)
model_2 = ParkNet_2().to(device)
model_3 = ParkNet_3().to(device)
model_4 = ParkNet_4().to(device)

# %%
def objective(trial) :
    model_choice = trial.suggest_categorical('model_choice', ['ParkNet_1', 'ParkNet_2', 'ParkNet_3', 'ParkNet_4'])

    if model_choice == 'ParkNet_1':
        model = model_1
        lr = trial.suggest_float('lr', 1e-6, 1e-2)
        epochs = trial.suggest_int('epochs', 30, 300)
    elif model_choice == 'ParkNet_2':
        model = model_2
        lr = trial.suggest_float('lr', 1e-6, 1e-2)
        epochs = trial.suggest_int('epochs', 30, 300)
    elif model_choice == 'ParkNet_3':
        model = model_3
        lr = trial.suggest_float('lr', 1e-6, 1e-2)
        epochs = trial.suggest_int('epochs', 30, 300)
    elif model_choice == 'ParkNet_4':
        model = model_4
        lr = trial.suggest_float('lr', 1e-6, 1e-2)
        epochs = trial.suggest_int('epochs', 30, 300)
        
    _, val_loss_history = fit(model, device, lr, trainloader, valloader, testloader, epochs=epochs, early_stopping_patience=10)
    return val_loss_history[-1]

# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

# %%
print("Best hyperparameters: ", study.best_params)
print("Best validation loss: ", study.best_value)

# %% [markdown]
# # Model info

# %% [markdown]
# Select the best model

# %%
model_choice = study.best_params['model_choice']
if model_choice == 'ParkNet_1':
    model = ParkNet_1().to(device)
elif model_choice == 'ParkNet_2':
    model = ParkNet_2().to(device)
elif model_choice == 'ParkNet_3':
    model = ParkNet_3().to(device)
elif model_choice == 'ParkNet_4':
    model = ParkNet_4().to(device)

best_lr = study.best_params['lr']
best_epochs = study.best_params['epochs']

# %%
train_loss_history, val_loss_history = fit(model, device, best_lr, trainloader, valloader, testloader, epochs=best_epochs, early_stopping_patience=10)

# %%
for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")

# %%
summary(model, input_data=(1, X.shape[1]))

# %% [markdown]
# ## Number of Computations

# %% [markdown]
# ### MACs

# %%
macs, params = get_model_complexity_info(model, (1, X.shape[1]), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=True, verbose=True)
# MACs → Multiply–Accumulate operations, how much computation the model needs, Parameters → how much memory the model needs
# print_per_layer_stat=True - function to show complexity layer by layer
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# %% [markdown]
# ### FLOPs

# %%
random_tensor=torch.randn(1, X.shape[1]).to(device)

flops = FlopCountAnalysis(model, random_tensor)

print('Total FLOPs',flops.total())

print('FLOPs by Operator',flops.by_operator())

print('FLOPs by Module',flops.by_module())

print('FLOPs by Module and Operator',flops.by_module_and_operator())

# %%
print(flop_count_table(flops))

# %%
torchview.draw_graph(model, input_size=(1, X.shape[1]), device=device, graph_name="ParkNet", save_graph=True)
plt.figure(figsize=(5, 10))
plt.imshow(plt.imread('ParkNet.gv.png'))

# %%
# Create a dummy input tensor
dummy_input = torch.randn(1, X.shape[1]).to(device)

# Perform a forward pass to get the computational graph
output = model(dummy_input)

# Visualize the computational graph using torchviz
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('ParkNet_CompGraph')
plt.figure(figsize=(5, 10))
plt.imshow(plt.imread('ParkNet_CompGraph.png'))

# %% [markdown]
# # Plot

# %%
plt.figure(figsize=(10, 6))
plt.title('Loss curve')

plt.plot(train_loss_history, label=f'Train Loss')
plt.plot(val_loss_history, label=f'Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# # Test

# %%
def compute_r2_score( y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score.item()

# %%
model.eval()

total = 0
correct = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs
        correct += (preds == outputs).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_r2_score = compute_r2_score(torch.tensor(all_preds), torch.tensor(all_labels))

print(f'Test R2 Score: {test_r2_score:.4f}')

# %%
# Generate two scatter plots for two outputs from the model
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(all_labels[:, 0], all_preds[:, 0], alpha=0.5)
plt.xlabel('True motor_UPDRS')
plt.ylabel('Predicted motor_UPDRS')
plt.title('True vs Predicted Values motor_UPDRS')

plt.subplot(1, 2, 2)
plt.scatter(all_labels[:, 1], all_preds[:, 1], alpha=0.5)
plt.xlabel('True total_UPDRS')
plt.ylabel('Predicted total_UPDRS')
plt.title('True vs Predicted Values total_UPDRS')

plt.tight_layout()
plt.show()


