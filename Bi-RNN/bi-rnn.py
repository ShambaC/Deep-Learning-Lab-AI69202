# %% [markdown]
# ## Get all the imports

# %%
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(27)
random.seed(27)
np.random.seed(27)

sns.set_theme(style="ticks")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Download the dataset and split into train, validation and test

# %%
airpassengers_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data_dir = Path("./data_downloaded")
data_dir.mkdir(parents=True, exist_ok=True)
dataset_path = data_dir / "AirPassengers_fresh.csv"

# Fresh download for this notebook run (no dependency on repository datasets).
df = pd.read_csv(airpassengers_url)
df.to_csv(dataset_path, index=False)

df.columns = ["Month", "#Passengers"]
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

df_train = df.loc[df["Month"].dt.year < 1957].copy()
df_val = df.loc[(df["Month"].dt.year >= 1957) & (df["Month"].dt.year < 1959)].copy()
df_test = df.loc[df["Month"].dt.year >= 1959].copy()

print("Dataset saved to:", dataset_path.as_posix())
print("Training Split with number of samples:", len(df_train))
display(df_train.head())
print("\nValidation Split with number of samples:", len(df_val))
display(df_val.head())
print("\nTest Split with number of samples:", len(df_test))
display(df_test.head())

# %%
# Scale the values appropriately
scaler = MinMaxScaler()

# pandas>=3 enforces dtype safety; cast first so float assignments are valid
df_train = df_train.astype({"#Passengers": "float64"})
df_val = df_val.astype({"#Passengers": "float64"})
df_test = df_test.astype({"#Passengers": "float64"})

df_train.loc[:, "#Passengers"] = scaler.fit_transform(df_train[["#Passengers"]])[:, 0]
df_val.loc[:, "#Passengers"] = scaler.transform(df_val[["#Passengers"]])[:, 0]
df_test.loc[:, "#Passengers"] = scaler.transform(df_test[["#Passengers"]])[:, 0]

# %% [markdown]
# ## Check out the data trend

# %%
plt.figure(figsize=(12, 5))
sns.lineplot(
    x=np.arange(len(df_train)),
    y=scaler.inverse_transform(df_train[["#Passengers"]])[:, 0],
    label="Train Split"
)
sns.lineplot(
    x=np.arange(len(df_train), len(df_train) + len(df_val)),
    y=scaler.inverse_transform(df_val[["#Passengers"]])[:, 0],
    label="Val Split"
)

train_ticks = [i for idx, i in enumerate(df_train["Month"].dt.strftime("%Y-%m").tolist()) if idx % 5 == 0]
val_ticks = [i for idx, i in enumerate(df_val["Month"].dt.strftime("%Y-%m").tolist()) if (idx + 1) % 5 == 0] + [df_val["Month"].dt.strftime("%Y-%m").tolist()[-1]]
plt.xticks(np.arange(0, len(df_train) + len(df_val) + 1, 5), train_ticks + val_ticks, rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Number of Passengers", fontsize=14)
plt.legend()
plt.show()

# %% [markdown]
# ## Design the data module responsible for data processing and generating data loaders

# %%
class DataModule:
    def __init__(self, window_size, stride, batch_size):
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

    def convert_data_to_sequence_format(self, data):
        X, Y = list(), list()
        for i in range(0, len(data) - self.window_size, self.stride):
            X.append(data[i:i + self.window_size])
            Y.append(data[i + self.window_size])
        X, Y = np.asarray(X), np.asarray(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, Y

    def data_loader(self, X, Y, shuffle=True):
        dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

# %% [markdown]
# ## Design a Custom Bidirectional RNN cell

# %%
class BiRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation="tanh", bias=True):
        super(BiRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
        }[activation]

        self.input_layer_forward = torch.nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.hidden_layer_forward = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

        self.input_layer_backward = torch.nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.hidden_layer_backward = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def _run_direction(self, x, reverse=False):
        ht = torch.rand((x.shape[0], self.hidden_size), device=x.device)
        ht = ht / (torch.sqrt(torch.sum(torch.square(ht), dim=0)) + 1e-8)
        output = list()

        indices = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])
        for i in indices:
            if reverse:
                ht = self.activation(self.input_layer_backward(x[:, i]) + self.hidden_layer_backward(ht))
                output.insert(0, ht)
            else:
                ht = self.activation(self.input_layer_forward(x[:, i]) + self.hidden_layer_forward(ht))
                output.append(ht)

        output = torch.stack(output)
        output = torch.swapdims(output, 0, 1)
        return output

    def forward(self, x):
        forward_output = self._run_direction(x, reverse=False)
        backward_output = self._run_direction(x, reverse=True)
        return torch.cat([forward_output, backward_output], dim=-1)

# %% [markdown]
# ## Define the Custom Bidirectional RNN model

# %%
class CustomBiRNN(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, activation, bias, device):
        super(CustomBiRNN, self).__init__()
        self.num_layers = num_layers

        self.BiRNN_layers = list()
        for i in range(num_layers):
            if i == 0:
                self.BiRNN_layers.append(BiRNNCell(input_size, hidden_size, activation, bias).to(device))
            else:
                self.BiRNN_layers.append(BiRNNCell(2 * hidden_size, hidden_size, activation, bias).to(device))
        self.BiRNN_layers = torch.nn.Sequential(*self.BiRNN_layers)

        self.linear = torch.nn.Linear(2 * hidden_size, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.BiRNN_layers[i](x)
        x = self.linear(x[:, -1]).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.optimizer.zero_grad()
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), preds.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        return loss.item(), preds.detach().cpu().numpy()

# %% [markdown]
# ## Get the data loaders

# %%
window_size, stride = 24, 1
batch_size = 128 if device.type == "cuda" else 32

data_module = DataModule(window_size, stride, batch_size)

# Build one continuous series, then slice windows for train/val/test targets.
full_series = np.concatenate(
    [
        df_train["#Passengers"].values,
        df_val["#Passengers"].values,
        df_test["#Passengers"].values,
    ],
    axis=0,
 )

X_all, Y_all = data_module.convert_data_to_sequence_format(full_series)

n_train = len(df_train) - window_size
n_val = len(df_val)
n_test = len(df_test)

X_train = X_all[:n_train]
Y_train = Y_all[:n_train]
print("Train data shape:", X_train.shape, Y_train.shape)

X_val = X_all[n_train:n_train + n_val]
Y_val = Y_all[n_train:n_train + n_val]
print("Val data shape:", X_val.shape, Y_val.shape)

X_test = X_all[n_train + n_val:n_train + n_val + n_test]
Y_test = Y_all[n_train + n_val:n_train + n_val + n_test]
print("Test data shape:", X_test.shape, Y_test.shape)

train_data_loader = data_module.data_loader(X_train, Y_train, shuffle=True)
val_data_loader = data_module.data_loader(X_val, Y_val, shuffle=False)
test_data_loader = data_module.data_loader(X_test, Y_test, shuffle=False)

# %% [markdown]
# ## Perform training and validation with our custom Bidirectional RNN

# %%
num_layers = 2
input_size = X_train.shape[-1]
hidden_size = 256 if device.type == "cuda" else 96
num_epochs = 350

model = CustomBiRNN(num_layers, input_size, hidden_size, "tanh", True, device).to(device)

train_epoch_loss = list()
val_epoch_loss = list()
min_val_loss = np.inf

for epoch in tqdm(range(num_epochs), desc="Training Custom BiRNN"):
    train_iter_loss = list()
    model.train()
    for batch_idx, batch in enumerate(train_data_loader):
        batch = (batch[0].to(device), batch[1].to(device))
        loss, _ = model.training_step(batch, batch_idx)
        train_iter_loss.append(loss)
    train_epoch_loss.append(sum(train_iter_loss) / len(train_iter_loss))

    val_iter_loss = list()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_data_loader):
            batch = (batch[0].to(device), batch[1].to(device))
            loss, _ = model.validation_step(batch, batch_idx)
            val_iter_loss.append(loss)
    val_epoch_loss.append(sum(val_iter_loss) / len(val_iter_loss))

    if val_epoch_loss[-1] < min_val_loss:
        min_val_loss = val_epoch_loss[-1]
        best_custom_model = deepcopy(model)

print("Best Validation Loss (Custom BiRNN):", min_val_loss)

# %% [markdown]
# ## Plot the training and validation losses

# %%
sns.lineplot(x=np.arange(num_epochs), y=train_epoch_loss, label="Train loss")
sns.lineplot(x=np.arange(num_epochs), y=val_epoch_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Get all the predictions on the train, validation and test sets

# %%
def predict_model(model, data_loader, device):
    model.eval()
    total_predictions = list()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            batch = (batch[0].to(device), batch[1].to(device))
            _, preds = model.validation_step(batch, batch_idx)
            total_predictions.extend(preds)
    return np.asarray(total_predictions)

train_eval_loader = data_module.data_loader(X_train, Y_train, shuffle=False)
custom_train_predictions = predict_model(best_custom_model, train_eval_loader, device)
custom_val_predictions = predict_model(best_custom_model, val_data_loader, device)
custom_test_predictions = predict_model(best_custom_model, test_data_loader, device)

custom_train_predictions_inv = scaler.inverse_transform(custom_train_predictions.reshape(-1, 1))[:, 0]
custom_val_predictions_inv = scaler.inverse_transform(custom_val_predictions.reshape(-1, 1))[:, 0]
custom_test_predictions_inv = scaler.inverse_transform(custom_test_predictions.reshape(-1, 1))[:, 0]

Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))[:, 0]
Y_val_inv = scaler.inverse_transform(Y_val.reshape(-1, 1))[:, 0]
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))[:, 0]

custom_train_mse = np.mean(np.square(custom_train_predictions_inv - Y_train_inv))
custom_val_mse = np.mean(np.square(custom_val_predictions_inv - Y_val_inv))
custom_test_mse = np.mean(np.square(custom_test_predictions_inv - Y_test_inv))

print("Custom BiRNN Training MSE:", custom_train_mse)
print("Custom BiRNN Validation MSE:", custom_val_mse)
print("Custom BiRNN Test MSE:", custom_test_mse)

# %% [markdown]
# ## Plot the ground truth and Custom BiRNN predictions

# %%
plt.figure(figsize=(12, 5))

y_train_plot = scaler.inverse_transform(df_train[["#Passengers"]].iloc[window_size:])[:, 0]
sns.lineplot(x=np.arange(len(y_train_plot)), y=y_train_plot, label="Train Ground Truth", color="#4c72b0")
sns.lineplot(x=np.arange(len(y_train_plot)), y=custom_train_predictions_inv, label="Train Predictions", color="#4c72b0", linestyle="--")

y_val_plot = scaler.inverse_transform(df_val[["#Passengers"]])[:, 0]
sns.lineplot(
    x=np.arange(len(y_train_plot), len(y_train_plot) + len(df_val)),
    y=y_val_plot,
    label="Val Ground Truth",
    color="#dd8452"
)
sns.lineplot(
    x=np.arange(len(y_train_plot), len(y_train_plot) + len(df_val)),
    y=custom_val_predictions_inv,
    label="Val Predictions",
    color="#dd8452",
    linestyle="--"
)

y_test_plot = scaler.inverse_transform(df_test[["#Passengers"]])[:, 0]
sns.lineplot(
    x=np.arange(len(y_train_plot) + len(df_val), len(y_train_plot) + len(df_val) + len(df_test)),
    y=y_test_plot,
    label="Test Ground Truth",
    color="red"
)
sns.lineplot(
    x=np.arange(len(y_train_plot) + len(df_val), len(y_train_plot) + len(df_val) + len(df_test)),
    y=custom_test_predictions_inv,
    label="Test Predictions",
    color="red",
    linestyle="--"
)

train_ticks = [i for idx, i in enumerate(df_train["Month"].dt.strftime("%Y-%m").tolist()[window_size:]) if idx % 5 == 0]
val_ticks = [i for idx, i in enumerate(df_val["Month"].dt.strftime("%Y-%m").tolist()) if (idx + 1) % 5 == 0] + [df_val["Month"].dt.strftime("%Y-%m").tolist()[-1]]
test_ticks = [i for idx, i in enumerate(df_test["Month"].dt.strftime("%Y-%m").tolist()) if (idx + 1) % 5 == 0] + [df_test["Month"].dt.strftime("%Y-%m").tolist()[-1]]
plt.xticks(np.arange(0, len(y_train_plot) + len(df_val) + len(df_test) + 1, 5), train_ticks + val_ticks + test_ticks, rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Number of Passengers", fontsize=14)
plt.legend()
plt.show()

# %% [markdown]
# ## Define the PyTorch Bidirectional RNN model

# %%
class PyTorchBiRNN(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, activation, bias, device):
        super(PyTorchBiRNN, self).__init__()
        self.RNN = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=activation,
            bias=bias,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
            device=device
        )

        self.linear = torch.nn.Linear(2 * hidden_size, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        num_directions = 2
        h0 = torch.rand((self.RNN.num_layers * num_directions, x.shape[0], self.RNN.hidden_size), device=x.device)
        h0 = h0 / (torch.sqrt(torch.sum(torch.square(h0), dim=0)) + 1e-8)
        x, _ = self.RNN(x, h0)
        x = self.linear(x[:, -1]).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.optimizer.zero_grad()
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), preds.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        return loss.item(), preds.detach().cpu().numpy()

# %% [markdown]
# ## Perform training and validation using PyTorch Bidirectional RNN

# %%
pt_model = PyTorchBiRNN(num_layers, input_size, hidden_size, "tanh", True, device).to(device)

pt_train_epoch_loss = list()
pt_val_epoch_loss = list()
pt_min_val_loss = np.inf

for epoch in tqdm(range(num_epochs), desc="Training PyTorch BiRNN"):
    train_iter_loss = list()
    pt_model.train()
    for batch_idx, batch in enumerate(train_data_loader):
        batch = (batch[0].to(device), batch[1].to(device))
        loss, _ = pt_model.training_step(batch, batch_idx)
        train_iter_loss.append(loss)
    pt_train_epoch_loss.append(sum(train_iter_loss) / len(train_iter_loss))

    val_iter_loss = list()
    pt_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_data_loader):
            batch = (batch[0].to(device), batch[1].to(device))
            loss, _ = pt_model.validation_step(batch, batch_idx)
            val_iter_loss.append(loss)
    pt_val_epoch_loss.append(sum(val_iter_loss) / len(val_iter_loss))

    if pt_val_epoch_loss[-1] < pt_min_val_loss:
        pt_min_val_loss = pt_val_epoch_loss[-1]
        best_pt_model = deepcopy(pt_model)

print("Best Validation Loss (PyTorch BiRNN):", pt_min_val_loss)

# %% [markdown]
# ## Plot the training and validation losses

# %%
sns.lineplot(x=np.arange(num_epochs), y=pt_train_epoch_loss, label="Train loss")
sns.lineplot(x=np.arange(num_epochs), y=pt_val_epoch_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluate PyTorch Bidirectional RNN and compare both models

# %%
pt_train_predictions = predict_model(best_pt_model, train_eval_loader, device)
pt_val_predictions = predict_model(best_pt_model, val_data_loader, device)
pt_test_predictions = predict_model(best_pt_model, test_data_loader, device)

pt_train_predictions_inv = scaler.inverse_transform(pt_train_predictions.reshape(-1, 1))[:, 0]
pt_val_predictions_inv = scaler.inverse_transform(pt_val_predictions.reshape(-1, 1))[:, 0]
pt_test_predictions_inv = scaler.inverse_transform(pt_test_predictions.reshape(-1, 1))[:, 0]

pt_train_mse = np.mean(np.square(pt_train_predictions_inv - Y_train_inv))
pt_val_mse = np.mean(np.square(pt_val_predictions_inv - Y_val_inv))
pt_test_mse = np.mean(np.square(pt_test_predictions_inv - Y_test_inv))

print("PyTorch BiRNN Training MSE:", pt_train_mse)
print("PyTorch BiRNN Validation MSE:", pt_val_mse)
print("PyTorch BiRNN Test MSE:", pt_test_mse)

comparison_df = pd.DataFrame({
    "Metric": ["Train MSE", "Validation MSE", "Test MSE"],
    "Custom BiRNN": [custom_train_mse, custom_val_mse, custom_test_mse],
    "PyTorch BiRNN": [pt_train_mse, pt_val_mse, pt_test_mse]
})

display(comparison_df)

comparison_df.set_index("Metric").plot(kind="bar", figsize=(9, 5), color=["#1f77b4", "#ff7f0e"])
plt.title("Custom BiRNN vs PyTorch BiRNN")
plt.ylabel("MSE")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# ## Utility functions for Bidirectional LSTM and GRU

# %%
def train_and_validate_model(model, train_loader, val_loader, num_epochs, desc="Training"):
    train_epoch_loss = list()
    val_epoch_loss = list()
    min_val_loss = np.inf

    for epoch in tqdm(range(num_epochs), desc=desc):
        train_iter_loss = list()
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = (batch[0].to(device), batch[1].to(device))
            loss, _ = model.training_step(batch, batch_idx)
            train_iter_loss.append(loss)
        train_epoch_loss.append(sum(train_iter_loss) / len(train_iter_loss))

        val_iter_loss = list()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = (batch[0].to(device), batch[1].to(device))
                loss, _ = model.validation_step(batch, batch_idx)
                val_iter_loss.append(loss)
        val_epoch_loss.append(sum(val_iter_loss) / len(val_iter_loss))

        if val_epoch_loss[-1] < min_val_loss:
            min_val_loss = val_epoch_loss[-1]
            best_model = deepcopy(model)

    return train_epoch_loss, val_epoch_loss, best_model, min_val_loss


def evaluate_regression_model(model, model_name):
    train_predictions = predict_model(model, train_eval_loader, device)
    val_predictions = predict_model(model, val_data_loader, device)
    test_predictions = predict_model(model, test_data_loader, device)

    train_predictions_inv = scaler.inverse_transform(train_predictions.reshape(-1, 1))[:, 0]
    val_predictions_inv = scaler.inverse_transform(val_predictions.reshape(-1, 1))[:, 0]
    test_predictions_inv = scaler.inverse_transform(test_predictions.reshape(-1, 1))[:, 0]

    train_mse = np.mean(np.square(train_predictions_inv - Y_train_inv))
    val_mse = np.mean(np.square(val_predictions_inv - Y_val_inv))
    test_mse = np.mean(np.square(test_predictions_inv - Y_test_inv))

    print(f"{model_name} Training MSE:", train_mse)
    print(f"{model_name} Validation MSE:", val_mse)
    print(f"{model_name} Test MSE:", test_mse)

    metrics = {
        "Train MSE": train_mse,
        "Validation MSE": val_mse,
        "Test MSE": test_mse,
    }
    return metrics

# %% [markdown]
# ## Design a Custom Bidirectional LSTM cell

# %%
class BiLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(BiLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer_forward = torch.nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_layer_forward = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.input_layer_backward = torch.nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_layer_backward = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def _step(self, x_t, h_t, c_t, input_layer, hidden_layer):
        gates = input_layer(x_t) + hidden_layer(h_t)
        i_t, f_t, g_t, o_t = torch.chunk(gates, 4, dim=-1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def _run_direction(self, x, reverse=False):
        h_t = torch.rand((x.shape[0], self.hidden_size), device=x.device)
        c_t = torch.rand((x.shape[0], self.hidden_size), device=x.device)
        h_t = h_t / (torch.sqrt(torch.sum(torch.square(h_t), dim=0)) + 1e-8)
        c_t = c_t / (torch.sqrt(torch.sum(torch.square(c_t), dim=0)) + 1e-8)

        output = list()
        indices = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])

        for i in indices:
            if reverse:
                h_t, c_t = self._step(
                    x[:, i], h_t, c_t, self.input_layer_backward, self.hidden_layer_backward
                )
                output.insert(0, h_t)
            else:
                h_t, c_t = self._step(
                    x[:, i], h_t, c_t, self.input_layer_forward, self.hidden_layer_forward
                )
                output.append(h_t)

        output = torch.stack(output)
        output = torch.swapdims(output, 0, 1)
        return output

    def forward(self, x):
        forward_output = self._run_direction(x, reverse=False)
        backward_output = self._run_direction(x, reverse=True)
        return torch.cat([forward_output, backward_output], dim=-1)


class CustomBiLSTM(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, bias, device):
        super(CustomBiLSTM, self).__init__()
        self.num_layers = num_layers

        self.BiLSTM_layers = list()
        for i in range(num_layers):
            if i == 0:
                self.BiLSTM_layers.append(BiLSTMCell(input_size, hidden_size, bias).to(device))
            else:
                self.BiLSTM_layers.append(BiLSTMCell(2 * hidden_size, hidden_size, bias).to(device))
        self.BiLSTM_layers = torch.nn.Sequential(*self.BiLSTM_layers)

        self.linear = torch.nn.Linear(2 * hidden_size, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.BiLSTM_layers[i](x)
        x = self.linear(x[:, -1]).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.optimizer.zero_grad()
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), preds.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        return loss.item(), preds.detach().cpu().numpy()

# %% [markdown]
# ## Perform training and validation with our custom Bidirectional LSTM

# %%
bilstm_num_layers = 2
bilstm_input_size = X_train.shape[-1]
bilstm_hidden_size = 256 if device.type == "cuda" else 96
bilstm_num_epochs = 240 if device.type == "cuda" else 120

custom_bilstm_model = CustomBiLSTM(
    bilstm_num_layers, bilstm_input_size, bilstm_hidden_size, True, device
).to(device)

custom_bilstm_train_loss, custom_bilstm_val_loss, best_custom_bilstm_model, custom_bilstm_best_val = train_and_validate_model(
    custom_bilstm_model,
    train_data_loader,
    val_data_loader,
    bilstm_num_epochs,
    desc="Training Custom BiLSTM",
)

print("Best Validation Loss (Custom BiLSTM):", custom_bilstm_best_val)

# %% [markdown]
# ## Plot the training and validation losses

# %%
sns.lineplot(x=np.arange(bilstm_num_epochs), y=custom_bilstm_train_loss, label="Train loss")
sns.lineplot(x=np.arange(bilstm_num_epochs), y=custom_bilstm_val_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Define the PyTorch Bidirectional LSTM model

# %%
class PyTorchBiLSTM(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, bias, device):
        super(PyTorchBiLSTM, self).__init__()
        self.LSTM = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
            device=device,
        )

        self.linear = torch.nn.Linear(2 * hidden_size, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        num_directions = 2
        h0 = torch.rand(
            (self.LSTM.num_layers * num_directions, x.shape[0], self.LSTM.hidden_size),
            device=x.device,
        )
        c0 = torch.rand(
            (self.LSTM.num_layers * num_directions, x.shape[0], self.LSTM.hidden_size),
            device=x.device,
        )
        h0 = h0 / (torch.sqrt(torch.sum(torch.square(h0), dim=0)) + 1e-8)
        c0 = c0 / (torch.sqrt(torch.sum(torch.square(c0), dim=0)) + 1e-8)

        x, _ = self.LSTM(x, (h0, c0))
        x = self.linear(x[:, -1]).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.optimizer.zero_grad()
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), preds.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        return loss.item(), preds.detach().cpu().numpy()

# %% [markdown]
# ## Perform training and validation using PyTorch Bidirectional LSTM

# %%
pt_bilstm_model = PyTorchBiLSTM(
    bilstm_num_layers, bilstm_input_size, bilstm_hidden_size, True, device
).to(device)

pt_bilstm_train_loss, pt_bilstm_val_loss, best_pt_bilstm_model, pt_bilstm_best_val = train_and_validate_model(
    pt_bilstm_model,
    train_data_loader,
    val_data_loader,
    bilstm_num_epochs,
    desc="Training PyTorch BiLSTM",
)

print("Best Validation Loss (PyTorch BiLSTM):", pt_bilstm_best_val)

# %% [markdown]
# ## Plot the training and validation losses

# %%
sns.lineplot(x=np.arange(bilstm_num_epochs), y=pt_bilstm_train_loss, label="Train loss")
sns.lineplot(x=np.arange(bilstm_num_epochs), y=pt_bilstm_val_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluate Bidirectional LSTM models and compare

# %%
custom_bilstm_metrics = evaluate_regression_model(best_custom_bilstm_model, "Custom BiLSTM")
pt_bilstm_metrics = evaluate_regression_model(best_pt_bilstm_model, "PyTorch BiLSTM")

bilstm_comparison_df = pd.DataFrame({
    "Metric": ["Train MSE", "Validation MSE", "Test MSE"],
    "Custom BiLSTM": [
        custom_bilstm_metrics["Train MSE"],
        custom_bilstm_metrics["Validation MSE"],
        custom_bilstm_metrics["Test MSE"],
    ],
    "PyTorch BiLSTM": [
        pt_bilstm_metrics["Train MSE"],
        pt_bilstm_metrics["Validation MSE"],
        pt_bilstm_metrics["Test MSE"],
    ],
})

display(bilstm_comparison_df)

bilstm_comparison_df.set_index("Metric").plot(
    kind="bar", figsize=(9, 5), color=["#2b8cbe", "#fdae6b"]
)
plt.title("Custom BiLSTM vs PyTorch BiLSTM")
plt.ylabel("MSE")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# ## Design a Custom Bidirectional GRU cell

# %%
class BiGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(BiGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer_forward = torch.nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden_layer_forward = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.input_layer_backward = torch.nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden_layer_backward = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def _step(self, x_t, h_t, input_layer, hidden_layer):
        x_proj = input_layer(x_t)
        h_proj = hidden_layer(h_t)

        x_r, x_z, x_n = torch.chunk(x_proj, 3, dim=-1)
        h_r, h_z, h_n = torch.chunk(h_proj, 3, dim=-1)

        r_t = torch.sigmoid(x_r + h_r)
        z_t = torch.sigmoid(x_z + h_z)
        n_t = torch.tanh(x_n + r_t * h_n)
        h_t = (1 - z_t) * n_t + z_t * h_t
        return h_t

    def _run_direction(self, x, reverse=False):
        h_t = torch.rand((x.shape[0], self.hidden_size), device=x.device)
        h_t = h_t / (torch.sqrt(torch.sum(torch.square(h_t), dim=0)) + 1e-8)

        output = list()
        indices = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])

        for i in indices:
            if reverse:
                h_t = self._step(
                    x[:, i], h_t, self.input_layer_backward, self.hidden_layer_backward
                )
                output.insert(0, h_t)
            else:
                h_t = self._step(
                    x[:, i], h_t, self.input_layer_forward, self.hidden_layer_forward
                )
                output.append(h_t)

        output = torch.stack(output)
        output = torch.swapdims(output, 0, 1)
        return output

    def forward(self, x):
        forward_output = self._run_direction(x, reverse=False)
        backward_output = self._run_direction(x, reverse=True)
        return torch.cat([forward_output, backward_output], dim=-1)


class CustomBiGRU(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, bias, device):
        super(CustomBiGRU, self).__init__()
        self.num_layers = num_layers

        self.BiGRU_layers = list()
        for i in range(num_layers):
            if i == 0:
                self.BiGRU_layers.append(BiGRUCell(input_size, hidden_size, bias).to(device))
            else:
                self.BiGRU_layers.append(BiGRUCell(2 * hidden_size, hidden_size, bias).to(device))
        self.BiGRU_layers = torch.nn.Sequential(*self.BiGRU_layers)

        self.linear = torch.nn.Linear(2 * hidden_size, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.BiGRU_layers[i](x)
        x = self.linear(x[:, -1]).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.optimizer.zero_grad()
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), preds.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        return loss.item(), preds.detach().cpu().numpy()

# %% [markdown]
# ## Perform training and validation with our custom Bidirectional GRU

# %%
bigru_num_layers = 2
bigru_input_size = X_train.shape[-1]
bigru_hidden_size = 256 if device.type == "cuda" else 96
bigru_num_epochs = 220 if device.type == "cuda" else 110

custom_bigru_model = CustomBiGRU(
    bigru_num_layers, bigru_input_size, bigru_hidden_size, True, device
).to(device)

custom_bigru_train_loss, custom_bigru_val_loss, best_custom_bigru_model, custom_bigru_best_val = train_and_validate_model(
    custom_bigru_model,
    train_data_loader,
    val_data_loader,
    bigru_num_epochs,
    desc="Training Custom BiGRU",
)

print("Best Validation Loss (Custom BiGRU):", custom_bigru_best_val)

# %% [markdown]
# ## Plot the training and validation losses

# %%
sns.lineplot(x=np.arange(bigru_num_epochs), y=custom_bigru_train_loss, label="Train loss")
sns.lineplot(x=np.arange(bigru_num_epochs), y=custom_bigru_val_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Define the PyTorch Bidirectional GRU model

# %%
class PyTorchBiGRU(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, bias, device):
        super(PyTorchBiGRU, self).__init__()
        self.GRU = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
            device=device,
        )

        self.linear = torch.nn.Linear(2 * hidden_size, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        num_directions = 2
        h0 = torch.rand(
            (self.GRU.num_layers * num_directions, x.shape[0], self.GRU.hidden_size),
            device=x.device,
        )
        h0 = h0 / (torch.sqrt(torch.sum(torch.square(h0), dim=0)) + 1e-8)

        x, _ = self.GRU(x, h0)
        x = self.linear(x[:, -1]).squeeze(-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.optimizer.zero_grad()
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), preds.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.float())
        return loss.item(), preds.detach().cpu().numpy()

# %% [markdown]
# ## Perform training and validation using PyTorch Bidirectional GRU

# %%
pt_bigru_model = PyTorchBiGRU(
    bigru_num_layers, bigru_input_size, bigru_hidden_size, True, device
).to(device)

pt_bigru_train_loss, pt_bigru_val_loss, best_pt_bigru_model, pt_bigru_best_val = train_and_validate_model(
    pt_bigru_model,
    train_data_loader,
    val_data_loader,
    bigru_num_epochs,
    desc="Training PyTorch BiGRU",
)

print("Best Validation Loss (PyTorch BiGRU):", pt_bigru_best_val)

# %% [markdown]
# ## Plot the training and validation losses

# %%
sns.lineplot(x=np.arange(bigru_num_epochs), y=pt_bigru_train_loss, label="Train loss")
sns.lineplot(x=np.arange(bigru_num_epochs), y=pt_bigru_val_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluate Bidirectional GRU models and compare

# %%
custom_bigru_metrics = evaluate_regression_model(best_custom_bigru_model, "Custom BiGRU")
pt_bigru_metrics = evaluate_regression_model(best_pt_bigru_model, "PyTorch BiGRU")

bigru_comparison_df = pd.DataFrame({
    "Metric": ["Train MSE", "Validation MSE", "Test MSE"],
    "Custom BiGRU": [
        custom_bigru_metrics["Train MSE"],
        custom_bigru_metrics["Validation MSE"],
        custom_bigru_metrics["Test MSE"],
    ],
    "PyTorch BiGRU": [
        pt_bigru_metrics["Train MSE"],
        pt_bigru_metrics["Validation MSE"],
        pt_bigru_metrics["Test MSE"],
    ],
})

display(bigru_comparison_df)

bigru_comparison_df.set_index("Metric").plot(
    kind="bar", figsize=(9, 5), color=["#41ab5d", "#f16913"]
)
plt.title("Custom BiGRU vs PyTorch BiGRU")
plt.ylabel("MSE")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0)
plt.show()


