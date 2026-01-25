# %% [markdown]
# # Installs

# %%
! pip install -q ucimlrepo ipywidgets

# %% [markdown]
# # Imports

# %%
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

import pandas as pd
import numpy as np

# %% [markdown]
# # EDA

# %%
df = fetch_ucirepo(id=183).data.original

# %%
df.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], inplace=True)
df.replace('?', np.nan, inplace=True)
# Convert OtherPerCap column to numeric (it is string for some reason)
df['OtherPerCap'] = pd.to_numeric(df['OtherPerCap'], errors='coerce')
print(f"Dataset shape: {df.shape}")

# %%
total_missing_values = df.isnull().sum().sort_values(ascending=False)
percent_missing_values = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_values, percent_missing_values], axis=1, keys=['Total', 'Percent'])
missing_data.head(24)

# %% [markdown]
# Remove the fields that have high number of missing values

# %%
df.drop((missing_data[missing_data['Total'] > 1600]).index, axis=1, inplace=True)
df.shape

# %% [markdown]
# Fill the NaN value of racePctWhite with the Median value

# %%
df['OtherPerCap'].fillna(df['OtherPerCap'].median(), inplace=True)

# %% [markdown]
# Segregate data

# %%
X = df.drop(columns=['ViolentCrimesPerPop'])
y = df['ViolentCrimesPerPop']

# %% [markdown]
# # Test-Train split

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

len(X_train), len(X_test), len(X_val), len(X)

# %% [markdown]
# # Neural Network Scratch Implementation

# %%
class Linear:
  """
  Linear Layer
  """

  def __init__(self, input_size: int, output_size: int) -> None:
    """
    Initialize the linear layer.

    Args:
      input_size (int): The size of the input.
      output_size (int): The size of the output.
    """

    # He initialization of the weight matrix
    self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
    self.bias = np.zeros((1, output_size))
    self.outputs = np.zeros((1, output_size))

  def __call__(self, input: np.ndarray) -> None:
    """
    Forward pass of the linear layer.
    """

    self.outputs = np.dot(input, self.weights) + self.bias
    return self.outputs
  
class ReLU:
  """
  ReLU Activation Function
  """

  def __init__(self) -> None:
    """
    Initialize the ReLU activation function.
    """
    self.outputs = None

  def __call__(self, input: np.ndarray) -> None:
    """
    Forward pass of the ReLU activation function.
    """

    self.outputs = np.maximum(0, input)
    return self.outputs
  
class deReLU:
  """
  Derivative of ReLU Activation Function
  """

  def __init__(self) -> None:
    """
    Initialize the derivative of ReLU activation function.
    """
    self.outputs = None

  def __call__(self, input: np.ndarray) -> None:
    """
    Forward pass of the derivative of ReLU activation function.
    """

    self.outputs = np.where(input > 0, 1, 0)
    return self.outputs
  
class HuberLoss:
  """
  Huber Loss Function
  """

  def __init__(self, delta: float = 1.0) -> None:
    """
    Initialize the Huber loss function.

    Args:
      delta (float): The threshold parameter for Huber loss.
    """

    self.delta = delta

  def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Huber loss.

    Args:
      y_true: True labels.
      y_pred: Predicted labels.

    Returns:
      float: Computed Huber loss.
    """

    error = y_true - y_pred
    is_small_error = np.abs(error) <= self.delta
    squared_loss = 0.5 * (error ** 2)
    linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))
  
  def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the Huber loss.

    Args:
      y_true: True labels.
      y_pred: Predicted labels.

    Returns:
      np.ndarray: Derivative of Huber loss.
    """

    error = y_pred - y_true
    is_small_error = np.abs(error) <= self.delta
    small_error_grad = error
    large_error_grad = self.delta * np.sign(error)
    return np.where(is_small_error, small_error_grad, large_error_grad) / y_true.shape[0]

# %%
class NeuralNetwork:
  """
  Neural Network implemented from scratch using numpy
  """
  def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.001, huber_delta: float = 1.0) -> None:
    """
    Initialize the neural network.

    Args:
      input_size (int): The size of the input layer.
      output_size (int): The size of the output layer.
      learning_rate (float): The learning rate for gradient descent.
      huber_delta (float): The delta parameter for Huber loss.
    """

    self.learning_rate = learning_rate

    # Layers
    self.linear1 = Linear(input_size, 32)
    self.relu1 = ReLU()
    self.linear2 = Linear(32, 16)
    self.relu2 = ReLU()
    self.output_layer = Linear(16, output_size)

    # Loss function
    self.loss_fn = HuberLoss(delta=huber_delta)

  def forward(self, X: np.ndarray) -> np.ndarray:
    """
    Forward pass of the neural network.

    Args:
      X (np.ndarray): Input data.

    Returns:
      np.ndarray: Output predictions.
    """

    out = self.linear1(X)
    out = self.relu1(out)
    out = self.linear2(out)
    out = self.relu2(out)
    out = self.output_layer(out)
    return out
  
  def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the loss.

    Args:
      y_true (np.ndarray): True labels.
      y_pred (np.ndarray): Predicted labels.

    Returns:
      float: Computed loss.
    """

    return self.loss_fn(y_true, y_pred)
  
  def compute_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Compute gradients using backpropagation.

    Args:
      X (np.ndarray): Input data.
      y_true (np.ndarray): True labels.
      y_pred (np.ndarray): Predicted labels.
    """

    m = y_true.shape[0]
    loss = self.loss_fn(y_true, y_pred)
    self.dL_dy = self.loss_fn.derivative(y_true, y_pred)

    # Backpropagation through output layer
    self.dL_dW3 = np.dot(self.relu2.outputs.T, self.dL_dy)
    self.dL_db3 = np.sum(self.dL_dy, axis=0, keepdims=True)
    self.dL_dA2 = np.dot(self.dL_dy, self.output_layer.weights.T)
    self.dL_dZ2 = self.dL_dA2 * deReLU()(self.linear2.outputs)

    # Backpropagation through second hidden layer
    self.dL_dW2 = np.dot(self.relu1.outputs.T, self.dL_dZ2)
    self.dL_db2 = np.sum(self.dL_dZ2, axis=0, keepdims=True)

    self.dL_dA1 = np.dot(self.dL_dZ2, self.linear2.weights.T)
    self.dL_dZ1 = self.dL_dA1 * deReLU()(self.linear1.outputs)

    # Backpropagation through first hidden layer
    self.dL_dW1 = np.dot(X.T, self.dL_dZ1)
    self.dL_db1 = np.sum(self.dL_dZ1, axis=0, keepdims=True)

  def optimizer_step(self) -> None:
    """
    Update weights and biases using gradient descent.

    Args:
      dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3: Gradients of weights and biases.
    """

    self.linear1.weights -= self.learning_rate * self.dL_dW1
    self.linear1.bias -= self.learning_rate * self.dL_db1
    self.linear2.weights -= self.learning_rate * self.dL_dW2
    self.linear2.bias -= self.learning_rate * self.dL_db2
    self.output_layer.weights -= self.learning_rate * self.dL_dW3
    self.output_layer.bias -= self.learning_rate * self.dL_db3

  def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, patience: int = 5, batch_size: int = 32) -> None:
    """
    Train the neural network with mini batch gradient descent and early stopping.

    Args:
      X (np.ndarray): Training data.
      y (np.ndarray): Training labels.
      X_val (np.ndarray): Validation data.
      y_val (np.ndarray): Validation labels.
      epochs (int): Number of training epochs.
      patience (int): Patience for early stopping based on validation loss.
      batch_size (int): Size of each training batch.
    """

    # Maintain an array of val and train losses and return them
    train_losses = []
    val_losses = []
    epoch = 0

    best_val_loss = float('inf')
    patience_counter = 0
    n_samples = X.shape[0]

    with tqdm(total=epochs) as pbar:
      for epoch in range(epochs):
        permutation = np.random.permutation(n_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, n_samples, batch_size):
          X_batch = X_shuffled[i:i+batch_size]
          y_batch = y_shuffled[i:i+batch_size]

          y_pred = self.forward(X_batch)
          self.compute_grad(X_batch, y_batch, y_pred)
          self.optimizer_step()

        # Compute training and validation loss
        train_pred = self.forward(X)
        train_loss = self.compute_loss(y, train_pred)

        val_pred = self.forward(X_val)
        val_loss = self.compute_loss(y_val, val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # tqdm.write(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          patience_counter = 0
        else:
          patience_counter += 1
          if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        pbar.set_description(f"Epoch: {epoch+1}")
        pbar.set_postfix_str(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        pbar.update(1)

      pbar.close()
      return train_losses, val_losses
  
  def save(self, filepath: str) -> None:
    """
    Save the model parameters to a file using pickle

    Args:
      filepath (str): Path to the file where the model will be saved.
    """
    import pickle
    with open(filepath, 'wb') as f:
      pickle.dump(self, f)
      print(f"Model saved to {filepath}")

  def load(self, filepath: str) -> None:
    """
    Load the model parameters from a file using pickle

    Args:
      filepath (str): Path to the file from which the model will be loaded.
    """
    import pickle
    with open(filepath, 'rb') as f:
      model = pickle.load(f)
      self.linear1 = model.linear1
      self.relu1 = model.relu1
      self.linear2 = model.linear2
      self.relu2 = model.relu2
      self.output_layer = model.output_layer
      self.loss_fn = model.loss_fn
      print(f"Model loaded from {filepath}")
  
  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using the trained neural network.

    Args:
      X (np.ndarray): Input data.

    Returns:
      np.ndarray: Predicted labels.
    """

    return self.forward(X)
  
  def R2_score(self, X: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the R^2 score.

    Args:
      X (np.ndarray): Input data.
      y_true (np.ndarray): True labels.

    Returns:
      float: R^2 score.
    """

    y_pred = self.predict(X)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# %% [markdown]
# # Training and Evaluation

# %% [markdown]
# ## Models

# %%
model_1 = NeuralNetwork(input_size=X_train.shape[1], output_size=1, learning_rate=0.001, huber_delta=0.1)
model_2 = NeuralNetwork(input_size=X_train.shape[1], output_size=1, learning_rate=0.001, huber_delta=0.5)
model_3 = NeuralNetwork(input_size=X_train.shape[1], output_size=1, learning_rate=0.001, huber_delta=1.0)
model_4 = NeuralNetwork(input_size=X_train.shape[1], output_size=1, learning_rate=0.001, huber_delta=2.0)
model_5 = NeuralNetwork(input_size=X_train.shape[1], output_size=1, learning_rate=0.001, huber_delta=5.0)

# %%
# Train the models and store the losses in a dict
models = {
    'model_1': model_1,
    'model_2': model_2,
    'model_3': model_3,
    'model_4': model_4,
    'model_5': model_5
}

histories = {}

for name, model in models.items():
    print(f"Training {name} with Huber delta = {model.loss_fn.delta}")
    train_losses, val_losses = model.fit(X_train.values, y_train.values.reshape(-1, 1), X_val.values, y_val.values.reshape(-1, 1), epochs=100, patience=5, batch_size=32)
    histories[name] = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    # Save model
    model.save(f"{name}_huber_delta_{model.loss_fn.delta}.pkl")

# %% [markdown]
# ## Plotting

# %%
plt.figure(figsize=(12, 8))
for name, history in histories.items():
    plt.plot(history['train_losses'], label=f'{name} Train Loss (delta={models[name].loss_fn.delta})')
    plt.plot(history['val_losses'], label=f'{name} Validation Loss (delta={models[name].loss_fn.delta})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves for Different Huber Delta Values')
plt.legend()
plt.show()

# %% [markdown]
# # Calculating R2 score for all models on validation set

# %%
# Calculate and print R2 scores on the validation set
r2_scores = {}
for name, model in models.items():
    r2 = model.R2_score(X_val.values, y_val.values.reshape(-1, 1))
    r2_scores[name] = r2
    print(f"{name} R^2 Score on Validation Set (delta={model.loss_fn.delta}): {r2:.4f}")

# %%
# Best model based on R2 score and its performance on the test set
best_model_name = max(r2_scores, key=r2_scores.get)
best_model = models[best_model_name]
preds = best_model.predict(X_test.values)
test_r2 = best_model.R2_score(X_test.values, y_test.values.reshape(-1, 1))
print(f"Best Model: {best_model_name} with R^2 Score on Test Set: {test_r2:.4f}")

# %%
# Plot prediction of best model vs true values in a scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, preds, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('NumPy Model: Predicted vs True Values')
plt.show()

# %% [markdown]
# # PyTorch Implementation

# %% [markdown]
# ## Import PyTorch

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# %% [markdown]
# ## Define PyTorch Neural Network

# %%
class PyTorchNeuralNetwork(nn.Module):
    """
    PyTorch Neural Network with the same architecture as the NumPy implementation
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
        """
        super(PyTorchNeuralNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(16, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output predictions.
        """
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.output_layer(out)
        return out

# %% [markdown]
# ## Initialize PyTorch Model

# %%
# Get the best delta value from the earlier training
best_delta = best_model.loss_fn.delta
print(f"Using best delta value: {best_delta}")

# Initialize PyTorch model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
device = torch.device('cpu')

pytorch_model = PyTorchNeuralNetwork(input_size=X_train.shape[1], output_size=1).to(device)

# Use Huber loss with the best delta
criterion = nn.HuberLoss(delta=best_delta)

# Use SGD optimizer with the same learning rate
learning_rate = 0.001
optimizer = optim.SGD(pytorch_model.parameters(), lr=learning_rate)

# %% [markdown]
# ## Prepare Data for PyTorch

# %%
# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values).to(device)
y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1)).to(device)
X_val_tensor = torch.FloatTensor(X_val.values).to(device)
y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1)).to(device)
X_test_tensor = torch.FloatTensor(X_test.values).to(device)
y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1)).to(device)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %% [markdown]
# ## Train PyTorch Model

# %%
def train_pytorch_model(model, train_loader, X_train, y_train, X_val, y_val, criterion, optimizer, epochs=100, patience=5):
    """
    Train the PyTorch model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        X_train, y_train: Training data tensors
        X_val, y_val: Validation data tensors
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs
        patience: Early stopping patience
    
    Returns:
        Lists of training and validation losses
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            # Training phase
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Evaluation phase
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                train_loss = criterion(train_outputs, y_train).item()
                
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
            
            pbar.set_description(f"Epoch: {epoch+1}")
            pbar.set_postfix_str(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            pbar.update(1)
    
    pbar.close()
    return train_losses, val_losses

# Train the model
pytorch_train_losses, pytorch_val_losses = train_pytorch_model(pytorch_model, train_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, criterion, optimizer, epochs=100, patience=5)

# %% [markdown]
# ## Plot Training and Validation Loss Curves

# %%
plt.figure(figsize=(10, 6))
plt.plot(pytorch_train_losses, label='Training Loss', linewidth=2)
plt.plot(pytorch_val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'PyTorch Model: Training and Validation Loss (Huber delta={best_delta})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Compute R² Score on Test Set

# %%
def compute_r2_score(model, X, y_true):
    """
    Compute R² score for PyTorch model.
    
    Args:
        model: PyTorch model
        X: Input tensor
        y_true: True labels tensor
    
    Returns:
        R² score
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
    
    # Convert to numpy for computation
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    ss_total = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    ss_residual = np.sum((y_true_np - y_pred_np) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return r2

# Compute R² score on test set
pytorch_test_r2 = compute_r2_score(pytorch_model, X_test_tensor, y_test_tensor)
print(f"PyTorch Model R² Score on Test Set: {pytorch_test_r2:.4f}")

# %% [markdown]
# ## Prediction vs Ground Truth Scatter Plot

# %%
# Generate predictions
pytorch_model.eval()
with torch.no_grad():
    pytorch_preds = pytorch_model(X_test_tensor).cpu().numpy()

y_test_np = y_test.values

# Create scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test_np, pytorch_preds, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('PyTorch Model: Predicted vs True Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare R² Scores: PyTorch vs NumPy Implementation

# %%
# Compare R² scores
print("=" * 60)
print("R² Score Comparison on Test Set")
print("=" * 60)
print(f"NumPy Implementation:    {test_r2:.4f}")
print(f"PyTorch Implementation:  {pytorch_test_r2:.4f}")
print(f"Difference:              {abs(pytorch_test_r2 - test_r2):.4f}")
print("=" * 60)

# Create a bar chart for comparison
plt.figure(figsize=(10, 6))
models_comparison = ['NumPy\nImplementation', 'PyTorch\nImplementation']
r2_values = [test_r2, pytorch_test_r2]
colors = ['#2E86AB', '#A23B72']

bars = plt.bar(models_comparison, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars, r2_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('R² Score', fontsize=12)
plt.title('Test Set R² Score Comparison: NumPy vs PyTorch', fontsize=14, fontweight='bold')
plt.ylim([min(r2_values) - 0.05, max(r2_values) + 0.05])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


