# %% [markdown]
# # Imports

# %%
import torch
import random
import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from torchinfo import summary

from tqdm.notebook import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from torchtext.vocab import build_vocab_from_iterator

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# %% [markdown]
# ### Set seed and device

# %%
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# %% [markdown]
# # Load dataset

# %%
df_raw = pd.read_csv("Reviews.csv", on_bad_lines='skip', engine='python', quotechar='"')
print(f"Raw dataset shape: {df_raw.shape}")
df_raw.head()

# %% [markdown]
# ## Preprocessing

# %%
df = df_raw[['Text', 'Score']].copy()
df.columns = ['text', 'rating']
print(f"Shape after column selection: {df.shape}")

# %% [markdown]
# ### Cleanup

# %%
df.dropna(subset=['text', 'rating'], inplace=True)
df = df[df['text'].str.strip() != '']

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['rating'], inplace=True)
df['rating'] = df['rating'].astype(int)

df = df[df['rating'].between(1, 5)]

# %%
df.reset_index(drop=True, inplace=True)
print(f"Shape after cleaning: {df.shape}")
df.head()

# %% [markdown]
# ### Text processing with NLTK

# %%
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text: str):

    text = text.lower()

    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens

df['tokens'] = df['text'].apply(preprocess_text)

# %%
print("Sample review:")
print(df['text'].iloc[0])
print("\nPreprocessed tokens:")
print(df['tokens'].iloc[0])

# %% [markdown]
# ### Build vocabulary

# %%
def yield_tokens(token_lists):
    for token_list in token_lists:
        yield token_list

vocab = build_vocab_from_iterator(
    yield_tokens(df['tokens']),
    specials=['<unk>', '<pad>'],
    min_freq=2
)
vocab.set_default_index(vocab['<unk>'])

print(f"Vocabulary size: {len(vocab):,}")

# %%
df['token_ids'] = df['tokens'].apply(lambda toks: vocab(toks))

print("Sample token IDs (first review):")
print(df['token_ids'].iloc[0])

# %% [markdown]
# ### Plotting distribution

# %%
rating_counts = df['rating'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
for i, v in enumerate(rating_counts.values):
    ax.text(i, v + 30, str(v), ha='center', fontsize=10)
plt.title('Distribution of Review Ratings (1-5)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Splitting dataset

# %%
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df['rating']
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df['rating']
)

# %%
print(f"Training   samples : {len(train_df):>6}  ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation samples : {len(val_df):>6}  ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test       samples : {len(test_df):>6}  ({len(test_df)/len(df)*100:.1f}%)")

# %% [markdown]
# # Dataloaders and Custom Dataset class

# %%
PAD_IDX    = vocab['<pad>']
BATCH_SIZE = 64

class ReviewDataset(Dataset):

    def __init__(self, token_ids, labels):
        self.token_ids = [torch.tensor(ids, dtype=torch.long) for ids in token_ids]
        # Labels are converted to 0-indexed (0-4) to work with CrossEntropyLoss
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]

def collate_fn(batch):
    """Pad sequences within the batch to equal length."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    return padded, torch.stack(labels)

# %%
train_dataset = ReviewDataset(train_df['token_ids'].tolist(), (train_df['rating'] - 1).tolist())
val_dataset   = ReviewDataset(val_df['token_ids'].tolist(),   (val_df['rating']   - 1).tolist())
test_dataset  = ReviewDataset(test_df['token_ids'].tolist(),  (test_df['rating']  - 1).tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn)

print(f"Train batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")
print(f"Test  batches : {len(test_loader)}")

# %% [markdown]
# # Model
# The problem is chosen as 5 class classification problem as different ratings carry different sentiments.

# %%
VOCAB_SIZE    = len(vocab)
EMBED_DIM     = 128
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
NUM_CLASSES   = 5
DROPOUT       = 0.4
EPOCHS        = 20
LEARNING_RATE = 1e-3

# %%
class GRUCellModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes,
                 dropout=0.0, pad_idx=0):
        super(GRUCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.cells = nn.ModuleList([
            nn.GRUCell(embed_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        embedded = self.dropout(self.embedding(x))

        batch_size, seq_len, _ = embedded.size()


        h = [torch.zeros(batch_size, self.hidden_size).to(x.device)
             for _ in range(self.num_layers)]


        for t in range(seq_len):
            x_t = embedded[:, t, :]
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](x_t, h[layer])
                x_t = h[layer]
                if layer < self.num_layers - 1:
                    x_t = self.dropout(x_t)

        out = self.fc(self.dropout(h[-1]))
        return out

# %%
gru_cell_model = GRUCellModel(
    VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
    DROPOUT, pad_idx=PAD_IDX
).to(device)

print(gru_cell_model)

# %%
summary(gru_cell_model, input_data=torch.zeros((BATCH_SIZE, 50), dtype=torch.long).to(device))

# %% [markdown]
# # Training

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gru_cell_model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=EPOCHS, model_path='best_model.pth'):
    train_losses = []
    val_losses   = []
    best_val_loss = float('inf')

    epoch_pbar = tqdm(range(num_epochs), desc=f'Training {type(model).__name__}')
    for epoch in epoch_pbar:

        model.train()
        running_loss = 0.0

        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
        for inputs, labels in batch_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_path)

        epoch_pbar.set_postfix({
            'Train Loss': f'{epoch_train_loss:.4f}',
            'Val Loss':   f'{epoch_val_loss:.4f}'
        })

    model.load_state_dict(torch.load(model_path))
    tqdm.write(f'Training complete. Best Val Loss: {best_val_loss:.4f}')
    return train_losses, val_losses, model

# %%
cell_train_losses, cell_val_losses, gru_cell_model = train_model(
    gru_cell_model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs=EPOCHS, model_path='best_gru_cell.pth'
)

# %%
plt.figure(figsize=(10, 6))
plt.plot(cell_train_losses, label='Train Loss',      color='blue')
plt.plot(cell_val_losses,   label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GRUCell Model: Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid()
plt.show()

# %%
def predict_ratings(model, data_loader, desc='Predicting'):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=desc):
            inputs  = inputs.to(device)
            outputs = model(inputs)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Convert 0-indexed classes back to 1-5 ratings
    return np.array(all_labels) + 1, np.array(all_preds) + 1

cell_gt, cell_pred = predict_ratings(gru_cell_model, test_loader, desc='Predicting (GRUCell)')

# %%
srocc_cell, pval_cell = spearmanr(cell_gt, cell_pred)
print(f"GRUCell Model — SROCC on Test Set : {srocc_cell:.4f}  (p-value: {pval_cell:.4e})")

# %%
plt.figure(figsize=(8, 5))
jitter = np.random.uniform(-0.15, 0.15, size=len(cell_gt))
plt.scatter(cell_gt + jitter, cell_pred + jitter, alpha=0.25, s=10, color='steelblue')
plt.plot([1, 5], [1, 5], 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Ground Truth Rating')
plt.ylabel('Predicted Rating')
plt.title(f'GRUCell: Ground Truth vs. Predicted Ratings\nSROCC = {srocc_cell:.4f}')
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([1, 2, 3, 4, 5])
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# # PYTORCH MODULE

# %%
class GRUModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes,
                 dropout=0.0, pad_idx=0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T)
        embedded = self.dropout(self.embedding(x))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(embedded, h0)
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

# %%
gru_model = GRUModel(
    VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
    DROPOUT, pad_idx=PAD_IDX
).to(device)

print(gru_model)

# %%
total_params_gru = sum(p.numel() for p in gru_model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters (nn.GRU Model): {total_params_gru:,}")

criterion_gru = nn.CrossEntropyLoss()
optimizer_gru = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
scheduler_gru = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gru, mode='min', factor=0.5, patience=3)

# %%
gru_train_losses, gru_val_losses, gru_model = train_model(
    gru_model, train_loader, val_loader, criterion_gru, optimizer_gru, scheduler_gru,
    num_epochs=EPOCHS, model_path='best_gru.pth'
)

# %% [markdown]
# ## Validation

# %%
plt.figure(figsize=(10, 6))
plt.plot(gru_train_losses, label='Train Loss',      color='green')
plt.plot(gru_val_losses,   label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('nn.GRU Model: Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid()
plt.show()

# %%
gru_gt, gru_pred = predict_ratings(gru_model, test_loader, desc='Predicting (nn.GRU)')

srocc_gru, pval_gru = spearmanr(gru_gt, gru_pred)
print(f"nn.GRU Model — SROCC on Test Set : {srocc_gru:.4f}  (p-value: {pval_gru:.4e})")

# %%
plt.figure(figsize=(8, 5))
jitter = np.random.uniform(-0.15, 0.15, size=len(gru_gt))
plt.scatter(gru_gt + jitter, gru_pred + jitter, alpha=0.25, s=10, color='mediumseagreen')
plt.plot([1, 5], [1, 5], 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Ground Truth Rating')
plt.ylabel('Predicted Rating')
plt.title(f'nn.GRU: Ground Truth vs. Predicted Ratings\nSROCC = {srocc_gru:.4f}')
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([1, 2, 3, 4, 5])
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# # Comparison

# %%
comparison_df = pd.DataFrame({
    'Model':            ['GRUCell Model', 'nn.GRU Model'],
    'SROCC':            [round(srocc_cell, 4), round(srocc_gru, 4)],
    'Trainable Params': [
        sum(p.numel() for p in gru_cell_model.parameters() if p.requires_grad),
        sum(p.numel() for p in gru_model.parameters()      if p.requires_grad)
    ]
})

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.barplot(x='Model', y='SROCC', data=comparison_df, palette='viridis', ax=axes[0])
axes[0].set_ylim(0, 1.0)
axes[0].set_title('SROCC on Test Set')
axes[0].set_ylabel('SROCC')
for i, row in comparison_df.iterrows():
    axes[0].text(i, row['SROCC'] + 0.01, f"{row['SROCC']:.4f}", ha='center', fontsize=11)

epochs_axis = list(range(1, EPOCHS + 1))
axes[1].plot(epochs_axis, cell_train_losses, label='GRUCell Train', color='blue',  linestyle='-')
axes[1].plot(epochs_axis, cell_val_losses,   label='GRUCell Val',   color='blue',  linestyle='--')
axes[1].plot(epochs_axis, gru_train_losses,  label='nn.GRU Train',  color='green', linestyle='-')
axes[1].plot(epochs_axis, gru_val_losses,    label='nn.GRU Val',    color='green', linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training & Validation Loss Comparison')
axes[1].legend(fontsize=8)
axes[1].grid()

plt.suptitle('GRUCell vs nn.GRU — Side-by-Side Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# 


