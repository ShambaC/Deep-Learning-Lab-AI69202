# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.utils.data as data
from torch.utils.data import random_split

from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

import numpy as np

from tqdm.notebook import tqdm

import os
import random

# %% [markdown]
# ### Set device and seed

# %%
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device
print(device)

# %% [markdown]
# # Load Dataset

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.EuroSAT(root='./data', transform=transform, download=True)

# %% [markdown]
# ## EDA

# %%
# Distribution of classes

class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

class_counts = {}
for _, label in tqdm(dataset):
    label = idx_to_class[label]

    if label in class_counts:
        class_counts[label] += 1
    else:
        class_counts[label] = 1



print("Class distribution:")
for k, v in class_counts.items():
    print(f"{k}: {v}")

# %%
labels = class_counts.keys()
counts = class_counts.values()

fig, ax = plt.subplots(figsize=(10, 5))
ax.pie(counts, labels=labels, autopct="%1.1f%%")
plt.show()

# %% [markdown]
# ## Splitting

# %%
# 70-15-15 split of dataset into train, val and test sets using torch random split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# %% [markdown]
# ## Data Augmentation

# %%
class customDataset(data.Dataset):
    """
    Custom dataset class to apply augmentations to the dataset.
    """
    def __init__(self, dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) :
        x, y = self.dataset[idx]

        # Apply transformations to the image
        if self.transform:
            x = self.transform(x)
        
        return x, y

# %%
# Augmentations to apply: flipping, rotation, random crop and resize
transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
])

train_dataset_aug = customDataset(train_dataset, transform=transform_aug)
val_dataset_aug = customDataset(val_dataset, transform=transform_aug)
test_dataset_enc = customDataset(test_dataset, transform=None)

# Sizes of the splits
print(f"Train dataset size: {len(train_dataset_aug)}")
print(f"Validation dataset size: {len(val_dataset_aug)}")
print(f"Test dataset size: {len(test_dataset_enc)}")

train_loader = data.DataLoader(train_dataset_aug, batch_size=128, shuffle=True)
val_loader = data.DataLoader(val_dataset_aug, batch_size=128, shuffle=False)
test_loader = data.DataLoader(test_dataset_enc, batch_size=1, shuffle=False)

# %% [markdown]
# # Training

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # A shortcut connection to match the dimensions of the input and output
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = self.conv2(out)
        out = torch.flatten(out, 1) # flatten the output
        return out

# %%
model = ResNet(num_classes=10).to(device)
print(model)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
num_epochs = 25
train_losses = []
val_losses = []
model=model.to(device)

pbar = tqdm(range(num_epochs), desc="Training")
for epoch in pbar:
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    pbar.set_postfix(train_loss=f"{train_losses[-1]:.4f}", val_loss=f"{val_losses[-1]:.4f}")

# Plot the training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluation

# %%
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    confusion_mat = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, precision, recall, confusion_mat, all_preds, all_labels

# Evaluate the model
test_accuracy, test_f1, test_precision, test_recall, confusion_mat, all_preds, all_labels = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Confusion Matrix:\n{confusion_mat}")

# %% [markdown]
# # Deepfool MultiClass

# %%
def deepfool_attack(image, model, num_classes=10, overshoot=0.02, max_iter=50, device="cuda"):
    """
    DeepFool adversarial attack for a single image (multiclass case).

    Args:
        image: input tensor [C,H,W]
        model: trained classifier
        num_classes: number of classes
        overshoot: small factor to pass the boundary
        max_iter: maximum DeepFool iterations

    Returns:
        perturbed_image
        total_perturbation
        num_iterations
    """

    model.eval()

    # Input image x
    image = image.clone().detach().to(device)

    # Initialize x0 <- x, i <- 0
    x_i = image.clone().detach()
    x_i.requires_grad = True
    i = 0

    # Compute initial prediction k_hat(x_0)
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        _, label = torch.max(logits, 1)
    label = label.item()

    # Total perturbation r_hat
    r_tot = torch.zeros_like(image)

    # Current predicted label
    logits = model(x_i.unsqueeze(0))
    _, current = torch.max(logits, 1)
    current = current.item()

    # While k_hat(x_i) = k_hat(x_0)
    while current == label and i < max_iter:

        x = x_i.unsqueeze(0)

        # Compute classifier outputs f_k(x_i)
        logits = model(x)
        f = logits[0]

        # Compute gradient of the original class ∇f_{k_hat(x_0)}(x_i)
        grad_orig = torch.autograd.grad(f[label], x, retain_graph=True)[0]

        min_pert = float("inf")
        w_best = None

        # For k != k_hat(x_0)
        for k in range(num_classes):

            if k == label:
                continue

            # w_k' ← ∇f_k(x_i) − ∇f_{k_hat(x_0)}(x_i)
            grad_k = torch.autograd.grad(f[k], x, retain_graph=True)[0]
            w_k = grad_k - grad_orig

            # f_k ← f_k(x_i) − f_{k_hat(x_0)}(x_i)
            f_k = f[k] - f[label]

            # Compute distance to decision boundary
            w_k_norm = torch.norm(w_k.flatten(), p=2)
            pert_k = torch.abs(f_k) / w_k_norm

            # Find class giving minimal perturbation
            if pert_k < min_pert:
                min_pert = pert_k
                w_best = w_k

        # r_i ← |f_l'| / ||w_l'||^2 * w_l'
        r_i = (min_pert + 1e-4) * w_best / torch.norm(w_best.flatten(), p=2)

        # Accumulate perturbation
        r_tot = r_tot + r_i.squeeze(0)

        # x_{i+1} ← x_i + r_i
        x_i = image + (1 + overshoot) * r_tot
        x_i = x_i.detach()
        x_i.requires_grad = True

        # i ← i + 1
        i += 1

        # Update predicted label
        logits = model(x_i.unsqueeze(0))
        _, current = torch.max(logits, 1)
        current = current.item()

    # Final adversarial example
    perturbed_image = (image + (1 + overshoot) * r_tot).detach()

    # Return r_hat = Sum_i (r_i)
    return perturbed_image, r_tot.detach(), i

# %%
def generate_adversarial_dataset(model, dataloader, device="cuda"):
    """
    Generates adversarial dataset using DeepFool.

    Returns:
        adv_images (tensor)
        adv_labels (tensor)
    """

    model.eval()

    adv_images = []
    adv_labels = []

    for images, labels in tqdm(dataloader, desc="Generating adversarial examples"):

        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):

            img = images[i]

            adv_img, _, _ = deepfool_attack(img, model, device=device)

            adv_images.append(adv_img.cpu())
            adv_labels.append(labels[i].cpu())

    adv_images = torch.stack(adv_images)
    adv_labels = torch.stack(adv_labels)

    return adv_images, adv_labels

# %%
def evaluate_model_adversarial(model, test_loader):
    model.eval()

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(test_loader, desc="Evaluating (Adversarial)"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.item())
        all_labels.append(labels.item())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    confusion_mat = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, precision, recall, confusion_mat, all_preds, all_labels


# Evaluate model on adversarial data
adv_images, adv_labels = generate_adversarial_dataset(model, test_loader, device)
adv_dataset = data.TensorDataset(adv_images, adv_labels)
adv_loader = data.DataLoader(adv_dataset, batch_size=1, shuffle=False)

adv_accuracy, adv_f1, adv_precision, adv_recall, adv_confusion, adv_preds, adv_labels = evaluate_model_adversarial(model, adv_loader)

print(f"Adversarial Accuracy: {adv_accuracy:.4f}")
print(f"Adversarial F1 Score: {adv_f1:.4f}")
print(f"Adversarial Precision: {adv_precision:.4f}")
print(f"Adversarial Recall: {adv_recall:.4f}")
print(f"Adversarial Confusion Matrix:\n{adv_confusion}")

# %% [markdown]
# ### Success rate

# %%
def deepfool_attack_success_rate(all_labels, clean_preds, adv_preds):
    all_labels = np.array(all_labels)
    clean_preds = np.array(clean_preds)
    adv_preds = np.array(adv_preds)

    # Samples correctly classified by the clean model
    initially_correct = clean_preds == all_labels

    # Among those, check which became incorrect after attack
    successful_attacks = adv_preds != all_labels

    # Intersection of the two conditions
    success_mask = initially_correct & successful_attacks

    if initially_correct.sum() == 0:
        return 0.0

    success_rate = success_mask.sum() / initially_correct.sum()

    return success_rate


success_rate = deepfool_attack_success_rate(all_labels, all_preds, adv_preds)

print(f"DeepFool Attack Success Rate: {success_rate:.4f}")

# %% [markdown]
# ## Using torchattack

# %%
import torchattack

attack = torchattack.DeepFool(model, num_classes=10, steps=50, overshoot=0.02)

adv_images = []
adv_labels = []

model.eval()

for inputs, labels in tqdm(test_loader, desc="Generating adversarial examples with TorchAttack"):
    inputs, labels = inputs.to(device), labels.to(device)

    adv = attack(inputs, labels)

    adv_images.append(adv.detach().cpu())
    adv_labels.append(labels.detach().cpu())

adv_images = torch.cat(adv_images)
adv_labels = torch.cat(adv_labels)

adv_dataset = data.TensorDataset(adv_images, adv_labels)
adv_loader = data.DataLoader(adv_dataset, batch_size=1, shuffle=False)

# Evaluate on adversarial samples
adv_accuracy_torch, adv_f1_torch, adv_precision_torch, adv_recall_torch, adv_confusion_torch, adv_preds_torch, adv_labels_torch = evaluate_model_adversarial(model, adv_loader)

print(f"Adversarial Accuracy (torchattacks): {adv_accuracy_torch:.4f}")
print(f"Adversarial F1 Score (torchattacks): {adv_f1_torch:.4f}")
print(f"Adversarial Precision (torchattacks): {adv_precision_torch:.4f}")
print(f"Adversarial Recall (torchattacks): {adv_recall_torch:.4f}")
print(f"Adversarial Confusion Matrix:\n{adv_confusion_torch}")

# %% [markdown]
# ## Compare success rate

# %%
def compute_attack_success_rate(labels, clean_preds, adv_preds):
    labels = np.array(labels)
    clean_preds = np.array(clean_preds)
    adv_preds = np.array(adv_preds)

    initially_correct = clean_preds == labels
    successful_attacks = adv_preds != labels

    success_mask = initially_correct & successful_attacks

    if initially_correct.sum() == 0:
        return 0.0

    return success_mask.sum() / initially_correct.sum()


# Success rates
custom_success = compute_attack_success_rate(all_labels, all_preds, adv_preds)
torch_success = compute_attack_success_rate(all_labels, all_preds, adv_preds_torch)

print("DeepFool Attack Success Rate Comparison")
print("---------------------------------------")
print(f"Custom DeepFool Implementation : {custom_success:.4f}")
print(f"TorchAttacks DeepFool          : {torch_success:.4f}")


