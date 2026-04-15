# %% [markdown]
# # ResNet and U-Net in Assignment-6 Style
# 
# This notebook mirrors your PyTorch coding style from Assignment 6: explicit model classes, explicit train and eval loops, tqdm progress bars, and metric plots.
# 
# Sample tasks:
# - ResNet for pet breed classification
# - U-Net for pet segmentation
# 
# Dataset is downloaded fresh into a timestamped folder under runtime_data.

# %%
import os
import random
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet

from tqdm.notebook import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

print('Device:', DEVICE)

# %%
# Hardware-aware parameters for Ryzen 5 9600X + RTX 5070 (12 GB).
# No transformer is used in this notebook; these settings are for CNN training.
CPU_COUNT = os.cpu_count() or 4

CFG = {
    'seed': SEED,
    'image_size': 128,
    'val_ratio': 0.2,
    'num_workers': min(8, max(2, CPU_COUNT // 2)),
    'pin_memory': DEVICE.type == 'cuda',
    'use_amp': DEVICE.type == 'cuda',

    'batch_size_resnet': 64 if DEVICE.type == 'cuda' else 16,
    'resnet_epochs': 8,
    'resnet_lr': 1e-3,

    'batch_size_unet': 16 if DEVICE.type == 'cuda' else 4,
    'unet_epochs': 12,
    'unet_lr': 1e-3,

    # Keep runtime practical for a sample experiment.
    'max_trainval_samples': 4000,
    'max_test_samples': 1000
}

print(CFG)

# %% [markdown]
# ## Fresh Data Download and DataLoaders

# %%
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

download_root = Path('runtime_data') / ('oxford_pets_fresh_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
download_root.mkdir(parents=True, exist_ok=True)

class OxfordPetsMultiTaskDataset(Dataset):
    def __init__(self, root, split, image_size=128, download=True):
        self.ds = OxfordIIITPet(
            root=str(root),
            split=split,
            target_types=['category', 'segmentation'],
            download=download
        )
        self.image_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.ds[idx]
        category, segmentation = target

        x = self.image_tf(image)

        seg_resized = self.mask_resize(segmentation)
        seg_np = np.array(seg_resized, dtype=np.uint8)
        mask_np = ((seg_np == 1) | (seg_np == 3)).astype(np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        y_class = int(category)
        return x, y_class, mask

def _subset_indices(total, max_samples, generator):
    if max_samples is None or max_samples >= total:
        return torch.arange(total)
    return torch.randperm(total, generator=generator)[:max_samples]

def build_dataloaders(cfg):
    gen = torch.Generator().manual_seed(cfg['seed'])

    trainval_full = OxfordPetsMultiTaskDataset(download_root, split='trainval', image_size=cfg['image_size'], download=True)
    test_full = OxfordPetsMultiTaskDataset(download_root, split='test', image_size=cfg['image_size'], download=True)

    trainval_idx = _subset_indices(len(trainval_full), cfg['max_trainval_samples'], gen).tolist()
    test_idx = _subset_indices(len(test_full), cfg['max_test_samples'], gen).tolist()

    trainval_subset = Subset(trainval_full, trainval_idx)
    test_subset = Subset(test_full, test_idx)

    val_size = max(1, int(len(trainval_subset) * cfg['val_ratio']))
    train_size = len(trainval_subset) - val_size
    if train_size < 1:
        raise ValueError('Increase max_trainval_samples to keep train split non-empty.')

    train_set, val_set = random_split(trainval_subset, [train_size, val_size], generator=gen)

    loader_kwargs = {
        'num_workers': cfg['num_workers'],
        'pin_memory': cfg['pin_memory'],
        'persistent_workers': cfg['num_workers'] > 0
    }

    cls_train_loader = DataLoader(train_set, batch_size=cfg['batch_size_resnet'], shuffle=True, **loader_kwargs)
    cls_val_loader = DataLoader(val_set, batch_size=cfg['batch_size_resnet'], shuffle=False, **loader_kwargs)
    cls_test_loader = DataLoader(test_subset, batch_size=cfg['batch_size_resnet'], shuffle=False, **loader_kwargs)

    seg_train_loader = DataLoader(train_set, batch_size=cfg['batch_size_unet'], shuffle=True, **loader_kwargs)
    seg_val_loader = DataLoader(val_set, batch_size=cfg['batch_size_unet'], shuffle=False, **loader_kwargs)
    seg_test_loader = DataLoader(test_subset, batch_size=cfg['batch_size_unet'], shuffle=False, **loader_kwargs)

    return {
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_subset,
        'cls_train_loader': cls_train_loader,
        'cls_val_loader': cls_val_loader,
        'cls_test_loader': cls_test_loader,
        'seg_train_loader': seg_train_loader,
        'seg_val_loader': seg_val_loader,
        'seg_test_loader': seg_test_loader
    }

data_bundle = build_dataloaders(CFG)

print('Fresh download root:', download_root)
print('Train samples:', len(data_bundle['train_set']))
print('Val samples:', len(data_bundle['val_set']))
print('Test samples:', len(data_bundle['test_set']))

# %%
x_batch, y_class_batch, y_mask_batch = next(iter(data_bundle['cls_train_loader']))
print('Image batch:', tuple(x_batch.shape))
print('Class batch:', tuple(y_class_batch.shape))
print('Mask batch:', tuple(y_mask_batch.shape))

# %% [markdown]
# ## ResNet Classifier

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.stem_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def autocast_context():
    if DEVICE.type == 'cuda':
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()

def run_resnet_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)
    for images, labels, _ in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context():
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc

def train_resnet(model, train_loader, val_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG['use_amp']) if DEVICE.type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epoch_pbar = tqdm(range(epochs), desc='ResNet Epochs')
    for epoch in epoch_pbar:
        train_loss, train_acc = run_resnet_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler)
        val_loss, val_acc = run_resnet_epoch(model, val_loader, criterion, optimizer=None, scaler=None)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        epoch_pbar.set_description('ResNet Epoch ' + str(epoch + 1) + '/' + str(epochs))
        epoch_pbar.set_postfix({
            'train_loss': format(train_loss, '.4f'),
            'val_loss': format(val_loss, '.4f'),
            'train_acc': format(train_acc, '.4f'),
            'val_acc': format(val_acc, '.4f')
        })

    return history

resnet_model = ResNetClassifier(num_classes=37).to(DEVICE)
resnet_history = train_resnet(
    model=resnet_model,
    train_loader=data_bundle['cls_train_loader'],
    val_loader=data_bundle['cls_val_loader'],
    epochs=CFG['resnet_epochs'],
    lr=CFG['resnet_lr']
)

resnet_test_loss, resnet_test_acc = run_resnet_epoch(
    model=resnet_model,
    loader=data_bundle['cls_test_loader'],
    criterion=nn.CrossEntropyLoss(),
    optimizer=None
)

print('ResNet test loss:', round(resnet_test_loss, 4))
print('ResNet test accuracy:', round(resnet_test_acc, 4))

e = np.arange(1, len(resnet_history['train_loss']) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(e, resnet_history['train_loss'], label='Train Loss')
plt.plot(e, resnet_history['val_loss'], label='Val Loss')
plt.title('ResNet Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(e, resnet_history['train_acc'], label='Train Acc')
plt.plot(e, resnet_history['val_acc'], label='Val Acc')
plt.title('ResNet Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# %%
resnet_model.eval()
images, labels, _ = next(iter(data_bundle['cls_test_loader']))
images = images.to(DEVICE)

with torch.no_grad():
    logits = resnet_model(images)
    preds = torch.argmax(logits, dim=1).cpu()

mean_t = torch.tensor(NORM_MEAN).view(3, 1, 1)
std_t = torch.tensor(NORM_STD).view(3, 1, 1)

plt.figure(figsize=(14, 6))
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    img = images[i].cpu() * std_t + mean_t
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title('GT ' + str(int(labels[i])) + ' | Pred ' + str(int(preds[i])))
    ax.axis('off')
plt.tight_layout()
plt.show()

torch.save(resnet_model.state_dict(), 'resnet_classifier_oxford_pets.pth')
print('Saved resnet_classifier_oxford_pets.pth')

# %% [markdown]
# ## U-Net Segmenter

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bridge = DoubleConv(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)

def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

def dice_metric(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_metric(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def run_unet_epoch(model, loader, bce_criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    pbar = tqdm(loader, leave=False)
    for images, _, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context():
                logits = model(images)
                bce = bce_criterion(logits, masks)
                dice_l = soft_dice_loss(logits, masks)
                loss = 0.5 * bce + 0.5 * dice_l

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        total_dice += dice_metric(logits, masks)
        total_iou += iou_metric(logits, masks)
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_dice = total_dice / max(1, total_batches)
    avg_iou = total_iou / max(1, total_batches)
    return avg_loss, avg_dice, avg_iou

def train_unet(model, train_loader, val_loader, epochs, lr):
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG['use_amp']) if DEVICE.type == 'cuda' else None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': []
    }

    epoch_pbar = tqdm(range(epochs), desc='UNet Epochs')
    for epoch in epoch_pbar:
        train_loss, train_dice, train_iou = run_unet_epoch(model, train_loader, bce_criterion, optimizer=optimizer, scaler=scaler)
        val_loss, val_dice, val_iou = run_unet_epoch(model, val_loader, bce_criterion, optimizer=None, scaler=None)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        epoch_pbar.set_description('UNet Epoch ' + str(epoch + 1) + '/' + str(epochs))
        epoch_pbar.set_postfix({
            'train_loss': format(train_loss, '.4f'),
            'val_loss': format(val_loss, '.4f'),
            'val_dice': format(val_dice, '.4f'),
            'val_iou': format(val_iou, '.4f')
        })

    return history

unet_model = UNet(in_channels=3, out_channels=1, base_channels=32).to(DEVICE)
unet_history = train_unet(
    model=unet_model,
    train_loader=data_bundle['seg_train_loader'],
    val_loader=data_bundle['seg_val_loader'],
    epochs=CFG['unet_epochs'],
    lr=CFG['unet_lr']
)

unet_test_loss, unet_test_dice, unet_test_iou = run_unet_epoch(
    model=unet_model,
    loader=data_bundle['seg_test_loader'],
    bce_criterion=nn.BCEWithLogitsLoss(),
    optimizer=None
)

print('UNet test loss:', round(unet_test_loss, 4))
print('UNet test dice:', round(unet_test_dice, 4))
print('UNet test IoU:', round(unet_test_iou, 4))

e = np.arange(1, len(unet_history['train_loss']) + 1)
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(e, unet_history['train_loss'], label='Train Loss')
plt.plot(e, unet_history['val_loss'], label='Val Loss')
plt.title('UNet Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(e, unet_history['train_dice'], label='Train Dice')
plt.plot(e, unet_history['val_dice'], label='Val Dice')
plt.title('UNet Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(e, unet_history['train_iou'], label='Train IoU')
plt.plot(e, unet_history['val_iou'], label='Val IoU')
plt.title('UNet IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.show()

# %%
unet_model.eval()
images, _, masks = next(iter(data_bundle['seg_test_loader']))
images = images.to(DEVICE)

with torch.no_grad():
    logits = unet_model(images)
    pred_masks = (torch.sigmoid(logits) > 0.5).float().cpu()

images_cpu = images.cpu()
masks_cpu = masks.cpu()

mean_t = torch.tensor(NORM_MEAN).view(3, 1, 1)
std_t = torch.tensor(NORM_STD).view(3, 1, 1)

plt.figure(figsize=(12, 10))
n = 4
for i in range(n):
    img = images_cpu[i] * std_t + mean_t
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

    ax1 = plt.subplot(n, 3, i * 3 + 1)
    ax1.imshow(img)
    ax1.set_title('Image')
    ax1.axis('off')

    ax2 = plt.subplot(n, 3, i * 3 + 2)
    ax2.imshow(masks_cpu[i, 0], cmap='gray')
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    ax3 = plt.subplot(n, 3, i * 3 + 3)
    ax3.imshow(pred_masks[i, 0], cmap='gray')
    ax3.set_title('Prediction')
    ax3.axis('off')

plt.tight_layout()
plt.show()

torch.save(unet_model.state_dict(), 'unet_segmenter_oxford_pets.pth')
print('Saved unet_segmenter_oxford_pets.pth')


