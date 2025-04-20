import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import *
import os
import time
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

CUR_DIR = os.getcwd()
BASE_DIR = os.path.join(CUR_DIR, "data")
TRAIN_CSV = os.path.join(BASE_DIR, "train_labels.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_labels.csv") # Seen characters, different images
UNSEEN_CSV = os.path.join(BASE_DIR, "unseen_labels.csv") # Unseen characters

MODEL_NAME = "resnet34"
NUM_CLASSES = 1 # Binary classification (Hero/Villain)
BATCH_SIZE = 64 # Can likely increase this on a 3080 Ti, monitor VRAM usage
NUM_EPOCHS = 50
LEARNING_RATE = 0.001 # Often start higher for scratch training
WEIGHT_DECAY = 1e-4 # Regularization is more important when training from scratch

class StarWarsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {"hero": 0, "villain": 1}

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.labels_frame.iloc[idx, 0]  # string
        img_path = str(Path(img_path).resolve())

        label_str = self.labels_frame.iloc[idx, 1]
        label = self.label_map[label_str]

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}. Skipping index {idx}.")
            return None
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Skipping index {idx}.")
            return None

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor

# --- Transformations ---
# ImageNet mean/std dev are less critical if not using ImageNet weights,
# but still a reasonable default. Could calculate dataset-specific mean/std.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Keep strong augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), # Slightly more aggressive crop maybe
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), # Slightly more rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # Stronger jitter
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
}

# --- DataLoader (Identical to previous version) ---
print("Creating Datasets and DataLoaders...")
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

train_dataset = StarWarsDataset(csv_file=TRAIN_CSV, transform=data_transforms['train'])
test_dataset = StarWarsDataset(csv_file=TEST_CSV, transform=data_transforms['val'])
unseen_dataset = StarWarsDataset(csv_file=UNSEEN_CSV, transform=data_transforms['val'])

dataloaders = {
    'train':  DataLoader(train_dataset,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, collate_fn=collate_fn_skip_none, pin_memory=True),
    'test':   DataLoader(test_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn_skip_none, pin_memory=True),
    'unseen': DataLoader(unseen_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn_skip_none, pin_memory=True)
}

dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset), 'unseen': len(unseen_dataset)}
print(f"Dataset sizes: {dataset_sizes}")


print("Setting up the model")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if MODEL_NAME == "resnet18":
    model = models.resnet18(weights=None)
elif MODEL_NAME == "resnet34":
     model = models.resnet34(weights=None)
elif MODEL_NAME == "resnet50":
     model = models.resnet50(weights=None)
# Add other architectures here if needed (e.g., VGG, DenseNet)
# elif MODEL_NAME == "vgg16_bn":
#     model = models.vgg16_bn(weights=None)
#     num_ftrs = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES) # Adjust VGG classifier
# elif MODEL_NAME == "densenet121":
#     model = models.densenet121(weights=None)
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Linear(num_ftrs, NUM_CLASSES) # Adjust DenseNet classifier
else:
    raise ValueError("Unsupported model name")

# Adjust the final layer for ResNet models (if not handled above for other types)
if 'resnet' in MODEL_NAME:
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss() # Still appropriate for binary classification

# Learning Rate Scheduler - Consider StepLR or CosineAnnealingLR for scratch training
# Option 1: StepLR
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # Decrease LR every 15 epochs
# Option 2: Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE/100)
# Option 3: ReduceLROnPlateau (still viable)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']: # Use 'test' set for validation during training
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                if inputs.nelement() == 0: continue

                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

            if total_samples == 0:
                epoch_loss = 0
                epoch_acc = 0
                print(f"Warning: No samples processed in phase {phase} for epoch {epoch}.")
            else:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                 # Step the scheduler after training epoch if it's not ReduceLROnPlateau
                if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                     scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                val_acc = epoch_acc

                # Step ReduceLROnPlateau scheduler based on validation metric
                if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                     scheduler.step(val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"*** New best model saved with validation accuracy: {best_acc:.4f} ***")
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, dataloader, criterion, phase_name="Evaluation"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    device = next(model.parameters()).device

    print(f"\n--- {phase_name} ---")

    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs.nelement() == 0: continue
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds_class = (torch.sigmoid(outputs) > 0.5).float()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_class == labels.data)
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds_class.cpu().numpy().flatten())

    if total_samples == 0:
        print("No samples found for evaluation.")
        return None, None

    eval_loss = running_loss / total_samples
    eval_acc = running_corrects.double() / total_samples
    print(f'Loss: {eval_loss:.4f} Acc: {eval_acc:.4f}')

    all_labels_int = np.array(all_labels).astype(int)
    all_preds_int = np.array(all_preds).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels_int, all_preds_int, average='binary', zero_division=0)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    cm = confusion_matrix(all_labels_int, all_preds_int)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hero', 'Villain'], yticklabels=['Hero', 'Villain'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{phase_name} Confusion Matrix')
    plt.show()

    return eval_loss, eval_acc.item()

if __name__ == "__main__":
    trained_model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation (Test Set) Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation (Test Set) Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    evaluate_model(trained_model, dataloaders['test'], criterion, phase_name="Evaluation on Test Set (Seen Characters)")
    evaluate_model(trained_model, dataloaders['unseen'], criterion, phase_name="Evaluation on Unseen Set (Unseen Characters)")

    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_save_path = os.path.join(BASE_DIR, f'{MODEL_NAME}_{cur_date_time}.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print("\nFinished.")