import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import os
import time
import copy
import datetime
import argparse
from tqdm import tqdm


def create_model(model_name, num_classes):
    """Creates the specified model architecture."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Modify the final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


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


def create_dataloaders(train_csv, test_csv, unseen_csv, num_workers=8, batch_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
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

    print("\nCreating Datasets and DataLoaders...")
    train_dataset  = StarWarsDataset(csv_file=train_csv,  transform=data_transforms['train'])
    test_dataset   = StarWarsDataset(csv_file=test_csv,   transform=data_transforms['val'])
    unseen_dataset = StarWarsDataset(csv_file=unseen_csv, transform=data_transforms['val'])

    dataloaders = {
        'train' : DataLoader(train_dataset,  batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn_skip_none, pin_memory=True),
        'test'  : DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_skip_none, pin_memory=True),
        'unseen': DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_skip_none, pin_memory=True)
    }
    print(f"Using Batch Size: {batch_size}, Num Workers: {num_workers}")

    dataset_sizes = {'train': len(train_dataset),
                     'test': len(test_dataset),
                     'unseen': len(unseen_dataset)}
    print(f"Dataset sizes: {dataset_sizes}")

    return dataloaders


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=15):
    since = time.time()

    best_val_loss        = float('inf')
    epochs_no_improve    = 0
    early_stop_triggered = False

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch     = -1

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training for up to {num_epochs} epochs with patience={patience}")

    for epoch in range(num_epochs):
        if early_stop_triggered:
            print("Early stopping triggered in previous epoch. Halting.")
            break

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                if inputs.nelement() == 0:
                    continue

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
                # Step scheduler based on epoch (for StepLR, CosineAnnealingLR)
                if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                     scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                current_val_loss = epoch_loss

                # Step ReduceLROnPlateau scheduler based on validation loss/metric if using it
                if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_loss)

                if current_val_loss < best_val_loss:
                    print(f"Validation loss decreased ({best_val_loss:.4f} --> {current_val_loss:.4f}). Saving model...")
                    best_val_loss = current_val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss did not decrease. Patience: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement on validation loss.")
                    early_stop_triggered = True
                    break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    if best_epoch != -1:
        print(f'Best Validation Loss: {best_val_loss:4f} achieved at epoch {best_epoch}')
        print(f"Loading best model weights from epoch {best_epoch}")
        model.load_state_dict(best_model_wts)
    else:
        print("Training stopped before the first validation epoch completed or no improvement ever occurred.")

    return model, history


def evaluate_model(model, dataloader, criterion, phase_name="Evaluation", plot_dir='plots', start_time='default_time'):
    model.eval()
    running_loss     = 0.0
    running_corrects = 0
    total_samples    = 0
    all_labels       = []
    all_preds        = []
    device           = next(model.parameters()).device

    print(f"\n--- {phase_name} ---")

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{phase_name} Progress", leave=False):
            if inputs.nelement() == 0:
                continue
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

    safe_phase_name = phase_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    cm_filename = os.path.join(plot_dir, f'confusion_matrix_{safe_phase_name}_{start_time}.png')
    try:
        plt.savefig(cm_filename, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_filename}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    plt.close()

    return eval_loss, eval_acc.item()



def saliency_map(model, dataloader, device, phase_name="Saliency Map", plot_dir='plots', start_time='default_time', target_class=None):
    """
    Generates an averaged saliency map across the entire dataloader and saves the plot.
    """
    print(f"\n--- {phase_name} ---")
    model.eval()
    averaged_saliency_map = None
    num_images = 0

    for inputs, _ in tqdm(dataloader, desc="Generating Saliency Maps"):
        inputs = inputs.to(device)
        for input_image in inputs:
            input_image.requires_grad = True
            output = model(input_image.unsqueeze(0))  # Add batch dimension

            if target_class is None:
                target_class = torch.argmax(output).item()
                print(f"No target class specified; using target class {target_class}")

            # Compute the gradient of the target class output with respect to the input image
            gradient = torch.autograd.grad(output[0, target_class], input_image)[0]

            # Aggregate the gradients (e.g., take the absolute value or square)
            saliency_map = torch.abs(gradient).mean(dim=0)

            # Normalize the saliency map to [0, 1]
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            if averaged_saliency_map is None:
                averaged_saliency_map = saliency_map
            else:
                averaged_saliency_map += saliency_map
            num_images += 1

    # Average the saliency map
    averaged_saliency_map /= num_images

    # Convert to numpy for visualization
    saliency_map_np = averaged_saliency_map.cpu().numpy()

    # Plot the saliency map
    plt.figure(figsize=(8, 8))
    plt.imshow(saliency_map_np, cmap='hot')
    plt.axis('off')
    plt.title(f'Averaged Saliency Map For Data Class {target_class}')

    # Save the plot
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'averaged_saliency_map_class_{target_class}_{start_time}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Averaged saliency map saved to {plot_path}")


def occlude_right(image_batch, occlusion_ratio):
    B, C, H, W = image_batch.shape
    assert H == W, "Images must be square"

    occlusion_start = int(W * (1-occlusion_ratio))
    
    # Generate random noise for the occluded area for the entire batch
    noise = torch.rand((B, C, H, W - occlusion_start), dtype=image_batch.dtype, device=image_batch.device)
    
    # Clone the original image batch to modify
    occluded_batch = image_batch.clone()
    
    # Replace the rightmost 25% of each image with noise
    occluded_batch[:, :, :, occlusion_start:] = noise
    
    return occluded_batch

def occlude_center(image_batch, occlusion_ratio):
    B, C, H, W = image_batch.shape
    assert H == W, "Images must be square"
    
    # Ratio defines the fraction of width/height for the central occlusion
    occlusion_width = int(W * occlusion_ratio)
    occlusion_height = int(H * occlusion_ratio)
    
    start_x = (W - occlusion_width) // 2
    start_y = (H - occlusion_height) // 2
    
    # Generate random noise for the occluded area for the entire batch
    noise = torch.rand((B, C, occlusion_height, occlusion_width), dtype=image_batch.dtype, device=image_batch.device)
    
    # Clone the original image batch to modify
    occluded_batch = image_batch.clone()
    
    # Replace the center of each image with noise
    occluded_batch[:, :, start_y:start_y + occlusion_height, start_x:start_x + occlusion_width] = noise
    
    return occluded_batch

def occlude_edges(image_batch, occlusion_ratio):
    B, C, H, W = image_batch.shape
    assert H == W, "Images must be square"

    # Ratio defines the thickness of the border relative to width/height
    occlusion_thickness_h = int(H * occlusion_ratio)
    occlusion_thickness_w = int(W * occlusion_ratio)

    # Ensure thickness isn't too large (<= half the dimension)
    occlusion_thickness_h = min(occlusion_thickness_h, H // 2)
    occlusion_thickness_w = min(occlusion_thickness_w, W // 2)

    # Generate random noise for the occluded edges for the entire batch
    noise_top = torch.rand((B, C, occlusion_thickness_h, W), dtype=image_batch.dtype, device=image_batch.device)
    noise_bottom = torch.rand((B, C, occlusion_thickness_h, W), dtype=image_batch.dtype, device=image_batch.device)

    side_noise_height = H - 2 * occlusion_thickness_h
    # Handle cases where thickness is large
    if side_noise_height < 0: side_noise_height = 0

    noise_left = torch.rand((B, C, side_noise_height, occlusion_thickness_w), dtype=image_batch.dtype, device=image_batch.device)
    noise_right = torch.rand((B, C, side_noise_height, occlusion_thickness_w), dtype=image_batch.dtype, device=image_batch.device)

    # Clone the original image batch to modify
    occluded_batch = image_batch.clone()

    # Replace the top and bottom edges with noise
    # Apply top and bottom noise
    occluded_batch[:, :, :occlusion_thickness_h, :] = noise_top
    occluded_batch[:, :, H - occlusion_thickness_h:, :] = noise_bottom

    # Replace the left and right edges with noise
    # Apply side noise only if side_noise_height > 0 (avoids corners)
    if side_noise_height > 0:
        occluded_batch[:, :, occlusion_thickness_h:H - occlusion_thickness_h, :occlusion_thickness_w] = noise_left
        occluded_batch[:, :, occlusion_thickness_h:H - occlusion_thickness_h, W - occlusion_thickness_w:] = noise_right

    return occluded_batch

def perturbate_model(model, dataloader, criterion, phase_name="Perturbation", plot_dir='plots', start_time='default_time', occlusion_rate=0.25, occluded_area='right'):
    model.eval()
    running_loss     = 0.0
    running_corrects = 0
    total_samples    = 0
    all_labels       = []
    all_preds        = []
    device           = next(model.parameters()).device

    assert occlusion_rate >= 0, "Invalid Occlusion Rate"
    assert occlusion_rate < 1, "Invalid Occlusion Rate"

    print(f"\n--- {phase_name} ---")

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{phase_name} Progress", leave=False):
            if inputs.nelement() == 0:
                continue

            if occluded_area == 'right':
                inputs = occlude_right(inputs, occlusion_rate)
                print(f'Occluding {occlusion_rate * 100}% of {occluded_area}')
            elif occluded_area == 'center':
                # square rooted such that the occluded area is of `occlusion_rate` of image
                inputs = occlude_center(inputs, occlusion_rate ** (1/2))
                print(f'Occluding {occlusion_rate * 100}% of {occluded_area}')
            elif occluded_area == 'edges':
                # let original occlusion rate be k, then remainder area is (1-2k)^2
                # let desired occlusion rate BY PERCENTAGE OF AREA OCCLUDED be b
                # let desired occlusion rate for occlude_edges be x
                # then (1-2x)^2=1-b; because b<1, 1-2x = sqrt(1-b), so x = (1-sqrt(1-b))/2
                inputs = occlude_edges(inputs, (1 - (1 - occlusion_rate) ** (1/2)) / 2)
                print(f'Occluding {occlusion_rate * 100}% of {occluded_area}')
            elif occluded_area == 'visualize':
                # Call the standalone visualization function
                visualize_occlusions(model, dataloader, plot_dir, start_time, device, occlusion_rate=occlusion_rate)
                # Skip the rest of the evaluation after visualization
                return None, None
            else:
                raise ValueError(f'Invalid occluded_area specified: {occluded_area}. Must be one of "right", "center", "edges", or "visualize".')

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

    safe_phase_name = phase_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    cm_filename = os.path.join(plot_dir, f'confusion_matrix_occlusion_{occluded_area}_{occlusion_rate}_{safe_phase_name}_{start_time}.png')
    try:
        plt.savefig(cm_filename, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_filename}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    plt.close()

    return eval_loss, eval_acc.item()


def visualize_occlusions(model, dataloader, plot_dir, start_time, device, occlusion_rate=0.25):
    """
    Visualizes the effect of different occlusion methods on a sample image.
    Takes the first image from the first batch provided by the dataloader.
    """
    print(f"\n--- Visualizing Occlusions (Rate: {occlusion_rate}) ---")
    model.eval()
    with torch.no_grad():
        try:
            inputs, _ = next(iter(dataloader)) # Get the first batch
            if inputs.nelement() == 0:
                print("First batch is empty, cannot visualize.")
                return
        except StopIteration:
            print("Dataloader is empty, cannot visualize.")
            return

        inputs = inputs.to(device)
        original_image = inputs[0].cpu()  # Take the first image from the batch

        # Apply occlusion methods using the standalone functions
        occluded_right_batch = occlude_right(original_image.unsqueeze(0), occlusion_rate)
        occluded_center_batch = occlude_center(original_image.unsqueeze(0), occlusion_rate ** (1/2))
        # see perturbate_model() for details on this occlusion rate
        occluded_edges_batch = occlude_edges(original_image.unsqueeze(0), (1 - (1 - occlusion_rate) ** (1/2)) / 2)

        # Get the occluded images (remove batch dim)
        occluded_right = occluded_right_batch[0].cpu()
        occluded_center = occluded_center_batch[0].cpu()
        occluded_edges = occluded_edges_batch[0].cpu()

        # Denormalize for visualization if necessary (assuming standard ImageNet normalization)
        # This requires knowing the normalization parameters used in create_dataloaders
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def denormalize(tensor):
            tensor = tensor * std + mean
            tensor = torch.clamp(tensor, 0, 1) # Clamp values to [0, 1]
            return tensor.permute(1, 2, 0).numpy() # Convert to HWC for matplotlib

        # Convert images to numpy for visualization
        original_np = denormalize(original_image)
        occluded_right_np = denormalize(occluded_right)
        occluded_center_np = denormalize(occluded_center)
        occluded_edges_np = denormalize(occluded_edges)

        # Plot the images
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Occlusion Visualization (Rate: {occlusion_rate})', fontsize=16)

        axes[0].imshow(original_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(occluded_right_np)
        axes[1].set_title('Occluded Right')
        axes[1].axis('off')

        axes[2].imshow(occluded_center_np)
        axes[2].set_title('Occluded Center')
        axes[2].axis('off')

        axes[3].imshow(occluded_edges_np)
        axes[3].set_title('Occluded Edges')
        axes[3].axis('off')

        # Save the plot
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'occlusion_visualization_rate_{occlusion_rate}_{start_time}.png')
        try:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Occlusion visualization saved to {plot_path}")
        except Exception as e:
            print(f"Error saving occlusion visualization plot: {e}")
        plt.close(fig)


if __name__ == "__main__":
    CUR_DIR    = os.getcwd()
    BASE_DIR   = os.path.join(CUR_DIR, "data")

    TRAIN_CSV  = os.path.join(BASE_DIR, "train_labels.csv")
    TEST_CSV   = os.path.join(BASE_DIR, "test_labels.csv")    # Seen characters, different images
    UNSEEN_CSV = os.path.join(BASE_DIR, "unseen_labels.csv")  # Unseen characters

    MODEL_DIR  = os.path.join(CUR_DIR,  "models")
    PLOT_DIR   = os.path.join(CUR_DIR,  "plots")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    START_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    EARLY_STOPPING_PATIENCE = 25

    MODEL_NAME    = "resnet34"
    NUM_CLASSES   = 1
    NUM_EPOCHS    = 100  # We use early stopping so this is just an upper bound
    LEARNING_RATE = 0.001
    WEIGHT_DECAY  = 1e-4

    BATCH_SIZE  = 128
    NUM_WORKERS = 12

    parser = argparse.ArgumentParser(description='Train or evaluate a Star Wars Hero/Villain classifier.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'perturbate'], help='Operation mode: "train", "eval", "perturbate" (default: train)')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model state_dict file (.pth) for evaluation.')
    parser.add_argument('--arch', type=str, default=MODEL_NAME, choices=['resnet18', 'resnet34', 'resnet50'], help=f'Model architecture to use or evaluate (required if mode is eval and different from default {MODEL_NAME}).')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f'Batch size for dataloaders (default: {BATCH_SIZE}); 32, 64, 128 are all good options. Adjust according to VRAM capacity')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help=f'Number of workers for dataloaders (default: {NUM_WORKERS}); 4, 8, 12 are all good options. Adjust according to core count of CPU.')
    args = parser.parse_args()

    BATCH_SIZE  = args.batch_size
    NUM_WORKERS = args.num_workers
    MODEL_NAME  = args.arch

    START_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = create_dataloaders(TRAIN_CSV, TEST_CSV, UNSEEN_CSV, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    criterion = nn.BCEWithLogitsLoss()

    if args.mode == 'train':
        print(f"\n--- Running in Training Mode (Arch: {MODEL_NAME}) ---")

        model = create_model(MODEL_NAME, NUM_CLASSES)
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE/100)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # Decrease LR every 15 epochs
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)

        trained_model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)

        print("\nSaving training history plot...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation (Test Set) Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation (Test Set) Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        history_plot_filename = os.path.join(PLOT_DIR, f'training_history_{MODEL_NAME}_{START_TIME}.png')
        try:
            plt.savefig(history_plot_filename)
            print(f"Training history plot saved to {history_plot_filename}")
        except Exception as e:
            print(f"Error saving training history plot: {e}")
        plt.close()

        evaluate_model(trained_model, dataloaders['test'], criterion, phase_name=f"Post-trainig Eval on Test Set (Arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME)
        evaluate_model(trained_model, dataloaders['unseen'], criterion, phase_name=f"Post-training Eval on Unseen Set (Arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME)

        model_save_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_{START_TIME}.pth')
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Trained model saved to {model_save_path}")

    elif args.mode == 'eval' or args.mode == 'perturbate':
        print(f"\n--- Running in Evaluation Mode (Arch: {MODEL_NAME}) ---")
        if not args.load_model:
            print("Error: --load_model path must be specified in eval mode.")
            exit(1)
        if not os.path.exists(args.load_model):
            print(f"Error: Model file not found at {args.load_model}")
            exit(1)

        model_to_evaluate = create_model(MODEL_NAME, NUM_CLASSES)

        print(f"Loading model weights from: {args.load_model}")
        try:
            model_to_evaluate.load_state_dict(torch.load(args.load_model, map_location=device))
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Ensure the loaded model architecture (--arch) matches the saved weights.")
            exit(1)

        model_to_evaluate = model_to_evaluate.to(device)
        model_to_evaluate.eval()

        # Generate and save the averaged saliency map
        if args.mode == 'perturbate':
            perturbate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Perturbation on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME, occluded_area='visualize')
            # saliency_map(model_to_evaluate, dataloaders['test'], device, phase_name=f"Saliency Map on Test Set", plot_dir=PLOT_DIR, start_time=START_TIME, target_class=0)
            # evaluate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Eval on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME)
            # perturbate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Perturbation on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME)
            # perturbate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Perturbation on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME, occluded_area='right')
            # perturbate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Perturbation on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME, occluded_area='center')
            perturbate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Perturbation on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME, occluded_area='edges')
        else:
            evaluate_model(model_to_evaluate, dataloaders['test'], criterion, phase_name=f"Eval on Test Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME)
            evaluate_model(model_to_evaluate, dataloaders['unseen'], criterion, phase_name=f"Eval on Unseen Set (Loaded model arch {MODEL_NAME})", plot_dir=PLOT_DIR, start_time=START_TIME)
    else:
        print(f"Error: Invalid mode '{args.mode}' specified.")

