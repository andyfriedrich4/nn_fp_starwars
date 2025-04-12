import os
import shutil
import random
import csv

# Define paths
base_dir = "C:\\Users\\andyf\\OneDrive\\Documents\\UT\\Spring25\\NN\\final_project"
categories = ["heros", "villains"]
output_dirs = {
    "train": os.path.join(base_dir, "train"),
    "test": os.path.join(base_dir, "test"),
    "unseen": os.path.join(base_dir, "unseen_characters")
}

# Ensure output directories exist
for split in output_dirs.values():
    os.makedirs(split, exist_ok=True)

# Prepare CSV files
train_csv = os.path.join(base_dir, "train_labels.csv")
test_csv = os.path.join(base_dir, "test_labels.csv")
unseen_csv = os.path.join(base_dir, "unseen_labels.csv")

train_data = []
test_data = []
unseen_data = []

# Select some characters to be in the unseen test set
unseen_fraction = 0.2  # 20% of characters will be unseen
unseen_characters = {}

for category in categories:
    char_folders = sorted(os.listdir(os.path.join(base_dir, category)))
    unseen_count = max(1, int(len(char_folders) * unseen_fraction))
    unseen_characters[category] = random.sample(char_folders, unseen_count)

    for char in char_folders:
        print(char)
        char_path = os.path.join(base_dir, category, char)
        if not os.path.isdir(char_path):
            continue

        images = sorted(os.listdir(char_path))
        random.shuffle(images)
        total_images = len(images)
        split_idx = int(0.8 * total_images)

        if char in unseen_characters[category]:
            # Move all images to unseen set
            unseen_char_path = os.path.join(output_dirs["unseen"], char)
            os.makedirs(unseen_char_path, exist_ok=True)
            for img in images:
                img_src = os.path.join(char_path, img)
                img_dest = os.path.join(unseen_char_path, img)
                shutil.copy(img_src, img_dest)
                unseen_data.append([img_dest, category[:-1]])  # "heros" -> "hero", "villains" -> "villain"
        else:
            # Split into train (80%) and test (20%)
            train_path = os.path.join(output_dirs["train"], category, char)
            test_path = os.path.join(output_dirs["test"], category, char)

            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            for img in images[:split_idx]:
                img_src = os.path.join(char_path, img)
                img_dest = os.path.join(train_path, img)
                shutil.copy(img_src, img_dest)
                train_data.append([img_dest, category[:-1]])

            for img in images[split_idx:]:
                img_src = os.path.join(char_path, img)
                img_dest = os.path.join(test_path, img)
                shutil.copy(img_src, img_dest)
                test_data.append([img_dest, category[:-1]])

# Write CSV files
for file, data in zip([train_csv, test_csv, unseen_csv], [train_data, test_data, unseen_data]):
    with open(file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(data)

print("Dataset successfully split into train, test, and unseen_characters sets.")
print("CSV files generated for training and evaluation.")
