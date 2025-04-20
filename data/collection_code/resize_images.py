from PIL import Image
import os

def center_crop(img):
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    return img.crop((left, top, right, bottom))

input_root = "train_large"
output_root = "train"
target_size = (256, 256) 

os.makedirs(output_root, exist_ok=True)

for dirpath, _, filenames in os.walk(input_root):
    for filename in filenames:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(dirpath, filename)

            relative_path = os.path.relpath(dirpath, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB") 
                    img = center_crop(img)
                    img_resized = img.resize(target_size)
                    img_resized.save(output_path)
                    print(f"Saved resized image to: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")


