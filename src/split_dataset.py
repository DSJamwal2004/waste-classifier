import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "data/raw"
DEST_DIR = "data/processed"

SPLIT_RATIO = (0.7, 0.15, 0.15)  # train, val, test

for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)
    
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    train_split = int(0.7 * len(images))
    val_split = int(0.85 * len(images))

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for split in splits:
        dest_path = os.path.join(DEST_DIR, split, category)
        os.makedirs(dest_path, exist_ok=True)

        for img in splits[split]:
            src = os.path.join(category_path, img)
            dst = os.path.join(dest_path, img)

            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error copying {src}: {e}")

print("✅ Dataset split completed!")