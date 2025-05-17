import os
import shutil
import random

def split_busi_dataset(busi_root, train_ratio=0.8, seed=42):
    """
    Splits the BUSI dataset into train and val folders.

    Args:
        busi_root (str): Path to the BUSI folder containing 'images' and 'masks'.
        train_ratio (float): Fraction of data to use for training (e.g. 0.8 for 80%).
        seed (int): Random seed for reproducibility.
    """
    images_dir = os.path.join(busi_root, "images")
    masks_dir = os.path.join(busi_root, "masks")

    # Output folders: BUSI/train/images, BUSI/train/masks, BUSI/val/images, BUSI/val/masks
    train_images_dir = os.path.join(busi_root, "train", "images")
    train_masks_dir = os.path.join(busi_root, "train", "masks")
    val_images_dir = os.path.join(busi_root, "valid", "images")
    val_masks_dir = os.path.join(busi_root, "valid", "masks")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    # Get list of image files
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Shuffle for random splitting
    random.seed(seed)
    random.shuffle(image_files)

    # Determine split index
    train_size = int(len(image_files) * train_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    print(f"Total images: {len(image_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")

    # Copy files for the training set
    for f in train_files:
        src_img = os.path.join(images_dir, f)
        dst_img = os.path.join(train_images_dir, f)
        shutil.copy(src_img, dst_img)

        # Copy corresponding mask if it exists
        mask_file = f  # Assuming mask has the same filename
        src_mask = os.path.join(masks_dir, mask_file)
        if os.path.exists(src_mask):
            dst_mask = os.path.join(train_masks_dir, mask_file)
            shutil.copy(src_mask, dst_mask)

    # Copy files for the validation set
    for f in val_files:
        src_img = os.path.join(images_dir, f)
        dst_img = os.path.join(val_images_dir, f)
        shutil.copy(src_img, dst_img)

        # Copy corresponding mask if it exists
        mask_file = f
        src_mask = os.path.join(masks_dir, mask_file)
        if os.path.exists(src_mask):
            dst_mask = os.path.join(val_masks_dir, mask_file)
            shutil.copy(src_mask, dst_mask)

    print("Finished splitting BUSI dataset into train and val folders.")

if __name__ == "__main__":
    busi_root = "/mnt/HDD1/lamle/tuong/thesis/dataset/ultrasound/UCLM"  # Change this to your BUSI folder path
    split_busi_dataset(busi_root, train_ratio=0.8, seed=42)
