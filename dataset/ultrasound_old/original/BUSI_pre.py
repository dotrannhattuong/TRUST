import os
import cv2
import numpy as np
import shutil
import re

# Input root directory: the original BUSI folder with subfolders [benign, malignant, normal]
input_root = "/mnt/HDD1/lamle/tuong/thesis/dataset/ultrasound/BUSI_ori"  # Change this to your actual BUSI folder path

# Output directory: new folder BUSI_new with images/ and masks/ subfolders
output_root = "/mnt/HDD1/lamle/tuong/thesis/dataset/ultrasound/BUSI"  # Change this to your desired output path
out_images_dir = os.path.join(output_root, "images")
out_masks_dir = os.path.join(output_root, "masks")

# Create output directories if they don't exist
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_masks_dir, exist_ok=True)

# Classes (subfolders) in the original BUSI folder
classes = ["benign", "malignant", "normal"]

# Initialize counters to create unique filenames for each class
counters = {
    "benign": 1,
    "malignant": 1,
    "normal": 1
}

for cls in classes:
    class_dir = os.path.join(input_root, cls)
    if not os.path.isdir(class_dir):
        print(f"Class folder not found: {class_dir}")
        continue

    # List all files in this class folder
    files_in_class = os.listdir(class_dir)

    # Identify base images (those without '_mask' in the filename)
    base_images = [f for f in files_in_class
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_mask" not in f]


    # Sort base_images numerically using a key lambda that extracts the number inside parentheses.
    sorted_base_images = sorted(
        base_images,
        key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)) if re.search(r'\((\d+)\)', x) else 0
    )

    for base_image_name in sorted_base_images:
        image_path = os.path.join(class_dir, base_image_name)
        base_name_no_ext, ext = os.path.splitext(base_image_name)

        # Find all mask files that correspond to this image. They are assumed to start with the base name and contain '_mask'
        related_masks = []
        for f in files_in_class:
            if f.startswith(base_name_no_ext) and "_mask" in f and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                related_masks.append(f)
        # Sort the list to have a consistent order (optional)
        related_masks.sort()

        # Merge all masks for this image
        merged_mask_gray = None
        for mask_file in related_masks:
            mask_path = os.path.join(class_dir, mask_file)
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                print(f"Could not load mask file: {mask_path}")
                continue
            # If first mask, initialize the merged mask
            if merged_mask_gray is None:
                merged_mask_gray = mask_gray.copy()
            else:
                # Merge by OR: pixel becomes 255 if it is 255 in any mask
                merged_mask_gray = cv2.bitwise_or(merged_mask_gray, mask_gray)

        # If no mask is found for this image, you can decide how to handle it.
        # For this example, we create an all-black mask of the same size as the image.
        if merged_mask_gray is None:
            # Load the image to get its shape
            temp_img = cv2.imread(image_path)
            if temp_img is None:
                print(f"Could not load image to determine mask size: {image_path}")
                continue
            merged_mask_gray = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

        # Create a 3-channel color version of the merged mask
        mask_color = np.zeros((merged_mask_gray.shape[0], merged_mask_gray.shape[1], 3), dtype=np.uint8)
        if cls == "benign":
            # For benign, color the 255 pixels with green (BGR: (0,255,0))
            mask_color[merged_mask_gray == 255] = (0, 255, 0)
        elif cls == "malignant":
            # For malignant, color the 255 pixels with red (BGR: (0,0,255))
            mask_color[merged_mask_gray == 255] = (0, 0, 255)
        else:  # For normal, simply convert grayscale to BGR (or adjust as desired)
            mask_color = cv2.cvtColor(merged_mask_gray, cv2.COLOR_GRAY2BGR)

        # Get new file names using the counter for this class
        idx = counters[cls]
        counters[cls] += 1
        new_image_name = f"{cls}_{idx}.png"
        new_mask_name = f"{cls}_{idx}.png"

        # Copy the original image to the new images folder
        out_image_path = os.path.join(out_images_dir, new_image_name)
        shutil.copy(image_path, out_image_path)

        # Save the merged and colorized mask to the new masks folder
        out_mask_path = os.path.join(out_masks_dir, new_mask_name)
        cv2.imwrite(out_mask_path, mask_color)

        print(f"[{cls}] Processed '{base_image_name}' with {len(related_masks)} mask(s) -> {new_image_name}, {new_mask_name}")

print("Finished merging and colorizing BUSI masks into BUSI!")

print(f"Malignant images count: {counters['malignant'] - 1}")
print(f"Benign images count: {counters['benign'] - 1}")
print(f"Normal images count: {counters['normal'] - 1}")
