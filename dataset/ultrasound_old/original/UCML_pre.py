import os
import cv2
import numpy as np
import shutil

# Define input directories for masks and images
mask_dir = "/mnt/HDD1/lamle/tuong/thesis/dataset/ultrasound/UCLM_ori/masks"
image_dir = "/mnt/HDD1/lamle/tuong/thesis/dataset/ultrasound/UCLM_ori/images"

# Define output directories for processed masks and images
out_dir = "/mnt/HDD1/lamle/tuong/thesis/dataset/ultrasound/UCLM"
out_masks_dir = os.path.join(out_dir, "masks")
out_images_dir = os.path.join(out_dir, "images")

# Create output directories if they do not exist
os.makedirs(out_masks_dir, exist_ok=True)
os.makedirs(out_images_dir, exist_ok=True)

# Initialize counters for each category
malignant_counter = 0
benign_counter = 0
normal_counter = 0

# Loop over each mask file in the mask directory
for mask_file in sorted(os.listdir(mask_dir)):
    # Process only image files (png, jpg, jpeg)
    if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    mask_path = os.path.join(mask_dir, mask_file)
    
    # Load the mask image using OpenCV (BGR format)
    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        print(f"Could not load mask file: {mask_path}")
        continue

    # Check for red and green pixels in the mask
    # In BGR, red is [0, 0, 255] and green is [0, 255, 0]
    has_red = np.any(np.all(mask_img == [0, 0, 255], axis=-1))
    has_green = np.any(np.all(mask_img == [0, 255, 0], axis=-1))

    assert not (has_red and has_green), "Mask should not contain both red and green pixels"
    
    # Determine the new file name based on mask content
    if has_red:
        new_filename = f"malignant_{malignant_counter}.png"
        malignant_counter += 1
    elif has_green:
        new_filename = f"benign_{benign_counter}.png"
        benign_counter += 1
    else:
        new_filename = f"normal_{normal_counter}.png"
        normal_counter += 1
    
    # Copy and rename the mask file to the output masks directory
    out_mask_path = os.path.join(out_masks_dir, new_filename)
    shutil.copy(mask_path, out_mask_path)
    
    # Process the corresponding image file from the images folder
    image_path = os.path.join(image_dir, mask_file)  # assuming the same filename
    
    assert os.path.exists(image_path), f"Image file not found: {image_path}"

    if os.path.exists(image_path):
        out_image_path = os.path.join(out_images_dir, new_filename)
        shutil.copy(image_path, out_image_path)
        print(f"Processed {mask_file} -> {new_filename} (both mask and image)")
        

print("Finished processing all files!")

ori_images_len = len(os.listdir(image_dir))
new_images_len = len(os.listdir(out_images_dir))
assert ori_images_len == new_images_len, "Original and new images count do not match!"

ori_masks_len = len(os.listdir(mask_dir))
new_masks_len = len(os.listdir(out_masks_dir))
assert ori_masks_len == new_masks_len, "Original and new masks count do not match!"

print(f"Original images count: {ori_images_len}")
print(f"New images count: {new_images_len}")
print(f"Malignant images count: {malignant_counter}")
print(f"Benign images count: {benign_counter}")
print(f"Normal images count: {normal_counter}")
