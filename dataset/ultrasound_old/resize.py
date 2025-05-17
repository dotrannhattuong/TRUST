from PIL import Image
import os

def resize_images(input_dir, output_dir, size=(256, 256), is_mask=False):
    """
    Resize images or masks and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing input images or masks.
        output_dir (str): Directory to save resized images or masks.
        size (tuple): Target resize dimensions (width, height).
        is_mask (bool): Set to True if processing masks (to use NEAREST interpolation).
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_dir, filename))
            if is_mask:
                # Use NEAREST interpolation for masks to preserve label values
                img = img.resize(size, resample=Image.NEAREST)
            else:
                # Use BILINEAR interpolation for normal images
                img = img.resize(size, resample=Image.BILINEAR)
            img.save(os.path.join(output_dir, filename))
            print(f"Resized and saved {filename} to {output_dir}")

# Resize train images
resize_images(
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/UCLM/train/images',
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/256/UCLM/train/images',
    size=(256, 256),
    is_mask=False
)
resize_images(
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/UCLM/train/masks',
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/256/UCLM/train/masks',
    size=(256, 256),
    is_mask=True
)

# Resize valid images
resize_images(
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/UCLM/valid/images',
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/256/UCLM/valid/images',
    size=(256, 256),
    is_mask=False
)
resize_images(
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/UCLM/valid/masks',
    '/mnt/HDD1/tuong/TRUST/dataset/ultrasound/256/UCLM/valid/masks',
    size=(256, 256),
    is_mask=True
)
