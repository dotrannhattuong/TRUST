import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.randaugment import RandAugment, ResizeImage

class ImageDataset(Dataset):
    def __init__(self, images_dir, transform=None, transform_w=None):
        """
        Args:
            images_dir (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_dir = os.path.join(images_dir, "imgs")
        self.image_files = [f for f in os.listdir(self.images_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = transform
        self.transform_w = transform_w

        # Find unique class names from the filenames (splitting by underscore)
        self.classes = sorted(set([f.split("_")[0] for f in self.image_files]))
        # Create a mapping from class name to integer id
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        output = {}

        file_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(image)
        if self.transform_w is not None:
            img_w = self.transform_w(image)
            output['img_w'] = img_w
        
        output['img'] = img
        output['path'] = img_path
        output["target"] = self.class_to_id[file_name.split("_")[0]]

        return output

##### Transformation #####
def train_transform(resize_size=256):
    return transforms.Compose([
            ResizeImage(resize_size),
            transforms.ToTensor()
        ])

def test_transform(resize_size=256):
    return transforms.Compose([
            ResizeImage(resize_size),
            transforms.ToTensor()
        ])

def train_w_transform(resize_size=256):
    return transforms.Compose([
            RandAugment(3, 5),
            ResizeImage(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
