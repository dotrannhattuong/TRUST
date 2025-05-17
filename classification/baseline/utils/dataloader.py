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
        self.image_files = [os.path.join(self.images_dir, f)
                            for f in os.listdir(self.images_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        

        self.transform = transform
        self.transform_w = transform_w

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        output = {}

        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(image)
        if self.transform_w is not None:
            img_w = self.transform_w(image)
            output['img_w'] = img_w
        
        output['img'] = img
        output['path'] = img_path

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
