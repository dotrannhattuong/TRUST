import argparse

import torch
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm
from termcolor import colored

from utils.dataloader import ImageList
from utils.preprocess import val_transform
from utils.utils import validate

parser = argparse.ArgumentParser(description="Classification")
# Data parameters
parser.add_argument("--test_dir", type=str, default="/mnt/HDD1/tuong/TRUST/baseline_TR_semi_AP/results/BUSI2UDIAT")
parser.add_argument("--resize_size", type=int, default=256,
                    help="Resize size")
parser.add_argument("--crop_size", type=int, default=224,
                    help="Crop size")
# Model parameters
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/vit_Breast-UDIAT/best.pth",
                    help="Pretrained model")
# Training parameters
parser.add_argument("--bz", type=int, default=16, help="Batch size")
# Device parameters
parser.add_argument("--device", default="cuda:7", help="Device")

args = parser.parse_args()

# Set device for training
device = torch.device(args.device)

# Load datasets using custom dataloader classes
print(colored(f"Loading datasets from {args.test_dir}", color="blue", force_color=True))
resize_size = args.resize_size
crop_size = args.crop_size
test_dataset = ImageList(args.test_dir, transform_w=val_transform(resize_size, crop_size))
num_classes = test_dataset.num_classes
print(f"Number of classes: {num_classes}")

test_loader = DataLoader(
    test_dataset, batch_size=args.bz, shuffle=False, num_workers=4,
    drop_last=False, pin_memory=True
)

##### Loading Model #####
print(colored(f"Loading model: {args.model}", color="red",
                force_color=True))
if args.model == 'resnet34':
    model = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes=num_classes)
elif args.model == 'resnet50':
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_classes)
elif args.model == 'resnet101':
    model = timm.create_model('resnet101.a1_in1k', pretrained=True, num_classes=num_classes)
elif args.model == 'vit':
    model = timm.create_model('timm/vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, num_classes=num_classes)
else:
    raise ValueError(f"Model {args.model} not supported")

print(colored(f"Loading pretrained weights from {args.checkpoints}", color="blue", force_color=True))
model.load_state_dict(torch.load(args.checkpoints, map_location=device))
model = model.to(device)
model.eval()

val_acc, AUC = validate(model, test_loader, device)
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Validation AUC: {AUC * 100:.2f}%")
