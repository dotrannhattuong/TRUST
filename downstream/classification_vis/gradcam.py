from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms.functional as TF
import argparse
import os
import torch
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2

from utils.dataloader import ImageList
from utils.preprocess import val_transform
from utils.utils import validate

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser(description="Classification with t-SNE and GradCAM Visualization")
# Data parameters
parser.add_argument("--test_dir", type=str, default="/mnt/HDD1/tuong/TRUST/baseline_TR_semi_AP/cp_BUSI2UCLM_TR_semi20_AP_1024/results/results_15.0k")
parser.add_argument("--resize_size", type=int, default=256, help="Resize size")
parser.add_argument("--crop_size", type=int, default=224, help="Crop size")
# Model parameters
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/vit_Breast-UCLM/best.pth", help="Pretrained model")
# Training parameters
parser.add_argument("--bz", type=int, default=16, help="Batch size")
# Device parameters
parser.add_argument("--device", default="cuda:1", help="Device")
# Save folder
parser.add_argument("--save_dir", type=str, default="./vis", help="Directory to save visualization")

args = parser.parse_args()
device = torch.device(args.device)

# ================== Load Dataset ==================
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

print(colored(f"Loading model: {args.model}", color="red", force_color=True))
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

model.load_state_dict(torch.load(args.checkpoints, map_location=device))
model = model.to(device)
model.eval()

val_acc, AUC = validate(model, test_loader, device)
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Validation AUC: {AUC * 100:.2f}%")
print(colored("Extracting features for t-SNE...", color="green", force_color=True))

def extract_features(model, dataloader, device):
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            inputs = batch['img_w']
            labels = batch['target']

            inputs = inputs.to(device)
            try:
                outputs = model.forward_features(inputs)  # For ViT
            except:
                outputs = model.global_pool(model.forward_features(inputs))  # For ResNet

            features_list.append(outputs.cpu())
            labels_list.append(labels)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    features = features.view(features.size(0), -1)   # <=== FLATTEN về 2D
    features = features.numpy()
    labels = labels.numpy()
    return features, labels

features, labels = extract_features(model, test_loader, device)
# ========== Get First Image ==========
batch = next(iter(test_loader))
inputs = batch['img_w'].to(device)
labels = batch['target']

input_tensor = inputs[0].unsqueeze(0)
label = labels[0].item()

# Convert to numpy image for overlay
input_image = inputs[0].detach().cpu().permute(1, 2, 0).numpy()
input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

# ========== Reshape Function for ViT ==========
def reshape_transform(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]  # Skip [CLS] token
    return tensor.reshape(tensor.size(0), height, width, tensor.size(2)).permute(0, 3, 1, 2)

# ========== Loop over selected blocks ==========
layers_to_test = []
for i in range(len(model.blocks)):
    layers_to_test.append((f"block{i}_attn", model.blocks[i]))  # full attention module
    # print(blocks[i])
# Optionally include qkv for the last block:
layers_to_test.append(("blocks[-1].attn.qkv", model.blocks[-1].attn.qkv))

# ========== GradCAM per Layer ==========
save_dir = os.path.join(args.save_dir, "gradcam_test_layers")
os.makedirs(save_dir, exist_ok=True)

for layer_name, layer_module in layers_to_test:
    try:
        cam = GradCAMPlusPlus(
            model=model,
            target_layers=[layer_module],
            # use_cuda="cuda" in args.device,
            reshape_transform=reshape_transform
        )

        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        vis = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
        save_path = os.path.join(save_dir, f"gradcam_{layer_name.replace('.', '_')}.png")
        plt.imsave(save_path, vis)
        print(colored(f"Saved Grad-CAM for {layer_name} at {save_path}", "cyan", force_color=True))
    except Exception as e:
        print(colored(f"Error processing {layer_name}: {e}", "red", force_color=True))
