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

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser(description="Classification with t-SNE and GradCAM Visualization")
# Data parameters
parser.add_argument("--test_dir", type=str, default="/mnt/HDD1/tuong/TRUST/baseline_TR_semi_AP/cp_BUSI2UCLM_TR_semi20_AP_1024/results/results_15.0k")
parser.add_argument("--resize_size", type=int, default=224, help="Resize size")
parser.add_argument("--crop_size", type=int, default=224, help="Crop size")
# Model parameters
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/vit_Breast-UCLM/best.pth", help="Pretrained model")
# Training parameters
parser.add_argument("--bz", type=int, default=16, help="Batch size")
# Device parameters
parser.add_argument("--device", default="cuda:1", help="Device")
# Save folder
parser.add_argument("--save_dir", type=str, default="./vis/gradcam_BUSI2UCLM_TRUST", help="Directory to save visualization")

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

# print(colored("Running t-SNE...", color="green", force_color=True))
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# features_2d = tsne.fit_transform(features)

# Create save directory
# tsne_dir = os.path.join(args.save_dir, "tsne")
# os.makedirs(tsne_dir, exist_ok=True)

# plt.figure(figsize=(8, 8))
# for label in np.unique(labels):
#     idxs = labels == label
#     plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=f'Class {label}', alpha=0.7)
# plt.legend()
# plt.title('t-SNE Visualization')
# plt.savefig(os.path.join(tsne_dir, "tsne.png"))
# plt.close()
# print(colored(f"Saved t-SNE visualization at {tsne_dir}/tsne.png", color="cyan", force_color=True))

# ================== Grad-CAM Visualization ==================
print(colored("Running Grad-CAM...", color="green", force_color=True))

def reshape_transform(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]
    batch_size, num_tokens, dim = tensor.shape
    return tensor.reshape(batch_size, height, width, dim).permute(0, 3, 1, 2)
# Grad-CAM setup
if args.model == 'vit':
    target_layers = [model.blocks[4].attn]
    reshape_fn = reshape_transform
else:
    target_layers = [model.layer4[-1]]
    reshape_fn = None

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_fn)

gradcam_dir = os.path.join(args.save_dir)
os.makedirs(gradcam_dir, exist_ok=True)

# Apply Grad-CAM to every image
# with torch.no_grad():
for batch in tqdm(test_loader, desc="Grad-CAM"):
    inputs = batch['img_w'].to(device)
    labels = batch['target']
    paths = batch['path']  # Ensure ImageList returns this!

    for idx in range(inputs.size(0)):
        input_tensor = inputs[idx].unsqueeze(0)
        input_image = inputs[idx].detach().cpu().permute(1, 2, 0).numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

        targets = [ClassifierOutputTarget(labels[idx].item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        # threshold = 0.5  # hoặc bạn có thể chỉnh ngưỡng này
        # mask = grayscale_cam > threshold
        # grayscale_cam = np.clip((grayscale_cam - threshold) / (1 - threshold), 0, 1)
        visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

        # Get original image filename (without extension)
        filename = os.path.basename(paths[idx])
        filename_no_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(gradcam_dir, f"{filename_no_ext}_gradcam.png")

        plt.imsave(save_path, visualization)
