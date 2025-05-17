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
from sklearn.preprocessing import StandardScaler

from utils.dataloader import ImageList
from utils.preprocess import val_transform

# Argument parser
parser = argparse.ArgumentParser(description="t-SNE on logits across domains")
parser.add_argument("--resize_size", type=int, default=256)
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--checkpoints", type=str, default="/mnt/HDD1/tuong/TRUST/downstream/classification_vis/checkpoints/vit_Breast-UCLM/best.pth")
parser.add_argument("--bz", type=int, default=16)
parser.add_argument("--device", default="cuda:6")
parser.add_argument("--save_dir", type=str, default="./vis")
args = parser.parse_args()

device = torch.device(args.device)

# Domain list
domains = [
    ("BUSBRA", "/mnt/HDD1/tuong/TRUST/dataset/Breast-BUSBRA"),
    ("UCLM", "/mnt/HDD1/tuong/TRUST/dataset/Breast-UCLM"),
    ("BUSI", "/mnt/HDD1/tuong/TRUST/dataset/Breast-BUSI"),
    # ("TRUST", "data/TRUST/UDIAT-UCLM"),
    ("UDIAT", "/mnt/HDD1/tuong/TRUST/dataset/Breast-UDIAT"),
]

# Load model
print(colored(f"Loading model: {args.model}", "red", force_color=True))
if args.model == 'resnet34':
    model = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes=2)
elif args.model == 'resnet50':
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=2)
elif args.model == 'resnet101':
    model = timm.create_model('resnet101.a1_in1k', pretrained=True, num_classes=2)
elif args.model == 'vit':
    model = timm.create_model('timm/vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, num_classes=2)
else:
    raise ValueError(f"Model {args.model} not supported")

# Load checkpoint
model.load_state_dict(torch.load(args.checkpoints, map_location=device))
model = model.to(device)
model.eval()

# Extract logits
def extract_logits(model, dataloader, device):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            inputs = batch['img_w'].to(device)
            outputs = model(inputs)  # Direct logits
            # outputs = model.forward_features(inputs)
            print(outputs)
            logits_list.append(outputs.cpu())
    logits = torch.cat(logits_list, dim=0)
    logits = logits.view(logits.size(0), -1).numpy()
    return logits

# Collect logits and labels
all_logits = []
all_labels = []

for name, path in domains:
    print(colored(f"Processing {name}", "blue", force_color=True))
    dataset = ImageList(path, transform_w=val_transform(args.resize_size, args.crop_size))
    loader = DataLoader(dataset, batch_size=args.bz, shuffle=False, num_workers=4, pin_memory=True)

    logits = extract_logits(model, loader, device)
    all_logits.append(logits)
    all_labels.extend([name] * len(logits))

# Stack logits and apply t-SNE
logits_concat = np.vstack(all_logits)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
logits_2d = tsne.fit_transform(logits_concat)
logits_2d = StandardScaler().fit_transform(logits_2d)

# Plotting
print(colored("Plotting t-SNE on logits...", "magenta", force_color=True))
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'orange', 'purple']

domain_set = sorted(set(all_labels))
for idx, domain in enumerate(domain_set):
    inds = [i for i, label in enumerate(all_labels) if label == domain]
    plt.scatter(
        logits_2d[inds, 0],
        logits_2d[inds, 1],
        label=domain,
        color=colors[idx % len(colors)],
        alpha=0.7,
        marker='o',
        edgecolors='k',
        linewidths=0.3
    )

plt.title("t-SNE on Logits (All Domains)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.tight_layout()

# Save
save_dir = os.path.join(args.save_dir, "tsne_logits_combined")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "tsne_all_domains_logits.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(colored(f"Saved plot at {save_path}", "cyan", force_color=True))
