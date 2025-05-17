import argparse
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.transformer import SourceEncoder, TargetEncoder, TokenDriven
import models.network as network 
from models.vgg import vgg
from models.decoder import decoder

from torchvision.utils import save_image

from utils.dataloader import ImageDataset, train_transform, test_transform
from utils.scheduler import adjust_learning_rate, warmup_learning_rate

from termcolor import colored
from log_utils.utils import ReDirectSTD

from torch.utils.data import DataLoader
from utils.utils import set_seed
import shutil

from classification.utils import validate
from classification.dataloader import ImageList
from classification.preprocess import cls_train_transform, cls_val_transform
import timm

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--source_dir', type=str, default='../dataset/Breast-BUSI/train',
                    help='source domain')
parser.add_argument('--target_dir', type=str, default='../dataset/Breast-UDIAT/train',
                    help='target domain')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--batch_size', dest='bz', type=int, default=3)
parser.add_argument('--num_prompts', type=int, default=1024)

### VGG model ###
parser.add_argument('--vgg_weights', type=str, default='../vgg_normalised.pth')

### Output ###
parser.add_argument('--save_dir', type=str, default='./results',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1234)

parser.add_argument('--device', type=str, default='cuda:1')

args = parser.parse_args()
###########################

##### Set random seed #####
set_seed(args.seed)

### Config log file ###
source_name = args.source_dir.split('/')[-2].replace('Breast-', '')
target_name = args.target_dir.split('/')[-2].replace('Breast-', '')
task = f"{source_name}2{target_name}"
print(colored(f"Task: {task}", 'red', force_color=True))
save_dir = f"{args.save_dir}/{task}"
os.makedirs(save_dir, exist_ok=True)

# Model
network_weights = f"./cp_{task}_TR_semi20_AP_1024/best_network_acc.pth"
cls_weights = f"../downstream/classification/checkpoints/vit_Breast-{target_name}/best.pth"

### Testing Dataset ###
source_test_dataset = ImageDataset(args.source_dir, transform=test_transform(args.img_size))
target_test_dataset = ImageDataset(args.target_dir, transform=test_transform(args.img_size))
source_test_loader = DataLoader(source_test_dataset, batch_size=args.bz, num_workers=4, shuffle=False, drop_last=False)
target_test_loader = DataLoader(target_test_dataset, batch_size=args.bz, num_workers=4, shuffle=False, drop_last=True)
len_test_source = len(source_test_loader)
len_test_target = len(target_test_loader)
print(colored(f"Source test dataset length: {len(source_test_dataset)}, Target test dataset length: {len(target_test_dataset)}", 'cyan', force_color=True))

##### Model loading #####
device = torch.device(args.device)
### Load VGG model ###
print(colored("Loading VGG model...", 'red', force_color=True))
vgg.load_state_dict(torch.load(args.vgg_weights, map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

### Classification Downstream ###
cls_model = timm.create_model('timm/vit_base_patch16_224', pretrained=True, num_classes=2)
print(colored(f"Loading Classification model {cls_weights}", 'red', force_color=True))
cls_model.load_state_dict(torch.load(cls_weights, map_location=device))
cls_model = cls_model.to(device)
cls_model.eval()

# Freeze all the parameters in the model
for param in cls_model.parameters():
    param.requires_grad = False

ce_loss = nn.CrossEntropyLoss().to(device)
cls_transform = cls_train_transform(args.img_size, 224)

### Style Transfer Network ###
source_encoder = SourceEncoder(img_size=args.img_size, patch_size=args.patch_size, num_prompts=args.num_prompts)
target_encoder = TargetEncoder(img_size=args.img_size, patch_size=args.patch_size)
TR_module = TokenDriven(num_decoder_layers=1)

with torch.no_grad():
    network = network.Network(source_encoder, target_encoder, TR_module, decoder, vgg)
print(colored(f"Loading Network model {network_weights}", 'red', force_color=True))
network.load_state_dict(torch.load(network_weights, map_location=device))

network.eval()
network.to(device)
#########################

iter_test_target = iter(target_test_loader)
for i, batch_test_source in tqdm(enumerate(source_test_loader)):
    ### Get data ###
    # Source #
    source_test_imgs = batch_test_source["img"].to(device)
    source_test_paths = batch_test_source["path"]

    # Target #
    if i % len_test_target == 0:
        iter_test_target = iter(target_test_loader)

    batch_test_target = next(iter_test_target)
    target_test_imgs = batch_test_target["img"].to(device)
    ################

    ##### inference #####
    with torch.no_grad():
        assert source_test_imgs.size(0) == target_test_imgs[:source_test_imgs.size(0)].size(0), "Source and target batch size should be equal"
        Ics, *_ = network(source_test_imgs, target_test_imgs[:source_test_imgs.size(0)])
    
    for idx, img in enumerate(Ics):
        img_name = source_test_paths[idx].split('/')[-1]                  

        output_name = os.path.join(save_dir, img_name)
        save_image(img.cpu(), output_name)

#### Testing ####
log_str = f"Testing at {save_dir}"
print(colored(log_str, color="magenta", force_color=True))

test_dataset = ImageList(save_dir, transform_w=cls_val_transform(256, 224)) # args.target_dir
test_loader = DataLoader(test_dataset, batch_size=args.bz, shuffle=False, drop_last=False)
val_acc, val_AUC = validate(cls_model, test_loader, device)

print(f"Test accuracy: {val_acc*100:.2f}, Test AUC: {val_AUC*100:.2f}")
   

