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

from segmentation.models.model_dict import get_model
from segmentation.utils.data_us import JointTransform2D, ImageToImage2D_Test, ImageToImage2D
from segmentation.utils.config import get_config
from segmentation.utils.evaluation import get_eval
from segmentation.utils.loss_functions.sam_loss import get_criterion
from segmentation.utils.generate_prompts import get_click_prompt

##### Parser #####
parser = argparse.ArgumentParser(description='Medical Image Style Transfer')

### Dataset ###
parser.add_argument('--source_dir', type=str, default='../../dataset/Breast-UCLM/train',
                    help='source domain')
parser.add_argument('--target_dir', type=str, default='../../dataset/Breast-BUSI/train',
                    help='target domain')
parser.add_argument('--source_test_dir', type=str, default='../../dataset/Breast-UCLM/valid')
parser.add_argument('--source_semi_dir', type=str, default="/mnt/HDD1/tuong/TRUST/dataset/semi_seg")


parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--num_prompts', type=int, default=1024)

### Training parameters ###
parser.add_argument('--max_iters', type=int, default=20000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='bz', type=int, default=3)

### Loss parameters ###
parser.add_argument('--c_weight', type=float, default=1.0,
                    help='content loss weight')
parser.add_argument('--s_weight', type=float, default=1.0,
                    help='style loss weight')

### VGG model ###
parser.add_argument('--vgg_weights', type=str, default='../../vgg_normalised.pth')
parser.add_argument('--seg_weights', type=str, default='../../downstream/segmentation/SAMUS/checkpoints2/Breast-BUSI/SAMUS_best.pth')
parser.add_argument('--sam_ckpt', type=str, default='../../downstream/segmentation/SAMUS/checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
parser.add_argument('-encoder_input_size', type=int, default=256)
parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')

### Output ###
parser.add_argument('--save_dir', type=str, default='./cp_UCLM2BUSI_TR_semi40_AP',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=1000)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu', type=str, default='3')

args = parser.parse_args()
###########################
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

##### Set random seed #####
set_seed(args.seed)

### Config log file ###
args.save_dir = f"{args.save_dir}_{args.num_prompts}"
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.save_dir+"/test", exist_ok=True)

log_path = os.path.join(args.save_dir, "log" + ".txt")
re = ReDirectSTD(log_path, "stdout", True)

log_tensorboard = log_path.replace(".txt","")
if os.path.exists(log_tensorboard):
    shutil.rmtree(log_tensorboard)
writer = SummaryWriter(log_dir=log_tensorboard)

result_path = os.path.join(args.save_dir, 'results')
os.makedirs(result_path, exist_ok=True)

##### Data loading #####
log_str = 'Dataset Information: %s to %s' % (args.source_dir.split("/")[-2], args.target_dir.split("/")[-2])
print(colored(log_str, 'green', force_color=True))
log_str = f"Source: {args.source_dir} Target: {args.target_dir}"
print(colored(log_str, 'yellow', force_color=True))

### Training Dataset ###
source_dataset = ImageDataset(args.source_dir, transform=train_transform(args.img_size))
target_dataset = ImageDataset(args.target_dir, transform=train_transform(args.img_size))
source_loader = DataLoader(source_dataset, batch_size=args.bz, num_workers=4, shuffle=True, drop_last=True)
target_loader = DataLoader(target_dataset, batch_size=args.bz*2, num_workers=4, shuffle=True, drop_last=True)

log_str = "Source dataset length: %d, Target dataset length: %d" % (len(source_dataset), len(target_dataset))
print(colored(log_str, 'blue', force_color=True))

### Get Len (total_imgs / batch_size) ###
len_source = len(source_loader)
len_target = len(target_loader)

### Test Dataset ###
source_test_dataset = ImageDataset(args.source_test_dir, transform=test_transform(args.img_size))
target_test_dataset = ImageDataset(args.target_dir, transform=test_transform(args.img_size))
source_test_loader = DataLoader(source_test_dataset, batch_size=args.bz*2, num_workers=4, shuffle=False, drop_last=False)
target_test_loader = DataLoader(target_test_dataset, batch_size=args.bz*2, num_workers=4, shuffle=False, drop_last=True)
len_test_source = len(source_test_loader)
len_test_target = len(target_test_loader)
print(colored(f"Source test dataset length: {len(source_test_dataset)}, Target test dataset length: {len(target_test_dataset)}", 'cyan', force_color=True))

##### Model loading #####
device = torch.device(args.device)
### Load VGG model ###
print(colored("Loading VGG model...", 'red', force_color=True))
vgg.load_state_dict(torch.load(args.vgg_weights, map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

##### Downstream #####
task = args.target_dir.split('/')[-2].replace("Breast-", "")
print(colored(f"Task: {task}", 'green', force_color=True))
opt = get_config(task)

source_name = args.source_dir.split("/")[-2].replace("Breast-", "")
opt.data_path = args.source_semi_dir
opt.train_path = os.path.join(args.source_semi_dir, source_name, "semi.txt")

print(colored(f"Loading dataset: {opt.data_path}", 'green', force_color=True))
print(colored(f"Loading dataset: {opt.train_path}", 'green', force_color=True))

### Semi Dataset ###
tf_train = JointTransform2D(img_size=args.img_size, low_img_size=128, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                            p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
tf_val = JointTransform2D(img_size=args.img_size, low_img_size=128, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)

source_semi_dataset = ImageToImage2D(opt.data_path, opt.train_path, train_transform(args.img_size), tf_val, img_size=args.encoder_input_size)
source_semi_loader = DataLoader(source_semi_dataset, batch_size=args.bz, shuffle=True, drop_last=True)
len_source_semi = len(source_semi_loader)
log_str = "Source semi dataset length: %d" % (len(source_semi_dataset))
print(colored(log_str, 'red', force_color=True))

### Segmentation Downstream ###
seg_model = get_model(args.modelname, args=args, opt=opt)
seg_model.to(device)

print(colored(f"Loading Segmentation model: {args.seg_weights}.", 'green', force_color=True))
checkpoint = torch.load(args.seg_weights, map_location=device)
new_state_dict = {}
for k,v in checkpoint.items():
    if k[:7] == 'module.':
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
seg_model.load_state_dict(new_state_dict)
seg_model.train()
criterion = get_criterion(modelname=args.modelname, opt=opt).to(device)

# Freeze all the parameters in the model
for param in seg_model.parameters():
    param.requires_grad = False

### Style Transfer Network ###
source_encoder = SourceEncoder(img_size=args.img_size, patch_size=args.patch_size, num_prompts=args.num_prompts)
target_encoder = TargetEncoder(img_size=args.img_size, patch_size=args.patch_size)
TR_module = TokenDriven(num_decoder_layers=1)

# with torch.no_grad():
network = network.Network(source_encoder, target_encoder, TR_module, decoder, vgg)

network.train()
network.to(device)
#########################
# network = nn.DataParallel(network, device_ids=[3,4])

optimizer = torch.optim.Adam([ 
                            {'params': network.source_encoder.parameters()},
                            {'params': network.target_encoder.parameters()},
                            {'params': network.TR_module.parameters()},
                            {'params': network.decoder.parameters()}
                            ], lr=args.lr)

best_dice, best_IoU = 0.0, 0.0

##### Training #####
for step in tqdm(range(args.max_iters)):
    if step < 1e4:
        warmup_learning_rate(optimizer, step, args.lr)
    else:
        adjust_learning_rate(optimizer, step, args.lr_decay)

    ### Load train data ###
    if step % len_source == 0:
        iter_source = iter(source_loader)
    if step % len_target == 0:
        iter_target = iter(target_loader)
    if step % len_source_semi == 0:
        iter_source_semi = iter(source_semi_loader)

    batch_source = next(iter_source)
    batch_target = next(iter_target)
    batch_source_semi = next(iter_source_semi)

    ### Get data ###
    source_imgs = batch_source["img"].to(device)
    target_imgs = batch_target["img"].to(device)
    # Semi
    imgs = batch_source_semi['img_style'].to(dtype = torch.float32, device=device)
    masks = batch_source_semi['low_mask'].to(dtype = torch.float32, device=device)
    bbox = torch.as_tensor(batch_source_semi['bbox'], dtype=torch.float32, device=device)
    pt = get_click_prompt(batch_source_semi, opt)
    
    n_semi = imgs.size(0)
    ########################
    ##### Style Transfer #####
    source_batch = torch.cat((source_imgs, imgs), 0)

    Ics, loss_c, loss_s = network(source_batch, target_imgs)
        
    loss_c = args.c_weight * loss_c
    loss_s = args.s_weight * loss_s

    ##### Downstream Segmentation #####
    # out, mask, low_mask = tf_train(Ics[n_semi:], masks)

    pred = seg_model(Ics[n_semi:], pt, bbox)
    seg_loss = criterion(pred, masks)

    loss = loss_c + loss_s + seg_loss

    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    ### Print log ###
    if step % args.log_interval == 0 or step == args.max_iters - 1:
        print("Train Step: [{}/{}] Loss: {:.4f} - Content: {:.4f} - Style: {:.4f} - Segment: {:.4f}".format(
            step, args.max_iters, 
            loss.sum().cpu().detach().numpy(), 
            loss_c.sum().cpu().detach().numpy(), 
            loss_s.sum().cpu().detach().numpy(),
            seg_loss.cpu().detach().numpy()
        ))
        
    if step % args.save_interval == 0 or step == args.max_iters - 1:
        ##### Save iamge #####
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(step),".jpg"
                    )
        out = torch.cat((source_batch, Ics),0)
        out = torch.cat((target_imgs, out),0)
        save_image(out, output_name)

    writer.add_scalar('loss_content', loss_c.sum().item(), step + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), step + 1)
    writer.add_scalar('loss_seg', seg_loss.item(), step + 1)
    writer.add_scalar('total_loss', loss.sum().item(), step + 1)

    if (step + 1) % args.test_interval == 0:
        network.eval()

        ### Save results ###
        path_save = f"{result_path}/results_{(step+1)/1000}k"
        os.makedirs(path_save, exist_ok=True)
        
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

                output_name = os.path.join(path_save, img_name)
                save_image(img.cpu(), output_name)

        network.train()

        #### Testing ####
        seg_model.eval()
        log_str = f"Testing at {path_save}"
        print(colored(log_str, color="magenta", force_color=True))

        test_path = f"{args.source_test_dir}.txt"
        test_dataset = ImageToImage2D_Test(path_save, test_path, tf_val, img_size=args.encoder_input_size, class_id=1)
        test_loader = DataLoader(test_dataset, batch_size=args.bz, shuffle=False, drop_last=False)

        dices, mean_dice, _, mean_IoU, val_losses = get_eval(test_loader, seg_model, criterion=criterion, opt=opt, args=args)
        seg_model.train()

        if mean_dice > best_dice:
            best_step_dice = step + 1
            best_dice = mean_dice
            log_str = f"Best Dice: {best_dice * 100:.2f}%"
            print(colored(log_str, color="red", force_color=True))

            torch.save(network.state_dict(), f"{args.save_dir}/best_network_dice.pth")

        if mean_IoU > best_IoU:
            best_step_IoU = step + 1
            best_IoU = mean_IoU
            log_str = f"Best IoU: {best_IoU:.2f}"
            print(colored(log_str, color="red", force_color=True))

            torch.save(network.state_dict(), f"{args.save_dir}/best_network_iou.pth")

        print(f"Validation Dice: {mean_dice * 100:.2f}% \t IoU: {mean_IoU:.2f}")
        log_str = f"Best Dice at {best_step_dice}: {best_dice * 100:.2f} \t Best IoU at {best_step_IoU}: {best_IoU:.2f}"
        print(colored(log_str, color="red", force_color=True))
                                                    
writer.close()
