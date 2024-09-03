import argparse
import logging
import os
import pprint
import random
import cv2

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import loralib as lora
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image


from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='custom', choices=['hypersim', 'vkitti', 'custom'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):                   # transform: 이미지 전처리 파이프라인
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for root, _, files in os.walk(self.data_dir):               # os.walk: 하위 디렉토리 탐색
            print(f"Searching in directory: {root}")
            for file in files:
                # 확장자 확인 및 모든 파일 이름 허용
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append(os.path.join(root, file))
        print(f"Found {len(data)} images in {self.data_dir}")
        return data                                                 # 이미지 파일의 수

    def __len__(self):
        return len(self.data)                                       # 데이터셋에 포함된 이미지 파일의 수 반환

    def __getitem__(self, idx):
        image_path = self.data[idx]
        # 이미지를 로드하고 전처리 적용
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # 예시로 반환하는 데이터
        return {'image': image}

def get_max_height(data_dir):
    max_height = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(os.path.join(root, file))
                height = image.size[1]  # 이미지의 높이
                if height > max_height:
                    max_height = height
    return max_height

def main():
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # Initialize other necessary components
    rank, world_size = 0, 1

    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    max_height = get_max_height("/home/san/d_drive/Depth-Anything-V2/custom/train")
    patch_size = 14
    new_height = (756 // patch_size) * patch_size
    new_width = (756 // patch_size) * patch_size

    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor(),
    ])

    # Dataset loading
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'custom':
        trainset = CustomDataset("/home/san/d_drive/Depth-Anything-V2/custom/train", transform=transform)
    else:
        raise NotImplementedError

    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=True)

    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'custom':
        valset = CustomDataset("/home/san/d_drive/Depth-Anything-V2/custom/val", transform=transform)
    else:
        raise NotImplementedError

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, shuffle=False)

    # Model and LoRA layer modification
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    modules_to_modify = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            modules_to_modify.append((name, module))

    for name, module in modules_to_modify:
        setattr(model, name, lora.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            merge_weights=False
        ))

    if args.pretrained_from:
        model.load_state_dict(
            {k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k},
            strict=False)

    model.cuda(rank)

    criterion = SiLogLoss().cuda(rank)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    for epoch in range(args.epochs):
        model.train()
        logger.info(f'===========> Epoch: {epoch}/{args.epochs}, d1: {previous_best["d1"]:.3f}, d2: {previous_best["d2"]:.3f}, d3: {previous_best["d3"]:.3f}')
        logger.info(f'===========> Epoch: {epoch}/{args.epochs}, abs_rel: {previous_best["abs_rel"]:.3f}, sq_rel: {previous_best["sq_rel"]:.3f}, rmse: {previous_best["rmse"]:.3f}, rmse_log: {previous_best["rmse_log"]:.3f}, log10: {previous_best["log10"]:.3f}, silog: {previous_best["silog"]:.3f}')

        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img = sample['image'].cuda()
            pred = model(img)

            if 'depth' in sample and 'valid_mask' in sample:
                depth = sample['depth'].cuda()
                valid_mask = sample['valid_mask'].cuda()
                loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            else:
                loss = torch.tensor(0.0, requires_grad=True).cuda()

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(f'Iter: {i}/{len(trainloader)}, LR: {optimizer.param_groups[0]["lr"]:.7f}, Loss: {loss.item():.3f}')


if __name__ == '__main__':
    main()
