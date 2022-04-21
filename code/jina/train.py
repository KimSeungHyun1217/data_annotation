import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
# from dataset import SceneTextDataset
from dataset import *

from model import EAST

import wandb
import random
import numpy as np

def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # print(f"seed : {seed}")

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR_2019'))
    parser.add_argument('--val_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/validation_2017'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=80)
    parser.add_argument('--save_interval', type=int, default=4)
    
    parser.add_argument('--wandb_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, val_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed, wandb_interval):
    
    # seed
    set_seed(seed)

    train_dataset = SceneTextDataset(data_dir, split='train')
    train_dataset = EASTDataset(train_dataset)

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1) # scheduler

    # wandb setting
    config = {
        'image_size':image_size, 'input_size':input_size, 'num_workers':num_workers, 'batch_size':batch_size,
        'learning_rate':learning_rate, 'epochs':max_epoch, 'seed':seed
    }
    wandb_name = 'jina_epoch_' + f'{max_epoch}' + '_lr_' + f'{learning_rate}'
    wandb.init(entity = 'yolo12', project = 'annotation', name = wandb_name, config = config)
    wandb.watch(model, log=all)
    
    # train
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()

        with tqdm(total=train_num_batches) as pbar:
            for iter, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict)
                                
                if (iter + 1) % wandb_interval == 0:
                    wandb.log({ "train/loss": loss.item(), 
                                "train/cls_loss": train_dict['Cls loss'],
                                "train/angle_loss": train_dict['Angle loss'],
                                "train/iou_loss": train_dict['IoU loss'],
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "epoch":epoch+1}, step=epoch * train_num_batches + iter)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)



def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
