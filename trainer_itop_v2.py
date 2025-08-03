"""
Module for training and evaluating the SPiKE model on the ITOP dataset.
"""

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from datasets.my_itop import ITOP
from utils import metrics, scheduler
import os
from const import skeleton_joints

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    
    def forward(self, input, target):
        # define weights
        w1 = 1.0
        w2 = 1.0
        w3 = 1.0

        # loss1: regression
        loss_1 = torch.mean(torch.abs(input - target))

        # loss2: bone length
        total_loss_2 = 0.0
        for b in range(input.shape[0]):
            for limb in skeleton_joints.joint_connections:
                #limb: (idx1, idx2, limb colour)
                p1 = limb[0]
                p2 = limb[1]
                len_input = torch.abs(input[b, p1] - input[b, p2]).sum()
                len_target = torch.abs(target[b, p1] - target[b, p2]).sum()
                total_loss_2 += torch.abs(len_input - len_target)      
        loss_2 = total_loss_2 / len(skeleton_joints.joint_connections) / input.shape[0]

        # loss3: symmetry
        total_loss_3 = 0.0
        for b in range(input.shape[0]):
            for limb in skeleton_joints.joint_symmetry:
                #limb: (idx1, idx2, limb colour)
                p1 = limb[0]
                p2 = limb[1]

                p3 = limb[2]
                p4 = limb[3]

                len_left = torch.abs(input[b, p1] - input[b, p2]).sum()
                len_right = torch.abs(target[b, p3] - target[b, p4]).sum()
                total_loss_3 += torch.abs(len_left - len_right)
        
        loss_3 = total_loss_3 / len(skeleton_joints.joint_connections) / input.shape[0]


        loss = loss_1 * w1 + loss_2 * w2 + loss_3

        return loss, loss_1, loss_2, loss_3


def train_one_epoch(
    pretrained_model,
    model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, threshold,
    out_dir,
):
    model.train()
    header = f"Epoch: [{epoch}]"
    total_loss, total_pck, total_map = 0.0, np.zeros(15), 0.0
    
    total_loss1, total_loss2, total_loss3 = 0.0, 0.0, 0.0
    
    cnt_bs = 0
    for clip, target, _ in tqdm(data_loader, desc=header):  # clip: [24, 3, 4096, 3], target: [24, 3, 15, 3]
        clip, target = clip.to(device), target.to(device)

        output = model(clip[:,-1].unsqueeze(1).contiguous(), target)
        output =  output.reshape(target[:,-1].shape)
   
        loss, loss1, loss2, loss3 = criterion(output, target[:,-1])
        
        pck, mean_map = metrics.joint_accuracy(output.unsqueeze(1), target[:,-1].unsqueeze(1), threshold)
        total_pck += pck.detach().cpu().numpy()
        total_map += mean_map.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()

        lr_scheduler.step()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    total_loss1 /= len(data_loader)
    total_loss2 /= len(data_loader)
    total_loss3 /= len(data_loader)
    
    return total_loss, total_pck, total_map, total_loss1, total_loss2, total_loss3

def evaluate(
        pretrained_model,
        model, criterion, data_loader, device, threshold
        ):
    model.eval()
    total_loss, total_pck, total_map = 0.0, np.zeros(15), 0.0
    
    total_loss1, total_loss2, total_loss3 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for clip, target, _ in tqdm(
            data_loader, desc="Validation" if data_loader.dataset.train else "Test"
        ):
            clip, target = clip.to(device, non_blocking=True), target.to(device, non_blocking=True)

            output = model(clip[:,-1].unsqueeze(1).contiguous(), target)
            output = output.reshape(target[:,-1].shape)

            loss, loss1, loss2, loss3 = criterion(output, target[:,-1])
            
            pck, mean_map = metrics.joint_accuracy(output.unsqueeze(1), target[:,-1].unsqueeze(1), threshold)
            total_pck += pck.detach().cpu().numpy()
            total_map += mean_map.detach().cpu().item()
            total_loss += loss.item()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    total_loss1 /= len(data_loader)
    total_loss2 /= len(data_loader)
    total_loss3 /= len(data_loader)

    return total_loss, total_pck, total_map, total_loss1, total_loss2, total_loss3

def load_data(config, mode="train"):
    """
    Load the ITOP dataset.

    Args:
        config (dict): The configuration dictionary.
        mode (str): The mode to load the data in ("train" or "test").

    Returns:
        tuple: A tuple containing the data loader(s) and the number of coordinate joints.
    """
    dataset_params = {
        "root": config["dataset_path"],
        "frames_per_clip": config["frames_per_clip"],
        "num_points": config["num_points"],
        "use_valid_only": config["use_valid_only"],
        "target_frame": config["target_frame"],
    }

    if mode == "train":
        dataset = ITOP(
            train=True, aug_list=config["PREPROCESS_AUGMENT_TRAIN"], **dataset_params
        )
        dataset_test = ITOP(
            train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["workers"],
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=config["batch_size"],
            num_workers=config["workers"]
        )
        
        return data_loader, data_loader_test, dataset.num_coord_joints

    dataset_test = ITOP(train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["batch_size"], num_workers=config["workers"])

    return data_loader_test, dataset_test.num_coord_joints

def create_criterion(config):
    """
    Create the loss criterion based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        torch.nn.Module: The loss function.
    """
    loss_type = config.get("loss_type", "std_cross_entropy")

    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "my":
        return MyLoss()
    raise ValueError("Invalid loss type. Supported types: 'l1', 'mse'.")

def create_optimizer_and_scheduler(config, model, data_loader):
    lr = config["lr"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    warmup_iters = config["lr_warmup_epochs"] * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in config["lr_milestones"]]
    lr_scheduler = scheduler.WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=config["lr_gamma"],
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )
    return optimizer, lr_scheduler
