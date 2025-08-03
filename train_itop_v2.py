"""
Module for training the SPiKE model on the ITOP dataset.
"""

import os
import sys
import argparse
import torch
import torch.utils.data
from torch import nn
from model import model_builder_v2
from trainer_itop_v2 import (
    train_one_epoch,
    evaluate,
    load_data,
    create_criterion,
    create_optimizer_and_scheduler,
)
from utils.config_utils import load_config, set_random_seed
from utils.distrib_utils import is_main_process

def main(arguments):
    config = load_config(arguments.config)

    import datetime
    now = datetime.datetime.now()
    dir_name = (now.strftime("%Y%m%d_%H:%M") 
                +"_"+ str(config["loss_type"]) 
                +"_d"+ str(config["device_args"]) 
                +"_e"+ str(config["epochs"])
                +"_b"+ str(config["batch_size"])
                +"_f"+ str(config["frames_per_clip"]) + "_" + str(config["target_frame"]) 
                +"_r"+ str(config["radius"]) 
                +"_n" + str(config["nsamples"]) 
                )
    if arguments.model is not None:
        dir_name = dir_name + "_wInit"
    
    dir_name = dir_name + "_self_2"

    if arguments.info is not None:
        dir_name = dir_name + "_" + arguments.info
    dir_name = dir_name

    out_dir = os.path.join(config["output_dir"], dir_name)
    #out_dir = os.path.join(config["output_dir"], dt_now.isoformat())
    print(f"output directory: {dir_name}")
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    torch.cuda.set_device(config["device_args"])
    device = torch.device(config["device_args"])
    print("device: ", device)

    set_random_seed(config["seed"])

    print(f"Loading data from {config['dataset_path']}")
    data_loader, data_loader_test, num_coord_joints = load_data(config)


    #''' Add
    if arguments.model is not None:
        # pretrained model (SPiKE)
        from model import model_builder
        pretrained_model = model_builder.create_model(config, num_coord_joints)
        pretrained_model.to(device)

        from const import path
        pretrained_model_path = os.path.join(path.EXPERIMENTS_PATH, arguments.model)
        print(f"Loading pretrained_model from {pretrained_model}")
        checkpoint = torch.load(pretrained_model_path, map_location="cpu")
        pretrained_model.load_state_dict(checkpoint["model"], strict=True)

        # target model
        model = model_builder_v2.create_model(config=config, num_coord_joints=num_coord_joints, backbone="pointnet++")
        model_without_ddp = model

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        model.to(device)
    elif arguments.model is None:
        pretrained_model = None
        model = model_builder_v2.create_model(config=config, num_coord_joints=num_coord_joints, backbone="pointnet++")
        model_without_ddp = model

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        model.to(device)
    #'''

    criterion = create_criterion(config)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)

    if config["resume"]:
        print(f"Loading model from {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        config["start_epoch"] = checkpoint["epoch"] + 1
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])


    print("Start training")
    min_loss = sys.maxsize
    eval_thresh = config["threshold"]

    best_train_loss, best_train_map, best_train_pck, best_val_loss, best_val_map, best_val_pck = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_epoch = 0

    list_train_loss, list_train_map, list_train_pck, list_val_loss, list_val_map, list_val_pck = [], [], [], [], [], []
    # Add MyLoss
    #list_train_loss1, list_train_loss2, list_train_loss3, list_val_loss1, list_val_loss2, list_val_loss3 = [], [], [], [], [], []

    for epoch in range(config["start_epoch"], config["epochs"]):
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        
        train_clip_loss, train_pck, train_map = train_one_epoch(
        # Add MyLoss
        #train_clip_loss, train_pck, train_map, train_loss1, train_loss2, train_loss3 = train_one_epoch(
            pretrained_model,
            model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, eval_thresh,
            out_dir
        )

        val_clip_loss, val_pck, val_map = evaluate(
        # Add MyLoss
        #val_clip_loss, val_pck, val_map, val_loss1, val_loss2, val_loss3 = evaluate(
            pretrained_model, 
            model, criterion, data_loader_test, device=device, threshold=eval_thresh
        )

        print(f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}")
        # Add MyLoss
        #print(f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}={train_loss1:.4f}+{train_loss2:.4f}+{train_loss3:.4f}")
        
        print(f"Epoch {epoch} - Train mAP: {train_map:.4f}")
        print(f"Epoch {epoch} - Train PCK: {train_pck}")
        
        print(f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}")
        # Add MyLoss
        #print(f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}={val_loss1:.4f}+{val_loss2:.4f}+{val_loss3:.4f}")
        
        print(f"Epoch {epoch} - Validation mAP: {val_map:.4f}")
        print(f"Epoch {epoch} - Validation PCK: {val_pck}")

        list_train_loss.append(train_clip_loss)
        list_train_map.append(train_map)
        list_train_pck.append(train_pck)
        list_val_loss.append(val_clip_loss)
        list_val_map.append(val_map)
        list_val_pck.append(val_pck)

        #list_train_loss1.append(train_loss1)
        #list_train_loss2.append(train_loss2)
        #list_train_loss3.append(train_loss3)
        #list_val_loss1.append(val_loss1)
        #list_val_loss2.append(val_loss2)
        #list_val_loss3.append(val_loss3)


        if config["output_dir"] and is_main_process():
            model_to_save = (model_without_ddp.module
                if isinstance(model_without_ddp, nn.DataParallel)
                else model_without_ddp
            )

            checkpoint = {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": config,
            }
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint.pth"))

            if val_clip_loss < min_loss:
                min_loss = val_clip_loss

                best_epoch      = epoch
                best_train_loss = train_clip_loss
                best_train_map  = train_map
                best_train_pck  = train_pck
                best_val_loss   = val_clip_loss
                best_val_map    = val_map
                best_val_pck    = val_pck

                #torch.save(checkpoint, os.path.join(config["output_dir"], "best_model.pth"))
                torch.save(checkpoint, os.path.join(out_dir, "best_model.pth"))


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check result
    print("~~~~~ best result ~~~~~")
    print(f"Epoch: {best_epoch}")
    print(f"Train Loss: {best_train_loss:.4f}")
    print(f"Train mAP: {best_train_map:.4f}")
    print(f"Train PCK: {best_train_pck}")
    print(f"Validation Loss: {best_val_loss:.4f}")
    print(f"Validation mAP: {best_val_map:.4f}")
    print(f"Validation PCK: {best_val_pck}")
    print("~~~~~~~~~~~~~~~~~~~~~~~")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # save result
    import matplotlib.pyplot as plt
    ''' Add MyLoss
    plt.plot(range(len(list_train_loss1)), list_train_loss1, label="train_loss1")
    plt.plot(range(len(list_train_loss2)), list_train_loss2, label="train_loss2")
    plt.plot(range(len(list_train_loss3)), list_train_loss3, label="train_loss3")
    plt.plot(range(len(list_val_loss1)), list_val_loss1, label="test_loss1")
    plt.plot(range(len(list_val_loss2)), list_val_loss2, label="test_loss2")
    plt.plot(range(len(list_val_loss3)), list_val_loss3, label="test_loss3")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.clf()
    plt.close()
    #'''
    
    plt.plot(range(len(list_train_loss)), list_train_loss, label="Train")
    plt.plot(range(len(list_val_loss)), list_val_loss, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.clf()
    plt.close()

    plt.plot(range(len(list_train_map)), list_train_map, label="Train")
    plt.plot(range(len(list_val_map)), list_val_map, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.savefig(os.path.join(out_dir, "mAP.png"))
    plt.clf()
    plt.close()

    import numpy as np
    list_train_pck = np.array(list_train_pck)
    list_val_pck = np.array(list_val_pck)
    #print("list_train_pck.shape: ", list_train_pck.shape)   #(epochs, 15)
    for n in range(list_train_pck.shape[1]):
        label_train = "Train:" + str(n)
        label_test  = "Test:" + str(n)
        plt.plot(range(len(list_train_pck)), list_train_pck[:,n], label=label_train)
        plt.plot(range(len(list_val_pck)), list_val_pck[:,n], label=label_test)
        plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("pck")
    plt.savefig(os.path.join(out_dir, "pck.png"))
    plt.clf()
    plt.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """
    model = model_builder_self_2.create_model(config=config, num_coord_joints=num_coord_joints, backbone="pointnet++")
    model.to(device)
    model_path = os.path.join(out_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    from tqdm import tqdm
    from utils import my_functions
    cnt = 0
    #temp_dir = "temp_frames"
    #os.makedirs(temp_dir, exist_ok=True)

    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0
    from utils import metrics
    with torch.no_grad():
        for clip, target, _ in tqdm(data_loader_test, desc="create video"):
            clip, target = clip.to(device, non_blocking=True), target.to(device, non_blocking=True)

            #''' Add
            if pretrained_model is not None:
                pretrained_output = pretrained_model(clip).reshape(target.shape)
                output = model(clip, pretrained_output).reshape(target.shape)
            elif pretrained_model is None:
                output, _ = model(clip[:,-1].unsqueeze(1), target[:,0:-1])
                output = output.reshape(target[:,-1,:,:].unsqueeze(1).shape)


            loss = criterion(output, target[:,-1,:,:].unsqueeze(1))
            pck, mean_ap = metrics.joint_accuracy(output, target[:,-1,:,:].unsqueeze(1), eval_thresh)
            total_pck += pck.detach().cpu().numpy()
            total_map += mean_ap.detach().cpu().item()
            total_loss += loss.item()
            
            ''' 動画用画像保存
            for n in range(clip.shape[0]):
                file_name = os.path.join(temp_dir, f"frame_{cnt+1:05d}.png")  # 5桁の番号でファイル名を付ける
                fig = my_functions.plot_point_cloud_and_joints(point_cloud=clip[n][config["frames_per_clip"]-1], 
                                                            pred_joints=output[n][0],
                                                            label_joints=target[n][0]
                                                            )
                my_functions.save_fig(fig=fig, file_name=file_name)
                cnt += 1    #'''

    total_loss /= len(data_loader_test)
    total_map /= len(data_loader_test)
    total_pck /= len(data_loader_test)

    print(f"Validation Loss: {best_val_loss:.4f} = {total_loss:.4f}"  )
    print(f"Validation mAP: {best_val_map:.4f} = {total_map:.4f}")
    print(f"Validation PCK: {best_val_pck} = {total_pck}")
    """
    
    '''# 画像を動画に変換
    import glob
    img_files  = glob.glob(os.path.join(temp_dir, '*.png'))
    img_files.sort()
    frames=len(img_files)
    assert frames != 0, 'not found image file'

    import cv2
    img = cv2.imread(img_files[0])
    h, w, _ = img.shape[:3]

    # 作成する動画
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(out_dir, "result.mp4"), codec, 5, (w, h),1)

    bar = tqdm(total=frames, dynamic_ncols=True)
    for f in img_files:
        # 画像を1枚ずつ読み込んで 動画へ出力する
        img = cv2.imread(f)
        writer.write(img)   
        bar.update(1)
    
    # 後始末
    bar.close()
    writer.release()
    import shutil
    shutil.rmtree(temp_dir)
    #'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE Training on ITOP dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/ITOP-SIDE/1",
        help="Path to the YAML config file",
    )

    #''' Add
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--info",
        type=str,
        default=None,
        help="add information to the directory name",
    )
    #'''

    args = parser.parse_args()
    main(args)
