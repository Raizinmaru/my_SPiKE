"""
Module for evaluating the SPiKE model on the ITOP dataset.
"""

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import model_builder_v2
from trainer_itop_v2 import load_data, create_criterion
from utils.config_utils import load_config, set_random_seed
from utils.metrics import joint_accuracy

from utils import my_functions

def evaluate(
        pre_model,
        config,
        save,
        model, criterion, data_loader, device, threshold):

    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0
    clip_losses = []

    # Add MyLoss
    #total_loss1, total_loss2, total_loss3 = 0.0, 0.0, 0.0

    total_inference_time = 0.0
    total_inference_time_pre = 0.0
    inference_cnt = 0
    inference_cnt_pre = 0

    initial_frames = 2
    #poses_cnt = 0
    poses_history = []
    switch = 0

    frame_cnt = 0
    log_video_id = 0

    if save is not None:
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(
            data_loader, desc="Validation" if data_loader.dataset.train else "Test"
        ):
            for clip, target, video_id in zip(batch_clips, batch_targets, batch_video_ids):
                frame_cnt += 1
                if frame_cnt < 3: continue

                clip = clip.unsqueeze(0).to(device, non_blocking=True)
                target = target.unsqueeze(0).to(device, non_blocking=True)
                #print(clip.shape)   #(1, 3, 4096, 3)
                #print(target.shape) #(1, 3, 15, 3)

                if abs(video_id[0][1].item() - log_video_id) > 1:
                    poses_history = []
                    
                log_video_id = video_id[0][1].item()

                start.record()
                # ~~~~~~~~~~
                output = pre_model(clip)
                # ~~~~~~~~~~
                end.record()
                torch.cuda.synchronize()
                total_inference_time_pre += start.elapsed_time(end)

                if len(poses_history) == initial_frames:
                    pose_seq = torch.cat((torch.stack(poses_history, dim=1), target[:,-1].unsqueeze(1)), dim=1)    # [1, 3, 15, 3]
                    #pose_seq = torch.cat((torch.stack(poses_history, dim=1), output.reshape(target[:,-1].shape).unsqueeze(0)), dim=1)    # [1, 3, 15, 3]
                    start.record()
                    # ~~~~~~~~~~
                    output = model(clip[:,-1].unsqueeze(1), pose_seq)
                    # ~~~~~~~~~~
                    end.record()
                    torch.cuda.synchronize()
                    total_inference_time += start.elapsed_time(end)
                    inference_cnt +=1
                    switch = 1
                else:
                    inference_cnt_pre +=1
                    switch = 0

                output =  output.reshape(target[:,-1].shape)

                poses_history.append(output)
                if len(poses_history) > initial_frames:
                    poses_history.pop(0) # 最古のposeを削除


                loss = criterion(output, target[:,-1])
                # Add MyLoss
                #loss, loss1, loss2, loss3 = criterion(output, target[:,-1])
                #loss, _, _, _ = criterion(output, target[:,-1])

                pck, mean_map = joint_accuracy(output.unsqueeze(1), target[:,-1].unsqueeze(1), threshold)
                total_pck += pck.detach().cpu().numpy()
                total_map += mean_map.detach().cpu().item()

                total_loss += loss.item()
                
                # Add MyLoss
                #total_loss1 += loss1.item()
                #total_loss2 += loss2.item()
                #total_loss3 += loss3.item()

                clip_losses.append(
                    (
                        video_id.cpu().detach().numpy(),
                        loss.item(),
                        clip.cpu().detach().numpy(),
                        target.cpu().detach().numpy(),
                        output.cpu().detach().numpy(),
                    )
                )

                if save is not None:
                    for b in range(clip.shape[0]):
                        file_name = os.path.join(tmp_dir, f"{frame_cnt:05d}_{video_id[b][0].item():02d}_{video_id[b][1].item():05d}_{switch}.png")  # 5桁の番号でファイル名を付ける
                        fig = my_functions.plot_point_cloud_and_joints(point_cloud=clip[b][-1], 
                                                                    pred_joints=output[b],
                                                                    #label_joints=target[b][-1]
                                                                    )
                        my_functions.save_fig(fig=fig, file_name=file_name)
                

        total_loss /= len(data_loader.dataset)
        total_map /= len(data_loader.dataset)
        total_pck /= len(data_loader.dataset)

        # Add MyLoss
        #total_loss1 /= len(data_loader.dataset)
        #total_loss2 /= len(data_loader.dataset)
        #total_loss3 /= len(data_loader.dataset)

        total_inference_time /= inference_cnt
        total_inference_time_pre /= (inference_cnt_pre+inference_cnt)
        print(f"Inference Times: {len(data_loader.dataset)}=={frame_cnt}=={inference_cnt}+{inference_cnt_pre}")
        print(f"Inference Time Pre: {total_inference_time_pre:.4f} ms")
        print(f"Inference Time: {total_inference_time:.4f} ms")

    return clip_losses, total_loss, total_map, total_pck
    # Add MyLoss
    #return clip_losses, total_loss, total_map, total_pck, total_loss1, total_loss2, total_loss3


def main(arguments):

    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"EPOCHS: {config['epochs']}")
    print(f"LOSS: {config.get('loss_type', 'std_cross_entropy')}")
    device = torch.device(0)
    print(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']} = device: {device}")
    set_random_seed(config["seed"])

    print(f"Loading data from {config['dataset_path']}")
    data_loader_test, num_coord_joints = load_data(config, mode="test")

    from const import path
    model_path = os.path.join(path.EXPERIMENTS_PATH, arguments.model)
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model = model_builder_v2.create_model(config, num_coord_joints, "pointnet++")
    model.to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    # spike
    from model import model_builder
    pre_model = model_builder.create_model(config, num_coord_joints)
    pre_model.to(device)
    pre_model_path = os.path.join(path.EXPERIMENTS_PATH, arguments.pre)
    print(f"Loading model from {pre_model_path}")
    pre_checkpoint = torch.load(pre_model_path, map_location="cpu")
    pre_model.load_state_dict(pre_checkpoint["model"], strict=True)

    criterion = create_criterion(config)

    losses, val_clip_loss, val_map, val_pck = evaluate(
    # Add MyLoss
    #losses, val_clip_loss, val_map, val_pck, val_loss1, val_loss2, val_loss3 = evaluate(
        pre_model,
        config,
        arguments.save,
        model, criterion, data_loader_test, device=device, threshold=config["threshold"], 
    )
    losses.sort(key=lambda x: x[1], reverse=True)

    print(f"Validation Loss: {val_clip_loss:.4f}")
    # Add MyLoss
    #print(f"Validation Loss: {val_clip_loss:.4f}={val_loss1:.4f}+{val_loss2:.4f}+{val_loss1:.3f}")
    print(f"Validation mAP: {val_map:.4f}")
    print(f"Validation PCK: {val_pck}")

    import sys
    sys.exit()


    temp_dir = "tmp"
    import glob
    img_files  = glob.glob(os.path.join(temp_dir, '*.png'))
    img_files.sort()
    frames=len(img_files)
    assert frames != 0, 'not found image file'

    import cv2
    img = cv2.imread(img_files[0])
    h, w, _ = img.shape[:3]

    # 作成する動画
    out_dir = os.path.dirname(arguments.model)
    print(f"out_dir: {out_dir}")  # 出力: dir1/dir2
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
    #import shutil
    #shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE Testing on ITOP dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/ITOP-SIDE/1",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--pre",
        type=str,
        default="experiments/ITOP-SIDE/1/log/20241225_09:48_l1_d0_e150_f3_last_r0.2_n32/best_model.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/ITOP-SIDE/1/log/best_model.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="bool",
    )

    args = parser.parse_args()
    main(args)
