import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import os
from helper import *
from tqdm import trange
from load_data import *
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
F_c = NeRF().to(device)
F_f = NeRF().to(device)
model_path = './model'
# Number of query points passed through the MLP at a time.
chunk_size = 1024 * 32
# ckpt
LOAD_ckpts = True
if LOAD_ckpts:
    ckpts = [os.path.join(model_path,f) for f in sorted(os.listdir(model_path)) if 'tar' in f]
    # 升序排列，选择最后一轮开始训练
    if len(ckpts)>0:
        print('Found ckpts',f'-->{ckpts}')
        checkpoint = torch.load(ckpts[-1], weights_only=False)
        step = checkpoint['global_step']
        F_c.load_state_dict(checkpoint['model_coarse_state_dict'])
        F_f.load_state_dict(checkpoint['model_fine_state_dict'])
    else:
        print('No ckpts found')
        # print('Train from scratch')

# load data
# 导入数据
splits = ['test']
imgs, poses, K = get_data(splits)
images = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
focal = K[0,0]
images = torch.tensor(images.astype(np.float32))
poses = torch.tensor(poses.astype(np.float32))
# img_size = images.shape[1]
H, W = images.shape[1:3]

# rendering args
# Near bound. See Section 4.
t_n = 2.0
# Far bound. See Section 4.
t_f = 6.0
# Number of coarse samples along a ray. See Section 5.3.
N_c = 64
# Number of fine samples along a ray. See Section 5.3.
N_f = 128
# Bins used to sample depths along a ray. See Equation (2) in Section 4.
t_i_c_gap = (t_f - t_n) / N_c
t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

inference_path = './inference'
if not os.path.exists(inference_path):
    os.makedirs(inference_path, exist_ok=True)

def calculate_psnr(gt_img, pred_img):
    """计算两幅图像之间的PSNR"""
    gt_img = (gt_img * 255).astype(np.uint8)
    pred_img = (pred_img * 255).astype(np.uint8)
    return psnr(gt_img, pred_img, data_range=255)

def run_one_inference(img, pose, idx):
    pose = pose.to(device)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    F_c.eval()
    F_f.eval()
    with torch.no_grad():
        (_, C_rs_f) = run_one_iter_of_nerf(
            rays_o,
            rays_d,
            t_i_c_bin_edges,
            t_i_c_gap,
            N_c,
            N_f,
            chunk_size,
            F_c,
            F_f,
        )
    gt_img = img.cpu().numpy()
    # 保存生成的图像
    generated_img = C_rs_f.detach().cpu().numpy()
    compare_fig = plt.figure()
    plt.subplot(121)
    plt.imshow(gt_img)
    plt.subplot(122)
    plt.imshow(generated_img)
    output_path = os.path.join(inference_path, f'{step}-pose{idx}.png')
    compare_fig.savefig(output_path)
    plt.close()
    
    # 计算PSNR
    current_psnr = calculate_psnr(gt_img, generated_img)
    
    return current_psnr, output_path

N = images.shape[0]
psnr_values = []
output_paths = []

for idx in trange(N):
    psnr_val, out_path = run_one_inference(images[idx], poses[idx], idx)
    psnr_values.append(psnr_val)
    output_paths.append(out_path)

# 计算并打印PSNR统计信息
if psnr_values:
    mean_psnr = np.mean(psnr_values)
    min_psnr = np.min(psnr_values)
    max_psnr = np.max(psnr_values)
    median_psnr = np.median(psnr_values)
    
    print("\nPSNR统计结果:")
    print(f"平均PSNR: {mean_psnr:.2f} dB")
    print(f"最小PSNR: {min_psnr:.2f} dB")
    print(f"最大PSNR: {max_psnr:.2f} dB")
    print(f"中位数PSNR: {median_psnr:.2f} dB")
    
    # 保存PSNR结果到文件
    psnr_file = os.path.join(inference_path, 'psnr_results.txt')
    with open(psnr_file, 'w', encoding='utf-8') as f:
        f.write(f"平均PSNR: {mean_psnr:.2f} dB\n")
        f.write(f"最小PSNR: {min_psnr:.2f} dB\n")
        f.write(f"最大PSNR: {max_psnr:.2f} dB\n")
        f.write(f"中位数PSNR: {median_psnr:.2f} dB\n\n")
        f.write("各图像PSNR值:\n")
        for path, val in zip(output_paths, psnr_values):
            f.write(f"{os.path.basename(path)}: {val:.2f} dB\n")
    
    print(f"\nPSNR结果已保存到: {psnr_file}")
else:
    print("没有计算任何PSNR值")