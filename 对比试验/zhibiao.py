import numpy as np
import os
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import uqi
import pandas as pd
import re  # 导入正则表达式模块

def compute_ergas(img1, img2, scale):
    d = img1 - img2
    ergasroot = 0
    for i in range(d.shape[2]):
        ergasroot = ergasroot + np.mean(d[:, :, i] ** 2) / np.mean(img1[:, :, i]) ** 2

    ergas = 100 / scale * np.sqrt(ergasroot / d.shape[2])
    return ergas

def compute_psnr(img1, img2):
    assert img1.ndim == 3 and img2.ndim == 3

    img_c, img_w, img_h = img1.shape
    img_c, img_w, img_h = img2.shape
    ref = img1.reshape(img_c, -1)
    tar = img2.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max1 = np.max(ref, 1)

    psnrall = 10 * np.log10(max1 ** 2 / msr)
    out_mean = np.mean(psnrall)
    return out_mean, max1

def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    var_sam = np.var(sam)
    return mSAM, var_sam

def MetricsCal(GT, P, scale):  # c,w,h
    m1, GTmax = compute_psnr(GT, P)  # bandwise mean psnr

    GT = GT.transpose(1, 2, 0)
    P = P.transpose(1, 2, 0)

    m2, _ = compute_sam(GT, P)  # sam
    m3 = compute_ergas(GT, P, scale)

    ssims = []
    for i in range(GT.shape[2]):
        ssimi = ssim(GT[:, :, i], P[:, :, i], data_range=P[:, :, i].max() - P[:, :, i].min())
        ssims.append(ssimi)
    m4 = np.mean(ssims)

    m5 = uqi(GT, P)

    return np.float64(m1), np.float64(m2), m3, m4, m5


def extract_scale_from_folder_name(folder_name, hr_prefix):
    """
    从文件夹名中提取 scale 参数（排除 hr_prefix 后的第一个数字）。
    例如：hr_prefix='scene1', folder_name='scene1_scale4_OUT' -> 4
    """
    # 去掉 hr_prefix 部分
    remaining_name = folder_name[len(hr_prefix):]
    # 查找第一个数字
    match = re.search(r'\d+', remaining_name)
    if match:
        return int(match.group())  # 返回提取的数字
    return None  # 如果没有找到数字，返回 None

def process_folders(hr_folder, out_folder, output_csv):
    # 初始化一个空的 DataFrame 用于存储结果
    results = pd.DataFrame(columns=['PSNR', 'SAM', 'ERGAS', 'SSIM', 'UQI'])

    # 遍历 HR 文件夹，获取所有子文件夹
    hr_subfolders = {}
    for subdir, _, files in os.walk(hr_folder):
        subdir_name = os.path.basename(subdir)
        # 直接使用子文件夹名作为键
        hr_subfolders[subdir_name] = subdir
        print(subdir_name)

    # 遍历 OUT 文件夹，获取所有子文件夹
    out_subfolders = {}
    for subdir, _, files in os.walk(out_folder):
        subdir_name = os.path.basename(subdir)
        # 遍历 hr_subfolders 的键，检查是否以 hr_subfolders 的键开头且紧跟着 '_'
        for hr_prefix in hr_subfolders:
            if subdir_name.startswith(hr_prefix + '_'):  # 确保 hr_prefix 后紧跟着 '_'
                # 提取 scale 参数
                scale = extract_scale_from_folder_name(subdir_name, hr_prefix)
                if scale is None:
                    print(f"Warning: No scale found in folder name {subdir_name}. Using default scale=8.")
                    scale = 8  # 默认值
                # 存储 OUT 子文件夹路径和 scale，以及对应的 HR 子文件夹路径
                if hr_prefix not in out_subfolders:
                    out_subfolders[hr_prefix] = []
                out_subfolders[hr_prefix].append((subdir, scale))
                break  # 找到匹配后跳出循环

    # 遍历匹配的子文件夹
    for hr_prefix, out_subdirs in out_subfolders.items():
        hr_subdir = hr_subfolders[hr_prefix]
        # 加载 HR 文件夹下的 .mat 文件
        hr_files = [f for f in os.listdir(hr_subdir) if f.endswith('.mat')]
        if len(hr_files) != 1:
            print(f"Warning: Subfolder {hr_prefix} does not contain exactly one .mat file in HR folder.")
            continue

        hr_file = os.path.join(hr_subdir, hr_files[0])
        hr_data = sio.loadmat(hr_file)
        if 'HR' not in hr_data:
            print(f"Warning: 'HR' key not found in {hr_file}.")
            continue

        gt = hr_data['HR']
        gt = gt.transpose(2, 0, 1)  # 调整维度顺序

        # 遍历对应的 OUT 子文件夹
        for out_subdir, scale in out_subdirs:
            # 加载 OUT 文件夹下的 .mat 文件
            out_files = [f for f in os.listdir(out_subdir) if f.endswith('.mat')]
            if len(out_files) != 1:
                print(f"Warning: Subfolder {out_subdir} does not contain exactly one .mat file in OUT folder.")
                continue

            out_file = os.path.join(out_subdir, out_files[0])
            out_data = sio.loadmat(out_file)
            if 'HR' not in out_data:
                print(f"Warning: 'HR' key not found in {out_file}.")
                continue

            out = out_data['HR']
            # 解包第一个维度为两个维度
            original_dim = int(np.sqrt(out.shape[0]))  # 假设第一个维度是 height * width
            out = out.reshape(original_dim, original_dim, -1)  # 解包为 (height, width, channels)
            out = out.transpose(2, 0, 1)  # 调整维度顺序为 (channels, height, width)

            # 计算指标
            psnr, sam, ergas, ssim_val, uqi_val = MetricsCal(gt, out, scale)

            # 将结果存入 DataFrame
            results.loc[out_subdir] = [psnr, sam, ergas, ssim_val, uqi_val]

    # 将结果保存到 CSV 文件
    results.to_csv(output_csv)
    print(f"Results saved to {output_csv}")

# 设置 HR 文件夹、OUT 文件夹和输出 CSV 文件路径
hr_folder = './UDALN_GRSL/data'  # HR 文件夹路径
out_folder = './uSDN/checkpoint'   # OUT 文件夹路径
output_csv = './uSDN_results.csv'  # 输出 CSV 文件路径

# 处理文件夹并保存结果
process_folders(hr_folder, out_folder, output_csv)

