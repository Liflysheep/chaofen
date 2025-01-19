import os
import torch
import scipy.io as scio
from typing import Callable, Dict, Union, List


class AISA:
    def __init__(self,
                 root_path: str,
                 window_size: int = None,  # 默认不进行窗口切片
                 verbose: bool = True,
                 hsi_num_channel: int = 31):  # hsi_num_channel 参数
        """
        高光谱和多光谱融合数据加载类的初始化方法

        参数:
            root_path (str): 存储 .mat 文件的根路径。
            window_size (int): 每个窗口的大小（例如 SwinTransformer 使用的 7x7 窗口）。
            verbose (bool): 是否打印详细信息。
            hsi_num_channel (int): 加载的波段数。用于选择 .mat 文件。
        """
        self.root_path = root_path
        self.window_size = window_size  # 保存窗口大小
        self.verbose = verbose
        self.hsi_num_channel = hsi_num_channel  # 高光谱通道数
        self.records = self.collect_records(root_path)  # 收集所有数据文件的路径

    def collect_records(self, root_path: str) -> List[Dict]:
        """
        收集所有的 .mat 文件路径和元数据。根据 hsi_num_channel 选择对应的文件。

        参数:
            root_path (str): 数据存储路径。

        返回:
            List[Dict]: 包含每个文件路径的字典列表。
        """
        records = []
        mat_files = [f for f in os.listdir(root_path) if f.endswith('.mat')]

        if len(mat_files) == 0:
            raise ValueError("The directory does not contain any .mat files.")

        # 只处理两个 .mat 文件
        if self.hsi_num_channel == 128:
            mat_file_path = [os.path.join(root_path, i) for i in mat_files if i == 'AISA_3.mat']  # 选择 128 波段的 .mat 文件
        elif self.hsi_num_channel == 31:
            mat_file_path = [os.path.join(root_path,i) for i in mat_files if i == 'AISA_31.mat']                
        records.append({'mat_file': mat_file_path[0]})

        if self.verbose:
            print(f"Found {len(records)} .mat files: {', '.join([file['mat_file'] for file in records])}")

        return records

    def window_partition(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        将输入张量按照 window_size 切成窗口。类似 Swin Transformer 的窗口划分方法。
        
        参数:
            tensor (torch.Tensor): 输入张量，形状为 (C, H, W)。
            window_size (int): 切片窗口的大小。如果为 None，则返回完整张量。
        
        返回:
            torch.Tensor: 如果指定 window_size，则返回切片后的张量，
            否则返回原始张量。
        """

        C, H, W = tensor.shape
        assert H % window_size == 0 and W % window_size == 0, \
            "Height and Width must be divisible by window_size"
        
        tensor = tensor.view(C, H // window_size, window_size, W // window_size, window_size)
        windows = tensor.permute(1, 3, 0, 2, 4).contiguous().view(-1, C, window_size, window_size)
        return windows
    
    def process_record(self, mat_file_path: str) -> List[Dict]:
        """
        从 .mat 文件中加载数据并转换为 PyTorch 张量，并根据窗口大小进行可选的窗口切片。

        参数:
            mat_file_path (str): .mat 文件的路径。

        返回:
            List[Dict]: 每个元素是一个子样本，包含 HR, HSI, MSI 数据张量和均值与标准差。
        """
        data = scio.loadmat(mat_file_path)

        # 提取 HR, HSI, MSI 数据
        HR = data['HR']  # 假设维度 (200, 200, 128)
        hsi = data['hsi']  # 假设维度 (25, 25, 128)
        msi = data['msi']  # 假设维度 (200, 200, 3)

        # 转换为 PyTorch 张量并调整维度
        HR_tensor = torch.tensor(HR).float().permute(2, 0, 1)  # (128, 200, 200)
        hsi_tensor = torch.tensor(hsi).float().permute(2, 0, 1)  # (128, 25, 25)
        msi_tensor = torch.tensor(msi).float().permute(2, 0, 1)  # (3, 200, 200)

        # 计算 HSI 的缩放比例
        H_hr = HR_tensor.shape[1]
        H_hsi = hsi_tensor.shape[1]
        hsi_scale = H_hr // H_hsi

        # 判断是否进行窗口切片
        if self.window_size is not None:
            # 使用窗口切片
            HR_windows = self.window_partition(HR_tensor, self.window_size)  # HR 和 MSI 使用相同的窗口大小
            MSI_windows = self.window_partition(msi_tensor, self.window_size)  # MSI 与 HR 相同的窗口
            HSI_windows = self.window_partition(hsi_tensor, self.window_size // hsi_scale)  # HSI 使用缩放后的窗口大小

            # 构造切片后的数据列表，每个切片计算独立的均值和标准差
            sliced_data = []
            for hr, hsi, msi in zip(HR_windows, HSI_windows, MSI_windows):
                # 计算每个切片的均值和标准差
                hsi_mean = hsi.mean(dim=[1, 2], keepdim=True)  # 每个 HSI 切片的均值 (C, 1, 1)
                hsi_std = hsi.std(dim=[1, 2], keepdim=True)    # 每个 HSI 切片的标准差 (C, 1, 1)
                msi_mean = msi.mean(dim=[1, 2], keepdim=True)  # 每个 MSI 切片的均值 (C, 1, 1)
                msi_std = msi.std(dim=[1, 2], keepdim=True)    # 每个 MSI 切片的标准差 (C, 1, 1)

                # 将切片后的张量和均值、标准差保存到字典中
                sliced_data.append({
                    'HR': hr,  # (C, window_size, window_size)
                    'HSI': hsi,  # (C, window_size // hsi_scale, window_size // hsi_scale)
                    'MSI': msi,  # (C, window_size, window_size)
                    'HSI_mean': hsi_mean,
                    'HSI_std': hsi_std,
                    'MSI_mean': msi_mean,
                    'MSI_std': msi_std
                })
        else:
            # 不进行窗口切片，直接返回完整张量
            sliced_data = [{
                'HR': HR_tensor,  # (C, H, W)
                'HSI': hsi_tensor,  # (C, H_hsi, W_hsi)
                'MSI': msi_tensor,  # (C, H, W)
                'HSI_mean': hsi_tensor.mean(dim=[1, 2], keepdim=True),  # 整体均值 (C, 1, 1)
                'HSI_std': hsi_tensor.std(dim=[1, 2], keepdim=True),    # 整体标准差 (C, 1, 1)
                'MSI_mean': msi_tensor.mean(dim=[1, 2], keepdim=True),  # 整体均值 (C, 1, 1)
                'MSI_std': msi_tensor.std(dim=[1, 2], keepdim=True)     # 整体标准差 (C, 1, 1)
            }]

        return sliced_data

    def __getitem__(self, index: int) -> Dict:
        """
        根据索引获取数据样本。

        参数:
            index (int): 数据索引。

        返回:
            Dict: 包含 HR, HSI, MSI 张量及其均值和标准差。
        """
        record_idx = index // len(self.process_record(self.records[0]['mat_file']))
        slice_idx = index % len(self.process_record(self.records[record_idx]['mat_file']))

        mat_file_path = self.records[record_idx]['mat_file']
        data = self.process_record(mat_file_path)

        return data[slice_idx]

    def __len__(self) -> int:
        """
        返回数据集大小。
        
        返回:
            int: 数据集中的样本数量。
        """
        return sum(len(self.process_record(record['mat_file'])) for record in self.records)



class Toys(AISA):
    def __init__(self,
                 root_path: str,
                 window_size: int = None,  # 默认不进行窗口切片
                 verbose: bool = True,
                 hsi_num_channel: int = 31,
                 LR_sacle : int = 16,
                 data_name : str = 'toys_ms',
                 ):
        """
        用于测试的 Toys 数据集类。

        参数:
            root_path (str): 存储 .mat 文件的根路径。
            window_size (int): 每个窗口的大小（例如 SwinTransformer 使用的 7x7 窗口）。
            verbose (bool): 是否打印详细信息。
            hsi_num_channel (int): 加载的波段数。用于选择 .mat 文件。
        """
        super().__init__(root_path, window_size, verbose, hsi_num_channel)
        self.records = self.collect_records(root_path)  # 收集所有数据文件的路径
        self.LR_scale = LR_sacle
        self.data_name = data_name

    def collect_records(self, root_path: str) -> List[Dict]:
        """
        收集所有的 .mat 文件路径和元数据。根据 hsi_num_channel 选择对应的文件。

        参数:
            root_path (str): 数据存储路径。

        返回:
            List[Dict]: 包含每个文件路径的字典列表。
        """
        records = []
        mat_files = [f for f in os.listdir(root_path) if f.endswith('.mat')]

        if len(mat_files) == 0:
            raise ValueError("The directory does not contain any .mat files.")

        for mat_files_path in mat_files:
            if mat_files_path.endswith(self.data_name):
                records.append({'mat_file': os.path.join(root_path, mat_files_path)})

        if self.verbose:
            print(f"Found {len(records)} .mat files: {', '.join([file['mat_file'] for file in records])}")

        return records            

    def process_record(self, mat_file_path: str) -> List[Dict]:
        """
        从 .mat 文件中加载数据并转换为 PyTorch 张量，并根据窗口大小进行可选的窗口切片。

        参数:
            mat_file_path (str): .mat 文件的路径。

        返回:
            List[Dict]: 每个元素是一个子样本，包含 HR, HSI, MSI 数据张量和均值与标准差。
        """
        data = scio.loadmat(mat_file_path)

        if self.LR_scale == 16:
            LR = data['LR_16']
        elif self.LR_scale == 32:
            LR = data['LR_32']
        else:
            raise ValueError("LR_scale must be 16 or 32")
        
        HR = data['HR']  # 假设维度 (512, 512, 31)
        RGB = data['RGB']  # 假设维度 (512, 512, 3)

        # 转换为 PyTorch 张量并调整维度
        HR_tensor = torch.tensor(HR).float().permute(2, 0, 1)  # (128, 200, 200)
        LR_tensor = torch.tensor(LR).float().permute(2, 0, 1)  # (128, 25, 25)
        RGB_tensor = torch.tensor(RGB).float().permute(2, 0, 1)  # (3, 200, 200)

        # 计算 HSI 的缩放比例
        H_hr = HR_tensor.shape[1]
        H_hsi = LR_tensor.shape[1]
        LR_scale = H_hr // H_hsi

        # 判断是否进行窗口切片
        if self.window_size is not None:
            # 使用窗口切片
            HR_windows = self.window_partition(HR_tensor, self.window_size)  # HR 和 MSI 使用相同的窗口大小
            RBG_windows = self.window_partition(RGB_tensor, self.window_size)  # MSI 与 HR 相同的窗口
            LR_windows = self.window_partition(LR_tensor, self.window_size // LR_scale)  # HSI 使用缩放后的窗口大小

            # 构造切片后的数据列表，每个切片计算独立的均值和标准差
            sliced_data = []
            for hr, lr, rgb in zip(HR_windows, LR_windows, RBG_windows):
                # 计算每个切片的均值和标准差
                if lr.shape[1] == 1:
                    lr_mean = lr.clone()  # 每个 HSI 切片的均值 (C, 1, 1)
                    lr_std = lr.clone()    # 每个 HSI 切片的标准差 (C, 1, 1)
                else:
                    lr_mean = lr.mean(dim=[1, 2], keepdim=True)  # 每个 HSI 切片的均值 (C, 1, 1)
                    lr_std = lr.std(dim=[1, 2], keepdim=True)    # 每个 HSI 切片的标准差 (C, 1, 1)
                rgb_mean = rgb.mean(dim=[1, 2], keepdim=True)  # 每个 MSI 切片的均值 (C, 1, 1)
                rgb_std = rgb.std(dim=[1, 2], keepdim=True)    # 每个 MSI 切片的标准差 (C, 1, 1)

                # 将切片后的张量和均值、标准差保存到字典中
                sliced_data.append({
                    'HR': hr,  # (C, window_size, window_size)
                    'HSI': lr,  # (C, window_size // hsi_scale, window_size // hsi_scale)
                    'MSI': rgb,  # (C, window_size, window_size)
                    'HSI_mean': lr_mean,
                    'HSI_std': lr_std,
                    'MSI_mean': rgb_mean,
                    'MSI_std': rgb_std
                })
        else:
            # 不进行窗口切片，直接返回完整张量
            sliced_data = [{
                'HR': HR_tensor,  # (C, H, W)
                'HSI': LR_tensor,  # (C, H_hsi, W_hsi)
                'MSI': RGB_tensor,  # (C, H, W)
                'HSI_mean': LR_tensor.mean(dim=[1, 2], keepdim=True),  # 整体均值 (C, 1, 1)
                'HSI_std': LR_tensor.std(dim=[1, 2], keepdim=True),    # 整体标准差 (C, 1, 1)
                'MSI_mean': RGB_tensor.mean(dim=[1, 2], keepdim=True),  # 整体均值 (C, 1, 1)
                'MSI_std': RGB_tensor.std(dim=[1, 2], keepdim=True)     # 整体标准差 (C, 1, 1)
            }]

        return sliced_data



if __name__ == "__main__":
    # 测试路径，替换为实际的 .mat 文件路径
    root_path = "./data"  # 请将此路径替换为实际存放 .mat 文件的路径


    # 初始化数据集
    dataset = Toys(root_path=root_path, window_size=32, verbose=True, hsi_num_channel=31, LR_sacle=32)

    # 打印数据集大小
    print(f"Dataset size: {len(dataset)}")
    
    # 打印前两个样本的所有特征的形状
    print("\nTesting shapes of features in first two samples:")
    for i in range(min(2, len(dataset))):  # 打印前两个样本
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  HR shape: {sample['HR'].shape}")
        print(f"  LR shape: {sample['HSI'].shape}")
        print(f"  RGB shape: {sample['MSI'].shape}")
        # print(f"  HSI_mean shape: {sample['HSI_mean']}")
        # print(f"  HSI_std shape: {sample['HSI_std']}")
        # print(f"  MSI_mean shape: {sample['MSI_mean']}")
        # print(f"  MSI_std shape: {sample['MSI_std']}")
    # # 初始化数据集
    # dataset = AISA(root_path=root_path, window_size=40, verbose=True, hsi_num_channel=31)

    # # 打印数据集大小
    # print(f"Dataset size: {len(dataset)}")
    
    # # 打印前两个样本的所有特征的形状
    # print("\nTesting shapes of features in first two samples:")
    # for i in range(min(2, len(dataset))):  # 打印前两个样本
    #     sample = dataset[i]
    #     print(f"\nSample {i}:")
    #     print(f"  HR shape: {sample['HR'].shape}")
    #     print(f"  HSI shape: {sample['HSI'].shape}")
    #     print(f"  MSI shape: {sample['MSI'].shape}")
    #     # print(f"  HSI_mean shape: {sample['HSI_mean']}")
    #     # print(f"  HSI_std shape: {sample['HSI_std']}")
    #     # print(f"  MSI_mean shape: {sample['MSI_mean']}")
    #     # print(f"  MSI_std shape: {sample['MSI_std']}")

