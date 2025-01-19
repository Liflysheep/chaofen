import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class MSDCNN(nn.Module):
    def __init__(self, scale, hsi_channels, msi_channels, **kwargs):
        super(MSDCNN, self).__init__()

        # 输入为 HS 和 MS 图像，因此通道数需要分别设置为 hsi_channels 和 msi_channels
        self.shallow_conv_1 = nn.Conv2d(
            in_channels=hsi_channels + msi_channels, out_channels=512, kernel_size=9, stride=1, padding=4)
        self.shallow_conv_2 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.shallow_conv_3 = nn.Conv2d(
            in_channels=256, out_channels=hsi_channels, kernel_size=5, stride=1, padding=2)  # 修改为hsi_channels
        self.relu = nn.ReLU()

        # 深层卷积层，适应更多的光谱通道
        self.deep_conv_1 = nn.Conv2d(
            in_channels=hsi_channels + msi_channels, out_channels=600, kernel_size=7, stride=1, padding=3)
        self.deep_conv_1_sacle_1 = nn.Conv2d(
            in_channels=600, out_channels=200, kernel_size=3, stride=1, padding=1)
        self.deep_conv_1_sacle_2 = nn.Conv2d(
            in_channels=600, out_channels=200, kernel_size=5, stride=1, padding=2)
        self.deep_conv_1_sacle_3 = nn.Conv2d(
            in_channels=600, out_channels=200, kernel_size=7, stride=1, padding=3)
        self.deep_conv_2 = nn.Conv2d(
            in_channels=600, out_channels=300, kernel_size=3, stride=1, padding=1)
        self.deep_conv_2_sacle_1 = nn.Conv2d(
            in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1)
        self.deep_conv_2_sacle_2 = nn.Conv2d(
            in_channels=300, out_channels=100, kernel_size=5, stride=1, padding=2)
        self.deep_conv_2_sacle_3 = nn.Conv2d(
            in_channels=300, out_channels=100, kernel_size=7, stride=1, padding=3)
        self.deep_conv_3 = nn.Conv2d(
            in_channels=300, out_channels=hsi_channels, kernel_size=5, stride=1, padding=2)  # 修改为hsi_channels

        self.interpolate = interpolate
        self.scale = scale

    def forward(self, hsi, msi,hsi_mean,hsi_std,msi_mean,msi_std):
        # # 打印输入变量的维度信息
        # print(f"hsi shape: {hsi.shape}")
        # print(f"msi shape: {msi.shape}")
        # print(f"hsi_mean shape: {hsi_mean.shape}")
        # print(f"hsi_std shape: {hsi_std.shape}")
        # print(f"msi_mean shape: {msi_mean.shape}")
        # print(f"msi_std shape: {msi_std.shape}")
        
        # 归一化输入
        hsi = (hsi - hsi_mean) / hsi_std
        msi = (msi - msi_mean) / msi_std

        # 将高光谱图像（HSI）和多光谱图像（MSI）拼接作为输入
        hsi = self.interpolate(hsi, scale_factor=self.scale, mode='bicubic')
        input_data = torch.cat([hsi, msi], dim=1)

        # 浅层卷积
        shallow_fea = self.relu(self.shallow_conv_1(input_data))
        shallow_fea = self.relu(self.shallow_conv_2(shallow_fea))
        shallow_out = self.shallow_conv_3(shallow_fea)  # 输出通道数为hsi_channels

        # 深层卷积
        deep_fea = self.relu(self.deep_conv_1(input_data))
        deep_fea_scale1 = self.relu(self.deep_conv_1_sacle_1(deep_fea))
        deep_fea_scale2 = self.relu(self.deep_conv_1_sacle_2(deep_fea))
        deep_fea_scale3 = self.relu(self.deep_conv_1_sacle_3(deep_fea))
        deep_fea_scale = torch.cat([deep_fea_scale1, deep_fea_scale2, deep_fea_scale3], dim=1)
        deep_fea_1 = torch.add(deep_fea, deep_fea_scale)
        deep_fea_2 = self.relu(self.deep_conv_2(deep_fea_1))
        deep_fea_2_scale1 = self.relu(self.deep_conv_2_sacle_1(deep_fea_2))
        deep_fea_2_scale2 = self.relu(self.deep_conv_2_sacle_2(deep_fea_2))
        deep_fea_2_scale3 = self.relu(self.deep_conv_2_sacle_3(deep_fea_2))
        deep_fea_2_scale = torch.cat([deep_fea_2_scale1, deep_fea_2_scale2, deep_fea_2_scale3], dim=1)
        deep_fea_3 = torch.add(deep_fea_2, deep_fea_2_scale)
        deep_out = self.deep_conv_3(deep_fea_3)  # 输出通道数为hsi_channels

        out = deep_out + shallow_out

        # 去归一化
        out = out * hsi_std + hsi_mean

        return out


if __name__ == "__main__":
    # 假设 HS 和 MS 图像的输入大小为：1x128x64x64（高光谱）和 1x3x64x64（多光谱）
    hsi = torch.randn(1, 128, 25, 25)  # 高光谱图像（假设有128个光谱通道）
    msi = torch.randn(1, 3, 200, 200)  # 多光谱图像（假设有3个通道）
    hsi_mean = torch.randn(1, 128, 1, 1)
    hsi_std = torch.randn(1, 128, 1, 1)
    msi_mean = torch.randn(1, 3, 1, 1)
    msi_std = torch.randn(1, 3, 1, 1)       

    # 创建模型，假设我们想进行8倍上采样，MSI的通道数为3，高光谱的通道数为128
    fusion_model = MSDCNN(scale=8, hsi_channels=128, msi_channels=3,
                                hsi_mean=hsi_mean, hsi_std=hsi_std,
                                msi_mean=msi_mean, msi_std=msi_std)
    
    # 进行前向推理
    output = fusion_model(hsi, msi,hsi_mean,hsi_std,msi_mean,msi_std)
    
    print("Output shape:", output.shape)  # 输出形状应为(1, 128, 200, 200)，即高光谱通道数与多光谱分辨率匹配


