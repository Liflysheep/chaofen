import torch
import torch.nn as nn
from .model import ChannelWiseEncoder,SpectralDecoder,PSF,SRF

class HyCoNet(nn.Module):
    def __init__(self, scale, hsi_channels, msi_channels, num_P, **kwargs):
        super(HyCoNet, self).__init__()
        self.scale = scale
        self.LrHSI_encoder2A = ChannelWiseEncoder(hsi_channels,num_P)
        self.HrMSI_encoder2A = ChannelWiseEncoder(msi_channels,num_P)
        self.HrMSI_A2LrMSI_A = PSF(self.scale,if_learn=False)
        self.A2Z = SpectralDecoder(num_P=num_P,num_L=hsi_channels,bias=False)
        self.HSI2MSI = SRF(imput_channels=hsi_channels, out_channels=msi_channels)
        self.HrMSI2LrMSI = PSF(self.scale,if_learn=False)



    def forward(self, Lr_HSI, Hr_MSI):

        # first lr process
        self.A_f_LrHSI = self.LrHSI_encoder2A(Lr_HSI)
        self.Za_hat = self.A2Z(self.A_f_LrHSI)
        #second msi process
        self.A_f_Hr_MSI = self.HrMSI_encoder2A(Hr_MSI)
        self.X_hat = self.A2Z(self.A_f_Hr_MSI)
        self.A_f_A = self.HrMSI_A2LrMSI_A(self.A_f_Hr_MSI)
        self.Zb_hat = self.A2Z(self.A_f_A)
        # third X_hat to Y_hat
        self.Y_hat = self.HSI2MSI(self.X_hat)
        self.lr_Y_hat_f_Z = self.HSI2MSI(Lr_HSI)     
        self.lr_Y_hat_f_Y = self.HrMSI2LrMSI(Hr_MSI)   

        return self.X_hat


if __name__ == "__main__":
    # 假设 HS 和 MS 图像的输入大小为：1x128x64x64（高光谱）和 1x3x64x64（多光谱）
    hsi = torch.randn(1, 128, 25, 25)  # 高光谱图像（假设有128个光谱通道）
    msi = torch.randn(1, 3, 200, 200)  # 多光谱图像（假设有3个通道）
    hsi_mean = torch.randn(1, 128, 1, 1)
    hsi_std = torch.randn(1, 128, 1, 1)
    msi_mean = torch.randn(1, 3, 1, 1)
    msi_std = torch.randn(1, 3, 1, 1)       

    # 创建模型，假设我们想进行8倍上采样，MSI的通道数为3，高光谱的通道数为128
    fusion_model = HyCoNet(scale=8, hsi_channels=128, msi_channels=3,num_P=128)
    
    # 进行前向推理
    output = fusion_model(hsi, msi)
    
    print("Output shape:", output.shape)  # 输出形状应为(1, 128, 200, 200)，即高光谱通道数与多光谱分辨率匹配


