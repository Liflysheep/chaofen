import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=[64, 128, 256], use_softmax=True):
        """
        基于 1×1 卷积的通道级编码器
        :param input_channels: 输入通道数
        :param output_channels: 输出通道数
        :param intermediate_channels: 中间层的通道数列表
        :param use_softmax: 是否在最后应用 Softmax
        """
        super(ChannelWiseEncoder, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        # 构建中间层，使用 1×1 卷积
        for inter_channels in intermediate_channels:
            layers.append(nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = inter_channels
        
        # 最后一层
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)
        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.net(x)
        if self.use_softmax:
            out = self.softmax(out)
        return out


class SpectralDecoder(nn.Module):
    def __init__(self, num_P, num_L,kernel_size=1, stride=1, padding=0, bias=False):
        super(SpectralDecoder, self).__init__()
        self.width = num_P
        self.out_channels = num_L

        # 定义一个 3D 卷积层
        self.conv3d = nn.Conv3d(
            in_channels=1,  # 输入通道数
            out_channels=self.out_channels,  # 输出通道数
            kernel_size=(self.width, 1, 1),  # 卷积核大小
            stride=1,  # 步幅
            padding=0,  # 无填充
            bias=bias
        )

    def forward(self, x):
        output = self.conv3d(x.unsqueeze(1)).squeeze(2)
        return output
    
class PSF(nn.Module):
    def __init__(self, scale,if_learn=False):
        super(PSF, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)
        self.if_learn = if_learn

    def forward(self, x):
        if self.if_learn==True:
            batch, channel, height, weight = list(x.size())
            return self.softmax(torch.cat([self.net(x[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1))
        else:
            return F.avg_pool2d(x, kernel_size=self.scale, stride=self.scale)#进行平均池化

class SRF(nn.Module):
    def __init__(self, imput_channels, out_channels,kernel_size=1, stride=1, padding=0, bias=False):
        super(SRF, self).__init__()
        self.width = imput_channels
        self.out_channels = out_channels

        # 定义一个 3D 卷积层
        self.conv3d = nn.Conv3d(
            in_channels=1,  # 输入通道数
            out_channels=self.out_channels,  # 输出通道数
            kernel_size=(self.width, kernel_size, stride),  # 卷积核大小
            stride=1,  # 步幅
            padding=0  # 无填充
        )

    def forward(self, x):
        output = self.conv3d(x.unsqueeze(1)).squeeze(2)


        return output    
    
class NonZeroClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1e8)

class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1)

class SumToOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0,10)
                w.div_(w.sum(dim=1,keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0,5)
