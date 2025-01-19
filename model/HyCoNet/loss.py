import torch
import torch.nn as nn
import torch.nn.functional as F

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        # 定义目标值为 1 的张量，作为缓冲区
        self.register_buffer('one', torch.tensor(1.0, dtype=torch.float))
        # 使用新版 reduction 参数，指定为 'sum'
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, input):
        """
        根据输入生成目标张量，目标张量的每个元素值为 1
        """
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def forward(self, input):
        """
        计算损失：输入的某一维的总和与目标值 1 的 L1 损失
        """
        # 对输入张量第 1 维求和
        input_sum = torch.sum(input, dim=1)
        # 生成与输入总和形状一致的目标张量
        target_tensor = self.get_target_tensor(input_sum)
        # 计算 L1 损失
        loss = self.loss(input_sum, target_tensor)
        return loss

def kl_divergence(p, q):

    # 修改 softmax 的调用方式
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2

class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self,input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


