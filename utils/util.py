from torchmetrics import Metric
import torch
import torch.nn.functional as F

# 自定义 ERGAS 指标
class ERGAS(Metric):
    """
    Implementation of ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    This metric is used to evaluate the quality of hyperspectral or multispectral image super-resolution.
    """
    higher_is_better = False  # Lower ERGAS indicates better performance

    def __init__(self, scale_factor: float, dist_sync_on_step: bool = False):
        """
        Initialize ERGAS metric.

        Args:
            scale_factor (float): Scale factor between the reference and the estimated images.
            dist_sync_on_step (bool): Synchronize metric states across processes at each forward step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.scale_factor = scale_factor

        # Add states to store cumulative RMSE^2/mean^2 and number of bands
        self.add_state("sum_rmse_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_bands", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with predicted and target images.

        Args:
            preds (torch.Tensor): Predicted image of shape (B, C, H, W).
            target (torch.Tensor): Reference image of shape (B, C, H, W).
        """
        if preds.shape != target.shape:
            raise ValueError("Predicted and target images must have the same shape.")

        # Compute per-band RMSE and mean
        batch_size, channels, _, _ = preds.shape
        rmse_per_band = torch.sqrt(torch.mean((preds - target) ** 2, dim=(2, 3)))  # Shape: (B, C)
        band_means = torch.mean(target, dim=(2, 3))  # Shape: (B, C)

        # Compute RMSE^2 / mean^2 for each band
        rmse_squared_div_mean_squared = (rmse_per_band ** 2) / (band_means ** 2)  # Shape: (B, C)

        # Aggregate across the batch dimension (mean or sum based on your logic)
        batch_agg_rmse_squared = torch.mean(rmse_squared_div_mean_squared, dim=0)  # Aggregate across batch

        # Sum up the results across all bands
        self.sum_rmse_squared += torch.sum(batch_agg_rmse_squared)  # Sum across bands
        self.count_bands += channels

    def compute(self):
        """
        Compute the final ERGAS value.
        """
        # Compute ERGAS using the accumulated values and explicitly dividing by N
        ergas = 100 * (1 / self.scale_factor) * torch.sqrt(self.sum_rmse_squared)
        return ergas

# 自定义 RMSE 指标
class RMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        更新状态变量，计算当前批次的平方误差。
        """
        # 确保 preds 和 target 的形状一致
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"

        # 计算平方误差并累积
        squared_error = torch.sum((preds - target) ** 2)
        self.sum_squared_error += squared_error

        # 累积样本数量
        self.total += target.numel()  # target 的总像素数量

    def compute(self):
        """
        计算 RMSE，根据累积的平方误差和总样本数。
        """
        return torch.sqrt(self.sum_squared_error / self.total)

# 自定义 SAM 指标
class SAM(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("angle_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, target):
        eps = 1e-10
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.size(1))
        target = target.permute(0, 2, 3, 1).reshape(-1, target.size(1))
        dot_product = torch.sum(pred * target, dim=-1)
        norm_pred = torch.norm(pred, dim=-1)
        norm_target = torch.norm(target, dim=-1)
        cos_theta = dot_product / (norm_pred * norm_target + eps)
        angles = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        self.angle_sum += torch.sum(angles)
        self.n_pixels += pred.size(0)

    def compute(self):
        return self.angle_sum / self.n_pixels

# 自定义 MRAE 指标
class MRAE(Metric):
    def __init__(self):
        super().__init__()
        # 累积 MRAE 总和和像素数量
        self.add_state("mrae_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, target):
        """
        更新 MRAE 指标的状态。计算每个像素的相对绝对误差。

        :param pred: 预测值 (batch_size, channels, height, width)
        :param target: 真实值 (batch_size, channels, height, width)
        """
        # 确保输入的预测值和真实值有相同的形状
        assert pred.shape == target.shape, "预测值和真实值的形状必须一致"

        # 展平张量，使其变成 (batch_size * height * width, channels)
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.size(1))
        target = target.permute(0, 2, 3, 1).reshape(-1, target.size(1))

        # 计算每个像素的绝对误差
        abs_error = torch.abs(target - pred)

        # 计算相对误差
        relative_error = abs_error / torch.abs(target)

        # 累积 MRAE 总和和总像素数
        self.mrae_sum += torch.sum(relative_error)
        self.n_pixels += relative_error.numel()  # 每个像素有多个通道，numel() 是总元素个数

    def compute(self):
        """
        计算最终的 MRAE 值。

        :return: 平均相对绝对误差
        """
        return self.mrae_sum / self.n_pixels


class PSNR(Metric):
    def __init__(
            self,
            data_range: float = 1.0,
            dist_sync_on_step: bool = False,
    ) -> None:
        # 初始化基类
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # PSNR 的最大像素值范围，一般为 1.0（归一化图像）或者 255（标准 8 位图像）
        self.data_range = data_range

        # 添加状态变量，用于存储累积的均方误差 (MSE) 和样本数量
        self.add_state("sum_psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        更新状态，计算预测和目标图像之间的均方误差。
        Args:
            preds (torch.Tensor): 预测的图像，形状为 (B, C, H, W)
            target (torch.Tensor): 目标图像，形状为 (B, C, H, W)
        """
        # 确保预测和目标的形状相同
        assert preds.shape == target.shape, "Predictions and target must have the same shape"
        batch, img_c, img_h, img_w = preds.shape
        ref = preds.reshape(batch, img_c, -1)
        tar = target.reshape(batch, img_c, -1)
        msr = torch.mean((ref - tar) ** 2, 2)  #[b,c]

        psnr = 10 * torch.log10(self.data_range ** 2 / msr)
        out_mean = torch.mean(psnr)
        # # 计算均方误差 (MSE)
        # mse = F.mse_loss(preds, target, reduction='sum')

        self.sum_psnr += out_mean

    def compute(self) -> torch.Tensor:
        """
        计算累积状态的 PSNR。
        Returns:
            torch.Tensor: 计算出的 PSNR 值
        """
        # 计算 PSNR
        psnr = self.sum_psnr
        return psnr

class UQI(Metric):
    def __init__(self):
        super().__init__()
        # 累积 UQI 总和和像素数量
        self.add_state("uqi_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, target):
        """
        更新 UQI 指标的状态。计算每个像素的 UQI。

        :param pred: 预测值 (batch_size, channels, height, width)
        :param target: 真实值 (batch_size, channels, height, width)
        """
        # 确保输入的预测值和真实值有相同的形状
        assert pred.shape == target.shape, "预测值和真实值的形状必须一致"

        # 展平张量，使其变成 (batch_size * height * width, channels)
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.size(1))
        target = target.permute(0, 2, 3, 1).reshape(-1, target.size(1))

        # 计算每个通道的均值和方差
        mu_pred = torch.mean(pred, dim=0)
        mu_target = torch.mean(target, dim=0)
        sigma_pred_sq = torch.var(pred, dim=0)
        sigma_target_sq = torch.var(target, dim=0)
        covariance = torch.mean((pred - mu_pred) * (target - mu_target), dim=0)

        # 计算 UQI 指标
        numerator = 4 * covariance * mu_pred * mu_target
        denominator = (mu_pred**2 + sigma_pred_sq) * (mu_target**2 + sigma_target_sq)

        uqi = numerator / denominator

        # 计算每个像素的 UQI 指标，并累积总和
        self.uqi_sum += torch.sum(uqi)
        self.n_pixels += uqi.numel()  # 每个像素有多个通道，numel() 是总元素个数

    def compute(self):
        """
        返回当前 UQI 的平均值
        """
        return self.uqi_sum / self.n_pixels


# 示例用法
if __name__ == "__main__":
    psnr_metric = PSNR(data_range=1.0)

    # 假设有两个批次的图像，大小为 (B, C, H, W)
    preds = torch.rand(1, 128, 200, 200)  # 模拟预测图像
    target = torch.rand(1, 128, 200, 200)  # 模拟目标图像

    # 更新指标
    psnr_metric.update(preds, target)

    # 计算 PSNR
    psnr_value = psnr_metric.compute()
    print(f"PSNR: {psnr_value.item()} dB")
