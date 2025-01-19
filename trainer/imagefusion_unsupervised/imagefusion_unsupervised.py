import logging
from typing import Any, Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from model.HyCoNet.loss import SumToOneLoss, SparseKLloss
from model.HyCoNet import model
from utils import util
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from .train_option import TrainOptions
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


_EVALUATE_OUTPUT = List[Dict[str, float]]

log = logging.getLogger('image_fusion')


def imagefusion_metrics(metric_list: List[str]):
    allowed_metrics = [
        'psnr', 'ssim', 'rmse', 'sam', 'ergas', 'mrae'
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose from {', '.join(allowed_metrics)}."
            )

    metric_dict = {
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0),
    }

    custom_metric_dict = {
        'psnr': util.PSNR(),
        'rmse': util.RMSE(),
        'sam': util.SAM(),
        'ergas': util.ERGAS(scale_factor=8),  # scale_factor 可以根据实际需求调整
        'mrae': util.MRAE(),
    }

    # 合并所有指标
    metric_dict.update(custom_metric_dict)

    # 根据输入列表筛选需要的指标
    metrics = {name: metric_dict[name] for name in metric_list}
    return MetricCollection(metrics)


class ImageFusionTrainer(pl.LightningModule):
    """
    用于高光谱和多光谱图像融合的训练器类。

    参数:
        model (nn.Module): 用于图像融合的模型。
        lr (float): 学习率。默认值为 1e-3。
        weight_decay (float): 权重衰减系数。默认值为 0.0。
        devices (int): 使用的设备数量。默认值为 1。
        accelerator (str): 加速器类型，可选 'cpu' 或 'gpu'。默认值为 'cpu'。
        metrics (List[str]): 评估指标列表。默认值为 ['psnr', 'ssim']。
    """

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-6,
                 weight_decay: float =1e-4,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ['psnr', 'ssim', 'rmse', 'sam', 'ergas', 'mrae'],
                 loss_fn: List[str] = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 args: TrainOptions = None,
                 ):
        super().__init__()
        self.model = model
        self.opt = args
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics
        self.optimizer = optimizer  # 使用自定义的优化器
        self.scheduler = scheduler  # 使用自定义的调度器

        # 选择loss
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.sparseloss = SparseKLloss()
        self.sum2oneloss = SumToOneLoss()


        # 初始化评估指标
        self.init_metrics(metrics)

    def init_metrics(self, metrics: List[str]) -> None:
        """
        初始化评估指标。

        参数:
            metrics (List[str]): 评估指标名称列表。
        """

        # self.train_loss = torchmetrics.MeanMetric()
        # self.val_loss = torchmetrics.MeanMetric()
        # self.test_loss = torchmetrics.MeanMetric()
        #
        # self.train_metrics = imagefusion_metrics(metrics)
        # self.val_metrics = imagefusion_metrics(metrics)
        # self.test_metrics = imagefusion_metrics(metrics)
        metric_dict = {
            'psnr': torchmetrics.PeakSignalNoiseRatio(),
            'ssim': torchmetrics.StructuralSimilarityIndexMeasure()
        }
        selected_metrics = {name: metric_dict[name] for name in metrics if name in metric_dict}

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = MetricCollection(selected_metrics)
        self.val_metrics = MetricCollection(selected_metrics)
        self.test_metrics = MetricCollection(selected_metrics)


    def fit(self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 2000,
        *args,
        **kwargs) -> Any:
        """
        训练模型。

        参数：
            train_loader (DataLoader): 训练数据的 DataLoader。
            val_loader (DataLoader): 验证数据的 DataLoader。
            max_epochs (int): 最大训练轮数，默认为 2000。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        返回：
            Any: 训练结果。
        """
        trainer = pl.Trainer(devices=self.devices,
                            accelerator=self.accelerator,
                            max_epochs=max_epochs,
                            *args,
                            **kwargs)
        # return trainer.fit(self, train_loader, val_loader,
        #                    ckpt_path="./logs/logs_HyCoNet/experiment/version_2/checkpoints/best_model_epochepoch=1299-train_losstrain_loss=4.43.ckpt")
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args, **kwargs) -> _EVALUATE_OUTPUT:
        """
        使用测试数据加载器对模型进行测试。

        参数:
            test_loader (DataLoader): 测试数据的 DataLoader。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        返回:
            _EVALUATE_OUTPUT: 每个DataLoader的评估输出(字典列表)。
        """
        # 创建 Trainer 实例用于测试
        trainer = pl.Trainer(
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        
        # 执行测试，并返回测试结果
        return trainer.test(self, test_loader)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        Lr_HSI = batch['Lr_HSI']
        Hr_MSI = batch['Hr_MSI']
        Hr_HSI_hat = self.model(Lr_HSI, Hr_MSI)
        #返回HR_hat
        return Hr_HSI_hat

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 模型前向传播
        Hr_HSI_hat = self(batch)
        Lr_HSI = batch['Lr_HSI']
        Hr_MSI = batch['Hr_MSI']
        Hr_HSI = batch['Hr_HSI']
        # 计算损失
        # base_loss
        self.loss_Z_Za_hat = self.L1Loss(Lr_HSI, self.model.Za_hat) * self.opt.lambda_A
        self.loss_Z_Zb_hat = self.L1Loss(Lr_HSI, self.model.Zb_hat) * self.opt.lambda_B
        self.loss_Y_Y_hat = self.L1Loss(Hr_MSI, self.model.Y_hat) * self.opt.lambda_C
        self.loss_lr_Y_hat_f_Y_lr_Y_hat_f_Z = self.L1Loss(self.model.lr_Y_hat_f_Y,self.model.lr_Y_hat_f_Z ) * self.opt.lambda_D
        self.loss_bass = self.loss_Z_Za_hat + self.loss_Z_Zb_hat + self.loss_Y_Y_hat + self.loss_lr_Y_hat_f_Y_lr_Y_hat_f_Z
        # sparse_loss
        self.loss_sparse_A_f_LrHSI = self.sparseloss(self.model.A_f_LrHSI)
        self.loss_sparse_A_f_Hr_MSI = self.sparseloss(self.model.A_f_Hr_MSI)
        self.loss_sparse_A_f_A = self.sparseloss(self.model.A_f_A)       
        self.loss_sparse = self.loss_sparse_A_f_LrHSI + self.loss_sparse_A_f_Hr_MSI + self.loss_sparse_A_f_A
        # sumtoone_loss
        self.loss_sum2one_A_f_LrHSI = self.sum2oneloss(self.model.A_f_LrHSI)
        self.loss_sum2one_A_f_Hr_MSI = self.sum2oneloss(self.model.A_f_Hr_MSI)
        self.loss_sum2one_A_f_A = self.sum2oneloss(self.model.A_f_A)   
        self.loss_sum2one = self.loss_sum2one_A_f_LrHSI + self.loss_sum2one_A_f_Hr_MSI + self.loss_sum2one_A_f_A

        loss = self.loss_bass + self.opt.lambda_E*self.loss_sparse + self.opt.lambda_F*self.loss_sum2one

        self.log("train_loss", 
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True,
                 sync_dist=True)

        # 计算并记录指标
        for metric_name, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[metric_name]}",
                     metric_value(Hr_HSI_hat, Hr_HSI),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True,
                     sync_dist=True)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        #进行梯度裁剪
        cliper_zeroone = model.ZeroOneClipper()
        #对端元进行裁剪
        self.model.A2Z.apply(cliper_zeroone)
        #对SRF进行裁剪
        self.model.HSI2MSI.apply(cliper_zeroone)

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True,
                 sync_dist=True
                 )
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True,
                     sync_dist=True
                     )

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    # def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     HR = batch['HR']
    #     HR_hat = self(batch)
                
    #     loss = self.loss_fn(HR_hat, HR)
    #     self.val_loss.update(loss)
    #     self.val_metrics.update(HR_hat, HR)

    #     return loss

    # def on_validation_epoch_end(self) -> None:
    #     self.log("val_loss",
    #              self.val_loss.compute(),
    #              prog_bar=False,
    #              on_epoch=True,
    #              on_step=False,
    #              logger=True,
    #              sync_dist=True)
    #     for i, metric_value in enumerate(self.val_metrics.values()):
    #         self.log(f"val_{self.metrics[i]}",
    #                  metric_value.compute(),
    #                  prog_bar=False,
    #                  on_epoch=True,
    #                  on_step=False,
    #                  logger=True,
    #                  sync_dist=True)

    #             # print the metrics
    #     str = "\n[Val] "
    #     for key, value in self.trainer.logged_metrics.items():
    #         if key.startswith("val_"):
    #             str += f"{key}: {value:.3f} "
    #     log.info(str + '\n')

    #     self.val_loss.reset()
    #     self.val_metrics.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        HR = batch['H']
        HR_hat = self(batch)
        loss = self.loss_fn(HR_hat, HR)

        self.test_loss.update(loss)
        self.test_metrics.update(HR_hat, HR)
        return loss
    
    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True,
                 sync_dist=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True,
                     sync_dist=True)
            
            # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        # 使用传入的优化器和调度器
        if self.optimizer and self.scheduler:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val_loss"  # 监控 val_loss，或者你想跟踪的其他指标
                }
            }
        else:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        return [optimizer], [scheduler]
    
    def predict_step(self,
                     batch: Dict[str, torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):

        HR_hat = self(batch)

        return HR_hat
       



