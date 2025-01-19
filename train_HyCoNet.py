import os
import subprocess
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from trainer.imagefusion_unsupervised.imagefusion_unsupervised import ImageFusionTrainer
from data_loader.dataloader_unsupervised import HyperspectralMultispectralDataset
from model.HyCoNet.HyCoNet import HyCoNet
from trainer.imagefusion_unsupervised.train_option import TrainOptions
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 设置 GPU 设备和 ulimit 参数
os.environ["CUDA_VISIBLE_DEVICES"] = "3,0,1,2"
subprocess.call("ulimit -n 65536", shell=True)

cpu_count = os.cpu_count()


def transform_label(x):
    return x - 1


logdict = 'logs_swincanet'

if __name__ == "__main__":
    # 添加命令行参数解析
    args = TrainOptions().parse()
    # 指定 .mat 文件所在的路径
    root_path = args.root_path

    # 创建数据集实例
    dataset = HyperspectralMultispectralDataset(root_path=root_path, window_size=None, verbose=True)

    # DataLoader 设置
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count // 2, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count // 2, persistent_workers=True
    )


    # 模型设置
    model = HyCoNet(scale=args.scale, hsi_channels=args.hsi_channels, msi_channels=args.msi_channels, num_P=args.num_P)
    # 假设模型的权重保存在 checkpoint_path 位置
    checkpoint_path = "/home2/lfy/chaofen/logs/logs_HyCoNet/experiment/version_0/checkpoints/last.ckpt"  # 这里的路径要替换为你保存模型的路径
    # TensorBoard Logger 设置
    logger = TensorBoardLogger(save_dir=f"{args.checkpoints_dir}", name="experiment")

    trainer = ImageFusionTrainer(
        model=model,
        lr=args.lr,
        weight_decay=0,
        devices=1,  # 设置 GPU 设备
        accelerator="gpu",  # 使用 GPU
        args=args,
    )

    # 提前停止回调
    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=args.lr_decay_patience, mode="min", verbose=True
    )

    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best_model_epoch{epoch:02d}-train_loss{train_loss:.2f}",
        save_weights_only=False,
        verbose=True,
        every_n_epochs=args.save_epoch_freq,
    )

    # 使用 Trainer 进行训练
    trainer.fit(
        train_loader,
        val_loader,
        log_every_n_steps=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        limit_val_batches=0,
    )

