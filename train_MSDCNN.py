import os
import subprocess
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from trainer.imagefusion_new import ImageFusionTrainer
from data_loader import AISA, Toys
from model.MSDCNN import MSDCNN
import argparse



# # 设置 GPU 设备和 ulimit 参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,1,3"
subprocess.call("ulimit -n 65536", shell=True)

cpu_count = os.cpu_count()


def transform_label(x):
    return x - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--root_path", type=str, default='./data', help="")
    parser.add_argument("--window_size", type=int, default=None, help="")
    parser.add_argument("--window_size", type=int, default=None, help="")
    parser.add_argument("--window_size", type=int, default=None, help="")
    parser.add_argument("--data_type", type=str, default='AISA', help="")
    parser.add_argument("--data_name", type=str, default='chart_and_stuffed_toy_ms', help="")
    # parser.add_argument("--data_name", type=str, default='stuffed_toys_ms', help="")
    parser.add_argument("--log_save_dir", type=str, default='./logs/logs_MSDCNN', help="")
    # 指定 .mat 文件所在的路径
    args = parser.parse_args()

    # 指定 .mat 文件所在的路径
    root_path = args.root_path  # 请将此路径替换为实际的 .mat 文件存储路径

    # 创建数据集实例
    if args.data_type == 'AISA':
        dataset = AISA(root_path=root_path, window_size=40, verbose=True,hsi_num_channel=128)
    elif args.data_type == 'Toys':
        dataset = Toys(root_path=root_path, window_size=40, verbose=True,LR_sacle=16,data_name=args.data_name)


    # 设定随机数
    generator2 = torch.Generator().manual_seed(42)

    # 划分数据集
    train_dataset, val_dataset = random_split(dataset, [0.8,0.2], generator=generator2)

    # DataLoader 设置
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=cpu_count // 2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cpu_count // 2, persistent_workers=True)

    print(f"Validation DataLoader length: {len(val_loader)}")

    # 模型设置
    model = MSDCNN(scale=8, hsi_channels=128, msi_channels=3)

    # TensorBoard Logger 设置
    logger = TensorBoardLogger(save_dir=args.log_save_dir, name='experiment')

    # # 优化器设置
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.999))

    trainer = ImageFusionTrainer(
        model=model,
        lr=1e-6,
        weight_decay=1e-4,
        devices=1,  # 设置 GPU 设备
        accelerator="gpu",  # 使用 GPU
        metrics= ['psnr', 'ssim', 'rmse', 'sam', 'ergas', 'mrae' ,'uqi'],
    )

    # 提前停止回调
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=1000,  # 提前停止的耐心参数
        mode='min',
        verbose=True
    )

    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best_model_epoch{epoch:02d}-val_loss{val_loss:.2f}",
        save_weights_only=False,
        verbose=True,
        every_n_epochs=10,
    )

    # 使用 Trainer 进行训练
    trainer.fit(
        train_loader,
        val_loader,
        log_every_n_steps=1,
        max_epochs=10000,  # 最大训练次数
        callbacks=[checkpoint_callback, early_stopping_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        # limit_val_batches=1,
        logger=logger,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1 # 每 5 个 epoch 执行一次验证
    )
    #
    # # 使用 Trainer 进行测试
    # score = trainer.test(
    #     val_loader,
    #     enable_progress_bar=True,
    #     enable_model_summary=True
    # )[0]


