#!/bin/bash

# 定义虚拟环境名称
ENV_NAME="leida"

# 检查 conda 是否已经安装
if ! command -v conda &> /dev/null
then
    echo "Conda 未安装或未在 PATH 中。请先安装 Anaconda 或 Miniconda。"
    exit 1
fi

# 激活 conda 环境
echo "正在激活虚拟环境：$ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"  # 加载 conda
conda activate $ENV_NAME

# 检查虚拟环境是否成功激活
if [ $? -eq 0 ]; then
    echo "虚拟环境 $ENV_NAME 已成功激活。"
else
    echo "激活虚拟环境失败。请检查是否存在环境 $ENV_NAME 或 Conda 安装是否正确。"
fi

# 定义实验名称和数据路径
experiment_name="experiment"
root_path="./data"
checkpoints_dir="logs/logs_HyCoNet"

# 训练相关参数
batch_size=1
val_batch_size=1
learning_rate=3e-4
epochs=15000
epoch_count=1
niter=2000
niter_decay=8000
lr_policy="lambda"
lr_decay_iters=1000
lr_decay_gamma=0.8
lr_decay_patience=200000
save_epoch_freq=20

# GPU设置
gpu_ids="0"

# 多线程支持
num_threads=0

# HSI 和 MSI 相关参数
scale=8
hsi_channels=128
msi_channels=3

# lambda 权重参数
lambda_A=10
lambda_B=10
lambda_C=10
lambda_D=0.01
lambda_E=0.01
lambda_F=10000
lambda_G=0.0
lambda_H=0.0
num_P=128
avg_crite_flag=""

# 如果需要启用 avg_crite 参数，将其置为 "--avg_crite"
if [ "$1" == "avg_crite" ]; then
    avg_crite_flag="--avg_crite"
fi

# 运行训练脚本
python train_HyCoNet.py \
    --name $experiment_name \
    --root_path $root_path \
    --checkpoints_dir $checkpoints_dir \
    --batch_size $batch_size \
    --val_batch_size $val_batch_size \
    --lr $learning_rate \
    --epochs $epochs \
    --epoch_count $epoch_count \
    --niter $niter \
    --niter_decay $niter_decay \
    --lr_policy $lr_policy \
    --lr_decay_iters $lr_decay_iters \
    --lr_decay_gamma $lr_decay_gamma \
    --lr_decay_patience $lr_decay_patience \
    --save_epoch_freq $save_epoch_freq \
    --gpu_ids $gpu_ids \
    --nThreads $num_threads \
    --scale $scale \
    --hsi_channels $hsi_channels \
    --msi_channels $msi_channels \
    --lambda_A $lambda_A \
    --lambda_B $lambda_B \
    --lambda_C $lambda_C \
    --lambda_D $lambda_D \
    --lambda_E $lambda_E \
    --lambda_F $lambda_F \
    --lambda_G $lambda_G \
    --lambda_H $lambda_H \
    --num_P $num_P \
    $avg_crite_flag
