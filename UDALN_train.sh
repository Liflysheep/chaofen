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
    exit 1
fi

# 定义实验名称和数据路径
#data_name="houston18"
#data_name="AISA_3"
#data_name="AISA_31"
#data_name="stuffed_toys_ms"
data_name="chart_and_stuffed_toy_ms"
#data_name="Chikusei"

# GPU设置
gpu_ids="1"
scale_factor=32

cd ./对比试验/UDALN_GRSL

# 运行训练脚本
python train_all_special.py \
    --data_name $data_name \
    --gpu_ids $gpu_ids \
    --scale_factor $scale_factor

# 检查脚本是否成功运行
if [ $? -eq 0 ]; then
    echo "训练脚本成功执行。"
else
    echo "训练脚本执行失败。请检查日志或脚本中的错误。"
    exit 1
fi

