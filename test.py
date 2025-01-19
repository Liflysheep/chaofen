import torch

# 指定检查点路径
checkpoint_path = "./logs/logs_HyCoNet/experiment/version_0/checkpoints/last.ckpt"

# 加载检查点
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 打印检查点的键
print(checkpoint.keys())

# 获取回调
callbacks = checkpoint.get('callbacks', [])
print(callbacks)

# 修改 EarlyStopping 的耐心值
early_stopping_key = "EarlyStopping{'monitor': 'train_loss', 'mode': 'min'}"
if early_stopping_key in checkpoint['callbacks']:
    checkpoint['callbacks'][early_stopping_key]['patience'] = 5000
    print(checkpoint['callbacks'][early_stopping_key]['patience'])

# 保存修改后的检查点
new_checkpoint_path = "./logs/logs_HyCoNet/experiment/version_0/checkpoints/modified_last.ckpt"
torch.save(checkpoint, new_checkpoint_path)

print(f"Modified checkpoint saved to {new_checkpoint_path}")