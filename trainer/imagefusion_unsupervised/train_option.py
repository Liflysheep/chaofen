import argparse
import os
import torch

class TrainOptions():
    def __init__(self):
        """初始化解析器"""
        self.parser = argparse.ArgumentParser(description="Base Options for Training")
        self.initialized = False

    def initialize(self):
        """定义基本的命令行参数"""
        # 基本配置
        self.parser.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment')
        self.parser.add_argument('--root_path', type=str, default='./data', help='Path to the dataset')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Where to save models')

        # 训练相关
        self.parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
        self.parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
        self.parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
        self.parser.add_argument('--epochs', type=int, default=10000, help='Maximum number of epochs')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='Starting epoch count')
        self.parser.add_argument('--niter', type=int, default=2000, help='Number of iterations at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=8000, help='Number of iterations to decay learning rate')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='Learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=1000, help='Decay learning rate every N iterations')
        self.parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='Gamma for learning rate decay')
        self.parser.add_argument('--lr_decay_patience', type=int, default=50, help='Patience for plateau-based LR scheduling')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='Frequency of saving checkpoints')

        # GPU 设置
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU ids, e.g., "0,1,2". Use "-1" for CPU.')

        # 多线程支持
        self.parser.add_argument('--nThreads', type=int, default=0, help='Number of threads for data loading')

        # HSI 和 MSI 相关参数
        self.parser.add_argument("--scale", type=int, default=8, help="Scale for MSDCNN model")
        self.parser.add_argument("--hsi_channels", type=int, default=128, help="Number of HSI channels")
        self.parser.add_argument("--msi_channels", type=int, default=3, help="Number of MSI channels")

        # 添加训练相关的 lambda 参数
        self.parser.add_argument('--lambda_A', type=float, default=10, help='weight for lr_lr')
        self.parser.add_argument('--lambda_B', type=float, default=10, help='weight for msi_msi beta')
        self.parser.add_argument('--lambda_C', type=float, default=10, help='weight for msi_s_lr alpha')
        self.parser.add_argument('--lambda_D', type=float, default=0.01, help='weight for sum2one mu')
        self.parser.add_argument('--lambda_E', type=float, default=0.01, help='weight for sparse nu')
        self.parser.add_argument('--lambda_F', type=float, default=100, help='weight for lrmsi gamma')
        self.parser.add_argument('--lambda_G', type=float, default=0.0, help='non')
        self.parser.add_argument('--lambda_H', type=float, default=0.0, help='non')
        self.parser.add_argument('--num_P', type=int, default=100)
        self.parser.add_argument('--avg_crite', action="store_true")

        self.isTrain = True
        self.initialized = True

    def parse(self):
        """解析命令行参数"""
        if not self.initialized:
            self.initialize()

        # 解析参数
        self.opt = self.parser.parse_args()

        # 处理 GPU 设置
        self.opt.gpu_ids = [int(id) for id in self.opt.gpu_ids.split(',') if int(id) >= 0]
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # 打印参数
        self.print_options(self.opt)

        return self.opt

    def print_options(self, opt):
        """打印参数并保存到文件"""
        message = '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            message += f'{str(k):>25}: {str(v):<30}\n'
        message += '----------------- End -------------------'
        print(message)

        # 保存到文件
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'w') as opt_file:
            opt_file.write(message + '\n')

if __name__ == '__main__':
    options = TrainOptions()
    options.initialize()  # 初始化基本参数
    args = options.parse()  # 解析参数