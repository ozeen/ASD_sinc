import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataload import ASDDataset
import utils
from src.model_mc import ClassVDD_newmobile_mel_only, ClassVDD_newmobile_mel_tgram_sinc

from trainer import Trainer
from torch.optim.lr_scheduler import LambdaLR
import math

sep = os.sep

def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def main():
    

    parser = argparse.ArgumentParser(description='resnet18_mel_sinc_tgram training')

    # 基础配置
    parser.add_argument('--version', type=str, default='1', help='model version name')
    parser.add_argument('--time_version', type=bool, default=False, help='add time stamp to version')
    parser.add_argument('--save_version_files', type=bool, default=False, help='save version files')
    parser.add_argument('--save_version_file_patterns', nargs='+', default=["*.py", "*.yaml"],
                        help='file patterns to save')
    parser.add_argument('--pass_dirs', nargs='+', default=['.', '_', 'runs', 'results'], help='dirs to skip')

    # 文件路径
    parser.add_argument('--train_dirs', nargs='+', default=['../dataset/DCASE2020/dev/fan/train',
                                                            '../dataset/DCASE2020/dev/pump/train',
                                                            '../dataset/DCASE2020/dev/slider/train',
                                                            '../dataset/DCASE2020/dev/ToyCar/train',
                                                            '../dataset/DCASE2020/dev/ToyConveyor/train',
                                                            '../dataset/DCASE2020/dev/valve/train',
                                                            '../dataset/DCASE2020/additional/fan/train',
                                                            '../dataset/DCASE2020/additional/pump/train',
                                                            '../dataset/DCASE2020/additional/slider/train',
                                                            '../dataset/DCASE2020/additional/ToyCar/train',
                                                            '../dataset/DCASE2020/additional/ToyConveyor/train',
                                                            '../dataset/DCASE2020/additional/valve/train',], help='training data directories')
    
    parser.add_argument('--valid_dirs', nargs='+', default=['../dataset/DCASE2020/dev/fan/test',
                                                            '../dataset/DCASE2020/dev/pump/test',
                                                            '../dataset/DCASE2020/dev/slider/test',
                                                            '../dataset/DCASE2020/dev/ToyCar/test',
                                                            '../dataset/DCASE2020/dev/ToyConveyor/test',
                                                            '../dataset/DCASE2020/dev/valve/test',
                                                           ], help='validation data directories')
    
    parser.add_argument('--test_dirs', nargs='+', default=['../dataset/DCASE2020/dev/fan/test',
                                                            '../dataset/DCASE2020/dev/pump/test',
                                                            '../dataset/DCASE2020/dev/slider/test',
                                                            '../dataset/DCASE2020/dev/ToyCar/test',
                                                            '../dataset/DCASE2020/dev/ToyConveyor/test',
                                                            '../dataset/DCASE2020/dev/valve/test',
                                                          ], help='test data directories')
    
    parser.add_argument('--result_dir', type=str, default='./results', help='result save directory')

    # 预处理
    parser.add_argument('--sr', type=int, default=16000, help='sample rate')
    parser.add_argument('--n_fft', type=int, default=2048, help='fft points')
    parser.add_argument('--n_mels', type=int, default=256, help='mel banks')
    parser.add_argument('--win_length', type=int, default=2048, help='window length')
    parser.add_argument('--hop_length', type=int, default=512, help='hop length')
    parser.add_argument('--power', type=float, default=2.0, help='spectrogram power')
    parser.add_argument('--secs', type=int, default=10, help='audio length in seconds')

    # 设备
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='cuda device ids')

    # 训练配置（与demo.sh一致）
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--valid_every_epochs', type=int, default=5, help='validate every N epochs')
    parser.add_argument('--early_stop_epochs', type=int, default=-1, help='early stop patience')
    parser.add_argument('--start_save_model_epochs', type=int, default=1, help='start save model epoch')
    parser.add_argument('--save_model_interval_epochs', type=int, default=5, help='save model interval')
    parser.add_argument('--start_scheduler_epoch', type=int, default=20, help='start lr scheduler epoch')
    parser.add_argument('--start_valid_epoch', type=int, default=0, help='start validation epoch')
    parser.add_argument('--warm_up', type=int, default=3, help='warm up epochs')

    parser.add_argument('--load_epoch', type=str, default='False', help='load epoch for test (best/number/False)')

    args = parser.parse_args()

    # ===================== 2. 日志与TensorBoard初始化 =====================
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'{time_str}-{args.version}' if (args.load_epoch == 'False' and args.time_version) else args.version
    print(f"当前版本：{args.version}")
    log_dir = f'runs/{args.version}'
    os.makedirs(log_dir, exist_ok=True)

    # 日志与writer
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    args.writer, args.logger = writer, logger
    args.logger.info(f"Start experiment: {args.version}")


    # ===================== 3. 训练主逻辑 =====================
    # 随机种子
    utils.setup_seed(args.random_seed)

    # 设备设置
    args.dp = False
    if not args.cuda or args.device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.device_ids[0]}')
        if len(args.device_ids) > 1:
            args.dp = True

    # 加载数据集
    
    args.meta2label, args.label2meta = utils.metadata_to_label(args.train_dirs)
    train_file_list = []
    
    for train_dir in args.train_dirs:
        train_file_list.extend(utils.get_filename_list(train_dir))

    train_dataset = ASDDataset(args, train_file_list, load_in_memory=False)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    # 模型初始化
    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f'Num classes: {args.num_classes}')

    net = ClassVDD_newmobile_mel_tgram_sinc(
        num_classes=args.num_classes,
        device=args.device,
        z_dim=128
    )
    net = net.to(args.device)

    # 优化器与学习率调度
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
    
    # scheduler = build_warmup_cosine_scheduler(
    #     optimizer=optimizer,
    #     warmup_epochs=args.warm_up,
    #     total_epochs=args.epochs,
    #     steps_per_epoch=len( train_dataloader)
    # )

    # 训练器
    trainer_tester = Trainer(
        args=args,
        net=net,
        optimizer=optimizer,
        scheduler=None,
        transform=train_dataset.transform
    )

    # 开始训练
    if args.load_epoch == 'False':
        trainer_tester.train(train_dataloader)


if __name__ == '__main__':
    main()