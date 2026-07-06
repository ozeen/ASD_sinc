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

sep = os.sep


def main():
    # ===================== 1. 命令行参数解析（全部从命令行传入）=====================
    parser = argparse.ArgumentParser(description='resnet18_mel_sinc_tgram training')

    # 基础配置
    parser.add_argument('--version', type=str, default='test', help='model version name')
    parser.add_argument('--time_version', type=bool, default=False, help='add time stamp to version')
    parser.add_argument('--save_version_files', type=bool, default=False, help='save version files')
    parser.add_argument('--save_version_file_patterns', nargs='+', default=["*.py", "*.yaml"], help='file patterns to save')
    parser.add_argument('--pass_dirs', nargs='+', default=['.', '_', 'runs', 'results'], help='dirs to skip')

    # 文件路径
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
                                                            '../dataset/DCASE2020/additional/valve/train', ],
                        help='training data directories')

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

    # 音频预处理
    parser.add_argument('--sr', type=int, default=16000, help='sample rate')
    parser.add_argument('--n_fft', type=int, default=2048, help='fft points')
    parser.add_argument('--n_mels', type=int, default=256, help='mel banks')
    parser.add_argument('--win_length', type=int, default=2048, help='window length')
    parser.add_argument('--hop_length', type=int, default=512, help='hop length')
    parser.add_argument('--power', type=float, default=2.0, help='spectrogram power')
    parser.add_argument('--secs', type=int, default=10, help='audio length in seconds')

    # 测试dataloader 参数
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')

    # 设备配置
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='cuda device ids')
    # 随机种子
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    # 测试/加载
    parser.add_argument('--checkpoint_dir',  default='./weights/RGC/sinc/best_checkpoint.pth.tar',type=str,  help='model checkpoint path')


    args = parser.parse_args()

    # ===================== 1. 日志与TensorBoard初始化 =====================
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'{time_str}-{args.version}'
    print(f"当前版本：{args.version}")
    log_dir = f'runs/{args.version}'
    os.makedirs(log_dir, exist_ok=True)

    # 日志与writer
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    args.writer, args.logger = writer, logger
    args.logger.info(f"Start experiment: {args.version}")

    # ===================== 2. 测试  =====================
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



    # 模型初始化
    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f'Num classes: {args.num_classes}')

    net =  ClassVDD_newmobile_mel_tgram_sinc(
        num_classes=args.num_classes,
        device=args.device,
        z_dim=128
    )
    
    
    # 模型加载已训练好权重
    net = net.to(args.device)
    model_path = args.checkpoint_dir #'weights/mel_only_weights_soft/best_checkpoint.pth.tar'
    net.load_state_dict(torch.load(model_path)['model'])
    
    
    
    # 优化器与学习率调度


    
    trainer = Trainer(
        args=args,
        net=net,
        optimizer=None,
        scheduler=None,
        transform=train_dataset.transform
    )
    
    # 测试

    trainer.valid()


if __name__ == '__main__':
    main()