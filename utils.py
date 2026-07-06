"""
functional functions
"""
import os
import re
import shutil
import glob
import yaml
import csv
import logging
import random
import numpy as np
import torch
import torchaudio
import itertools
from collections import OrderedDict
sep = os.sep


def load_yaml(file_path='./config.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params


def save_yaml_file(file_path, data: dict):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, encoding='utf-8', allow_unicode=True)


def save_load_version_files(path, file_patterns, pass_dirs=None):
    #    save latest version files
    if pass_dirs is None:
        pass_dirs = ['.', '_', 'runs', 'results']
    copy_files(f'.{sep}', 'runs/latest_project', file_patterns, pass_dirs)
    copy_files(f'.{sep}', os.path.join(path, 'project'), file_patterns, pass_dirs)


def save_csv(file_path, data: list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


# 复制目标文件到目标路径
def copy_files(root_dir, target_dir, file_patterns, pass_dirs=['.git']):
    # print(root_dir, root_dir.split(sep), [name for name in root_dir.split(sep) if name != ''])
    os.makedirs(target_dir, exist_ok=True)
    len_root = len([name for name in root_dir.split(sep) if name != ''])
    for root, _, _ in os.walk(root_dir):
        cur_dir = sep.join(root.split(sep)[len_root:])
        first_dir_name = cur_dir.split(sep)[0]
        if first_dir_name != '':
            if (first_dir_name in pass_dirs) or (first_dir_name[0] in pass_dirs): continue
        # print(len_root, root, cur_dir)
        target_path = os.path.join(target_dir, cur_dir)
        os.makedirs(target_path, exist_ok=True)
        files = []
        for file_pattern in file_patterns:
            file_path_pattern = os.path.join(root, file_pattern)
            files += sorted(glob.glob(file_path_pattern))
        for file in files:
            target_path_file = os.path.join(target_path, os.path.split(file)[-1])
            shutil.copyfile(file, target_path_file)


def save_model_state_dict(file_path, epoch=None, net=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': net.state_dict() if net else None,
    }
    torch.save(state_dict, file_path)


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_filename_list(dir_path, pattern='*', ext='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list


def set_type(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


def get_machine_id_list(data_dir):
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in get_filename_list(data_dir)])
    )))
    return machine_id_list


def metadata_to_label(data_dirs):
    meta2label = {}
    label2meta = {}
    label = 0

    # 逐类别去检索数据的元信息
    for data_dir in data_dirs:
        machine = data_dir.split('/')[-2]
        id_list = get_machine_id_list(data_dir)
        for id_str in id_list:
            meta = machine + '-' + id_str
            meta2label[meta] = label
            label2meta[label] = meta
            label += 1
            # meta2label是元信息对标签的对照关系：例如 'fan-id_00' 对应标签 0
            # label2meta是标签对应元信息关系
    return meta2label, label2meta


def metadata_to_label_hierarchy(data_dirs):
    meta2label = OrderedDict()
    label2meta = OrderedDict()


    # 先得到机械类型列表
    machine_list = []
    for data_dir in data_dirs:
        machine = data_dir.split('/')[-2]
        machine_list.append(machine)
    machine_list = list(set(machine_list))


    # 生成字典：键为元素，值为索引（标签）
    machine2label = OrderedDict()
    for idx, machine_ in enumerate(machine_list):
        machine2label[machine_] = idx


    machine_num_classes = len(machine_list) # 机械类别数量




    # 再得到idlist
    id_label = 0
    for data_dir in data_dirs:
        machine = data_dir.split('/')[-2]
        id_list = get_machine_id_list(data_dir)
        for id_str in id_list:
            meta = machine + '-' + id_str
            machine_label = machine2label[machine]
            meta2label[meta] = [machine_label, id_label] # 获取该设备对应的标签以及id标签，叠加作为标签
            label2meta[f'{machine_label}-{id_label}'] = meta
            id_label += 1

    # 存放每个机械类型下ID数量的列表
    machine_id_count = OrderedDict()
    for data_dir in data_dirs:
        machine = data_dir.split('/')[-2]
        count_id = sum(1 for key in meta2label.keys() if machine in key)
        machine_id_count[machine] = count_id

    return meta2label, label2meta, machine_id_count, machine_num_classes



def create_test_file_list(target_dir,
                          id_name,
                          dir_name='test',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


if __name__ == '__main__':
    print(get_filename_list('../Fastorch', ext='py'))
