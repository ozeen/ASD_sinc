import os
import re
import glob
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
import random
import pandas as pd

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


class ASDDataset(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory



        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    def transform(self, filename):

        machine = filename.split('/')[-2]

        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]



        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label



    def __len__(self):
        return len(self.file_list)


