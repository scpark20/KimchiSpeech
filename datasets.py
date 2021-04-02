import librosa
import librosa.display
import numpy as np
import torch
import os
from os import listdir
from os.path import isfile, join, isdir
import ntpath
from torch.utils.data import DataLoader
import warnings
from g2p_en import G2p
#import pytorch_lightning as pl

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    mel = np.dot(spc, mel_basis.T)
    return np.log10(np.maximum(1e-5, mel)).T

class LibriDataset(torch.utils.data.Dataset):
    def __init__(self, hp, split='train'):
        self.hp = hp
        self.split = split
        self.data_files = self._get_data_files(hp.dataset, hp.data_dir, hp.data_file)
        self.mel_matrix = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
        self.g2p_en = G2p()
        
    def _get_data_files(self, root_dir):
        dirs = ['103', '150', '200', '250']
        dirs = [join(root, f) for f in dirs if isdir(join(root, f))]

        sub_dirs = []
        for dir in dirs:
            sub_dirs += [join(dir, f) for f in listdir(dir) if isdir(join(dir, f))]

        files = []
        for dir in sub_dirs:
            files += [join(dir, f) for f in listdir(dir) if isfile(join(dir, f)) and '.normalized.txt' in f]
           
        data_files = []
        for file in files:
            with open(file, 'r') as f:
                l = f.readline()
            data_files.append(())

        
    

class LJDataset(torch.utils.data.Dataset):
    def __init__(self, hp, split='train'):
        self.hp = hp
        self.split = split
        self.data_files = self._get_data_files(hp.dataset, hp.data_dir, hp.data_file)
        self.mel_matrix = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
        self.g2p_en = G2p()
        
    def _get_data_files(self, dataset, root_dir, file):
        if dataset == 'lj':
            metadata = root_dir + 'metadata.csv'

            data_files = []
            with open(metadata, 'r') as f:
                l = f.readline().strip()
                while l:
                    l = l.split('|')
                    if self.split == 'test':
                        if 'LJ001' in l[0] or 'LJ002' in l[0]:
                            pass
                        else:
                            l = f.readline().strip()
                            continue
                            
                    elif self.split == 'valid':
                        if 'LJ003' in l[0]:
                            pass
                        else:
                            l = f.readline().strip()
                            continue
                    
                    elif self.split == 'train':
                        if 'LJ001' in l[0] or 'LJ002' in l[0] or 'LJ003' in l[0]:
                            l = f.readline().strip()
                            continue
                        else:
                            pass
                    else:
                        pass
                    
                    wav_file = root_dir + 'wavs/' + l[0] + '.wav'
                    text = l[2]
                    data_files.append((wav_file, text))
                    l = f.readline().strip()

            return data_files  
        
        elif dataset == 'kss':
            metadata = root_dir + file

            data_files = []
            with open(metadata, 'r') as f:
                l = f.readline().strip()
                while l:
                    l = l.split('|')
                    wav_file = root_dir + l[0]
                    text = l[2]
                    data_files.append((wav_file, text))
                    l = f.readline().strip()

            return data_files
    
    def _get_mel(self, data_file):
        wav, _ = librosa.core.load(data_file, sr=22050)
        #wav, _ = librosa.effects.trim(wav, top_db=40)
        
        with warnings.catch_warnings():
            mel = logmelfilterbank(wav, sampling_rate=22050, fft_size=1024, hop_size=256, fmin=80, fmax=7600)
    
        if self.hp.mel_norm:
            mel = (mel + 5) / 5
            
        return mel
    
    def _get_utf8_values(self, text):
        if self.hp.g2p:
            if self.hp.dataset == 'lj':
                text_array = self.g2p_en(text)
                text = ""
                for t in text_array:
                    text += t
                
        #text = g2p(text)
        text_utf = text.encode()
        ts = [0]
        for t in text_utf:
            ts.append(t)
        if self.hp.eos_token:
            ts.append(256)    
        else:
            ts.append(0)
        utf8_values = np.array(ts)
        
        return utf8_values
        
        
    def __getitem__(self, index):
        wav, _ = librosa.core.load(self.data_files[index][0], sr=22050)
        mel = self._get_mel(self.data_files[index][0])
        string = self.data_files[index][1]
        text = self._get_utf8_values(string)
        
        return torch.LongTensor(text), torch.FloatTensor(mel), string, wav
        
    def __len__(self):
        return len(self.data_files)
    
class TextMelCollate():
    
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hp):
        self.hp = hp
        
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        
        outputs = {}
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        strings = []
        wavs = []
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            strings.append(batch[ids_sorted_decreasing[i]][2])
            wavs.append(batch[ids_sorted_decreasing[i]][3])
        outputs['text'] = text_padded
        outputs['text_lengths'] = input_lengths
        outputs['strings'] = strings
        outputs['wavs'] = wavs
            
        # include mel padded and gate padded
        num_mels = batch[0][1].size(0)    
        max_target_len = max([x[1].shape[1] for x in batch])
        #max_target_len = 1024
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        if self.hp.mel_norm:
            mel_padded.fill_(0)
        else:
            mel_padded.fill_(-5)
            
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            
        outputs['mels'] = mel_padded
        outputs['mel_lengths'] = output_lengths

        return outputs
    
# class LJDataModule(pl.LightningDataModule):
#     def __init__(self, hp):
#         super().__init__()
#         self.hp = hp
        
#     def setup(self, stage=None):
#         self.train_dataset = LJDataset(self.hp.root_dir)
#         self.collate_fn = TextMelCollate()
        
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_dataset,
#                                            batch_size=self.hp.batch_size,
#                                            num_workers=self.hp.num_workers,
#                                            shuffle=True,
#                                            collate_fn=self.collate_fn)