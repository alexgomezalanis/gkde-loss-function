from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.io import wavfile
import soundfile as sf 

PA_CLASSES = {'-': 0, 'AA': 1, 'AB': 2, 'AC': 3, 'BA': 4, 'BB': 5, 'BC': 6, 'CA': 7, 'CB': 8, 'CC': 9}

Fs = 16000 # Hz

class LCNN_Dataset(Dataset):
  """LCNN numpy dataset."""

  def __init__(
    self,
    csv_file,
    root_dir,
    n_filts,
    n_frames,
    window,
    shift,
    dataset,
    is_evaluating_la,
    num_classes,
    dataframe=None,
    normalize=False):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the wav files.
        n_
        window length (float): Window length in seconds
        frame shift (float): Frame shift in seconds
    """
    if dataframe is not None:
      self.wavfiles_frame = dataframe
    else:
      self.wavfiles_frame = pd.read_csv(csv_file, sep=' ')
    self.root_dir = root_dir
    self.n_filts = n_filts
    self.n_frames = n_frames
    self.nperseg = window * Fs
    self.noverlap = self.nperseg - shift * Fs
    self.nfft = int(n_filts * 2)
    self.dataset = dataset
    self.is_evaluating_la = is_evaluating_la
    self.num_classes = num_classes
    self.normalize = normalize

  def __len__(self):
    return len(self.wavfiles_frame)

  def __getitem__(self, idx):
    label = self.wavfiles_frame['label'][idx]
    nameFile = self.wavfiles_frame['wav'][idx]
    if self.is_evaluating_la:
      if label == 'bonafide':
        label = 'A00'
      else:
        label = self.wavfiles_frame['target'][idx]
      label = label[1:]
      target = int(label)
    else:
      if self.num_classes == 10:
        target = PA_CLASSES[self.wavfiles_frame['target'][idx]]
      else:
        target = 0 if label == 'bonafide' else 1

    db = 'la-challenge' if self.is_evaluating_la else 'pa-challenge'
    path_db = os.path.join(self.root_dir, db)
    if self.dataset != 'test':
      if self.is_evaluating_la:
        file_dir = 'S' + str(target)
      else:
        file_dir = 'genuine' if label == 'bonafide' else 'spoof'
      file_path = os.path.join(path_db, 'flac-files', self.dataset, file_dir, nameFile + '.flac')
    else:
      file_path = os.path.join(path_db, 'flac-files', self.dataset, 'ASVspoof2019_' + nameFile[:2] + '_eval_v1/flac', nameFile + '.flac')

    # STFT features
    stft = np.zeros((self.n_filts, self.n_frames), dtype=np.float32) # [height, width]

    data, fs = sf.read(file_path, dtype='float32')

    f, t, Sxx =  signal.stft(data, fs, window='blackman', nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
    Sxx = np.abs(Sxx)
    features = np.where(Sxx > 1e-10, np.log10(Sxx), -10)

    feat = features[:self.n_filts, :]
    numFrames = len(feat[0])

    if numFrames < self.n_frames:
      nCuts = 1
      while (numFrames * nCuts < self.n_frames):
        stft[:, numFrames * (nCuts - 1) : numFrames * nCuts] = feat[:, :]
        nCuts += 1
      stft[:, numFrames * (nCuts - 1) : self.n_frames] = feat[:, :self.n_frames - numFrames * (nCuts - 1)]
    else:
      stft = feat[:, :self.n_frames]

    if self.normalize:
      median = np.mean(stft, axis=1)
      std = np.std(stft, axis=1)
      stft = (np.transpose(stft) - median) / std
    stft = np.reshape(stft, (1, self.n_filts, self.n_frames))

    sample = (stft, target, nameFile)

    return sample