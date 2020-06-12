import numpy as np
from torch.utils.data import Sampler

class CustomSampler(Sampler):
  def __init__(self, data_source, shuffle, num_classes):
    self.df = self.data_source.wavfiles_frame
    self.shuffle = shuffle
    self.num_classes = num_classes

  def getIndices(self):
    if self.num_classes == 2:
      labels = np.unique(self.df['label']).tolist()
      digit_indices = [np.where(self.df['label'] == i)[0] for i in labels]
    else:
      labels = np.unique(self.df['target']).tolist()
      digit_indices = [np.where(self.df['target'] == i)[0] for i in labels]
    if self.shuffle:
      for i in range(len(digit_indices)):
        np.random.shuffle(digit_indices[i])
    min_size = np.size(digit_indices[0])
    for i in range(1, len(digit_indices)):
      size = np.size(digit_indices[i])
      min_size = size if size < min_size else min_size
    return digit_indices, min_size

  def __iter__(self):
    digit_indices, min_size = self.getIndices()
    num_classes = len(digit_indices)
    indices = []
    for i in range(min_size):
      indices += [digit_indices[n][i] for n in range(num_classes)]
    return iter(indices)

  def __len__(self):
    digit_indices, min_size = self.getIndices()
    return min_size * len(digit_indices)