import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.nn import init

class mfm(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
    super(mfm, self).__init__()
    self.out_channels = out_channels
    if type == 1:
      self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
      init.xavier_uniform_(self.filter.weight)
      self.filter.bias.data.fill_(0.1)
    else:
      self.filter = nn.Linear(in_channels, 2 * out_channels)
      init.xavier_uniform_(self.filter.weight)
      self.filter.bias.data.fill_(0.1)

  def forward(self, x):
    x = self.filter(x)
    out = torch.split(x, self.out_channels, 1)
    return torch.max(out[0], out[1])

class LCNN(nn.Module):
  def __init__(self, emb_size, num_classes):
    super(LCNN, self).__init__()
    self.features = nn.Sequential(
      mfm(1, 8, 5, 1, 2),
      nn.BatchNorm2d(8),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
      mfm(8, 8, 1, 1, 0),
      nn.BatchNorm2d(8),
      mfm(8, 16, 3, 1, 1),
      #nn.BatchNorm2d(16),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
      mfm(16, 16, 1, 1, 0),
      nn.BatchNorm2d(16),
      mfm(16, 16, 3, 1, 1),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
      #nn.BatchNorm2d(16),
      mfm(16, 16, 1, 1, 0),
      nn.BatchNorm2d(16),
      mfm(16, 16, 3, 1, 1),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )
    self.mfm = mfm(16 * 25 * 16, emb_size, type=0)
    self.batch_norm = nn.BatchNorm1d(emb_size)
    self.fc5 = nn.Linear(emb_size, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = F.dropout(x, p=0.7, training=self.training)
    embedding = self.mfm(x)
    embedding = self.batch_norm(embedding)
    out = self.fc5(embedding)
    return out, embedding

