import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randint

class TripletLoss(nn.Module):
  """
  Online Triplets loss
  Takes a batch of embeddings and corresponding labels.
  Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
  triplets
  """

  def __init__(self, margin):
    super(TripletLoss, self).__init__()
    self.margin = margin
    self.criterion = nn.TripletMarginLoss(margin=margin, p=2)

  def triplet_selector(self, embeddings, target):
    classes = np.unique(target)
    num_classes = len(classes)
    digit_indices = [np.where(target == i)[0] for i in classes]
    anchor_indices = []
    negative_indices = []
    positive_indices = []
    for index_pos, arr_pos in enumerate(digit_indices):
      len_arr_pos = len(arr_pos)
      for i in range(len_arr_pos):
        anchor = embeddings[arr_pos[i]]
        anchor_idx = arr_pos[i]
        # Choose randomly the positive sample
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
          positive_idx = arr_pos[randint(0, len_arr_pos - 1)]
        pos_dist = torch.dist(anchor, embeddings[positive_idx], 2)
        # Choose randomly a negative sample which lies inside the margin
        count = 0
        min_dist = float('Inf')
        while count < 150:
          index_neg = randint(0, num_classes - 1)
          if index_neg != index_pos:
            count += 1
            arr_neg = digit_indices[index_neg]
            n = randint(0, len(arr_neg) - 1)
            neg_idx = arr_neg[n]
            neg_dist = torch.dist(anchor, embeddings[neg_idx], 2)
            if pos_dist < neg_dist and neg_dist - self.margin < pos_dist:
              negative_idx = neg_idx
              break
            elif neg_dist < min_dist:
              negative_idx = neg_idx
              min_dist = neg_dist
        anchor_indices.append(anchor_idx)
        negative_indices.append(negative_idx)
        positive_indices.append(positive_idx)
    return anchor_indices, positive_indices, negative_indices      

  def forward(self, embeddings, target, size_average=True):
    anchor_indices, positive_indices, negative_indices = self.triplet_selector(embeddings, target)

    loss = self.criterion(embeddings[anchor_indices], embeddings[positive_indices], embeddings[negative_indices])

    return loss