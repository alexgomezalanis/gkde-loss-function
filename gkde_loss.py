import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import shuffle
from torch.distributions.multivariate_normal import MultivariateNormal

class KernelDensityLoss(nn.Module):
  """
  Kernel Density Gaussian loss
  Takes a batch of embeddings and corresponding labels.
  """

  def __init__(
    self,
    device,
    emb_size=64,
    scale_matrix=True,
    init_w=10.0,
    init_b=0.1,
    init_bandwidth=1.0,
    loss_method='softmax',
    margin_triplet=1.0,
    optimize_bandwidth=True,
    num_classes=7,
    kde_log=False
  ):
    super(KernelDensityLoss, self).__init__()
    if optimize_bandwidth:
      self.bandwidths = nn.Parameter(torch.tensor(num_classes * [init_bandwidth]).to(device))
    else:
      self.bandwidths = num_classes * [init_bandwidth]

    self.scale_matrix = scale_matrix
    self.kde_log = kde_log
    if scale_matrix:
      self.w = nn.Parameter(torch.tensor(init_w).to(device))
      self.b = nn.Parameter(torch.tensor(init_b).to(device))

    self.emb_size = emb_size
    self.distributions = []
    self.probs = []
    self.digit_indices = []
    self.device = device
    self.num_classes = num_classes
    self.loss_method = loss_method
    self.margin_triplet = torch.FloatTensor([margin_triplet])
    self.margin_triplet = self.margin_triplet.to(device)

    assert self.loss_method in ['softmax', 'contrast', 'triplet', 'softmax_contrast', 'all']

    if self.loss_method == 'softmax':
      self.embed_loss = self.softmax_loss
    elif self.loss_method == 'contrast':
      if kde_log:
        self.embed_loss = self.contrast_loss_log
      else:
        self.embed_loss = self.contrast_loss
    elif self.loss_method == 'triplet':
      self.embed_loss = self.triplet_loss

  def get_likelihood(self, index_utt, index_class):
    probs = []
    for j in self.digit_indices[index_class]:
      if index_utt != j:
        index_i = min(index_utt, j)
        index_j = max(index_utt, j)
        probs.append(torch.exp(-self.distances[index_i][index_j] / (2 * self.bandwidths[index_class])))
    likelihood = torch.mean(torch.stack(probs))
    if self.kde_log:
      return torch.log(likelihood)
    return likelihood

  def triplet_loss(self, embeddings):    
    negative_probs = []
    positive_probs = []
    for index_pos, arr_pos in enumerate(self.digit_indices):
      len_arr_pos = len(arr_pos)
      for pos_idx in range(len_arr_pos):
        pos_prob = self.probs[index_pos,pos_idx,index_pos]
        positive_probs.append(pos_prob)
        # Choose randomly a negative sample which lies inside the margin
        neg_classes = [n for n in range(self.num_classes) if n != index_pos]
        shuffle(neg_classes)
        max_prob = float('-inf')
        is_semihard = False
        for index_neg in neg_classes:
          arr_neg = list(range(len(self.digit_indices[index_neg])))
          shuffle(arr_neg)
          for neg_idx in arr_neg:
            neg_prob = self.probs[index_neg,neg_idx,index_pos]
            if (pos_prob > neg_prob and neg_prob + self.margin_triplet > pos_prob):
              is_semihard = True
              break
            elif neg_prob > max_prob:
              max_prob = neg_prob
        if is_semihard:
          negative_probs.append(neg_prob)
        else:
          negative_probs.append(max_prob)

    positive_probs = torch.stack(positive_probs)
    negative_probs = torch.stack(negative_probs)
    L = F.relu(negative_probs - positive_probs + self.margin_triplet)
    loss = L.sum()

    return loss

  def softmax_loss(self, embeddings):
    # N spoofing classes, M utterances per class
    N, M, _ = list(self.probs.size())

    L = []
    for j in range(N):
      L_row = []
      for i in range(M):
        L_row.append(-F.log_softmax(self.probs[j,i], 0)[j])
      L_row = torch.stack(L_row)
      L.append(L_row)
    L_torch = torch.stack(L)

    return L_torch.sum()

  def contrast_loss(self, embeddings):
    # N spoofing classes, M utterances per class
    N, M, _ = list(self.probs.size())

    L = []
    for j in range(N):
      L_row = []
      for i in range(M):
        probs_to_classes = torch.sigmoid(self.probs[j,i])
        excl_probs_to_classes = torch.cat((probs_to_classes[:j], probs_to_classes[j+1:]))
        L_row.append(1.0 - torch.sigmoid(self.probs[j,i,j]) + torch.max(excl_probs_to_classes))
      L_row = torch.stack(L_row)
      L.append(L_row)
    L_torch = torch.stack(L)

    return L_torch.sum()

  def contrast_loss_log(self, embeddings):
    # N spoofing classes, M utterances per class
    N, M, _ = list(self.probs.size())

    L = []
    for j in range(N):
      L_row = []
      for i in range(M):
        probs_to_classes = self.probs[j,i]
        excl_probs_to_classes = torch.cat((probs_to_classes[:j], probs_to_classes[j+1:]))
        L_row.append(torch.max(excl_probs_to_classes) - self.probs[j,i,j])
      L_row = torch.stack(L_row)
      L.append(L_row)
    L_torch = torch.stack(L)

    return F.relu(L_torch).sum()

  def forward(self, embeddings, target, size_average=True):
    classes = np.unique(target)
    self.num_classes = len(classes)
    self.digit_indices = [np.where(target == i)[0] for i in range(self.num_classes)]

    self.distances = [[0] * len(target) for _ in range(len(target))]
    for i in range(len(target)):
      for j in range(i+1, len(target)):
        self.distances[i][j] = torch.dist(embeddings[i], embeddings[j], 2)

    probs = []
    for class_idx, class_indices in enumerate(self.digit_indices):
      probs_row = []
      for utt_idx, utterance in enumerate(class_indices):
        probs_col = []
        for class_centroid in range(self.num_classes):
          probs_col.append(self.get_likelihood(utterance, class_centroid))
        probs_col = torch.stack(probs_col)
        probs_row.append(probs_col)
      probs_row = torch.stack(probs_row)
      probs.append(probs_row)
    self.probs = torch.stack(probs)

    if self.scale_matrix:
      torch.clamp(self.w, 1e-6)
      self.probs = self.w * self.probs + self.b

    if self.loss_method == 'all':
      loss = self.softmax_loss(embeddings) + self.contrast_loss(embeddings) + self.triplet_loss(embeddings)
    elif self.loss_method == 'softmax_contrast':
      if self.kde_log:
        loss = self.softmax_loss(embeddings) + self.contrast_loss_log(embeddings)
      else:
        loss = self.softmax_loss(embeddings) + self.contrast_loss(embeddings)
    else:
      loss = self.embed_loss(embeddings)

    return loss