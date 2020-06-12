from __future__ import print_function, division
import argparse
import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from model import LCNN
from train import train
from eval import eval
from angular_softmax_loss import AngularPenaltySMLoss
from triplet_loss import TripletLoss
from ge2e import GE2ELoss
from utils.checkpoint import load_checkpoint, create_directory
from get_data_backend import get_data_backend
from get_data_softmax import get_data_softmax
from gkde_loss import KernelDensityLoss

# Training settings
parser = argparse.ArgumentParser(description='LCNN ASVspoof 2019')
parser.add_argument('--batch-size', type=int, default=280, metavar='N',
                    help='input batch size for training (default: 14)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 14)')
parser.add_argument('--epochs', type=int, default=7, metavar='N',
                    help='number of epochs for early stopping (default: 5)')
parser.add_argument('--num-data-workers', type=int, default=7,
                    help='How many processes to load data')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--version', type=str, default='v1',
                    help='Version to save the model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=4, nargs="*",
                    help='how many eval processes to use (default: 4)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--num-filts', type=int, default=256,
                    help='How many filters to compute STFT')
parser.add_argument('--num-frames', type=int, default=400,
                    help='How many frames to compute STFT')
parser.add_argument('--window-length', type=float, default=0.025,
                    help='Window Length to compute STFT (s)')
parser.add_argument('--frame-shift', type=float, default=0.010,
                    help='Frame Shift to compute STFT (s)')
parser.add_argument('--margin-triplet', type=float, default=1.0,
                    help='Prob. margin for triplet loss')
parser.add_argument('--emb-size', type=int, default=64, metavar='N',
                    help='embedding size')
parser.add_argument('--load-epoch', type=int, default=-1,
                    help='Saved epoch to load and start training')
parser.add_argument('--eval-epoch', type=int, default=-1,
                    help='Epoch to load and evaluate')
parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
parser.add_argument('--eval', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to extract the xvectors')
parser.add_argument('--backend', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to prepare embeddings for backend')
parser.add_argument('--is-la', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train Logical or Physical Access')
parser.add_argument('--num-classes', type=int, default=7, metavar='N',
                    help='Number of training classes (2, 7, 10)')
parser.add_argument('--loss-method', type=str, default='softmax',
                    help='softmax, angular_softmax_sphereface, angular_softmax_cosface, triplet, ge2e, kde-softmax, kde-contrast, kde-softmax_contrast, kde-triplet, kde-all')
parser.add_argument('--optimize-bandwidth', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to optimize the bandwidth of the KDE')
parser.add_argument('--scale-matrix', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to scale the KDE matrix')
parser.add_argument('--bandwidth', type=float, default=1.0,
                    help='Initialization of KDE bandwidth. v1 -> False; v2 -> True')
parser.add_argument('--normalize', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to normalize the utterance: mean and variance normalization')
parser.add_argument('--kde-log', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to use the log with KDE')

rootPath = os.getcwd()
                  
if __name__ == '__main__':
  args = parser.parse_args()
  print(args)

  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  mp.set_start_method('spawn')

  torch.manual_seed(args.seed)

  model = LCNN(args.emb_size, args.num_classes).to(device)

  if args.loss_method == 'softmax':
    criterion = nn.CrossEntropyLoss()
  elif args.loss_method == 'angular_softmax_sphereface':
    criterion = AngularPenaltySMLoss(in_features=args.emb_size, out_features=args.num_classes, device=device, loss_type='sphereface')
  elif args.loss_method == 'angular_softmax_cosface':
    criterion = AngularPenaltySMLoss(in_features=args.emb_size, out_features=args.num_classes, device=device, loss_type='cosface')
  elif args.loss_method == 'triplet':
    criterion = TripletLoss(margin=args.margin_triplet)
  elif args.loss_method == 'ge2e':
    criterion = GE2ELoss(device=device)
  elif args.loss_method.startswith('kde'):
    criterion = KernelDensityLoss(
      emb_size=args.emb_size,
      device=device,
      loss_method=args.loss_method.split('-')[1],
      init_bandwidth=args.bandwidth,
      optimize_bandwidth=args.optimize_bandwidth,
      margin_triplet=args.margin_triplet,
      num_classes=args.num_classes,
      scale_matrix=args.scale_matrix,
      kde_log=args.kde_log
    )

  params = list(model.parameters()) + list(criterion.parameters())
  optimizer = optim.Adam(params, lr=args.lr)

  # Model and embeddings path
  dirSpoof = 'LA' if args.is_la else 'PA'
  dirEmbeddings = 'lcnn_' + args.version + '_loss_' + args.loss_method + '_emb_' + str(args.emb_size) + '_classes_' + str(args.num_classes)
  model_location = os.path.join(rootPath, 'models', dirSpoof, dirEmbeddings)
  create_directory(model_location)

  if (args.load_epoch != -1):
    path_model_location = os.path.join(model_location, 'epoch-' + str(args.load_epoch) + '.pt')
    model, optimizer, criterion, start_epoch = load_checkpoint(model, optimizer, criterion, path_model_location)
  else:
    start_epoch = 0

  if args.train:
    train(
      args=args,
      model=model,
      start_epoch=start_epoch,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
      model_location=model_location
    )
  
  embeddings_location = os.path.join(rootPath, 'embeddings', dirSpoof, dirEmbeddings)
  softmax_location = os.path.join(rootPath, 'scores', dirSpoof, dirEmbeddings)
  # Create embeddings and scores directories
  create_directory(embeddings_location)
  create_directory(softmax_location)

  if args.eval:
    if args.eval_epoch != -1:
      path_model_location = os.path.join(model_location, 'epoch-' + str(args.eval_epoch) + '.pt')
    else:
      path_model_location = os.path.join(model_location, 'best.pt')

    model, optimizer, criterion, eval_epoch = load_checkpoint(model, optimizer, criterion, path_model_location)

    eval(
      args=args,
      model=model,
      embeddings_location=embeddings_location,
      softmax_location=softmax_location,
      device=device,
      mp=mp)

  if args.backend:
    print('Get data for backend')
    for kind in ['training', 'development', 'test']:
      print(kind)
      get_data_backend(
        path=embeddings_location,
        kind=kind,
        args=args
      )

  if args.loss_method == 'softmax':
    print('Get softmax scores')
    for kind in ['training', 'development', 'test']:
      print(kind)
      get_data_softmax(
        path=softmax_location,
        kind=kind,
        args=args
      )
    

  
  