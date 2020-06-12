import os
import torch

def load_checkpoint(model, optimizer, criterion, filename):
  # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
  start_epoch = 0
  if os.path.isfile(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion.load_state_dict(checkpoint['criterion'])
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
  else:
    print("=> no checkpoint found at '{}'".format(filename))
  return model, optimizer, criterion, start_epoch

def create_directory(location):
  try:
    os.mkdir(location)
  except OSError:  
    print ("Creation of the directory %s failed" % location)
  else:  
    print ("Successfully created the directory %s " % location)

