from __future__ import print_function, division
import os
import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from dataset import LCNN_Dataset
from utils.checkpoint import create_directory

rootPath = os.getcwd()

DICT_NUM_CLASSES = {
  'LA': { 'training': 7, 'development': 7, 'test': 20 },
  'PA': { 'training': 10, 'development': 10, 'test': 10 }
}

DICT_PROTOCOLS = {
  'LA': { 'training': 'train_la.csv', 'development': 'dev_la.csv', 'test': 'eval_la.csv' },
  'PA': { 'training': 'train_pa.csv', 'development': 'dev_pa.csv', 'test': 'eval_pa.csv'}
}

def test_epoch(model, device, data_loader, db_set, dirEmbeddings, dirSoftmax, db, loss_optimized):
  with torch.no_grad():
    for batch_idx, sample in enumerate(data_loader):
      (stft, labels, nameFiles) = sample
      stft = stft.to(device)
      softmax, embeddings = model(stft)
      embeddings = embeddings.cpu().numpy()
      softmax = softmax.cpu().numpy()
      labels = labels.numpy()
      for n, nameFile in enumerate(nameFiles):
        np.save(os.path.join(dirEmbeddings, db, db_set, 'S' + str(labels[n]), nameFile + '.npy'), embeddings[n])
        if loss_optimized == 'softmax':
          np.save(os.path.join(dirSoftmax, db, db_set, 'S' + str(labels[n]), nameFile + '.npy'), softmax[n])

# db: LA or PA -> Embeddings being evaluated
# db_set: training, development or test -> Dataset to evaluate
def eval(embeddings_location, softmax_location, args, model, device, mp):
  model.eval()

  db = 'LA' if args.is_la else 'PA'
  db_location = os.path.join(embeddings_location, db)
  create_directory(db_location)
  create_directory(os.path.join(softmax_location, db))

  for db_set in ['training', 'development', 'test']:
    set_location = os.path.join(db_location, db_set)
    create_directory(set_location)
    create_directory(os.path.join(softmax_location, db, db_set))
    num_classes = DICT_NUM_CLASSES[db][db_set]
    for n in range(num_classes):
      class_location = os.path.join(set_location, 'S' + str(n))
      create_directory(class_location)
      create_directory(os.path.join(softmax_location, db, db_set, 'S' + str(n)))

  for db_set in ['training', 'development', 'test']:
    print('Eval embeddings ' + db + ' ' + db_set)
    processes = []
    protocol = DICT_PROTOCOLS[db][db_set]
    df = pd.read_csv('./protocols/' + protocol, sep=' ')
    numRows = len(df.index)
    rows_p = numRows // args.num_processes

    for p in range(args.num_processes):
      if (p == args.num_processes - 1):
        df_p = df.iloc[p*rows_p:, :].reset_index().copy()
      else:
        df_p = df.iloc[p * rows_p : (p+1) * rows_p, :].reset_index().copy()
      dataset = LCNN_Dataset(
        csv_file='',
        root_dir='/home2/alexgomezalanis',
        n_filts=args.num_filts,
        n_frames=args.num_frames,
        window=args.window_length,
        shift=args.frame_shift,
        dataset=db_set,
        is_evaluating_la=db == 'LA',    # Embeddings to evaluate
        dataframe=df_p,
        num_classes=args.num_classes,
        normalize=args.normalize)
    
      loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_data_workers)

      process = mp.Process(target=test_epoch, args=(model, device, loader, db_set, embeddings_location, softmax_location, db, args.loss_method))
      process.start()
      processes.append(process)
    
    for p in processes:
      p.join()