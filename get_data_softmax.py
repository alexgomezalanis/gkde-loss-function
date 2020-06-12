import numpy as np
import sys
import glob
import os
import torch
import torch.nn.functional as F

root = os.getcwd()

DICT_CLASSES = {
  'LA': { 'training': [i for i in range(7)], 'development': [i for i in range(7)], 'test': [0] + [i for i in range(7, 20)] },
  'PA': { 'training': [i for i in range(10)], 'development': [i for i in range(10)], 'test': [i for i in range(10)] }
}

def get_data_softmax(kind: str, path: str, args: dict) -> None:
  #dirIvectors = '/identity-vectors/' + path + '/'
  db = 'LA' if args.is_la else 'PA'
  classes = DICT_CLASSES[db][kind]

  for num_class in classes:
    print("Class: " + str(num_class))
    devScores = []
    devLabels = []
    devLabelsBinary = []
    fileNames = glob.glob(os.path.join(path, db, kind, 'S' + str(num_class) + '/*.npy'))

    for utterance in range(len(fileNames)):
      ut = np.load(fileNames[utterance])
      score = F.softmax(torch.from_numpy(np.reshape(ut, (1, -1))), dim=1)[0, 0].numpy()
      devScores.append(score)
      devLabels.append(num_class)
      if (num_class == 0):
        devLabelsBinary.append(0)
      else:
        devLabelsBinary.append(1)
    X_dev = np.asarray(devScores)
    Y_dev = np.asarray(devLabels, dtype=np.int32)
    Y_binary_dev = np.asarray(devLabelsBinary, dtype=np.int32)
    np.save(os.path.join(path, db, kind, 'X_' + str(num_class) + '.npy'), X_dev)
    np.save(os.path.join(path, db, kind, 'Y_' + str(num_class) + '.npy'), Y_dev)
    np.save(os.path.join(path, db, kind, 'Y_binary_' + str(num_class) + '.npy'), Y_binary_dev)
