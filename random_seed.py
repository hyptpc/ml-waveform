#!/usr/bin/env python3

'''
Helpers to make training runs reproducible.
'''

import random

import numpy as np
import torch

WORKER_SEED_MODULUS = 2**32

#______________________________________________________________________________
def set_seed(seed=0):
  ''' seed every random number generator used by this project '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

#______________________________________________________________________________
def seed_worker(worker_id):
  ''' seed worker for DataLoader '''
  worker_seed = torch.initial_seed() % WORKER_SEED_MODULUS
  np.random.seed(worker_seed)
  random.seed(worker_seed)

#______________________________________________________________________________
def make_generator(seed=0):
  ''' generator to pass to DataLoader for reproducible shuffling '''
  generator = torch.Generator()
  generator.manual_seed(seed)
  return generator
