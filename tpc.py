#!/usr/bin/env python3

'''
Read TPC waveforms straight out of a root file with uproot.

CERN ROOT is not required, and no intermediate wav or pt file is
written: the waveforms become tensors in memory.
'''

import logging

import numpy as np
import torch
import torch.nn.functional as F
import uproot
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 12500000 # 12.5 MHz
ADC_FULL_SCALE = 2**12 # 12 bit ADC
# Every pad is read out with the same window, so a fixed length keeps
# the waveforms batchable.
WAVEFORM_LENGTH = 170
ALL_EVENTS = -1
TREE_NAME = 'tpc'
BRANCHES = ['runnum', 'evnum', 'rpadTpc', 'rwavTpc']

#______________________________________________________________________________
def read_tree(file_path, max_events=ALL_EVENTS):
  ''' read the TPC branches, or None when the file cannot be read '''
  try:
    logger.info(f'open {file_path}')
    tree = uproot.open(f'{file_path}:{TREE_NAME}')
    entry_stop = None if max_events == ALL_EVENTS else max_events
    return tree.arrays(BRANCHES, entry_stop=entry_stop, library='ak')
  except (uproot.KeyInFileError, FileNotFoundError, OSError) as e:
    logger.error(e)
    return None

#______________________________________________________________________________
def normalize(raw, subtract_baseline=True):
  ''' scale the ADC counts, optionally subtracting the baseline '''
  waveform = torch.from_numpy(np.asarray(raw, dtype=np.float32))
  waveform = waveform / ADC_FULL_SCALE
  if subtract_baseline:
    waveform = waveform - waveform.mean()
  return waveform

#______________________________________________________________________________
def fix_length(waveform, length=WAVEFORM_LENGTH):
  ''' pad or trim a waveform so that a batch can be stacked '''
  n_samples = waveform.size(-1)
  if n_samples == length:
    return waveform
  if n_samples > length:
    return waveform[..., :length]
  return F.pad(waveform, (0, length - n_samples))

#______________________________________________________________________________
class TpcWaveformDataset(Dataset):
  ''' one item per pad hit, read directly from a root file

  Items are (waveform, meta) where waveform is [channel, time] and
  meta carries the run, event and pad numbers so that a prediction can
  be traced back to the detector.
  '''

  def __init__(self, file_path, max_events=ALL_EVENTS,
               length=WAVEFORM_LENGTH, transform=None):
    self.tree = read_tree(file_path, max_events)
    self.length = length
    self.transform = transform
    self.index = []
    if self.tree is None:
      return
    for i in range(len(self.tree)):
      for j in range(len(self.tree['rpadTpc'][i])):
        self.index.append((i, j))
    logger.info(f'{len(self.tree)} events, {len(self.index)} waveforms')

  def __len__(self):
    return len(self.index)

  def __getitem__(self, item):
    i, j = self.index[item]
    waveform = fix_length(normalize(self.tree['rwavTpc'][i][j]),
                          self.length)
    if self.transform is not None:
      waveform = self.transform(waveform)
    meta = {'runnum': int(self.tree['runnum'][i]),
            'evnum': int(self.tree['evnum'][i]),
            'pad': int(self.tree['rpadTpc'][i][j])}
    return waveform.unsqueeze(0), meta
