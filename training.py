#!/usr/bin/env python3

'''
Feed TPC waveforms from a root file into a training loop.

The waveforms are read directly with uproot, so no wav or pt
conversion step is involved. The model and the labels for the TPC data
are not defined yet, so this script stops after reporting what the
dataset yields.
'''

import argparse
import logging
import logging.config
import os

import yaml

import torch
import torchaudio
from torch.utils.data import DataLoader

import random_seed
import tpc

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

# Linear spectrogram, not mel: the mel scale is tuned to human hearing
# and means nothing at the 12.5 MHz sampling rate of the TPC.
N_FFT = 64
HOP_LENGTH = 16
BATCH_SIZE = 32
SEED = 0

#______________________________________________________________________________
def build_loader(file_path, max_events=tpc.ALL_EVENTS, spectrogram=False,
                 batch_size=BATCH_SIZE):
  ''' data loader over every pad hit of the root file '''
  transform = None
  if spectrogram:
    transform = torchaudio.transforms.Spectrogram(n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH)
  dataset = tpc.TpcWaveformDataset(file_path, max_events=max_events,
                                   transform=transform)
  if len(dataset) == 0:
    logger.error(f'no waveform found in {file_path}')
    return None
  return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    worker_init_fn=random_seed.seed_worker,
                    generator=random_seed.make_generator(SEED))

#______________________________________________________________________________
def run(file_path, max_events=tpc.ALL_EVENTS, spectrogram=False,
        batch_size=BATCH_SIZE):
  ''' report what one pass over the dataset delivers '''
  random_seed.set_seed(SEED)
  loader = build_loader(file_path, max_events, spectrogram, batch_size)
  if loader is None:
    return
  n_batches = 0
  n_samples = 0
  minimum = None
  maximum = None
  for data, meta in loader:
    if n_batches == 0:
      logger.info(f'batch shape: {tuple(data.size())}')
      logger.info(f'meta keys: {sorted(meta)}')
    n_batches += 1
    n_samples += data.size(0)
    batch_min = float(data.min())
    batch_max = float(data.max())
    minimum = batch_min if minimum is None else min(minimum, batch_min)
    maximum = batch_max if maximum is None else max(maximum, batch_max)
  logger.info(f'{n_samples} waveforms in {n_batches} batches')
  logger.info(f'value range: [{minimum:.6f}, {maximum:.6f}]')
  logger.warning('no model defined yet: stopping before training')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help='file path of input root file')
  parser.add_argument('--max-events', type=int, default=tpc.ALL_EVENTS,
                      help='number of events to read (-1 for all)')
  parser.add_argument('--spectrogram', action='store_true',
                      help='feed spectrograms instead of raw waveforms')
  parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                      help='number of waveforms per batch')
  parsed, unparsed = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path=parsed.file_path, max_events=parsed.max_events,
      spectrogram=parsed.spectrogram, batch_size=parsed.batch_size)
