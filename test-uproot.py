#!/usr/bin/env python3

'''
Test uproot I/O performance of root file.
'''

import argparse
from lauda import stopwatch
import logging
import logging.config
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from memory_profiler import profile
import os
import yaml

import torch
import torchaudio
import uproot

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#______________________________________________________________________________
@stopwatch
@profile
def process_event(event):
  run_number = event['runnum']
  evnum = event['evnum']
  rpadTpc = event['rpadTpc']
  rwavTpc = event['rwavTpc']
  for i in range(len(rpadTpc)):
    waveform = torch.Tensor(rwavTpc[i]) / (2**12)
    waveform = (waveform - waveform.mean())
    logger.debug(waveform)

#______________________________________________________________________________
def run(file_path, sample_rate=12500000):
  ''' run using root file '''
  try:
    logger.info(f'open {file_path}')
    for chunk in uproot.iterate(file_path+':tpc'):
      logger.info(f'{chunk}')
      for event in chunk:
        process_event(event)
  except (uproot.KeyInFileError, IsADirectoryError) as e:
    logger.error(e)
    return

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help=('file path of input file (.root)'))
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path=parsed.file_path)
