#!/usr/bin/env python3

import argparse
import logging
import logging.config
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

import torch
import torchaudio
import torchaudio.transforms as transforms

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#______________________________________________________________________________
def run(file_path, cutoff_frequency):
  waveform, sample_rate = torchaudio.load(file_path)
  logger.info(f'Shape of waveform [channel, time]: {waveform.size()}')
  logger.info(f'Sample rate of waveform: {sample_rate}')
  logger.info(f'Cutoff frequency for lowpass filter: {cutoff_frequency}')
  waveform = waveform[0] # show only first channel
  waveform = (waveform - waveform.mean()) #/waveform.std() # normalize
  filtered_waveform = torchaudio.functional.lowpass_biquad(
    waveform=waveform, sample_rate=sample_rate, cutoff_freq=cutoff_frequency)
  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  plt.plot(waveform.t().numpy())
  plt.title('waveform')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  plt.subplot(2, 1, 2)
  plt.plot(filtered_waveform.numpy())
  plt.title(f'filtered waveform (lowpass cutoff freq.={cutoff_frequency})')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  plt.tight_layout()
  plt.savefig('lowpass_filter.png')
  plt.show()

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help=('file path of input wav'))
  parser.add_argument('cutoff_frequency', type=float, nargs='?',
                      default=800,
                      help=('file path of input wav'))
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path = parsed.file_path,
      cutoff_frequency = parsed.cutoff_frequency)
