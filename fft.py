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

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#______________________________________________________________________________
def run(file_path):
  waveform, sample_rate = torchaudio.load(file_path)
  logger.info(f'Shape of waveform [channel, time]: {waveform.size()}')
  logger.info(f'Sample rate of waveform: {sample_rate}')
  waveform = waveform[0] # first channel
  waveform = (waveform - waveform.mean())/waveform.std() # normalize
  fft_result = torch.fft.fft(waveform)
  freq_axis = torch.fft.fftfreq(waveform.size(0), d=1./sample_rate)
  logger.debug(f'fft result: {fft_result}')
  logger.debug(f'fft axis: {freq_axis}')
  positive_freq_mask = (freq_axis >= 0) # mask negative freq
  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  plt.plot(waveform.t().numpy())
  plt.title(f'waveform : {os.path.basename(file_path)}')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  plt.subplot(2, 1, 2)
  plt.plot(freq_axis[positive_freq_mask].numpy(),
           torch.abs(fft_result[positive_freq_mask]).numpy())
  plt.title('fft')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Amplitude')
  plt.tight_layout()
  plt.savefig('fft.png')
  plt.show()

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help=('file path of input wav'))
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path = parsed.file_path)
