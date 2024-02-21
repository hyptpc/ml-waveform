#!/usr/bin/env python3

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
def run(filename, spec_transform):
  waveform, sample_rate = torchaudio.load(filename)
  waveform = waveform[0] # show only first channel
  logger.info(f'Shape of waveform [channel, time]: {waveform.size()}')
  logger.info(f'Sample rate of waveform: {sample_rate}')
  data = waveform.t().numpy()
  spec = spec_transform(waveform).log2()
  logger.info(f'Shape of spectrogram: {spec.size()}')
  logger.debug(waveform.t().numpy())
  logger.debug(spec)

  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  plt.plot(waveform.t().numpy())
  plt.title('waveform')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  plt.subplot(2, 1, 2)
  plt.imshow(spec.numpy(), cmap='viridis', aspect='auto', origin='lower')
  plt.title('Spectrogram')
  plt.xlabel('Time')
  plt.ylabel('Frequency Bin')
  plt.colorbar(format='%+2.0f dB')
  plt.tight_layout()
  plt.show()

#______________________________________________________________________________
if __name__ == '__main__':
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(filename = 'under_transition.wav',
      spec_transform=transforms.Spectrogram())
