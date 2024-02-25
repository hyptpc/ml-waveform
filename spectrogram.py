#!/usr/bin/env python3

'''
Load wav file and display spectrogram
'''

import argparse
import logging
import logging.config
import matplotlib.pyplot as plt
import os
import yaml

import torch
import torchaudio
import torchaudio.transforms as transforms

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#______________________________________________________________________________
def run(file_path, spec_transform):
  ''' run '''
  waveform, sample_rate = torchaudio.load(file_path)
  logger.info(f'Shape of waveform [channel, time]: {waveform.size()}')
  logger.info(f'Sample rate of waveform: {sample_rate}')
  waveform = waveform[0] # show only first channel
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
  plt.imshow(spec.numpy(), aspect='auto', origin='lower')
  plt.title('Spectrogram')
  plt.xlabel('Time')
  plt.ylabel('Frequency')
  plt.colorbar(format='%+2.0f dB')
  plt.tight_layout()
  plt.savefig('spectrogram.png')
  plt.show()

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help='file path of input wav')
  parser.add_argument('n_fft', nargs='?', type=int, default=30,
                      help='size of fft')
  parser.add_argument('hop_length', nargs='?', type=int, default=10,
                      help='number of sample between frames')
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path = parsed.file_path,
      spec_transform=transforms.Spectrogram(n_fft=parsed.n_fft,
                                            hop_length=parsed.hop_length))
