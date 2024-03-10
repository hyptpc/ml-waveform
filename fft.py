#!/usr/bin/env python3

'''
Load wav/root file and do fft.
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
def fft(waveform, sample_rate, show=True, outfig='fft.png'):
  ''' fft '''
  fft_result = torch.fft.fft(waveform)
  freq_axis = torch.fft.fftfreq(waveform.size(0), d=1./sample_rate)
  logger.debug(f'fft result: {fft_result}')
  logger.debug(f'fft axis: {freq_axis}')
  positive_freq_mask = (freq_axis >= 0) # mask negative freq
  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  plt.plot(waveform.t().numpy())
  plt.title(f'waveform')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  plt.subplot(2, 1, 2)
  plt.plot(freq_axis[positive_freq_mask].numpy(),
           torch.abs(fft_result[positive_freq_mask]).numpy())
  plt.title('fft')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Amplitude')
  plt.tight_layout()
  if show:
    plt.savefig(outfig)
    plt.show()
  else:
    outfig.savefig()
  plt.close()

#______________________________________________________________________________
def run(file_path):
  ''' run '''
  _, input_ext = os.path.splitext(file_path)
  if input_ext == '.wav':
    run_wav(file_path)
  elif input_ext == '.root':
    run_root(file_path)
  else:
    logger.error(f'unknown file type: {file_path}')

#______________________________________________________________________________
def run_root(file_path, sample_rate=12500000):
  ''' run using root file '''
  try:
    logger.info(f'open {file_path}')
    tree = uproot.concatenate(file_path+':tpc',
                              filter_name=['runnum', 'evnum',
                                           'rpadTpc', 'rwavTpc'],
                              library='ak')
  except uproot.KeyInFileError as e:
    logger.error(e)
    return
  run_number = tree['runnum'][0]
  evnum = tree['evnum']
  rpadTpc = tree['rpadTpc']
  rwavTpc = tree['rwavTpc']
  pdf = PdfPages('fft.pdf')
  for i in range(len(tree)):
    logger.debug(f'run{run_number:05d} ev{evnum[i]:06d} {len(rpadTpc[i])}')
    for j in range(len(rpadTpc[i])):
      logger.info(f'{rpadTpc[i][j]}, {rwavTpc[i][j]}')
      waveform = torch.Tensor(rwavTpc[i][j]) / (2**12)
      waveform = (waveform - waveform.mean())
      fft(waveform, sample_rate, show=False, outfig=pdf)
    break
  pdf.close()

#______________________________________________________________________________
def run_wav(file_path):
  ''' run using wav file '''
  waveform, sample_rate = torchaudio.load(file_path)
  logger.info(f'Shape of waveform [channel, time]: {waveform.size()}')
  logger.info(f'Sample rate of waveform: {sample_rate}')
  waveform = waveform[0] # first channel
  waveform = (waveform - waveform.mean())
  fft(waveform, sample_rate)

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help=('file path of input file (.wav or .root)'))
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path=parsed.file_path)
