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

import audio_io
import tpc

top_dir = os.path.dirname(os.path.abspath(__file__))
# Reading every event of a run produces a very large pdf, so stop at
# the first one unless the caller asks for more.
DEFAULT_MAX_EVENTS = 1
logger = logging.getLogger(__name__)

#______________________________________________________________________________
@stopwatch
@profile
def fft(waveform, sample_rate, show=True, outfig='fft.png'):
  ''' fft '''
  fft_result = torch.fft.fft(waveform)
  freq_axis = torch.fft.fftfreq(waveform.size(-1), d=1./sample_rate)
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
  if isinstance(outfig, PdfPages):
    outfig.savefig()
  else:
    plt.savefig(outfig)
  if show:
    plt.show()
  plt.close()

#______________________________________________________________________________
def run(file_path, max_events=DEFAULT_MAX_EVENTS):
  ''' run '''
  _, input_ext = os.path.splitext(file_path)
  if input_ext == '.wav':
    run_wav(file_path)
  elif input_ext == '.root':
    run_root(file_path, max_events=max_events)
  else:
    logger.error(f'unknown file type: {file_path}')

#______________________________________________________________________________
def run_root(file_path, sample_rate=tpc.SAMPLE_RATE,
             max_events=DEFAULT_MAX_EVENTS):
  ''' run using root file '''
  dataset = tpc.TpcWaveformDataset(file_path, max_events=max_events)
  if len(dataset) == 0:
    logger.error(f'no waveform found in {file_path}')
    return
  with PdfPages('fft.pdf') as pdf:
    for waveform, meta in dataset:
      logger.info(f"run{meta['runnum']:05d} ev{meta['evnum']:06d} "
                  f"pad{meta['pad']:04d}")
      fft(waveform[0], sample_rate, show=False, outfig=pdf)

#______________________________________________________________________________
def run_wav(file_path):
  ''' run using wav file '''
  waveform, sample_rate = audio_io.load(file_path)
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
  parser.add_argument('--max-events', type=int,
                      default=DEFAULT_MAX_EVENTS,
                      help='number of events to read (-1 for all)')
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path=parsed.file_path, max_events=parsed.max_events)
