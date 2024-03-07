#!/usr/bin/env python3

'''
read root file and convert to wav.
'''

import argparse
import logging
import logging.config
import os
import yaml

import ROOT

import torch
import torchaudio

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

sample_rate = 12500000 # 12.5 MHz
bits_per_sample = 16 # signed 16 bit

#______________________________________________________________________________
def run(input_path):
  ''' run process '''
  logger.info('start run')
  ROOT.gROOT.Reset()
  ROOT.gROOT.SetBatch()
  output_dir = os.path.dirname(input_path)
  f = ROOT.TFile.Open(input_path)
  if f == None or not f.IsOpen():
    logger.warning(f'failed to open {input_path}')
    return
  logger.info(f'open {input_path}')
  run_number = int(os.path.basename(input_path)[3:8])
  tree = f.Get('tpc')
  for event in tree:
    logger.info(f'{event.rpadTpc.size()} {event.rwavTpc.size()}')
    for i in range(event.rpadTpc.size()):
      output_file = os.path.join(
        output_dir,
        f'run{run_number:05.0f}_ev{event.evnum:08d}_' +
        f'pad{event.rpadTpc[i]:04d}.wav')
      w = torch.Tensor(event.rwavTpc[i]) / (2**12)
      w = w.reshape(1, w.size(-1)) # [channel, time]
      logger.debug(f'waveform : {w.size()}\n{w}')
      logger.info(f'write {output_file}')
      torchaudio.save(output_file, src=w,
                      sample_rate=sample_rate,
                      bits_per_sample=bits_per_sample)
  logger.info('done')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path',
                      help='input root file')
  parsed, unpased = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  if os.path.isfile(parsed.input_path):
    run(parsed.input_path)
  else:
    logger.error('cannot find valid input path')
    exit(1)
