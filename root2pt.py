#!/usr/bin/env python3

'''
read root file and convert to wav using uproot.
'''

import argparse
import logging
import logging.config
import os
import yaml

import uproot

import torch

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

sample_rate = 12500000 # 12.5 MHz
bits_per_sample = 16 # signed 16 bit

#______________________________________________________________________________
def run(input_path):
  ''' run process '''
  logger.info('start run')
  output_dir = os.path.dirname(input_path)
  try:
    logger.info(f'open {input_path}')
    tree = uproot.concatenate(input_path+':tpc',
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
  output_file = os.path.join(
    output_dir, f'run{run_number:05.0f}.pt')
  logger.info(f'write {output_file}')
  torch.save(tree, output_file)
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
