#!/usr/bin/env python3

'''
Cache the TPC branches of a root file as a pt file.

Optional: the training pipeline reads root files directly, see tpc.py.
This is only worth it when the same events are read over and over.
'''

import argparse
import logging
import logging.config
import os

import yaml

import torch

import tpc

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#______________________________________________________________________________
def run(input_path, output_file=None, max_events=tpc.ALL_EVENTS):
  ''' run process '''
  tree = tpc.read_tree(input_path, max_events)
  if tree is None:
    return
  if output_file is None:
    run_number = int(tree['runnum'][0])
    output_file = os.path.join(
      os.path.dirname(os.path.abspath(input_path)),
      f'run{run_number:05d}.pt')
  logger.info(f'write {output_file}')
  torch.save(tree, output_file)
  logger.info('done')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path',
                      help='input root file')
  parser.add_argument('--output', default=None,
                      help='output pt file (default: next to input)')
  parser.add_argument('--max-events', type=int, default=tpc.ALL_EVENTS,
                      help='number of events to read (-1 for all)')
  parsed, unparsed = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(input_path=parsed.input_path, output_file=parsed.output,
      max_events=parsed.max_events)
