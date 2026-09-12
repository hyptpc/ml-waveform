#!/usr/bin/env python3

'''
Export TPC waveforms of a root file as wav files.

This is an escape hatch for external audio tools only; the training
pipeline reads root files directly, see tpc.py. Reading goes through
uproot, so CERN ROOT is not required.
'''

import argparse
import logging
import logging.config
import os

import yaml

import audio_io
import tpc

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

#______________________________________________________________________________
def run(input_path, output_dir=None, max_events=tpc.ALL_EVENTS):
  ''' write one wav file per pad hit '''
  tree = tpc.read_tree(input_path, max_events)
  if tree is None:
    return
  if output_dir is None:
    output_dir = os.path.dirname(os.path.abspath(input_path))
  os.makedirs(output_dir, exist_ok=True)
  n_written = 0
  for i in range(len(tree)):
    run_number = int(tree['runnum'][i])
    evnum = int(tree['evnum'][i])
    for j in range(len(tree['rpadTpc'][i])):
      pad = int(tree['rpadTpc'][i][j])
      output_file = os.path.join(
        output_dir,
        f'run{run_number:05d}_ev{evnum:08d}_pad{pad:04d}.wav')
      # Keep the raw scaling: the baseline is subtracted at training
      # time, and doing it here would be lost to the wav quantization.
      waveform = tpc.normalize(tree['rwavTpc'][i][j],
                               subtract_baseline=False)
      logger.debug(f'write {output_file}')
      audio_io.save(output_file, waveform.reshape(1, -1),
                    tpc.SAMPLE_RATE)
      n_written += 1
  logger.info(f'wrote {n_written} wav files to {output_dir}')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path',
                      help='input root file')
  parser.add_argument('--output-dir', default=None,
                      help='output directory (default: next to input)')
  parser.add_argument('--max-events', type=int, default=tpc.ALL_EVENTS,
                      help='number of events to read (-1 for all)')
  parsed, unparsed = parser.parse_known_args()
  log_conf = os.path.join(top_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(input_path=parsed.input_path, output_dir=parsed.output_dir,
      max_events=parsed.max_events)
