#!/usr/bin/env python3

'''
Pads that must not be used as the common noise reference.

Mirrors tpc::IsDead() and tpc::Noise() of the k18analyzer
TPCPadHelper.hh. Run extract_pads.py to refresh excluded_pads.json.
'''

import json
import logging
import os

top_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

PADS_FILE = os.path.join(top_dir, 'excluded_pads.json')
# IsDead() covers the two former, Noise() the latter.
DEAD_KEYS = ('padOnCenterFrame', 'deadChannel')
NOISE_KEYS = ('padOnSectionFrame',)

#______________________________________________________________________________
def load(path=PADS_FILE):
  ''' pad lists as written by extract_pads.py, {} when unavailable '''
  try:
    with open(path, 'r') as f:
      return json.load(f)
  except (FileNotFoundError, OSError, ValueError) as e:
    logger.warning(f'{e}: no pad will be excluded')
    return {}

#______________________________________________________________________________
def excluded(path=PADS_FILE):
  ''' union of the dead and the noisy pads '''
  data = load(path)
  pads = set()
  for key in DEAD_KEYS + NOISE_KEYS:
    pads.update(data.get(key, []))
  return pads
