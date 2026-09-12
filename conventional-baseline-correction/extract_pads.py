#!/usr/bin/env python3

'''
Extract the excluded pad lists from the k18analyzer header.

TPCPadHelper.hh keeps the pad ids as plain C++ arrays, so they are
parsed out once and cached as json. Run this again when the analyzer
is updated.
'''

import argparse
import json
import os
import re

top_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_HEADER = os.path.expanduser(
  '~/work/k18analyzer/e72/include/TPCPadHelper.hh')
DEFAULT_OUTPUT = os.path.join(top_dir, 'excluded_pads.json')
# padOnCenterFrame and deadChannel drive IsDead(), padOnSectionFrame
# drives Noise(); both disqualify a pad as the noise reference.
ARRAY_NAMES = ('padOnCenterFrame', 'deadChannel', 'padOnSectionFrame')

#______________________________________________________________________________
def parse_array(source, name):
  ''' pad ids of one `static const Int_t <name>[] = {...};` array '''
  pattern = rf'static\s+const\s+Int_t\s+{name}\s*\[\s*\]\s*=\s*\{{'
  match = re.search(pattern, source)
  if match is None:
    raise ValueError(f'{name} not found')
  end = source.index('};', match.end())
  body = source[match.end():end]
  body = re.sub(r'//[^\n]*', '', body) # drop the section comments
  return sorted({int(t) for t in re.findall(r'-?\d+', body)})

#______________________________________________________________________________
def run(header=DEFAULT_HEADER, output=DEFAULT_OUTPUT):
  ''' write the pad lists as json '''
  with open(header, 'r') as f:
    source = f.read()
  pads = {name: parse_array(source, name) for name in ARRAY_NAMES}
  pads['source'] = header
  with open(output, 'w') as f:
    json.dump(pads, f, indent=2)
  for name in ARRAY_NAMES:
    print(f'{name}: {len(pads[name])} pads')
  print(f'wrote {output}')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--header', default=DEFAULT_HEADER,
                      help='path of TPCPadHelper.hh')
  parser.add_argument('--output', default=DEFAULT_OUTPUT,
                      help='output json file')
  parsed, unparsed = parser.parse_known_args()
  run(header=parsed.header, output=parsed.output)
