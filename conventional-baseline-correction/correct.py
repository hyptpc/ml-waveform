#!/usr/bin/env python3

'''
Run the conventional common noise correction on one TPC event.

Reads the root file with uproot, picks the reference channel, corrects
every channel and writes a diagnostic pdf.
'''

import argparse
import logging
import logging.config
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

import baseline
import pads

top_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(top_dir)
sys.path.insert(0, parent_dir) # tpc.py lives in the repository root

import tpc

logger = logging.getLogger(__name__)

DEFAULT_EVENT = 0
DEFAULT_N_EXAMPLES = 4
DEFAULT_OUTFIG = 'baseline-correction.pdf'
FIG_SIZE = (12, 7)

#______________________________________________________________________________
def read_event(file_path, event=DEFAULT_EVENT):
  ''' raw ADC waveforms and pad ids of one event '''
  tree = tpc.read_tree(file_path, max_events=event + 1)
  if tree is None:
    return None
  if event >= len(tree):
    logger.error(f'event {event} not in {file_path}')
    return None
  waveforms = np.asarray(tree['rwavTpc'][event].to_list(), dtype=float)
  pad_ids = np.asarray(tree['rpadTpc'][event].to_list(), dtype=int)
  logger.info(f"run{int(tree['runnum'][event]):05d} "
              f"ev{int(tree['evnum'][event]):06d} "
              f'{waveforms.shape[0]} channels '
              f'{waveforms.shape[1]} time buckets')
  return waveforms, pad_ids

#______________________________________________________________________________
def report(result, waveforms, pad_ids, config):
  ''' log what the correction did '''
  reference = result.reference
  logger.info(f'reference: channel {reference} pad {pad_ids[reference]}')
  logger.info(f'  amplitude {amplitude_of(waveforms, reference, config):.1f}'
              f'  rms {rms_of(waveforms, reference, config):.1f}')
  before = tail_rms(waveforms, config)
  after = tail_rms(result.corrected, config)
  logger.info(f'tail rms (in the fit region, so in-sample): '
              f'{before.mean():.2f} -> {after.mean():.2f}')
  head_before = head_rms(waveforms, config)
  head_after = head_rms(result.corrected, config)
  logger.info(f'pre-signal rms (outside the fit region): '
              f'{head_before.mean():.2f} -> {head_after.mean():.2f} '
              f'({100 * (1 - head_after.mean() / head_before.mean()):.1f}'
              f'% lower)')
  for name, values in (('adc offset', result.params[:, 0]),
                       ('scale', result.params[:, 1]),
                       ('time offset', result.params[:, 2])):
    logger.info(f'{name}: mean {values.mean():.3f} '
                f'[{values.min():.3f}, {values.max():.3f}]')
  return before, after

#______________________________________________________________________________
def amplitude_of(waveforms, index, config):
  ''' signal window amplitude of one channel '''
  return baseline.amplitude(waveforms[index], config)

#______________________________________________________________________________
def rms_of(waveforms, index, config):
  ''' signal window rms of one channel '''
  return baseline.rms(waveforms[index], config.min_time_bucket,
                      config.max_time_bucket)

#______________________________________________________________________________
def tail_rms(waveforms, config):
  ''' rms of the fit region, where only common noise should remain '''
  tail = waveforms[:, config.max_time_bucket:config.n_time_bucket]
  return np.std(tail, axis=1)

#______________________________________________________________________________
def head_rms(waveforms, config):
  ''' rms before the signal window: never seen by the fit, so this is
  the out-of-sample check that the common noise really is removed '''
  head = waveforms[:, :config.min_time_bucket]
  return np.std(head, axis=1)

#______________________________________________________________________________
def plot_reference(pdf, result, waveforms, pad_ids, config):
  ''' the channel chosen as the common noise reference '''
  plt.figure(figsize=FIG_SIZE)
  plt.plot(waveforms[result.reference], c='k', label='raw')
  plt.plot(result.template, c='r',
           label='template (pedestal subtracted)')
  plt.axvspan(config.min_time_bucket, config.max_time_bucket,
              color='b', alpha=.08, label='signal window')
  plt.axvspan(config.max_time_bucket, config.n_time_bucket,
              color='g', alpha=.12, label='fit region')
  plt.title(f'common noise reference: pad {pad_ids[result.reference]}')
  plt.xlabel('Time bucket')
  plt.ylabel('ADC ch')
  plt.legend()
  plt.tight_layout()
  pdf.savefig()
  plt.close()

#______________________________________________________________________________
def plot_examples(pdf, result, waveforms, pad_ids, config, n_examples):
  ''' the channels with the largest signal after correction '''
  peak = result.corrected[:, config.min_time_bucket:].max(axis=1)
  order = np.argsort(peak)[::-1][:n_examples]
  for i in order:
    plt.figure(figsize=FIG_SIZE)
    plt.subplot(2, 1, 1)
    plt.plot(waveforms[i], c='k', label='raw')
    plt.plot(result.fitted[i], c='r', label='fitted baseline')
    plt.axvspan(config.max_time_bucket, config.n_time_bucket,
                color='g', alpha=.12, label='fit region')
    offset, scale, shift = result.params[i]
    plt.title(f'pad {pad_ids[i]}: offset={offset:.1f} '
              f'scale={scale:.3f} shift={shift:.2f}')
    plt.ylabel('ADC ch')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(result.corrected[i], c='b', label='corrected')
    plt.axhline(0., c='gray', lw=.8)
    plt.xlabel('Time bucket')
    plt.ylabel('ADC ch')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

#______________________________________________________________________________
def plot_summary(pdf, result, before, after):
  ''' distributions over all channels of the event '''
  plt.figure(figsize=FIG_SIZE)
  plt.subplot(2, 2, 1)
  bins = np.linspace(0, max(before.max(), after.max()), 50)
  plt.hist(before, bins=bins, color='k', histtype='step', label='raw')
  plt.hist(after, bins=bins, color='b', histtype='step',
           label='corrected')
  plt.xlabel('Pre-signal RMS (ADC ch)')
  plt.ylabel('Channels')
  plt.legend()
  for i, (name, values) in enumerate((('adc offset', result.params[:, 0]),
                                      ('scale', result.params[:, 1]),
                                      ('time offset',
                                       result.params[:, 2]))):
    plt.subplot(2, 2, i + 2)
    plt.hist(values, bins=50, color='r', histtype='step')
    plt.xlabel(name)
    plt.ylabel('Channels')
  plt.tight_layout()
  pdf.savefig()
  plt.close()

#______________________________________________________________________________
def run(file_path, event=DEFAULT_EVENT, outfig=DEFAULT_OUTFIG,
        n_examples=DEFAULT_N_EXAMPLES, weighted=True, save=None):
  ''' correct one event and write the diagnostic pdf '''
  config = baseline.Config(weighted=weighted)
  data = read_event(file_path, event)
  if data is None:
    return
  waveforms, pad_ids = data
  if waveforms.shape[1] != config.n_time_bucket:
    logger.error(f'{waveforms.shape[1]} time buckets, '
                 f'expected {config.n_time_bucket}')
    return
  result = baseline.correct_event(waveforms, pad_ids, pads.excluded(),
                                  config)
  if result is None:
    return
  report(result, waveforms, pad_ids, config)
  with PdfPages(outfig) as pdf:
    plot_reference(pdf, result, waveforms, pad_ids, config)
    plot_examples(pdf, result, waveforms, pad_ids, config, n_examples)
    plot_summary(pdf, result, head_rms(waveforms, config),
                 head_rms(result.corrected, config))
  logger.info(f'wrote {outfig}')
  if save is not None:
    np.savez(save, raw=waveforms, corrected=result.corrected,
             fitted=result.fitted, params=result.params,
             pad=pad_ids, reference=result.reference,
             template=result.template)
    logger.info(f'wrote {save}')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path',
                      help='file path of input root file')
  parser.add_argument('--event', type=int, default=DEFAULT_EVENT,
                      help='event index to correct')
  parser.add_argument('--outfig', default=DEFAULT_OUTFIG,
                      help='output pdf file')
  parser.add_argument('--n-examples', type=int,
                      default=DEFAULT_N_EXAMPLES,
                      help='number of example channels to draw')
  parser.add_argument('--unweighted', action='store_true',
                      help='plain least squares instead of the ROOT '
                           'sqrt(content) bin errors')
  parser.add_argument('--save', default=None,
                      help='also write the waveforms to an npz file')
  parsed, unparsed = parser.parse_known_args()
  log_conf = os.path.join(parent_dir, 'logging_config.yml')
  with open(log_conf, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f))
  run(file_path=parsed.file_path, event=parsed.event,
      outfig=parsed.outfig, n_examples=parsed.n_examples,
      weighted=not parsed.unweighted, save=parsed.save)
