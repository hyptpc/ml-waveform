#!/usr/bin/env python3

'''
Common noise correction for TPC waveforms.

Port of TPCRawData::CorrectBaselineTPC() from the k18analyzer
(e72/src/TPCRawData.cc). One channel that looks free of any hit is
taken as the reference of the common noise, and every channel is then
corrected by an offset, a scale and a time shift of that reference.

Waveforms are handled in raw ADC counts, like the original.
'''

import dataclasses
import logging

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Defaults follow UserParam_e72_tpc_0 of the k18analyzer.
NUM_OF_TIME_BUCKET = 170
MIN_TIME_BUCKET = 15
MAX_TIME_BUCKET = 150
MIN_BASE_RMS = 20.
# Parameter limits of the original TF1 fit.
ADC_OFFSET_LIMITS = (0., 4000.)
SCALE_LIMITS = (-5., 5.)
TIME_OFFSET_LIMITS = (-10., 10.)
MAX_FIT_CALLS = 2000

#______________________________________________________________________________
@dataclasses.dataclass
class Config:
  ''' the UserParam entries the correction depends on '''
  n_time_bucket: int = NUM_OF_TIME_BUCKET
  min_time_bucket: int = MIN_TIME_BUCKET
  max_time_bucket: int = MAX_TIME_BUCKET
  min_base_rms: float = MIN_BASE_RMS
  # The original fits a ROOT histogram, whose default bin error is
  # sqrt(content); set False for plain unweighted least squares.
  weighted: bool = True

#______________________________________________________________________________
@dataclasses.dataclass
class Result:
  ''' outcome of one event '''
  reference: int # channel index used as the common noise reference
  template: np.ndarray # [n_time_bucket], pedestal removed
  corrected: np.ndarray # [n_channel, n_time_bucket]
  fitted: np.ndarray # [n_channel, n_time_bucket], the baselines
  params: np.ndarray # [n_channel, 3]: adc offset, scale, time offset

#______________________________________________________________________________
def amplitude(waveform, config=None):
  ''' excursion above the mean inside the signal window '''
  config = config or Config()
  window = waveform[config.min_time_bucket:config.max_time_bucket]
  return float(window.max() - window.mean())

#______________________________________________________________________________
def rms(waveform, start, stop):
  ''' TMath::RMS is the population standard deviation '''
  return float(np.std(waveform[start:stop]))

#______________________________________________________________________________
def select_reference(waveforms, pads=None, excluded=None, config=None):
  ''' channel index of the common noise reference, None if there is
  none

  Minimum amplitude method: the channel with the smallest excursion
  above its own mean is the one least likely to hold a hit. A channel
  quieter than min_base_rms is dead rather than clean, and dead or
  noisy pads are skipped.
  '''
  config = config or Config()
  excluded = excluded or set()
  best = None
  best_amplitude = None
  for i, waveform in enumerate(waveforms):
    if len(waveform) != config.n_time_bucket:
      logger.debug(f'channel {i}: unexpected length {len(waveform)}')
      continue
    if pads is not None and int(pads[i]) in excluded:
      continue
    if rms(waveform, config.min_time_bucket,
           config.max_time_bucket) <= config.min_base_rms:
      continue
    value = amplitude(waveform, config)
    if best_amplitude is None or value < best_amplitude:
      best_amplitude = value
      best = i
  return best

#______________________________________________________________________________
def build_template(waveform, config=None):
  ''' the reference waveform with its pedestal subtracted '''
  config = config or Config()
  waveform = np.asarray(waveform, dtype=float)
  pedestal = float(np.mean(waveform[:config.min_time_bucket]))
  return waveform - pedestal

#______________________________________________________________________________
def sample(template, index):
  ''' template value at integer index, zero outside the record '''
  values = np.zeros(len(index), dtype=float)
  inside = (index >= 0) & (index < len(template))
  values[inside] = template[index[inside]]
  return values

#______________________________________________________________________________
def evaluate(template, adc_offset, scale, time_offset, n_samples=None):
  ''' the fitted baseline at every time bucket

  Mirrors f_baseline(): the template is shifted by time_offset with a
  linear interpolation between the two neighbouring buckets.
  '''
  n_samples = len(template) if n_samples is None else n_samples
  index = np.arange(n_samples)
  floor = int(np.floor(time_offset))
  frac = time_offset - floor
  left = sample(template, index + floor)
  right = sample(template, index + floor + 1)
  return adc_offset + scale * ((1. - frac) * left + frac * right)

#______________________________________________________________________________
def fit_baseline(waveform, template, config=None):
  ''' fit the template onto the tail, where no signal is expected '''
  config = config or Config()
  waveform = np.asarray(waveform, dtype=float)
  index = np.arange(config.max_time_bucket, config.n_time_bucket)
  data = waveform[index]

  def model(x, adc_offset, scale, time_offset):
    full = evaluate(template, adc_offset, scale, time_offset,
                    len(waveform))
    # curve_fit hands x back as floats; they are exact bucket indices.
    return full[np.asarray(x, dtype=int)]

  lower = (ADC_OFFSET_LIMITS[0], SCALE_LIMITS[0], TIME_OFFSET_LIMITS[0])
  upper = (ADC_OFFSET_LIMITS[1], SCALE_LIMITS[1], TIME_OFFSET_LIMITS[1])
  start = (float(np.clip(data.mean(), *ADC_OFFSET_LIMITS)), 1., 0.)
  sigma = None
  if config.weighted:
    sigma = np.sqrt(np.abs(data))
    sigma[sigma == 0.] = 1.
  try:
    params, _ = curve_fit(model, index, data, p0=start,
                          bounds=(lower, upper), sigma=sigma,
                          max_nfev=MAX_FIT_CALLS)
  except (RuntimeError, ValueError) as e:
    logger.warning(f'fit failed, falling back to the initial values: {e}')
    params = np.array(start)
  return np.asarray(params, dtype=float)

#______________________________________________________________________________
def correct_event(waveforms, pads=None, excluded=None, config=None):
  ''' correct every channel of one event, None if no reference '''
  config = config or Config()
  waveforms = np.asarray(waveforms, dtype=float)
  reference = select_reference(waveforms, pads, excluded, config)
  if reference is None:
    logger.error('no reference was found')
    return None
  template = build_template(waveforms[reference], config)
  params = np.empty((len(waveforms), 3), dtype=float)
  fitted = np.empty_like(waveforms)
  for i, waveform in enumerate(waveforms):
    params[i] = fit_baseline(waveform, template, config)
    fitted[i] = evaluate(template, *params[i], len(waveform))
  return Result(reference=reference, template=template,
                corrected=waveforms - fitted, fitted=fitted,
                params=params)
