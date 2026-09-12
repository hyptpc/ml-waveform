#!/usr/bin/env python3

'''
Self checks for the port of TPCRawData::CorrectBaselineTPC().

Run directly: python3 test_baseline.py
'''

import numpy as np

import baseline

N_TB = baseline.NUM_OF_TIME_BUCKET
CONFIG = baseline.Config()
TOLERANCE = 1e-6

#______________________________________________________________________________
def check(name, condition):
  ''' report one assertion '''
  print(f'{"ok  " if condition else "FAIL"} {name}')
  assert condition, name

#______________________________________________________________________________
def test_sample_is_zero_outside():
  ''' f_baseline reads out of range bins as empty ones '''
  template = np.array([1., 2., 3.])
  values = baseline.sample(template, np.array([-1, 0, 2, 3]))
  check('sample() pads with zero',
        np.allclose(values, [0., 1., 3., 0.]))

#______________________________________________________________________________
def test_evaluate_integer_shift():
  ''' an integer time offset is a plain shift '''
  template = np.arange(5.)
  shifted = baseline.evaluate(template, 0., 1., 1.)
  check('evaluate() shifts by whole buckets',
        np.allclose(shifted, [1., 2., 3., 4., 0.]))

#______________________________________________________________________________
def test_evaluate_fractional_shift():
  ''' a fractional offset interpolates between two buckets '''
  template = np.array([0., 10., 20., 30.])
  shifted = baseline.evaluate(template, 0., 1., .5)
  check('evaluate() interpolates linearly',
        np.allclose(shifted, [5., 15., 25., 15.]))

#______________________________________________________________________________
def test_evaluate_offset_and_scale():
  ''' par[0] and par[1] are an offset and a scale '''
  template = np.array([1., 2., 3.])
  values = baseline.evaluate(template, 100., 2., 0.)
  check('evaluate() applies offset and scale',
        np.allclose(values, [102., 104., 106.]))

#______________________________________________________________________________
def test_build_template_removes_pedestal():
  ''' the pedestal is the mean before the signal window '''
  waveform = np.full(N_TB, 400.)
  waveform[CONFIG.min_time_bucket:] += 50.
  template = baseline.build_template(waveform, CONFIG)
  check('build_template() zeroes the pre-signal region',
        abs(template[:CONFIG.min_time_bucket].mean()) < TOLERANCE)

#______________________________________________________________________________
def make_waveforms(seed=0):
  ''' a synthetic event: common noise everywhere, a hit on one pad '''
  rng = np.random.default_rng(seed)
  buckets = np.arange(N_TB)
  common = 40. * np.sin(2. * np.pi * buckets / 12.)
  waveforms = np.empty((4, N_TB))
  for i, scale in enumerate((1., 1.5, .8, 1.2)):
    waveforms[i] = 400. + scale * common
  # A hit well inside the signal window, away from the fit region.
  waveforms[2, 60:80] += 900.
  return waveforms + rng.normal(0., 1., waveforms.shape)

#______________________________________________________________________________
def test_select_reference_takes_the_smallest_amplitude():
  ''' the minimum amplitude channel is the one without a hit '''
  waveforms = make_waveforms()
  reference = baseline.select_reference(waveforms, config=CONFIG)
  check('select_reference() avoids the channel with the hit',
        reference != 2)
  amplitudes = [baseline.amplitude(w, CONFIG) for w in waveforms]
  check('select_reference() returns the minimum amplitude channel',
        reference == int(np.argmin(amplitudes)))

#______________________________________________________________________________
def test_select_reference_skips_excluded_pads():
  ''' dead and noisy pads cannot become the reference '''
  waveforms = make_waveforms()
  pad_ids = np.array([10, 11, 12, 13])
  amplitudes = [baseline.amplitude(w, CONFIG) for w in waveforms]
  best = int(np.argmin(amplitudes))
  reference = baseline.select_reference(
    waveforms, pad_ids, {int(pad_ids[best])}, CONFIG)
  check('select_reference() skips an excluded pad', reference != best)

#______________________________________________________________________________
def test_select_reference_skips_quiet_channels():
  ''' a channel below min_base_rms is dead, not clean '''
  waveforms = make_waveforms()
  waveforms[0] = 400. # perfectly flat, rms == 0
  reference = baseline.select_reference(waveforms, config=CONFIG)
  check('select_reference() skips a channel below min_base_rms',
        reference != 0)

#______________________________________________________________________________
def test_correct_event_removes_the_common_noise():
  ''' the corrected waveforms are flat apart from the hit '''
  waveforms = make_waveforms()
  result = baseline.correct_event(waveforms, config=CONFIG)
  check('correct_event() found a reference', result is not None)
  head = result.corrected[:, :CONFIG.min_time_bucket]
  raw_head = waveforms[:, :CONFIG.min_time_bucket]
  check('correct_event() flattens the pre-signal region',
        np.std(head) < .2 * np.std(raw_head))
  hit = result.corrected[2, 60:80].mean()
  check('correct_event() keeps the hit amplitude',
        abs(hit - 900.) < 50.)

#______________________________________________________________________________
if __name__ == '__main__':
  test_sample_is_zero_outside()
  test_evaluate_integer_shift()
  test_evaluate_fractional_shift()
  test_evaluate_offset_and_scale()
  test_build_template_removes_pedestal()
  test_select_reference_takes_the_smallest_amplitude()
  test_select_reference_skips_excluded_pads()
  test_select_reference_skips_quiet_channels()
  test_correct_event_removes_the_common_noise()
  print('all checks passed')
