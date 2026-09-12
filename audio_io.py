#!/usr/bin/env python3

'''
Minimal wav I/O.

torchaudio 2.11 delegates decoding to torchcodec, which needs FFmpeg
installed system wide. soundfile ships libsndfile inside its wheel,
which covers the wav files used here without any system dependency.

wav is only an export format for external audio tools; the training
pipeline reads root files directly, see tpc.py.
'''

import numpy as np
import soundfile as sf
import torch

DEFAULT_SUBTYPE = 'PCM_16'

#______________________________________________________________________________
def load(file_path):
  ''' read a wav file as a [channel, time] float tensor '''
  data, sample_rate = sf.read(file_path, dtype='float32',
                              always_2d=True)
  return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate

#______________________________________________________________________________
def save(file_path, waveform, sample_rate, subtype=DEFAULT_SUBTYPE):
  ''' write a [channel, time] tensor as a wav file '''
  data = waveform.detach().cpu().numpy()
  sf.write(file_path, data.T, sample_rate, subtype=subtype)
