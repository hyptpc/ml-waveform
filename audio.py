#!/usr/bin/env python3

'''
Load a wav file and show its waveform and spectrogram.
'''

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchaudio

import audio_io

top_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_WAV = os.path.join(top_dir, 'under_transition.wav')
# Guard log2() against zero-valued spectrogram bins, which would
# otherwise produce -inf and flatten the whole color scale.
LOG_EPSILON = 1e-10

#______________________________________________________________________________
def show(file_path, outfig='audio.png'):
  ''' plot waveform and log-scaled spectrogram '''
  waveform, sample_rate = audio_io.load(file_path)
  print(f'Shape of waveform [channel, time]: {waveform.size()}')
  print(f'Sample rate of waveform: {sample_rate}')
  spec = torchaudio.transforms.Spectrogram()(waveform)
  plt.subplot(2, 1, 1)
  plt.plot(waveform.t().numpy())
  plt.subplot(2, 1, 2)
  plt.imshow(torch.log2(spec[0] + LOG_EPSILON).numpy(), aspect='auto')
  plt.colorbar()
  plt.tight_layout()
  plt.savefig(outfig)
  plt.show()
  plt.close()

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path', nargs='?', default=DEFAULT_WAV,
                      help='file path of input wav file')
  parsed, unparsed = parser.parse_known_args()
  show(file_path=parsed.file_path)
