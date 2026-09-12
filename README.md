# ml-waveform

Machine Learning for Waveform Analysis.

The goal is to extract signals from TPC waveforms with techniques
borrowed from audio analysis. The TPC data lives in ROOT files and is
read directly as tensors; a few audio example scripts are kept
alongside as references for the techniques themselves.

## Requirements

- Python 3.14.6, pinned in `.python-version`
- [pyenv](https://github.com/pyenv/pyenv)

No system level dependency is needed. In particular:

- **CERN ROOT is not required.** ROOT files are read with `uproot`,
  a pure Python package.
- **FFmpeg is not required.** `torchaudio` 2.11 delegates decoding to
  `torchcodec`, which needs a system FFmpeg, so wav I/O goes through
  `soundfile` instead (`libsndfile` ships inside its wheel). The
  `torchaudio` transforms are unaffected and still used.

## Setup

This repository lives in an iCloud Drive folder, so the virtual
environment is kept outside of it to avoid syncing gigabytes of
wheels.

```console
pyenv install 3.14.6   # once; .python-version then selects it
python -m venv ~/.venvs/ml-waveform
source ~/.venvs/ml-waveform/bin/activate
pip install -r requirements.txt
```

Activate the environment in every new shell:

```console
source ~/.venvs/ml-waveform/bin/activate
```

`torch`, `torchaudio` and `torchvision` are pinned to matching
versions in `requirements.txt`; upgrade the three together, since
the latter two are built against a specific `torch` ABI.

## Data flow

```
run*.root  --uproot-->  tpc.TpcWaveformDataset  -->  DataLoader
                                 |
                                 +-- torchaudio transforms (optional)
```

The ROOT file is the single source of truth. There is no intermediate
conversion step: `tpc.py` turns each pad hit into a `[channel, time]`
tensor on demand, and carries the run, event and pad numbers along so
that a prediction can be traced back to the detector.

wav was used as an intermediate format in earlier tests. It is no
longer part of the pipeline: it cannot carry the run/event/pad
metadata, it forces int16, and one file per pad does not scale.
`root2wav.py` still exports wav for external audio tools, and the
older converted files are kept in `wav-legacy.tar.gz`.

## Modules

- `tpc.py`
  Reads TPC waveforms from a ROOT file with uproot.
  `TpcWaveformDataset` yields `(waveform, meta)` per pad hit,
  `read_tree()`, `normalize()` and `fix_length()` are the primitives.
- `audio_io.py`
  wav load/save through `soundfile`.
- `random_seed.py`
  `set_seed()`, `seed_worker()` and `make_generator()` to make runs
  reproducible.
- `logging_config.yml`
  Logging configuration loaded by the CLI scripts.

## Scripts

### TPC

- `training.py file.root [--max-events N] [--spectrogram]
  [--batch-size N]`
  Feed the waveforms of a ROOT file into a training loop.
- `fft.py file.{root,wav} [--max-events N]`
  FFT of the TPC waveforms of a ROOT file, written to `fft.pdf`, or
  of a wav file. Only the first event is read by default; pass `-1`
  to read every event.
- `root2pt.py file.root [--output out.pt] [--max-events N]`
  Optional cache of the TPC branches as a pt file.
- `root2wav.py file.root [--output-dir DIR] [--max-events N]`
  Export one wav file per pad hit, for external audio tools.
- `test-uproot.py file.root`
  Measure the uproot I/O performance of a ROOT file.

### Audio references

These run on `under_transition.wav` and exist to try out techniques
before applying them to the TPC data.

- `audio.py [file.wav]`
  Plot a waveform together with its log-scaled spectrogram.
- `spectrogram.py file.wav [n_fft] [hop_length]`
  Display the spectrogram of a wav file.
- `highpass_filter.py file.wav [cutoff_hz]`
  High-pass filter a wav file.
- `lowpass_filter.py file.wav [cutoff_hz]`
  Low-pass filter a wav file.
- `example-rnn.py`
  RNN regression on synthetic amplitude / period data.
- `example-gtzan.py [--root DIR] [--epochs N]`
  CNN genre classification on GTZAN. The dataset is about 1.2 GB
  and is downloaded into `data/` on the first run.

## Documents

The ML design notes and the handover memo are kept in `private/`,
which is not tracked here.

## Conventional analysis

`conventional-baseline-correction/` holds the non-ML common noise
correction, ported from `TPCRawData::CorrectBaselineTPC()` of the
k18analyzer. It is the baseline that the machine learning approach has
to beat; see the README in that directory.

## Notes

- **Use a linear spectrogram for TPC data, not a mel spectrogram.**
  The mel scale is tuned to human hearing over 20 Hz - 20 kHz and
  carries no meaning at the 12.5 MHz TPC sampling rate. `example-
  gtzan.py` uses mel because GTZAN is music; `training.py` uses
  `torchaudio.transforms.Spectrogram`.
- A pad readout is only 170 samples long, so an STFT yields very few
  frames. Feeding the raw waveform to a 1D CNN or an RNN is likely
  the stronger baseline, with the spectrogram as a second view.

## Status

`training.py` stops before the model: the network and the labels for
the TPC data are not defined yet.
