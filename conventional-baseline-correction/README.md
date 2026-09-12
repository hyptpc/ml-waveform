# Conventional baseline correction

A non-ML common noise correction for TPC waveforms, kept here as the
baseline that the machine learning approach has to beat.

It is a Python port of `TPCRawData::CorrectBaselineTPC()` from the
k18analyzer (`e72/src/TPCRawData.cc`), so that the conventional result
can be produced from this repository without CERN ROOT.

## Algorithm

1. **Pick the reference channel.** Within one event, take the channel
   with the smallest excursion above its own mean over the signal
   window, `max(w[t0:t1]) - mean(w[t0:t1])`. That is the channel least
   likely to hold a hit, so what is left on it is the common noise.
   Channels quieter than `min_base_rms` are dead rather than clean and
   are skipped, as are dead and noisy pads.
2. **Build the template.** Subtract the pre-signal pedestal,
   `mean(w[0:t0])`, from the reference waveform.
3. **Fit the template onto every channel.** The model is
   `f(t) = offset + scale * template(t + shift)`, with a linear
   interpolation between buckets, fitted over the tail region
   `[t1, n)` where no signal is expected.
4. **Subtract.** The corrected waveform is `w(t) - f(t)`.

Waveforms are handled in raw ADC counts, like the original.

## Parameters

Defaults come from `UserParam_e72_tpc_0` of the k18analyzer:

| Name | Value | Meaning |
| --- | --- | --- |
| `n_time_bucket` | 170 | samples per channel |
| `min_time_bucket` | 15 | start of the signal window (`t0`) |
| `max_time_bucket` | 150 | end of the signal window (`t1`) |
| `min_base_rms` | 20 | a quieter channel is treated as dead |

The fit parameter limits follow the original `TF1`: offset in
`[0, 4000]`, scale in `[-5, 5]`, shift in `[-10, 10]`.

## Usage

```console
source ~/.venvs/ml-waveform/bin/activate
python correct.py ../root/run06127_TPCWaveform.root --event 0
```

Options: `--event N`, `--outfig out.pdf`, `--n-examples N`,
`--save out.npz`, `--unweighted`.

The diagnostic pdf holds the reference waveform and its template, the
channels with the largest corrected signal (raw, fitted baseline and
corrected), and the distributions over all channels of the event.

Run the self checks with:

```console
python test_baseline.py
```

## Files

- `baseline.py` — the algorithm: `select_reference()`,
  `build_template()`, `evaluate()`, `fit_baseline()`,
  `correct_event()`.
- `correct.py` — CLI: reads a root file, corrects one event, writes
  the diagnostic pdf.
- `pads.py` — the pads that `IsDead()` and `Noise()` reject.
- `excluded_pads.json` — those pad ids, generated.
- `extract_pads.py` — regenerates the json from `TPCPadHelper.hh`.
  Rerun it when the analyzer is updated.
- `test_baseline.py` — self checks of the ported semantics.

## Notes

- **The fit region is not an honest test of the correction.** The tail
  RMS drops because that is exactly where the template was fitted. The
  pre-signal region `[0, t0)` is never seen by the fit, so its RMS is
  the out-of-sample check. On event 0 of run 6127 it falls from 29.3
  to 15.7 ADC counts.
- A cluster of channels fits `scale ~ 0`: the fit found no common
  noise there. They are worth inspecting rather than trusting.
- The original fits a ROOT histogram, whose default bin error is
  `sqrt(content)`, so that weighting is reproduced by default. Pass
  `--unweighted` for plain least squares, which is arguably the better
  choice since FADC noise is not Poisson.
