# General cross-talk methods paper and synthetic waveforms

`crosstalk_method_paper.tex` is a new manuscript, with the original least-squares
derivation largely preserved. The introduction, implementation, synthetic
application, and conclusions are general. There are no experiment or detector-pair
identifiers. Correction results have not been fabricated: the figures show
uncorrected synthetic data and known injected truth. Add your correction results
in the evaluation subsection before presenting quantitative performance claims.

The three first presets reproduce the qualitative shapes of the supplied plots;
they are not fitted or digitized replicas of the measurements.

## Run

Requires Python 3.10 or newer and NumPy, SciPy, Matplotlib:

```bash
python -m pip install -r requirements.txt
python synthetic_crosstalk.py --case all --n-events 120 --out examples
python make_paper_figures.py
python verify_generator.py
pdflatex crosstalk_method_paper.tex
pdflatex crosstalk_method_paper.tex
```

The delivered examples contain 120 pairs per case, and plots display the first
20 events. The generator default is 600 events; increase this for performance
studies. Recreate the supplied figures with the exact command above. If you
change the example configurations or event counts, also update the corresponding
paragraph and table in the manuscript.

The LaTeX file prefers RevTeX 4.1 (as in the original source), then RevTeX 4.2.
An `article` fallback makes the same source compilable without RevTeX. The supplied
PDF preview uses this fallback; RevTeX installations will give the intended APS
formatting. The original external `gen/formatting` and `gen/references` files were
not attached; all required manuscript macros and bibliography entries are now
included. Add the final author list and affiliations. The ZIP contains all figure
files needed for compilation. The LaTeX file by itself also compiles, with figure
placeholders if the figure files are absent.

## Change a case

```bash
python synthetic_crosstalk.py --case bipolar --strength 0.08 --polarity -1 --n-events 2000 --out my_case
python synthetic_crosstalk.py --case negative --kind difference --delay-us 0.04 --out my_difference
```

For full control from a notebook or another script:

```python
from dataclasses import replace
import numpy as np
from synthetic_crosstalk import PRESETS, generate_pairs, plot_pair, save_case

cfg = replace(
    PRESETS['bipolar'],
    n_events=2000,
    strength=0.08,              # nominal peak cross-talk / source ratio
    polarity=-1,               # reverse the whole response
    delay_us=0.04,
    tau_us=0.10,
    source_noise_rms=3.0,
    target_noise_rms=6.0,
    target_noise_rho=0.4,       # colored target noise; 0 means white
    seed=42,
)
data = generate_pairs(cfg)
u, v = data['source'], data['target']  # shape: (events, samples)
train, val, test = (data[k] for k in ('train_index', 'val_index', 'test_index'))

# Fit your filter using u[train], v[train]; tune with val; evaluate with test.
# Do not train or tune using kernel_true or crosstalk_true.
# valid_mask excludes samples without the required recorded source prehistory.
fig = plot_pair(data)
fig.savefig('my_pairs.pdf', bbox_inches='tight')

# Or save measured data, truth, JSON configuration, PDF and PNG together:
save_case('my_case', cfg, 'my_case_output')
```

All Config fields can also be overridden in a JSON file:

```json
{"strength": 0.05, "polarity": -1, "tau_us": 0.1, "target_noise_rms": 4.0}
```

Pass it with `--config my_config.json`. Command-line options override JSON fields;
JSON fields override the selected preset. `--case all` applies overrides to every
case, including `null`, so setting a nonzero strength also changes the null case.

## Presets and parameters

| Preset | Response | Relation to supplied plots |
|---|---|---|
| `bipolar` | Positive lobe followed by a larger negative lobe | Bipolar input example |
| `negative` | Delayed negative pulse | Negative input example |
| `weak_bipolar` | Weaker reversed bipolar response | Weak bipolar input example |
| `positive` | Delayed positive pulse | Additional example |
| `tail` | Smoothed negative pulse with exponential tail | Additional example |
| `ringing` | Damped oscillation excited by a narrow source | Additional example |
| `echo` | Main pulse followed by an opposite-polarity echo | Additional example |
| `null` | Noise-only target, zero injected coupling | Control sample |

The mapping above refers to waveform morphology. No experimental channel names
or measured scores are assigned to the synthetic data.

| Parameter | Meaning |
|---|---|
| `strength`, `polarity` | Nominal coupling magnitude and full-response sign |
| `kind` | `direct`, `lowpass`, `bipolar`, `difference`, `ringing`, or `echo` |
| `delay_us`, `kernel_taps` | Fixed transfer delay and finite response support |
| `tau_us`, `frequency_mhz` | Exponential decay and ringing frequency |
| `echo_delay_us`, `echo_weight` | Delay and relative weight of the second echo |
| `separation_us` | Separation of impulses in `difference` coupling |
| `source_shape` | `asymmetric`, `gaussian`, `double_exp`, or `step` |
| `rise_us`, `fall_us` | Widths/time constants; their meaning depends on source shape |
| `shape_jitter`, `time_jitter_us` | Source-shape spread and common event-time spread |
| `source_noise_rms`, `target_noise_rms` | Stationary RMS of independent channel noise |
| `source_noise_rho`, `target_noise_rho` | AR(1) correlation within each channel |
| `target_probability`, `target_amplitude_range` | Genuine target-pulse occupancy and size |
| `target_locked` | Lock genuine target activity to the source (contamination test) |
| `gain_jitter`, `relative_delay_jitter_us` | Event-dependent transfer changes (stress tests) |
| `saturation` | Optional nonlinear coupling scale, `c -> L*tanh(c/L)` |
| `seed` | Deterministic random seed |

All times are in microseconds, frequencies in MHz, and amplitudes in arbitrary
units. For `asymmetric`, the widths are the left/right Gaussian sigmas; for
`gaussian`, only `rise_us` sets the sigma. For `double_exp`, they are rise and decay
time constants (`fall_us > rise_us`). For `step`, `peak_us` is the onset,
`rise_us` is the rise time constant, and `fall_us` is unused. `shape_jitter` is the
logarithmic width spread. `time_jitter_us` shifts the source and its induced
response together; it is different from relative source/target timing jitter.

`strength` is normalized once on a unit-height nominal source without a
precursor. It is NOT forced to be the peak ratio of every individual event.
Changing `source_shape` or its nominal widths rebuilds this normalization. To
compare different pulse families through exactly the same physical transfer,
keep the injected kernel fixed, rather than interpreting equal strength settings
across different reference shapes as identical electronics.

To reuse absolute coefficients without any renormalization:

```python
reference = generate_pairs(PRESETS['bipolar'])
changed_source = replace(PRESETS['bipolar'], source_shape='gaussian', rise_us=0.04)
data = generate_pairs(changed_source, kernel=reference['kernel_true'])
```

With `kernel=...`, the supplied coefficients already contain strength, polarity
and delay; those Config settings are bypassed. `kernel_supplied` records this
choice. The actual support is the length of the supplied kernel, regardless of
`kernel_taps`.

The generic kernel-shape interface is:

```python
cfg = replace(PRESETS['bipolar'], delay_us=0.0)
data = generate_pairs(cfg, kernel_shape=np.array([1.0, -0.2, -0.4, -0.4]))
```

This raw shape is normalized by the same strength convention and its time origin
is lag zero. Increase `kernel_taps` for longer tails, echoes, or delays. Truncated
kernels remain valid finite models, but a tail reaching the last tap can require
a longer support to model the intended continuous response accurately.

## Saved arrays

Each `examples/CASE.npz` can be opened with `np.load(..., allow_pickle=False)`.

| Key | Content |
|---|---|
| `time_us` | Sample time axis |
| `source`, `target` | Measured waveform pairs used by your method |
| `source_clean`, `source_noise` | Clean source and source measurement noise |
| `crosstalk_true` | Exact induced target contribution, including enabled stress effects |
| `target_physical`, `target_noise` | Genuine target signal and independent target noise |
| `target_ideal` | Genuine target signal plus the SAME noise, without cross-talk |
| `kernel_true`, `kernel_time_us` | Nominal injected FIR coefficients and their lag axis |
| `event_gain`, `event_delay_us` | Event-dependent transfer perturbations |
| `source_amplitude`, `source_peak_us` | Generated source amplitude parameter and time reference |
| `target_active` | Events with an injected genuine target pulse |
| `train_index`, `val_index`, `test_index` | Disjoint 60/20/20 event split |
| `valid_mask` | Samples whose nominal FIR prehistory is available in the saved source |
| `config_json` | Exact configuration as a JSON string |
| `kernel_supplied` | Whether absolute coefficients bypassed the kernel factory |

The exact identities are `target = target_physical + crosstalk_true + target_noise`
and `target_ideal = target_physical + target_noise`. Cross-talk is generated from
the clean source. Changing source measurement noise does not change the physical
cross-talk. A fixed nominal kernel is no longer an exact per-event model when
gain/timing jitter or saturation is enabled; use `crosstalk_true` for evaluation.
For a fitted filter longer than the injected one, recompute the valid sample
mask for the fitted support. Add margins for any fitted negative lags or relative
timing shifts. Truth arrays are for evaluation only.

## Signal preservation and stability tests

Train on isolated-source data, then generate independent test events with genuine
target signals. Fit and apply the correction with your existing code:

```python
train_data = generate_pairs(replace(PRESETS['bipolar'], n_events=2000, seed=101))
test_cfg = replace(PRESETS['bipolar'], n_events=1000, seed=102,
                   target_probability=0.5, target_amplitude_range=(20, 80))
test_data = generate_pairs(test_cfg)

# Separate, explicitly nonstationary/nonlinear tests:
timing_test = generate_pairs(replace(test_cfg, relative_delay_jitter_us=0.015))
gain_test = generate_pairs(replace(test_cfg, gain_jitter=0.15))
nonlinear_test = generate_pairs(replace(test_cfg, saturation=30.0))
contaminated_training = generate_pairs(replace(test_cfg, target_locked=True))
```

Compare your corrected target to `target_ideal`, and your predicted cross-talk to
`crosstalk_true`, over a declared window. Report waveform prediction error and
changes in reconstructed physical-pulse amplitude, area, and timing. Baseline
RMS alone cannot demonstrate preservation of physical signals. The manuscript
defines normalized error and suppression in dB; neither is a detection efficiency.

## Checks and drafting notes

`verify_generator.py` checks deterministic output, sign and gain changes, exact
component decomposition, causal convolution, fractional delays, independent
measurement noise, stationary colored noise, event splitting, alternate pulse
types, and stress modes. It does not test or claim correction performance.

The core equations and their labels were retained. Necessary clarifications cover
finite-record boundaries, source noise, training contamination, and joint fitting
when multiple sources are correlated. Pair relevance is evaluated independently
of training and has no universal hard threshold. The application section is now
a controlled synthetic benchmark. References to the DPLMS publication were
retained; a classical reference for reference-based interference cancellation was
added to place the established least-squares principle in context.
