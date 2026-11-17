"""Reproducible source/target waveform pairs for cross-talk filter studies.

All times are in microseconds; amplitudes are arbitrary units. No correction
filter is fitted here. The coupling acts on the CLEAN source; independent
measurement noise is then added to the source and target channels.

Quick start:
    python synthetic_crosstalk.py --case all --out examples
    python synthetic_crosstalk.py --case bipolar --strength .08 --polarity -1

Import:
    from dataclasses import replace
    from synthetic_crosstalk import PRESETS, generate_pairs, plot_pair
    cfg = replace(PRESETS['bipolar'], n_events=2000, strength=.08, seed=42)
    data = generate_pairs(cfg)
    u, v = data['source'], data['target']  # (events, samples)

The fixed kernel is normalized ONCE on the nominal unit-height source pulse:
strength = max(abs(convolve(reference_pulse, kernel))). Shape variations can
therefore change individual peak ratios. No event-by-event renormalization is
applied to the coupling. Arrays include the truth and disjoint train/validation/
test indices; use only the measured source and target for training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter


@dataclass(frozen=True)
class Config:
    n_events: int = 500
    n_samples: int = 768
    dt_us: float = 0.004
    start_us: float = 0
    peak_us: float = 1.6
    amplitude_range: tuple[float, float] = (250.0, 700.0)
    source_shape: str = 'asymmetric'  # asymmetric, gaussian, double_exp, step
    rise_us: float = 0.13           # sigma for asymmetric; time constant otherwise
    fall_us: float = 0.045          # right sigma, or decay constant for double_exp
    shape_jitter: float = 0.08      # log-normal fractional width spread
    time_jitter_us: float = 0.035   # common event-time spread, not channel jitter
    precursor_fraction: float = 0.08
    precursor_delay_us: float = 0.25
    kind: str = 'bipolar'          # direct, lowpass, bipolar, difference, ringing, echo
    strength: float = 0.14         # nonnegative nominal peak coupling ratio
    polarity: int = 1             # reverses the entire kernel
    delay_us: float = 0.024
    kernel_taps: int = 201
    tau_us: float = 0.07
    separation_us: float = 0.045   # difference of two delayed copies
    frequency_mhz: float = 8.0
    echo_delay_us: float = 0.22
    echo_weight: float = -0.55
    source_noise_rms: float = 3.0
    target_noise_rms: float = 5.0
    source_noise_rho: float = 0.0   # stationary AR(1) correlation coefficient
    target_noise_rho: float = 0.0
    target_probability: float = 0.0
    target_amplitude_range: tuple[float, float] = (20.0, 60.0)
    target_time_range_us: tuple[float, float] = (1.1, 2.4)
    target_rise_us: float = 0.012
    target_decay_us: float = 0.10
    target_locked: bool = False    # correlated-activity stress test
    target_delay_us: float = 0.05
    gain_jitter: float = 0.0       # stress test: event-dependent coupling gain
    relative_delay_jitter_us: float = 0.0  # stress test: channel timing instability
    saturation: float | None = None       # stress test: c -> L*tanh(c/L)
    seed: int = 20261112


PRESETS = {
    'bipolar': Config(),
    'negative': Config(kind='direct', polarity=-1, strength=0.12, delay_us=0.08,
                       amplitude_range=(350.0, 1050.0), rise_us=0.085,
                       fall_us=0.035, seed=20261113),
    'weak_bipolar': Config(polarity=-1, strength=0.035, seed=20261114),
    'positive': Config(kind='direct', strength=0.10, delay_us=0.04, seed=20261115),
    'tail': Config(kind='lowpass', polarity=-1, strength=0.12,
                   tau_us=0.16, seed=20261116),
    'ringing': Config(kind='ringing', strength=0.14, tau_us=0.20,
                      frequency_mhz=6.0, source_shape='gaussian', rise_us=0.025,
                      fall_us=0.025, time_jitter_us=0.008, precursor_fraction=0.0,
                      seed=20261117),
    'echo': Config(kind='echo', strength=0.12, echo_delay_us=0.23,
                   echo_weight=-0.65, seed=20261118),
    'null': Config(strength=0.0, seed=20261119),
}

TITLES = {
    'bipolar': 'Bipolar coupling', 'negative': 'Negative delayed pulse',
    'weak_bipolar': 'Weak reversed bipolar coupling', 'positive': 'Positive coupling',
    'tail': 'Exponential tail', 'ringing': 'Damped ringing',
    'echo': 'Opposite-polarity echo', 'null': 'No coupling',
}


def _validate(cfg):
    for name in ('n_events', 'n_samples', 'kernel_taps'):
        value = getattr(cfg, name)
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if cfg.n_events < 5 or not 2 <= cfg.kernel_taps < cfg.n_samples:
        raise ValueError('Need >=5 events and 2 <= kernel_taps < n_samples')
    for name in ('dt_us', 'rise_us', 'fall_us', 'tau_us', 'target_rise_us',
                 'target_decay_us', 'frequency_mhz', 'separation_us', 'echo_delay_us'):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('strength', 'delay_us', 'shape_jitter', 'time_jitter_us',
                 'precursor_fraction', 'precursor_delay_us', 'source_noise_rms',
                 'target_noise_rms', 'gain_jitter', 'relative_delay_jitter_us'):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('start_us', 'peak_us', 'target_delay_us', 'echo_weight'):
        if not np.isfinite(getattr(cfg, name)):
            raise ValueError(f'{name} must be finite')
    for name in ('amplitude_range', 'target_amplitude_range', 'target_time_range_us'):
        pair = getattr(cfg, name)
        if len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[1] < pair[0]:
            raise ValueError(f'{name} must be an ordered finite pair')
    if cfg.amplitude_range[0] <= 0 or cfg.target_amplitude_range[0] < 0:
        raise ValueError('Source amplitudes must be positive; target amplitudes nonnegative')
    if cfg.polarity not in (-1, 1) or not 0 <= cfg.target_probability <= 1:
        raise ValueError('polarity must be -1 or +1; target_probability must be in [0, 1]')
    if not all(abs(r) < 1 for r in (cfg.source_noise_rho, cfg.target_noise_rho)):
        raise ValueError('AR(1) correlations must satisfy abs(rho) < 1')
    if cfg.saturation is not None and (not np.isfinite(cfg.saturation) or cfg.saturation <= 0):
        raise ValueError('saturation must be positive or None')
    if not cfg.start_us < cfg.peak_us < cfg.start_us + (cfg.n_samples - 1) * cfg.dt_us:
        raise ValueError('The nominal source peak must lie inside the record')
    if cfg.delay_us >= (cfg.kernel_taps - 1) * cfg.dt_us:
        raise ValueError('Increase kernel_taps to cover delay_us')
    if cfg.source_shape not in ('asymmetric', 'gaussian', 'double_exp', 'step'):
        raise ValueError('Unknown source_shape')
    if cfg.source_shape == 'double_exp' and cfg.fall_us <= cfg.rise_us:
        raise ValueError('double_exp needs fall_us > rise_us')
    if cfg.target_decay_us <= cfg.target_rise_us:
        raise ValueError('target_decay_us must exceed target_rise_us')
    if cfg.kind == 'ringing' and cfg.frequency_mhz >= 0.5 / cfg.dt_us:
        raise ValueError('Ringing frequency must be below Nyquist')


def _double_exp(t, rise, decay):
    """Causal difference of exponentials, analytically normalized to unit peak."""
    t = np.asarray(t)
    a = np.maximum(t, 0.0)
    peak = np.log(decay / rise) / (1.0 / rise - 1.0 / decay)
    norm = np.exp(-peak / decay) - np.exp(-peak / rise)
    return np.where(t >= 0, (np.exp(-a / decay) - np.exp(-a / rise)) / norm, 0.0)


def _pulse(t, kind, rise, fall):
    """The asymmetric/gaussian/double_exp reference peaks at t=0."""
    if kind == 'asymmetric':
        return np.exp(-0.5 * (t / np.where(t < 0, rise, fall)) ** 2)
    if kind == 'gaussian':
        return np.exp(-0.5 * (t / rise) ** 2)
    if kind == 'double_exp':
        peak = np.log(fall / rise) / (1.0 / rise - 1.0 / fall)
        return _double_exp(t + peak, rise, fall)
    # For a step, t=0 is the onset and rise_us is the rise time constant.
    return np.where(t >= 0, -np.expm1(-np.maximum(t, 0.0) / rise), 0.0)


def _fractional_impulse(length, delay_samples):
    """Two-tap linear interpolation of a delayed unit impulse; no circular shift."""
    if not 0 <= delay_samples <= length - 1:
        raise ValueError('Requested lag exceeds the FIR support; increase kernel_taps')
    h = np.zeros(length)
    k = int(np.floor(delay_samples))
    fraction = delay_samples - k
    h[k] = 1.0 - fraction
    if fraction > 0:
        h[k + 1] = fraction
    return h


def make_kernel(cfg: Config, kernel_shape=None):
    """Build a fixed causal FIR, including polarity, delay and strength.

    kernel_shape optionally supplies a 1-D unnormalized shape instead of kind.
    Its first sample is lag zero. delay_us is added, and the same reference-pulse
    normalization is used. Do not pass a fitted filter if you intend a blind test.
    """
    _validate(cfg)
    delay = cfg.delay_us / cfg.dt_us
    base_len = cfg.kernel_taps - int(np.ceil(delay))
    t = np.arange(base_len) * cfg.dt_us
    delta = _fractional_impulse(base_len, 0.0)
    if kernel_shape is not None:
        shape = np.asarray(kernel_shape, dtype=float)
        if shape.ndim != 1 or not 1 <= shape.size <= base_len or not np.all(np.isfinite(shape)):
            raise ValueError('kernel_shape must be finite, 1-D and fit inside the FIR support')
        raw = np.pad(shape, (0, base_len - shape.size))
    elif cfg.kind == 'direct':
        raw = delta
    elif cfg.kind in ('lowpass', 'bipolar'):
        smooth = np.exp(-t / cfg.tau_us)
        smooth /= smooth.sum()
        raw = smooth if cfg.kind == 'lowpass' else delta - smooth
    elif cfg.kind == 'difference':
        raw = delta - _fractional_impulse(base_len, cfg.separation_us / cfg.dt_us)
    elif cfg.kind == 'ringing':
        raw = np.exp(-t / cfg.tau_us) * np.sin(2 * np.pi * cfg.frequency_mhz * t)
    elif cfg.kind == 'echo':
        raw = delta + cfg.echo_weight * _fractional_impulse(base_len, cfg.echo_delay_us / cfg.dt_us)
    else:
        raise ValueError(f'Unknown coupling kind: {cfg.kind}')
    # Shift using full convolution, not np.roll or a centered "same" crop.
    shift = _fractional_impulse(int(np.ceil(delay)) + 1, delay)
    raw = np.convolve(raw, shift)[:cfg.kernel_taps]
    raw = np.pad(raw, (0, cfg.kernel_taps - raw.size))
    reference_time = np.arange(-4 * cfg.n_samples, 4 * cfg.n_samples) * cfg.dt_us
    reference = _pulse(reference_time, cfg.source_shape, cfg.rise_us, cfg.fall_us)
    response = fftconvolve(reference, raw, mode='full')[:reference.size]
    norm = np.max(np.abs(response))
    if norm <= np.finfo(float).eps:
        raise ValueError('The kernel has zero response to the reference pulse')
    return cfg.polarity * cfg.strength * raw / norm


def _noise(rng, shape, rms, rho):
    """Stationary Gaussian AR(1); rms is the ensemble RMS, not sample-rescaled."""
    z = rng.normal(size=shape)
    z[:, 0] *= rms
    if shape[1] > 1:
        z[:, 1:] *= rms * np.sqrt(1.0 - rho * rho)
    return lfilter([1.0], [1.0, -rho], z, axis=1)


def generate_pairs(cfg: Config = Config(), kernel_shape=None, *, kernel=None):
    """Return measured pairs, hidden truth and metadata in a dict of NumPy arrays.

    source/target and all waveform components have shape (n_events, n_samples).
    target = target_physical + crosstalk_true + target_noise.
    target_ideal = target_physical + target_noise is the SAME-noise reference.
    kernel_true is exact for the nominal LTI case; the saved event gains/delays
    and saturation setting describe deliberate departures from that case.
    kernel optionally supplies absolute FIR coefficients, bypassing kind,
    strength, polarity and delay normalization. This preserves one physical
    transfer when changing the source-pulse family. Do not also set kernel_shape.
    """
    supplied_kernel = kernel is not None
    if supplied_kernel:
        _validate(cfg)
        kernel = np.asarray(kernel, dtype=float).copy()
        if kernel_shape is not None:
            raise ValueError('Use either kernel or kernel_shape, not both')
        if (kernel.ndim != 1 or not 1 <= kernel.size < cfg.n_samples
                or not np.all(np.isfinite(kernel))):
            raise ValueError('kernel must be finite, 1-D and shorter than the record')
    else:
        kernel = make_kernel(cfg, kernel_shape)
    pulse_rng, src_rng, tgt_rng, phys_rng, coupling_rng, split_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(6)
    ]
    time = cfg.start_us + np.arange(cfg.n_samples) * cfg.dt_us
    # Generate prehistory so a finite record does not create artificial edges.
    guard = kernel.size + int(np.ceil(8 * cfg.relative_delay_jitter_us / cfg.dt_us))
    extended_time = cfg.start_us + np.arange(-guard, cfg.n_samples + guard) * cfg.dt_us
    amplitudes = pulse_rng.uniform(*cfg.amplitude_range, cfg.n_events)
    peaks = cfg.peak_us + pulse_rng.normal(0.0, cfg.time_jitter_us, cfg.n_events)
    widths = np.exp(pulse_rng.normal(-0.5 * cfg.shape_jitter**2,
                                    cfg.shape_jitter, (cfg.n_events, 2)))
    precursors = pulse_rng.uniform(0.0, cfg.precursor_fraction, cfg.n_events)
    source_ext = np.empty((cfg.n_events, extended_time.size))
    for i in range(cfg.n_events):
        rise, fall = cfg.rise_us * widths[i, 0], cfg.fall_us * widths[i, 1]
        if cfg.source_shape == 'double_exp':
            # Vary the decay/rise gap positively to preserve a valid pulse.
            fall = rise + (cfg.fall_us - cfg.rise_us) * widths[i, 1]
        t = extended_time - peaks[i]
        shape = _pulse(t, cfg.source_shape, rise, fall)
        if cfg.source_shape != 'step':
            shape += precursors[i] * _pulse(t + cfg.precursor_delay_us,
                                           cfg.source_shape, rise, fall)
        source_ext[i] = amplitudes[i] * shape
    cross_ext = fftconvolve(source_ext, kernel[None, :], axes=1, mode='full')[:, :extended_time.size]
    event_gain = np.exp(coupling_rng.normal(-0.5 * cfg.gain_jitter**2,
                                          cfg.gain_jitter, cfg.n_events))
    event_delay = coupling_rng.normal(0.0, cfg.relative_delay_jitter_us, cfg.n_events)
    for i in range(cfg.n_events):
        if cfg.relative_delay_jitter_us:
            cross_ext[i] = np.interp(extended_time - event_delay[i], extended_time,
                                     cross_ext[i], left=0.0, right=0.0)
    cross_ext *= event_gain[:, None]
    if cfg.saturation is not None:
        cross_ext = cfg.saturation * np.tanh(cross_ext / cfg.saturation)
    crop = slice(guard, guard + cfg.n_samples)
    clean = source_ext[:, crop].copy()
    cross = cross_ext[:, crop].copy()
    shape = clean.shape
    source_noise = _noise(src_rng, shape, cfg.source_noise_rms, cfg.source_noise_rho)
    target_noise = _noise(tgt_rng, shape, cfg.target_noise_rms, cfg.target_noise_rho)
    physical = np.zeros(shape)
    active = phys_rng.random(cfg.n_events) < cfg.target_probability
    physical_amplitudes = phys_rng.uniform(*cfg.target_amplitude_range, cfg.n_events)
    physical_times = phys_rng.uniform(*cfg.target_time_range_us, cfg.n_events)
    if cfg.target_locked:
        physical_times = peaks + cfg.target_delay_us
        # Amplitude correlation deliberately violates a clean training selection.
        physical_amplitudes *= amplitudes / np.mean(cfg.amplitude_range)
    for i in np.flatnonzero(active):
        physical[i] = physical_amplitudes[i] * _double_exp(
            time - physical_times[i], cfg.target_rise_us, cfg.target_decay_us)
    indices = split_rng.permutation(cfg.n_events)
    n_train, n_val = int(0.6 * cfg.n_events), int(0.2 * cfg.n_events)
    data = dict(
        time_us=time, source=clean + source_noise, target=physical + cross + target_noise,
        source_clean=clean, source_noise=source_noise, crosstalk_true=cross,
        target_physical=physical, target_noise=target_noise, target_ideal=physical + target_noise,
        kernel_true=kernel, kernel_time_us=np.arange(kernel.size) * cfg.dt_us,
        source_amplitude=amplitudes, source_peak_us=peaks, event_gain=event_gain,
        event_delay_us=event_delay, target_active=active,
        train_index=indices[:n_train], val_index=indices[n_train:n_train + n_val],
        test_index=indices[n_train + n_val:],
        # Rows before this mask need prehistory unavailable in the saved source.
        valid_mask=np.arange(cfg.n_samples) >= kernel.size - 1,
        config_json=np.array(json.dumps(asdict(cfg))),
        kernel_supplied=np.array(supplied_kernel),
    )
    return data


def plot_pair(data, title='Synthetic waveform pairs', n_show=20, axes=None):
    """Plot measured ensembles, with the same color identifying each event."""
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.7), sharex=True,
                                 gridspec_kw={'hspace': 0.08}, layout='constrained')
    else:
        fig = axes[0].figure
    time = data['time_us']
    n_show = min(n_show, len(data['source']))
    colors = plt.get_cmap('tab20')(np.arange(n_show) % 20)
    for i, color in enumerate(colors):
        axes[0].plot(time, data['source'][i], color=color, lw=0.85, alpha=0.85)
        axes[1].plot(time, data['target'][i], color=color, lw=0.85, alpha=0.80)
    # Mean truth is illustrative; it is never an estimated correction.
    axes[1].plot(time, data['crosstalk_true'][:n_show].mean(axis=0), color='black',
                 ls='--', lw=1.3, label='Mean cross-talk (truth)', zorder=5)
    axes[0].set_ylabel('Source (a.u.)')
    axes[1].set_ylabel('Target (a.u.)')
    axes[1].set_xlabel(r'Time ($\mu$s)')
    axes[1].legend(title=title,loc='lower right',framealpha=0.9)
    for ax in axes:
        ax.axhline(0, color='0.4', lw=0.5, zorder=0)
        ax.tick_params(direction='in', top=True, right=True)
        ax.set_xlim(time[0], time[-1])
    return fig


def save_case(name, cfg, outdir, n_show=20):
    """Save NPZ data (allow_pickle=False), explicit JSON settings and PDF/PNG."""
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = generate_pairs(cfg)
    np.savez_compressed(outdir / f'{name}.npz', **data)
    (outdir / f'{name}.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    fig = plot_pair(data, TITLES.get(name, name), n_show)
    fig.savefig(outdir / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(outdir / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=['all', *PRESETS], default='all')
    parser.add_argument('--out', type=Path, default=Path('examples'))
    parser.add_argument('--config', type=Path, help='JSON field overrides for Config')
    parser.add_argument('--n-events', type=int)
    parser.add_argument('--n-show', type=int, default=20)
    parser.add_argument('--strength', type=float)
    parser.add_argument('--polarity', type=int, choices=[-1, 1])
    parser.add_argument('--kind', choices=['direct', 'lowpass', 'bipolar', 'difference', 'ringing', 'echo'])
    parser.add_argument('--delay-us', type=float)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.n_show < 1:
        parser.error('--n-show must be positive')
    overrides = json.loads(args.config.read_text()) if args.config else {}
    for name in ('n_events', 'strength', 'polarity', 'kind', 'delay_us'):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    names = list(PRESETS) if args.case == 'all' else [args.case]
    for index, name in enumerate(names):
        changes = dict(overrides)
        if args.seed is not None:
            changes['seed'] = args.seed + index
        cfg = replace(PRESETS[name], **changes)
        save_case(name, cfg, args.out, args.n_show)
        print(f'{name}: {cfg.n_events} waveform pairs -> {args.out}')


if __name__ == '__main__':
    main()
