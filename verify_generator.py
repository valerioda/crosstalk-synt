"""Scientific consistency checks; run with: python verify_generator.py."""
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import fftconvolve
from synthetic_crosstalk import Config, PRESETS, generate_pairs, make_kernel, _noise


def main():
    for name, config in PRESETS.items():
        cfg = replace(config, n_events=24)
        data = generate_pairs(cfg)
        repeated = generate_pairs(cfg)
        for key in data:
            assert_array_equal(data[key], repeated[key])
        assert data['source'].shape == (cfg.n_events, cfg.n_samples)
        assert_allclose(data['target'], data['target_physical'] + data['crosstalk_true'] + data['target_noise'])
        assert_allclose(data['target'] - data['crosstalk_true'], data['target_ideal'], atol=1e-12)
        assert_allclose(data['source'] - data['source_clean'], data['source_noise'], atol=1e-12)
        oracle = fftconvolve(data['source_clean'], data['kernel_true'][None, :], axes=1)[:, :cfg.n_samples]
        valid = data['valid_mask']
        assert_allclose(oracle[:, valid], data['crosstalk_true'][:, valid], atol=1e-10)
        # Sign, amplitude and noise changes must not silently change pulse/noise draws.
        opposite = generate_pairs(replace(cfg, polarity=-cfg.polarity))
        assert_array_equal(opposite['source'], data['source'])
        assert_array_equal(opposite['target_noise'], data['target_noise'])
        assert_allclose(opposite['crosstalk_true'], -data['crosstalk_true'], atol=1e-12)
        scaled = generate_pairs(replace(cfg, strength=2 * cfg.strength))
        assert_allclose(scaled['crosstalk_true'], 2 * data['crosstalk_true'], atol=1e-12)
        noisy = generate_pairs(replace(cfg, source_noise_rms=50))
        assert_array_equal(noisy['crosstalk_true'], data['crosstalk_true'])
        assert_array_equal(noisy['target'], data['target'])
        index = np.concatenate([data['train_index'], data['val_index'], data['test_index']])
        assert_array_equal(np.sort(index), np.arange(cfg.n_events))
        assert np.isfinite(data['target']).all()
        print(f'{name}: reproducibility, decomposition, causal convolution, sign, gain, noise and split OK')

    cfg = replace(Config(), n_events=24, kind='direct', delay_us=0.026, strength=0.2)
    kernel = make_kernel(cfg)
    assert_array_equal(np.flatnonzero(kernel), [6, 7])
    assert_allclose(kernel[6], kernel[7], rtol=1e-12)
    assert abs(make_kernel(Config()).sum()) < 1e-12
    assert_allclose(make_kernel(replace(cfg, delay_us=0), [1, 0, -0.5]).shape, (cfg.kernel_taps,))
    no_target = generate_pairs(cfg)
    absolute_kernel = np.array([0.1, -0.04, -0.06])
    fixed = generate_pairs(replace(cfg, source_shape='gaussian'), kernel=absolute_kernel)
    assert_array_equal(fixed['kernel_true'], absolute_kernel)
    assert bool(fixed['kernel_supplied'])
    target = generate_pairs(replace(cfg, target_probability=1.0))
    assert_array_equal(no_target['source'], target['source'])
    assert_array_equal(no_target['crosstalk_true'], target['crosstalk_true'])
    assert target['target_physical'].max() > 0
    for kind in ('gaussian', 'double_exp', 'step'):
        d = generate_pairs(replace(cfg, source_shape=kind, rise_us=0.02, fall_us=0.12))
        assert np.isfinite(d['source']).all()
    for changes in ({'gain_jitter': 0.2}, {'relative_delay_jitter_us': 0.02}, {'saturation': 15.0}):
        stressed = generate_pairs(replace(cfg, **changes))
        assert not np.allclose(stressed['crosstalk_true'], no_target['crosstalk_true'])
    for rho in (-0.6, 0.0, 0.8):
        noise = _noise(np.random.default_rng(1987), (1200, 512), 5.0, rho)
        assert abs(noise.std() - 5.0) < 0.04
        measured_rho = np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1]
        assert abs(measured_rho - rho) < 0.01
    assert not np.any(generate_pairs(replace(cfg, strength=0))['crosstalk_true'])
    print('Fractional delay, custom kernel, alternate pulses, physical activity, stress tests and stationary noise OK')
    print('All generator checks passed. No correction-performance claim is tested or implied.')


if __name__ == '__main__':
    main()
