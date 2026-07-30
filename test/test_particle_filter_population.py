"""
Regression test for particle population bookkeeping in the PF cascade.

Both correction steps inject extra first-state particles and are meant to drop
the same number again after resampling. Neither removal did, so both
populations grew for as long as the tracker ran. Per-frame cost is
O(population), so a long session kept getting slower.

Uses the repo's own test clip (10 s at 120 BPM), tracked on a loop to stand in
for a long session.

Usage:
    python test/test_particle_filter_population.py [minutes]
    python -m pytest test/test_particle_filter_population.py -v
"""

import os
import sys
import time

import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from BeatNetPlus.inference import BeatNetPlusInference

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, 'test_data', '808kick120bpm.mp3')
WEIGHTS = os.path.join(HERE, os.pardir, 'src', 'BeatNetPlus', 'models', 'generic_weights.pt')
CLIP_SECONDS = 10


def track(minutes, report_every=1):
    """Track the clip on a loop, reporting population sizes as it goes."""
    estimator = BeatNetPlusInference(
        WEIGHTS, mode='online', inference_model='PF', device='cpu')
    audio = librosa.load(AUDIO, sr=estimator.sample_rate)[0]
    pf = estimator.estimator
    report = []
    for minute in range(report_every, minutes + 1, report_every):
        elapsed = time.perf_counter()
        for _ in range(report_every * 60 // CLIP_SECONDS):
            estimator.process(audio)
        elapsed = time.perf_counter() - elapsed
        report.append((minute, len(pf.particles), len(pf.down_particles), elapsed))
    return pf, report


def test_populations_stay_nominal():
    pf, _ = track(1)
    assert len(pf.particles) == pf.particle_size
    assert len(pf.down_particles) == pf.down_particle_size


if __name__ == '__main__':
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pf, report = track(minutes, report_every=5 if minutes > 10 else 1)
    print(f'{"audio":>6} {"beat particles":>16} {"downbeat particles":>20} {"wall clock":>12}')
    for minute, beat, down, elapsed in report:
        print(f'{minute:>5}m {beat:>16} {down:>20} {elapsed:>11.1f}s')
    print(f'\nnominal: {pf.particle_size} beat / {pf.down_particle_size} downbeat particles')
