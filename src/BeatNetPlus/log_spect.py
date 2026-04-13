# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# Log-magnitude spectrogram feature extraction for BeatNet+.
# Differences from BeatNet: window length is 80ms (1764 samples) instead of 64ms (1411),
# producing 288-dim features instead of 272-dim.

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor
from BeatNetPlus.common import *


class LOG_SPECT(FeatureModule):
    """Log-magnitude spectrogram + spectral difference features.

    Pipeline: Audio -> Signal -> Frames -> STFT -> FilteredSpectrogram -> Log -> Difference
    Output dimension: 2 * num_filtered_bands (spectrogram + first-order difference)
    """

    def __init__(self, num_channels=1, sample_rate=22050, win_length=1764, hop_size=441,
                 n_bands=[24], mode='online'):
        sig = SignalProcessor(num_channels=num_channels, win_length=win_length,
                              sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        multi = ParallelProcessor([])
        frame_sizes = [win_length]
        num_bands = n_bands
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            if mode in ('online', 'offline'):
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size)
            else:  # stream, realtime
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size,
                                               num_frames=4)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        self.pipe = SequentialProcessor((sig, multi, np.hstack))

    def process_audio(self, audio):
        feats = self.pipe(audio)
        return feats.T