# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# Abstract base class for feature extraction modules.
# Shared between BeatNet and BeatNet+.

from abc import abstractmethod
import numpy as np
import librosa


class FeatureModule(object):
    """Generic music feature extraction module wrapper."""

    def __init__(self, sample_rate, hop_length, num_channels=1, decibels=True):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_channels = num_channels
        self.decibels = decibels

    def get_expected_frames(self, audio):
        return 1 + len(audio) // self.hop_length

    def get_sample_range(self, num_frames):
        max_samples = num_frames * self.hop_length - 1
        min_samples = max(1, max_samples - self.hop_length + 1)
        return np.arange(min_samples, max_samples + 1)

    @abstractmethod
    def process_audio(self, audio):
        return NotImplementedError

    def to_decibels(self, feats):
        return librosa.core.amplitude_to_db(feats, ref=np.max)

    def post_proc(self, feats):
        if self.decibels:
            feats = self.to_decibels(feats)
            feats = feats / 80
            feats = feats + 1
        feats = np.expand_dims(feats, axis=0)
        return feats

    def get_times(self, audio):
        num_frames = self.get_expected_frames(audio)
        frame_idcs = np.arange(num_frames + 1)
        return librosa.frames_to_time(frames=frame_idcs, sr=self.sample_rate,
                                       hop_length=self.hop_length)

    def get_sample_rate(self):
        return self.sample_rate

    def get_hop_length(self):
        return self.hop_length

    def get_num_channels(self):
        return self.num_channels

    @classmethod
    def features_name(cls):
        return cls.__name__