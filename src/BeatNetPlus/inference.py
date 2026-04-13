# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# BeatNet+ inference handler.
# Supports the same four modes as BeatNet: stream, realtime, online, offline.
# Uses a single BeatNetPlusBranch with pre-trained weights.
#
# Usage:
#   from BeatNetPlus.inference import BeatNetPlusInference
#   estimator = BeatNetPlusInference('weights.pt', mode='online', inference_model='PF')
#   output = estimator.process("audio_file.wav")

import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from madmom.features import DBNDownBeatTrackingProcessor

from BeatNetPlus.log_spect import LOG_SPECT
from BeatNetPlus.model import BeatNetPlusBranch
from BeatNetPlus.particle_filtering_cascade import particle_filter_cascade


class BeatNetPlusInference:
    """Main BeatNet+ inference handler.

    Parameters
    ----------
    weights : str
        Path to pre-trained model weights (.pt file).
    mode : str
        'stream', 'realtime', 'online', or 'offline'.
    inference_model : str
        'PF' (particle filtering, causal) or 'DBN' (dynamic Bayesian network, non-causal).
    plot : list
        Plot options: 'activations', 'beat_particles', 'downbeat_particles'.
    device : str
        'cpu', 'cuda', 'cuda:0', 'mps', etc.
    dim_in : int
        Feature dimension (default 288 for BeatNet+).
    num_cells : int
        LSTM hidden size (default 150).
    num_layers : int
        LSTM layers (default 4 for BeatNet+).
    """

    def __init__(self, weights, mode='online', inference_model='PF', plot=[],
                 device='cpu', dim_in=288, num_cells=150, num_layers=4):
        self.mode = mode
        self.inference_model = inference_model
        self.plot = plot
        self.device = device

        self.sample_rate = 22050
        self.hop_length = 441   # 20ms
        self.win_length = 1764  # 80ms

        # Feature extractor
        self.proc = LOG_SPECT(
            sample_rate=self.sample_rate, win_length=self.win_length,
            hop_size=self.hop_length, n_bands=[24], mode=self.mode)

        # Inference decoder
        if inference_model == 'PF':
            self.estimator = particle_filter_cascade(
                beats_per_bar=[], fps=50, plot=self.plot, mode=self.mode)
        elif inference_model == 'DBN':
            self.estimator = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4], fps=50, observation_lambda=16)
        else:
            raise ValueError(f'inference_model must be "PF" or "DBN", got "{inference_model}"')

        # Neural network
        self.model = BeatNetPlusBranch(dim_in, num_cells, num_layers, device)
        self.model.load_state_dict(
            torch.load(weights, map_location=device), strict=False)
        self.model.eval()

        # Streaming state
        if self.mode == 'stream':
            self.stream_window = np.zeros(
                self.win_length + 2 * self.hop_length, dtype=np.float32)
            import pyaudio
            self.stream = pyaudio.PyAudio().open(
                format=pyaudio.paFloat32, channels=1, rate=self.sample_rate,
                input=True, frames_per_buffer=self.hop_length)

    def process(self, audio_path=None):
        """Process audio and return beats/downbeats.

        Returns
        -------
        output : np.ndarray, shape (num_beats, 2)
            Column 0: beat time in seconds.
            Column 1: beat type (1 = downbeat, 2 = regular beat).
        """
        if self.mode == 'stream':
            return self._process_stream()
        elif self.mode == 'realtime':
            return self._process_realtime(audio_path)
        elif self.mode == 'online':
            return self._process_online(audio_path)
        elif self.mode == 'offline':
            return self._process_offline(audio_path)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def _load_audio(self, audio_path):
        if isinstance(audio_path, str):
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        elif len(np.shape(audio_path)) > 1:
            audio = np.mean(audio_path, axis=1)
        else:
            audio = audio_path
        return audio

    def _get_activations(self, feats):
        """Run NN on features, return (T, 2) activation array."""
        feats_t = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model.inference_forward(feats_t)[0]  # (3, T)
            probs = F.softmax(logits, dim=0)                   # softmax over classes
        probs = probs.cpu().numpy()
        return np.transpose(probs[:2, :])

    def _process_online(self, audio_path):
        audio = self._load_audio(audio_path)
        feats = self.proc.process_audio(audio).T  # (T, dim)
        feats = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model.inference_forward(feats)[0]  # (3, T)
            probs = F.softmax(logits, dim=0)                  # softmax over classes
        preds = probs.cpu().numpy()
        preds = np.transpose(preds[:2, :])

        if self.inference_model == 'PF':
            return self.estimator.process(preds)
        else:
            try:
                return self.estimator(preds)
            except (ValueError, IndexError) as e:
                # madmom DBN can fail on short files or with numpy compat issues
                import warnings
                warnings.warn(f"DBN decoding failed ({e}), returning empty output. "
                              f"Consider using inference_model='PF' instead.")
                return np.zeros((0, 2))

    def _process_offline(self, audio_path):
        if self.inference_model != 'DBN':
            raise ValueError('Offline mode requires inference_model="DBN"')
        return self._process_online(audio_path)

    def _process_realtime(self, audio_path):
        if self.inference_model != 'PF':
            raise ValueError('Realtime mode requires inference_model="PF"')
        audio = self._load_audio(audio_path)
        self.model.reset_hidden()
        counter = 0
        output = None
        total_frames = round(len(audio) / self.hop_length)

        while counter < total_frames:
            if counter < 2:
                pred = np.zeros([1, 2])
            else:
                start = self.hop_length * (counter - 2)
                end = self.hop_length * counter + self.win_length
                chunk = audio[start:end]
                feats = self.proc.process_audio(chunk).T[-1]
                feats = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(feats)[0]
                    probs = F.softmax(logits, dim=0)
                pred = np.transpose(probs.cpu().numpy()[:2, :])
            output = self.estimator.process(pred)
            counter += 1
        return output

    def _process_stream(self):
        if self.inference_model != 'PF':
            raise ValueError('Stream mode requires inference_model="PF"')
        self.model.reset_hidden()
        counter = 0
        while self.stream.is_active():
            hop = self.stream.read(self.hop_length)
            hop = np.frombuffer(hop, dtype=np.float32)
            self.stream_window = np.append(
                self.stream_window[self.hop_length:], hop)
            if counter < 5:
                pred = np.zeros([1, 2])
            else:
                feats = self.proc.process_audio(self.stream_window).T[-1]
                feats = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(feats)[0]
                    probs = F.softmax(logits, dim=0)
                pred = np.transpose(probs.cpu().numpy()[:2, :])
            output = self.estimator.process(pred)
            counter += 1