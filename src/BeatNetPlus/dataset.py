# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# PyTorch Datasets for BeatNet+ training.
# Supports multiple audio source modes for the three training strategies:
#   - Generic dual-branch: mix (main) + drumless_mix (aux)
#   - Auxiliary Freezing: vocal/drumless (student) + mix (teacher)
#   - Guided Fine-Tuning: vocal + fading accompaniment (single branch)
#
# Expected data directory structure (created by prepare_data.py):
#   {data_dir}/{DATASET}/tracks/{DATASET}#{split}#{track}.pkl
#   {data_dir}/{DATASET}/tracks_list.pkl
#
# Each pickle contains:
#   feats_mix: (288, T)          — features from full mixture
#   feats_drumless: (288, T)     — features from drumless mix (or None)
#   feats_vocal: (288, T)        — features from vocal stem (or None)
#   feats_drums: (288, T)        — features from drum stem (or None)
#   times: (T,)
#   ground_truth: (3, T)         — one-hot [beat, downbeat, non-beat]

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class BeatNetPlusDataset(Dataset):
    """Unified dataset for all BeatNet+ training modes.

    Parameters
    ----------
    track_ids : list of str
    tracks_dirs : dict mapping dataset name -> tracks directory path
    main_audio : str
        Audio source for main/student branch: 'mix', 'drumless_mix', 'vocal', 'drums'
    aux_audio : str or None
        Audio source for auxiliary/teacher branch. None for single-branch modes.
    seq_len : int or None
        Frames to crop per sample. None = full track.
    epoch : int
        Current epoch (used for GF decay scheduling).
    gf_decay_rate : float
        Guided fine-tuning decay rate. Only used when accompaniment_audio is set.
    accompaniment_audio : str or None
        Audio to blend with main_audio, fading over epochs (GF mode).
    seed : int
    """

    FEAT_KEYS = {
        'mix': 'feats_mix',
        'drumless_mix': 'feats_drumless',
        'vocal': 'feats_vocal',
        'drums': 'feats_drums',
    }

    def __init__(self, track_ids, tracks_dirs, main_audio='mix', aux_audio=None,
                 seq_len=None, epoch=0, gf_decay_rate=0.0, accompaniment_audio=None,
                 seed=42):
        self.track_ids = track_ids
        self.tracks_dirs = tracks_dirs
        self.main_audio = main_audio
        self.aux_audio = aux_audio
        self.seq_len = seq_len
        self.epoch = epoch
        self.gf_decay_rate = gf_decay_rate
        self.accompaniment_audio = accompaniment_audio
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        dataset_name = track_id.split('#')[0]
        tracks_dir = self.tracks_dirs[dataset_name]

        pkl_path = os.path.join(tracks_dir, track_id + '.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        main_key = self.FEAT_KEYS[self.main_audio]
        main_feats = data.get(main_key)
        if main_feats is None:
            main_feats = data['feats_mix']  # fallback to mix

        gt = data['ground_truth']
        times = data['times']

        # Guided fine-tuning: blend target with fading accompaniment
        if self.accompaniment_audio and self.gf_decay_rate > 0:
            acc_key = self.FEAT_KEYS[self.accompaniment_audio]
            acc_feats = data.get(acc_key)
            if acc_feats is not None:
                alpha = max(0.0, 1.0 - self.epoch * self.gf_decay_rate)
                main_feats = main_feats + alpha * acc_feats

        # Auxiliary branch features
        aux_feats = None
        if self.aux_audio:
            aux_key = self.FEAT_KEYS[self.aux_audio]
            aux_feats = data.get(aux_key)
            if aux_feats is None:
                aux_feats = data['feats_mix']

        # Crop or pad
        num_frames = main_feats.shape[-1]
        if self.seq_len is not None:
            if num_frames <= self.seq_len:
                pad = self.seq_len - num_frames
                main_feats = np.pad(main_feats, ((0, 0), (0, pad)))
                times = np.pad(times, (0, pad))
                gt = np.pad(gt, ((0, 0), (0, pad)))
                gt[2, num_frames:] = 1
                if aux_feats is not None:
                    aux_feats = np.pad(aux_feats, ((0, 0), (0, pad)))
            else:
                start = self.rng.randint(0, num_frames - self.seq_len)
                end = start + self.seq_len
                main_feats = main_feats[..., start:end]
                times = times[start:end]
                gt = gt[..., start:end]
                if aux_feats is not None:
                    aux_feats = aux_feats[..., start:end]

        result = {
            'main_feats': torch.from_numpy(main_feats.copy().astype(np.float32)),
            'times': torch.from_numpy(times.copy().astype(np.float32)),
            'ground_truth': torch.from_numpy(gt.copy().astype(np.float32)),
        }
        if aux_feats is not None:
            result['aux_feats'] = torch.from_numpy(aux_feats.copy().astype(np.float32))

        return result


def build_datasets(config):
    """Build train, validation, and test datasets from prepared data.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : BeatNetPlusDataset
    """
    data_dir = config['data_dir']
    ds_config = config['datasets']
    train_datasets = ds_config.get('train', [])
    test_datasets = ds_config.get('test', [])
    dataset_weights = config.get('dataset_weights', {})
    train_val_split = config.get('train_val_split', 0.9)
    seq_len = config.get('seq_len', 750)
    seed = config.get('seed', 42)
    main_audio = config.get('main_audio', 'mix')
    aux_audio = config.get('aux_audio', None)
    gf_decay_rate = config.get('gf_decay_rate', 0.0)
    accompaniment_audio = config.get('accompaniment_audio', None)

    rng = np.random.RandomState(seed)
    train_ids, val_ids, test_ids = [], [], []
    tracks_dirs = {}

    for ds_name in train_datasets:
        ds_dir = os.path.join(data_dir, ds_name)
        tracks_dir = os.path.join(ds_dir, 'tracks')
        tracks_dirs[ds_name] = tracks_dir

        manifest_path = os.path.join(ds_dir, 'tracks_list.pkl')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run prepare_data.py first for {ds_name}.")

        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)

        all_ids = []
        for split_ids in tracks_list.values():
            all_ids.extend(split_ids)

        rng.shuffle(all_ids)
        n_train = int(len(all_ids) * train_val_split)
        split_train = all_ids[:n_train]
        split_val = all_ids[n_train:]

        weight = dataset_weights.get(ds_name, 1)
        for _ in range(weight):
            train_ids.extend(split_train)
        val_ids.extend(split_val)

    for ds_name in test_datasets:
        ds_dir = os.path.join(data_dir, ds_name)
        tracks_dir = os.path.join(ds_dir, 'tracks')
        tracks_dirs[ds_name] = tracks_dir

        manifest_path = os.path.join(ds_dir, 'tracks_list.pkl')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run prepare_data.py first for {ds_name}.")

        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)
        for split_ids in tracks_list.values():
            test_ids.extend(split_ids)

    print(f"Dataset splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    common_kwargs = dict(tracks_dirs=tracks_dirs, main_audio=main_audio, seed=seed)

    train_dataset = BeatNetPlusDataset(
        train_ids, seq_len=seq_len, aux_audio=aux_audio,
        gf_decay_rate=gf_decay_rate, accompaniment_audio=accompaniment_audio,
        **common_kwargs)
    val_dataset = BeatNetPlusDataset(
        val_ids, seq_len=None, aux_audio=aux_audio, **common_kwargs)
    test_dataset = BeatNetPlusDataset(
        test_ids, seq_len=None, aux_audio=aux_audio, **common_kwargs)

    return train_dataset, val_dataset, test_dataset