# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# Data preparation for BeatNet+ training.
# Extracts LOG_SPECT features from multiple audio sources (mix, drumless, vocal, drums)
# and saves per-track pickle files with beat/downbeat annotations.
#
# Usage:
#   python -m BeatNetPlus.prepare_data --raw_dir /path/to/raw --dataset BALLROOM GTZAN
#   python -m BeatNetPlus.prepare_data --raw_dir /path/to/raw --dataset MUSDB18 --has_stems
#
# Expected raw directory structure:
#   Without stems (--has_stems not set):
#     {raw_dir}/{dataset_lower}/audio/{split}/{track}.wav
#     {raw_dir}/{dataset_lower}/annotations/{track}.beats
#     Demucs will be run to generate stems automatically.
#
#   With stems (--has_stems):
#     {raw_dir}/{dataset_lower}/audio/{split}/{track}/mix.wav
#     {raw_dir}/{dataset_lower}/audio/{split}/{track}/drums.wav
#     {raw_dir}/{dataset_lower}/audio/{split}/{track}/vocal.wav (or vocals.wav)
#     {raw_dir}/{dataset_lower}/audio/{split}/{track}/other.wav
#     {raw_dir}/{dataset_lower}/audio/{split}/{track}/bass.wav
#     {raw_dir}/{dataset_lower}/annotations/{track}.beats
#
# Output per-track pickle:
#   feats_mix: (288, T), feats_drumless: (288, T), feats_vocal: (288, T),
#   feats_drums: (288, T), times: (T,), ground_truth: (3, T)

import argparse
import os
import pickle
import sys
from collections import defaultdict

import librosa
import numpy as np
import yaml

from BeatNetPlus.log_spect import LOG_SPECT


def parse_beats_file(label_path):
    """Parse a .beats annotation file into beat and downbeat time arrays."""
    beats, downs = [], []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            time_sec = float(parts[0])
            beat_num = int(float(parts[1]))
            if beat_num == 1:
                downs.append(time_sec)
            else:
                beats.append(time_sec)
    return np.asarray(beats), np.asarray(downs)


def build_ground_truth(beats, downs, num_frames, sample_rate, hop_length):
    """Build (3, num_frames) one-hot ground truth: [beat, downbeat, non-beat]."""
    gt = np.zeros((3, num_frames), dtype=np.float32)

    if len(beats) > 0:
        beat_frames = librosa.time_to_frames(beats, sr=sample_rate, hop_length=hop_length)
        beat_frames = beat_frames[beat_frames < num_frames]
        gt[0, beat_frames] = 1

    if len(downs) > 0:
        down_frames = librosa.time_to_frames(downs, sr=sample_rate, hop_length=hop_length)
        down_frames = down_frames[down_frames < num_frames]
        gt[1, down_frames] = 1
        gt[0, down_frames] = 0  # downbeats are not also beats in row 0

    gt[2, np.sum(gt, axis=0) == 0] = 1
    assert int(np.sum(gt)) == num_frames
    return gt


def extract_features(audio, feature_extractor):
    """Extract features from audio, returning (dim, T) array."""
    if audio is None:
        return None
    feats = feature_extractor.process_audio(audio)  # returns (dim, T)
    return feats.astype(np.float32)


def run_demucs(audio_path, output_dir):
    """Run Demucs source separation on an audio file.

    Returns dict of stem paths: {mix, drums, vocal, bass, other}
    """
    import subprocess
    cmd = ['python', '-m', 'demucs', '--two-stems=drums', '-o', output_dir, audio_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try the full 4-stem separation
        cmd = ['python', '-m', 'demucs', '-o', output_dir, audio_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"    WARNING: Demucs not available, skipping source separation for {audio_path}")
            return None

    # Find the output directory (demucs creates model_name/track_name/)
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    for model_dir in os.listdir(output_dir):
        stem_dir = os.path.join(output_dir, model_dir, track_name)
        if os.path.isdir(stem_dir):
            stems = {}
            for f in os.listdir(stem_dir):
                name = os.path.splitext(f)[0].lower()
                stems[name] = os.path.join(stem_dir, f)
            return stems
    return None


def discover_splits(audio_dir):
    """Discover genre/split subdirectories under an audio directory."""
    splits = []
    for entry in sorted(os.listdir(audio_dir)):
        if os.path.isdir(os.path.join(audio_dir, entry)):
            splits.append(entry)
    return splits


def find_annotation(annotations_dir, track_name):
    """Find .beats annotation file for a track."""
    for ext in ['.beats', '.beat']:
        path = os.path.join(annotations_dir, track_name + ext)
        if os.path.exists(path):
            return path
    for f in os.listdir(annotations_dir):
        base = os.path.splitext(f)[0]
        if base.lower() == track_name.lower() and f.endswith(('.beats', '.beat')):
            return os.path.join(annotations_dir, f)
    return None


def prepare_dataset(dataset_name, raw_dir, data_dir, feature_extractor,
                    sample_rate, hop_length, has_stems=False, run_separation=False):
    """Prepare a single dataset."""
    dataset_lower = dataset_name.lower()
    dataset_raw = os.path.join(raw_dir, dataset_lower)
    audio_dir = os.path.join(dataset_raw, 'audio')
    annotations_dir = os.path.join(dataset_raw, 'annotations')

    if not os.path.isdir(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return

    tracks_dir = os.path.join(data_dir, dataset_name, 'tracks')
    os.makedirs(tracks_dir, exist_ok=True)
    demucs_dir = os.path.join(data_dir, dataset_name, 'demucs_cache')

    splits = discover_splits(audio_dir)
    if not splits:
        splits = ['default']

    tracks_list = defaultdict(list)
    total_processed = 0

    for split in splits:
        split_audio_dir = audio_dir if split == 'default' else os.path.join(audio_dir, split)

        if has_stems:
            # Stems mode: each track is a directory with mix.wav, drums.wav, etc.
            track_dirs = sorted([d for d in os.listdir(split_audio_dir)
                                 if os.path.isdir(os.path.join(split_audio_dir, d))])
            entries = [(d, None) for d in track_dirs]
        else:
            # Single-file mode: each track is a .wav/.mp3 file
            files = sorted([f for f in os.listdir(split_audio_dir)
                            if f.endswith(('.wav', '.mp3', '.flac'))])
            entries = [(os.path.splitext(f)[0], f) for f in files]

        print(f"  Split '{split}': {len(entries)} tracks")

        for track_name, audio_file in entries:
            track_id = f"{dataset_name}#{split}#{track_name}"
            label_path = find_annotation(annotations_dir, track_name) if os.path.isdir(annotations_dir) else None
            if label_path is None:
                continue

            # Load audio sources
            if has_stems:
                stem_dir = os.path.join(split_audio_dir, track_name)
                mix_path = _find_stem(stem_dir, ['mix', 'mixture'])
                drums_path = _find_stem(stem_dir, ['drums', 'drum'])
                vocal_path = _find_stem(stem_dir, ['vocals', 'vocal', 'voice'])
                bass_path = _find_stem(stem_dir, ['bass'])
                other_path = _find_stem(stem_dir, ['other', 'others', 'accompaniment'])

                mix_audio = _load_audio(mix_path, sample_rate)
                drums_audio = _load_audio(drums_path, sample_rate)
                vocal_audio = _load_audio(vocal_path, sample_rate)
                # drumless = everything except drums
                parts = [_load_audio(p, sample_rate) for p in [bass_path, vocal_path, other_path]]
                parts = [p for p in parts if p is not None]
                if parts:
                    min_len = min(len(p) for p in parts)
                    drumless_audio = sum(p[:min_len] for p in parts)
                else:
                    drumless_audio = None
            else:
                wav_path = os.path.join(split_audio_dir, audio_file)
                mix_audio = _load_audio(wav_path, sample_rate)
                drums_audio = None
                vocal_audio = None
                drumless_audio = None

                if run_separation and mix_audio is not None:
                    os.makedirs(demucs_dir, exist_ok=True)
                    stems = run_demucs(wav_path, demucs_dir)
                    if stems:
                        drums_audio = _load_audio(stems.get('drums'), sample_rate)
                        vocal_audio = _load_audio(stems.get('vocals') or stems.get('vocal'), sample_rate)
                        no_drums = stems.get('no_drums') or stems.get('other')
                        if no_drums:
                            drumless_audio = _load_audio(no_drums, sample_rate)

            if mix_audio is None:
                continue

            # Extract features
            feats_mix = extract_features(mix_audio, feature_extractor)
            num_frames = feats_mix.shape[1]

            feats_drumless = _safe_extract(drumless_audio, feature_extractor, num_frames)
            feats_vocal = _safe_extract(vocal_audio, feature_extractor, num_frames)
            feats_drums = _safe_extract(drums_audio, feature_extractor, num_frames)

            # Parse annotations and build ground truth
            beats, downs = parse_beats_file(label_path)
            gt = build_ground_truth(beats, downs, num_frames, sample_rate, hop_length)

            if gt[0].sum() + gt[1].sum() < 4 or gt[1].sum() < 2:
                continue

            times = librosa.frames_to_time(np.arange(num_frames), sr=sample_rate,
                                            hop_length=hop_length).astype(np.float32)

            data = {
                'feats_mix': feats_mix,
                'feats_drumless': feats_drumless,
                'feats_vocal': feats_vocal,
                'feats_drums': feats_drums,
                'times': times,
                'ground_truth': gt,
            }
            with open(os.path.join(tracks_dir, track_id + '.pkl'), 'wb') as f:
                pickle.dump(data, f)

            tracks_list[split].append(track_id)
            total_processed += 1

    manifest_path = os.path.join(data_dir, dataset_name, 'tracks_list.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(dict(tracks_list), f)

    print(f"  Done: {total_processed} tracks. Manifest: {manifest_path}")


def _find_stem(stem_dir, names):
    """Find a stem file by trying multiple name variants."""
    for name in names:
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(stem_dir, name + ext)
            if os.path.exists(path):
                return path
    return None


def _load_audio(path, sample_rate):
    """Load audio file, returning None if path is None or file doesn't exist."""
    if path is None or not os.path.exists(path):
        return None
    audio, _ = librosa.load(path, sr=sample_rate)
    return audio


def _safe_extract(audio, feature_extractor, target_frames):
    """Extract features, aligning to target_frames. Returns None if audio is None."""
    if audio is None:
        return None
    feats = extract_features(audio, feature_extractor)
    if feats.shape[1] > target_frames:
        feats = feats[:, :target_frames]
    elif feats.shape[1] < target_frames:
        feats = np.pad(feats, ((0, 0), (0, target_frames - feats.shape[1])))
    return feats


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for BeatNet+ training')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    parser.add_argument('--raw_dir', type=str, required=True, help='Raw dataset root directory')
    parser.add_argument('--dataset', type=str, nargs='+', default=None, help='Dataset names')
    parser.add_argument('--data_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--has_stems', action='store_true',
                        help='Datasets already have source-separated stems')
    parser.add_argument('--run_demucs', action='store_true',
                        help='Run Demucs source separation on single-file datasets')
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    sample_rate = config.get('sample_rate', 22050)
    hop_length = config.get('hop_length', 441)
    win_length = config.get('win_length', 1764)
    n_bands = config.get('n_bands', 24)
    data_dir = args.data_dir or config.get('data_dir', './data')

    if args.dataset:
        datasets = args.dataset
    else:
        ds_config = config.get('datasets', {})
        datasets = ds_config.get('train', []) + ds_config.get('test', [])

    if not datasets:
        print("ERROR: No datasets specified.")
        sys.exit(1)

    feature_extractor = LOG_SPECT(
        sample_rate=sample_rate, win_length=win_length, hop_size=hop_length,
        n_bands=[n_bands], mode='online')

    print(f"BeatNet+ data preparation")
    print(f"Features: LOG_SPECT (dim={config.get('feature_dim', 288)})")
    print(f"Sample rate: {sample_rate}, hop: {hop_length}, win: {win_length}")
    print(f"Output: {data_dir}\n")

    for ds in datasets:
        print(f"Processing: {ds}")
        prepare_dataset(ds, args.raw_dir, data_dir, feature_extractor,
                        sample_rate, hop_length, has_stems=args.has_stems,
                        run_separation=args.run_demucs)
        print()

    print("All done.")


if __name__ == '__main__':
    main()