# BeatNet+: Enhanced Joint Music Beat, Downbeat, Tempo, and Meter Tracking

BeatNet+ extends [BeatNet](https://github.com/mjhydri/BeatNet) with a deeper CRNN architecture and a novel two-branch training strategy that improves robustness across diverse music types, including non-percussive music and isolated singing voices.

**Key improvements over BeatNet:**
- Deeper recurrent block: **4-layer LSTM** (up from 2)
- **Dual-branch training** with MSE latent-matching loss for percussive-invariant representations
- **Auxiliary Freezing (AF)** and **Guided Fine-Tuning (GF)** adaptation strategies for challenging music types
- Source separation (Demucs) integration for training data preparation
- Achieves **80.62** beat F1 and **56.51** downbeat F1 on GTZAN (vs BeatNet's 75.44 / 46.69)

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Pre-trained Models](#pre-trained-models)
- [Inference](#inference)
  - [Online Mode (Causal, Particle Filtering)](#online-mode-causal-particle-filtering)
  - [Offline Mode (Non-Causal, DBN)](#offline-mode-non-causal-dbn)
  - [Realtime Mode (File-Based, Causal)](#realtime-mode-file-based-causal)
  - [Streaming Mode (Microphone)](#streaming-mode-microphone)
  - [Choosing a Model for Your Use Case](#choosing-a-model-for-your-use-case)
- [Training](#training)
  - [Overview: Multi-Step Training Pipeline](#overview-multi-step-training-pipeline)
  - [Step 0: Data Preparation](#step-0-data-preparation)
  - [Step 1: Generic Dual-Branch Training](#step-1-generic-dual-branch-training)
  - [Step 2a: Auxiliary Freezing (AF) Adaptation](#step-2a-auxiliary-freezing-af-adaptation)
  - [Step 2b: Guided Fine-Tuning (GF) Adaptation](#step-2b-guided-fine-tuning-gf-adaptation)
  - [Configuration Reference](#configuration-reference)
  - [Monitoring with TensorBoard](#monitoring-with-tensorboard)
  - [Resuming Training](#resuming-training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Annotation Format](#annotation-format)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Architecture

BeatNet+ is a fully **causal** system (online-capable) consisting of two stages:

**Stage 1 — Neural Network (CRNN):** Processes audio frame-by-frame and outputs beat/downbeat activation probabilities.

```
Audio (22050 Hz)
  -> Log-Magnitude Spectrogram (80ms window, 20ms hop, 30-17kHz, 24 bands/oct)
  -> 288-dim feature vector per frame (filtered spectrogram + spectral difference)
  -> Conv1d(1, 2, kernel=10) + ReLU + MaxPool1d(2)
  -> Linear(278, 150)
  -> 4-layer Unidirectional LSTM (hidden=150)
  -> Linear(150, 3) + Softmax
  -> [P(beat), P(downbeat), P(non-beat)] per frame
```

**Stage 2 — Particle Filtering:** A two-stage cascade particle filter infers beats, downbeats, tempo, and meter from the neural network activations. This is identical to BeatNet's inference algorithm.

**Training architecture (dual-branch):** During training, two identical CRNN branches process the same music piece simultaneously — the main branch sees the full mix, and the auxiliary branch sees the drumless version. An MSE loss ties their latent representations together, forcing the main branch to learn features invariant to percussive content.

---

## Installation

```bash
# Clone and install
git clone https://github.com/mjhydri/BeatNet-Plus.git
cd beatnet_plus
pip install -e .
```

**Dependencies:** numpy, librosa, madmom, torch, scipy, tensorboard, pyyaml, matplotlib.

> **Note on madmom compatibility:** madmom 0.16.1 has known issues with Python >= 3.10 and NumPy >= 1.24. See [Troubleshooting](#troubleshooting) for fixes.

---

## Pre-trained Models

Three pre-trained weight files are included in `src/BeatNetPlus/models/`:

| File | Model | Best For |
|------|-------|----------|
| `generic_weights.pt` | BeatNet+ generic (auxiliary branch) | General-purpose music with any level of percussion |
| `generic_main_weights.pt` | BeatNet+ generic (main branch) | Standard music with clear percussion |
| `af_non_percussive_weights.pt` | AF-adapted student | Non-percussive music, ambient, classical |

**Which weights to use:**
- For **general music** (pop, rock, electronic, jazz): use `generic_weights.pt`
- For **non-percussive or acoustic music** (classical, ambient, solo instruments): use `af_non_percussive_weights.pt`
- For **percussive-heavy music**: use `generic_main_weights.pt`

All weight files are `state_dict`-only format, directly loadable by `BeatNetPlusBranch`.

---

## Inference

BeatNet+ supports four inference modes matching the original BeatNet interface.

### Online Mode (Causal, Particle Filtering)

Processes the entire audio file at once using the causal neural network, then runs particle filtering for beat inference. Produces identical results to realtime mode but runs faster than real-time.

```python
from BeatNetPlus.inference import BeatNetPlusInference

estimator = BeatNetPlusInference(
    'src/BeatNetPlus/models/generic_weights.pt',
    mode='online',
    inference_model='PF',
    device='cpu'        # or 'cuda', 'mps'
)

output = estimator.process("path/to/audio.wav")

# output is a numpy array of shape (num_beats, 2):
#   Column 0: beat time in seconds
#   Column 1: beat type (1 = downbeat, 2 = regular beat)

for time_sec, beat_type in output:
    label = "DOWNBEAT" if beat_type == 1 else "beat"
    print(f"  {time_sec:.3f}s  {label}")
```

### Offline Mode (Non-Causal, DBN)

Uses madmom's Dynamic Bayesian Network for globally optimal beat/downbeat decoding. Slightly more accurate than PF but requires the entire audio upfront (not causal).

```python
estimator = BeatNetPlusInference(
    'src/BeatNetPlus/models/generic_weights.pt',
    mode='offline',
    inference_model='DBN'
)

output = estimator.process("path/to/audio.wav")
```

### Realtime Mode (File-Based, Causal)

Reads an audio file chunk-by-chunk and processes each chunk as it arrives, simulating real-time conditions. Uses particle filtering. The LSTM hidden state is maintained across chunks.

```python
estimator = BeatNetPlusInference(
    'src/BeatNetPlus/models/generic_weights.pt',
    mode='realtime',
    inference_model='PF'
)

output = estimator.process("path/to/audio.wav")
```

### Streaming Mode (Microphone)

Captures live audio from the system microphone and tracks beats in real-time. Requires `pyaudio`.

```python
estimator = BeatNetPlusInference(
    'src/BeatNetPlus/models/generic_weights.pt',
    mode='stream',
    inference_model='PF'
)

# Blocks and processes indefinitely until the stream is stopped
estimator.process()
```

> **Tip:** For best streaming results, ensure the microphone input is as loud as possible. The models are trained on mastered songs — low-volume or reverberant input degrades performance.

### Choosing a Model for Your Use Case

| Use Case | Weights | Mode | Inference |
|----------|---------|------|-----------|
| General beat tracking | `generic_weights.pt` | `online` | `PF` |
| Best offline accuracy | `generic_weights.pt` | `offline` | `DBN` |
| Live performance / DJ | `generic_weights.pt` | `stream` | `PF` |
| Classical / ambient | `af_non_percussive_weights.pt` | `online` | `PF` |
| Vocal melody tracking | `af_non_percussive_weights.pt` | `online` | `PF` |
| Audio analysis pipeline | `generic_weights.pt` | `online` | `PF` |

### Using Custom Trained Weights

After training your own model (see [Training](#training)), load the saved weights:

```python
estimator = BeatNetPlusInference(
    'output/generic/best_model_weights.pt',   # your trained weights
    mode='online',
    inference_model='PF'
)
output = estimator.process("audio.wav")
```

---

## Training

### Overview: Multi-Step Training Pipeline

BeatNet+ uses a multi-step training approach:

```
Step 1: Generic Dual-Branch Training
  Main branch (full mix) + Auxiliary branch (drumless mix)
  Loss = CE_main + CE_aux + λ * MSE(latent_main, latent_aux)
  → Produces the pre-trained BeatNet+ generic model
       |
       |--- Step 2a: Auxiliary Freezing (AF)
       |      Frozen teacher (generic weights) + Student (target domain)
       |      → Adapted model for vocals / non-percussive
       |
       |--- Step 2b: Guided Fine-Tuning (GF)
              Single branch initialized from generic weights
              Accompaniment faded out over epochs
              → Adapted model for vocals / non-percussive
```

### Step 0: Data Preparation

Before training, raw audio and annotations must be converted to pickled feature files.

**Expected raw directory structure:**

For datasets **without** pre-separated stems:
```
raw_datasets/
    ballroom/
        audio/
            ChaChaCha/
                track001.wav
                track002.wav
            Waltz/
                track003.wav
        annotations/
            track001.beats
            track002.beats
            track003.beats
```

For datasets **with** pre-separated stems (e.g., MUSDB18):
```
raw_datasets/
    musdb18/
        audio/
            train/
                track_name/
                    mix.wav        (or mixture.wav)
                    drums.wav
                    vocals.wav     (or vocal.wav)
                    bass.wav
                    other.wav
        annotations/
            track_name.beats
```

**Run data preparation:**

```bash
# Basic: extract features from mix audio only
python -m BeatNetPlus.prepare_data \
    --config src/BeatNetPlus/configs/generic.yaml \
    --raw_dir /path/to/raw_datasets \
    --dataset BALLROOM HAINSWORTH GTZAN ROCK_CORPUS

# With pre-separated stems (MUSDB18, URSing)
python -m BeatNetPlus.prepare_data \
    --raw_dir /path/to/raw_datasets \
    --dataset MUSDB18 URSING \
    --has_stems

# With automatic Demucs source separation (requires demucs installed)
python -m BeatNetPlus.prepare_data \
    --raw_dir /path/to/raw_datasets \
    --dataset BALLROOM \
    --run_demucs

# Custom output directory
python -m BeatNetPlus.prepare_data \
    --raw_dir /path/to/raw --dataset BALLROOM \
    --data_dir /path/to/prepared_data
```

This produces per-track pickle files containing:
- `feats_mix`: (288, T) — log-spectrogram features from full mixture
- `feats_drumless`: (288, T) — features from drumless mix (if available)
- `feats_vocal`: (288, T) — features from vocal stem (if available)
- `feats_drums`: (288, T) — features from drum stem (if available)
- `times`: (T,) — frame timestamps in seconds
- `ground_truth`: (3, T) — one-hot encoding [beat, downbeat, non-beat]

### Step 1: Generic Dual-Branch Training

Trains the core BeatNet+ model with two branches connected by MSE latent-matching loss.

```bash
python -m BeatNetPlus.train --config src/BeatNetPlus/configs/generic.yaml

# With GPU
python -m BeatNetPlus.train --config src/BeatNetPlus/configs/generic.yaml device=cuda

# Override hyperparameters via CLI
python -m BeatNetPlus.train --config src/BeatNetPlus/configs/generic.yaml \
    device=cuda batch_size=64 learning_rate=0.001
```

**What happens:**
1. Both branches are randomly initialized
2. Each batch: main branch receives full mix features, auxiliary branch receives drumless features
3. Loss: `L_CE(main) + L_CE(aux) + 200 * MSE(latent_main, latent_aux)`
4. Cross-entropy class weights: [60, 200, 1] for [beat, downbeat, non-beat]
5. Validates every `checkpoint_every` epochs using particle filtering or DBN
6. Early stopping when validation beat F-measure doesn't improve for `patience` epochs
7. Saves `best_model_weights.pt` (main branch state_dict, directly usable for inference)

**Outputs** (in `output/generic/`):
```
best_model_weights.pt          # Best main branch weights (use this for inference)
final_model_weights.pt         # Final main branch weights
checkpoint_epoch_N.pt          # Full checkpoints (model + optimizer, for resuming)
model_weights_epoch_N.pt       # Periodic weight snapshots
tensorboard/                   # Training logs
```

### Step 2a: Auxiliary Freezing (AF) Adaptation

Adapts the pre-trained BeatNet+ model to a challenging target domain (singing voices, non-percussive music) using a frozen teacher branch.

```bash
python -m BeatNetPlus.train \
    --config src/BeatNetPlus/configs/auxiliary_freezing.yaml \
    pretrained_weights=output/generic/best_model_weights.pt \
    device=cuda

# For non-percussive music adaptation
python -m BeatNetPlus.train \
    --config src/BeatNetPlus/configs/auxiliary_freezing.yaml \
    pretrained_weights=output/generic/best_model_weights.pt \
    main_audio=drumless_mix \
    output_dir=output/af_non_percussive

# For singing voice adaptation
python -m BeatNetPlus.train \
    --config src/BeatNetPlus/configs/auxiliary_freezing.yaml \
    pretrained_weights=output/generic/best_model_weights.pt \
    main_audio=vocal \
    output_dir=output/af_vocal
```

**What happens:**
1. Teacher branch is loaded with pre-trained generic weights and **frozen** (no gradient updates)
2. Student branch is randomly initialized
3. Teacher receives full mix features; student receives target domain features (vocal/drumless)
4. Loss: `L_CE(student) + λ * MSE(student_latent, teacher_latent)`
5. After training, the student branch is used for inference on the target domain

### Step 2b: Guided Fine-Tuning (GF) Adaptation

Fine-tunes the pre-trained model with gradual removal of accompaniment from the training data.

```bash
python -m BeatNetPlus.train \
    --config src/BeatNetPlus/configs/guided_finetuning.yaml \
    pretrained_weights=output/generic/best_model_weights.pt \
    device=cuda

# Custom decay rate (slower adaptation)
python -m BeatNetPlus.train \
    --config src/BeatNetPlus/configs/guided_finetuning.yaml \
    pretrained_weights=output/generic/best_model_weights.pt \
    gf_decay_rate=0.005
```

**What happens:**
1. Single branch initialized from pre-trained generic weights
2. At epoch `e`, training input = `vocal + max(0, 1 - e * γ) * accompaniment`
3. With `γ=0.01`: epoch 0 = full mix, epoch 50 = 50% accompaniment, epoch 100 = pure vocal
4. Standard cross-entropy loss (no MSE, no auxiliary branch)
5. The gradual data scheduling is the key innovation — prevents catastrophic forgetting

### Configuration Reference

All parameters are set in YAML config files and can be overridden via CLI (`key=value`).

| Parameter | Generic | AF | GF | Description |
|-----------|---------|-----|-----|-------------|
| `training_mode` | `generic` | `auxiliary_freezing` | `guided_finetuning` | Training strategy |
| `pretrained_weights` | — | Required | Required | Path to pre-trained weights |
| `sample_rate` | 22050 | 22050 | 22050 | Audio sample rate (Hz) |
| `hop_length` | 441 | 441 | 441 | STFT hop (20ms, 50fps) |
| `win_length` | 1764 | 1764 | 1764 | STFT window (80ms) |
| `feature_dim` | 288 | 288 | 288 | Log-spectrogram feature dimension |
| `num_cells` | 150 | 150 | 150 | LSTM hidden size |
| `num_layers` | 4 | 4 | 4 | LSTM layers |
| `batch_size` | 40 | 40 | 40 | Training batch size |
| `learning_rate` | 5e-4 | 5e-4 | 5e-4 | Adam optimizer learning rate |
| `seq_len` | 750 | 750 | 750 | Training excerpt (15s at 50fps) |
| `max_epochs` | 10000 | 5000 | 5000 | Maximum training epochs |
| `patience` | 20 | 20 | 20 | Early stopping patience |
| `class_weights` | [60,200,1] | [60,200,1] | [60,200,1] | CE weights: [beat, downbeat, non-beat] |
| `mse_lambda` | 200 | 200 | — | MSE latent loss weight |
| `gf_decay_rate` | — | — | 0.01 | Accompaniment fade rate per epoch |
| `main_audio` | `mix` | `vocal` | `vocal` | Main/student branch audio source |
| `aux_audio` | `drumless_mix` | `mix` | — | Aux/teacher branch audio source |
| `accompaniment_audio` | — | — | `drumless_mix` | GF accompaniment to fade |
| `checkpoint_every` | 10 | 10 | 10 | Epochs between checkpoints |
| `val_inference` | `DBN` | `DBN` | `DBN` | Validation inference method |
| `output_dir` | `./output/generic` | `./output/af_vocal` | `./output/gf_vocal` | Output directory |
| `device` | `cpu` | `cpu` | `cpu` | Device (cpu, cuda, mps) |
| `seed` | 42 | 42 | 42 | Random seed |
| `num_workers` | 4 | 4 | 4 | DataLoader workers |

### Monitoring with TensorBoard

```bash
tensorboard --logdir output/generic/tensorboard
```

Logged metrics:
- `train/loss` — Training loss per epoch
- `val/beat_f` — Validation beat F-measure
- `val/down_f` — Validation downbeat F-measure
- `test/beat_f`, `test/down_f` — Test set metrics (if test data available)
- `train/accompaniment_scale` — GF mode: current accompaniment scale factor

### Resuming Training

```bash
python -m BeatNetPlus.train \
    --config src/BeatNetPlus/configs/generic.yaml \
    --resume output/generic/checkpoint_epoch_100.pt
```

---

## Evaluation

Evaluate a trained model on test data with multiple inference methods and tolerance windows:

```bash
# Evaluate with both PF and DBN inference
python -m BeatNetPlus.evaluate \
    --weights output/generic/best_model_weights.pt \
    --config src/BeatNetPlus/configs/generic.yaml \
    --inference DBN PF

# Evaluate on specific datasets
python -m BeatNetPlus.evaluate \
    --weights src/BeatNetPlus/models/generic_weights.pt \
    --data_dir ./data \
    --test_datasets GTZAN \
    --device cuda

# Evaluate AF model
python -m BeatNetPlus.evaluate \
    --weights src/BeatNetPlus/models/af_non_percussive_weights.pt \
    --config src/BeatNetPlus/configs/auxiliary_freezing.yaml
```

Reports beat and downbeat F-measures at:
- **70ms tolerance** — Standard evaluation window
- **200ms tolerance** — More lenient, recommended for singing voice and non-percussive music

---

## Testing

The test suite validates the entire training and inference pipeline using synthetic toy data:

```bash
python test/test_training.py
```

**11 tests covering:**

| Test | What it validates |
|------|-------------------|
| `test_branch_shapes` | Single branch output dimensions and statelessness |
| `test_dual_branch_shapes` | Dual-branch forward pass produces 4 correct tensors |
| `test_dual_branch_loss` | Generic loss (CE_main + CE_aux + λ*MSE) computes correctly |
| `test_auxiliary_freezing` | Teacher is frozen, student is trainable, forward works |
| `test_guided_finetuning` | GF model loads pretrained weights, forward works |
| `test_dataset_dual_branch` | Dataset returns main + aux features with correct shapes |
| `test_dataset_gf_decay` | Accompaniment correctly fades over epochs |
| `test_generic_training_loop` | 3-epoch dual-branch training, loss decreases |
| `test_weight_compatibility` | Main branch weights load into standalone BeatNetPlusBranch |
| `test_validation_pipeline` | Full validation: model → DBN decoding → F-measure |
| `test_full_pipeline` | End-to-end: data → datasets → train → validate → save/load |

---

## Project Structure

```
beatnet_plus/
    setup.py
    README.md
    src/BeatNetPlus/
        __init__.py
        model.py                        # BeatNetPlusBranch, BeatNetPlus, AuxiliaryFreezing, GuidedFineTuning
        log_spect.py                    # 288-dim log-spectrogram features (80ms window)
        common.py                       # FeatureModule base class
        particle_filtering_cascade.py   # Two-stage cascade particle filter (from BeatNet)
        inference.py                    # Inference handler (stream, realtime, online, offline)
        train.py                        # Training script (generic, AF, GF modes)
        dataset.py                      # PyTorch Dataset with multi-source audio support
        prepare_data.py                 # Data preparation with optional Demucs separation
        evaluate.py                     # Evaluation with PF/DBN at 70ms/200ms tolerance
        configs/
            generic.yaml                # Generic dual-branch training config
            auxiliary_freezing.yaml     # AF adaptation config
            guided_finetuning.yaml      # GF adaptation config
        models/
            generic_weights.pt          # Pre-trained BeatNet+ generic (aux branch)
            generic_main_weights.pt     # Pre-trained BeatNet+ generic (main branch)
            af_non_percussive_weights.pt  # Pre-trained AF non-percussive adaptation
    test/
        test_training.py                # Comprehensive test suite (11 tests)
        test_data/
            808kick120bpm.mp3           # Test audio file
```

---

## Datasets

| Dataset | Tracks | Usage | Annotations | Stems |
|---------|--------|-------|-------------|-------|
| Ballroom | 699 | Train | Original | No (use Demucs) |
| Hainsworth | 220 | Train | Original | No (use Demucs) |
| Rock Corpus | 200 | Train | Original | No (use Demucs) |
| MUSDB18 | 150 | Train | Added (new) | Yes (4 stems) |
| URSing | 65 | Train | Added (new) | Yes |
| RWC Pop | 100 | Train | Revised | No (use Demucs) |
| RWC Jazz | 50 | Train | Revised | No (use Demucs) |
| RWC Royalty-free | 15 | Train | Revised | No (use Demucs) |
| **GTZAN** | **999** | **Test only** | **Original** | **No** |

GTZAN is used exclusively for testing — no model sees GTZAN data during training.

---

## Annotation Format

The `.beats` annotation format is one line per beat:

```
<time_in_seconds> <beat_number>
```

Where `beat_number == 1` indicates a **downbeat** (first beat of a measure), and any other value indicates a regular beat. Example:

```
0.520 1
1.040 2
1.540 3
2.060 4
2.560 1
3.080 2
```

This encodes a 4/4 time signature where beats at 0.520s and 2.560s are downbeats.

---

## Output Format

All inference modes return a numpy array of shape `(num_beats, 2)`:

| Column | Content |
|--------|---------|
| 0 | Beat time in seconds |
| 1 | Beat type: **1** = downbeat, **2** = regular beat |

Example output:
```python
array([[ 0.52, 1. ],    # downbeat at 0.52s
       [ 1.04, 2. ],    # beat at 1.04s
       [ 1.54, 2. ],    # beat at 1.54s
       [ 2.06, 2. ],    # beat at 2.06s
       [ 2.56, 1. ],    # downbeat at 2.56s
       ...])
```

---

## Troubleshooting

### madmom `ImportError: cannot import name 'MutableSequence' from 'collections'`

Fix: In your madmom installation, edit `madmom/processors.py`:
```python
# Change this:
from collections import MutableSequence
# To this:
from collections.abc import MutableSequence
```

### madmom `AttributeError: module 'numpy' has no attribute 'float'`

This occurs in madmom's compiled Cython extensions. Fix by adding to your Python's `sitecustomize.py`:
```python
import numpy as np
if not hasattr(np, 'float'): np.float = np.float64
if not hasattr(np, 'int'): np.int = np.int_
```

Alternatively, use Python 3.9 where these aliases still exist.

### DBN inference returns empty output on short files

This is a known madmom bug with `beats_per_bar=[2,3,4]` on very short audio. The inference code handles this gracefully with a warning. Use `inference_model='PF'` instead for short files.

### CUDA out of memory during training

Reduce `batch_size` or `seq_len` in the config:
```bash
python -m BeatNetPlus.train --config configs/generic.yaml batch_size=20 seq_len=500
```

### Demucs not found during data preparation

Install Demucs separately:
```bash
pip install demucs
```
Or use `--has_stems` if your dataset already has separated stems.

---

## Citation

```bibtex
@article{heydari2024beatnetplus,
  title={BeatNet+: Advancing Music Beat and Downbeat Tracking for Non-Percussive Music and Singing Voices},
  author={Heydari, Mojtaba and Cwitkowitz, Frank and Duan, Zhiyao},
  journal={Transactions of the International Society for Music Information Retrieval},
  year={2024}
}
```
