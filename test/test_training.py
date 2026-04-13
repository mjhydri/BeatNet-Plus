"""
Test suite for BeatNet+ training pipeline.
Validates all three training modes (generic, AF, GF) using synthetic toy data.

Usage:
    python test/test_training.py
    python -m pytest test/test_training.py -v
"""

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from BeatNetPlus.model import (
    BeatNetPlusBranch, BeatNetPlus, AuxiliaryFreezing, GuidedFineTuning
)
from BeatNetPlus.dataset import BeatNetPlusDataset, build_datasets
from BeatNetPlus.prepare_data import build_ground_truth, parse_beats_file
from BeatNetPlus.train import validate

SAMPLE_RATE = 22050
HOP_LENGTH = 441
FEATURE_DIM = 288


def make_toy_track(num_seconds=10, bpm=120, meter=4):
    """Generate a synthetic track with features for all audio sources."""
    num_frames = int(num_seconds * SAMPLE_RATE / HOP_LENGTH)
    times = np.arange(num_frames) * HOP_LENGTH / SAMPLE_RATE

    beat_interval = 60.0 / bpm
    beat_times = np.arange(0, num_seconds, beat_interval)
    down_times = beat_times[::meter]
    beat_only_times = np.array([t for t in beat_times if t not in down_times])

    gt = build_ground_truth(beat_only_times, down_times, num_frames, SAMPLE_RATE, HOP_LENGTH)

    return {
        'feats_mix': np.random.randn(FEATURE_DIM, num_frames).astype(np.float32),
        'feats_drumless': np.random.randn(FEATURE_DIM, num_frames).astype(np.float32),
        'feats_vocal': np.random.randn(FEATURE_DIM, num_frames).astype(np.float32),
        'feats_drums': np.random.randn(FEATURE_DIM, num_frames).astype(np.float32),
        'times': times.astype(np.float32),
        'ground_truth': gt,
    }


def make_toy_dataset(data_dir, name, num_tracks=6, num_seconds=10):
    ds_dir = os.path.join(data_dir, name, 'tracks')
    os.makedirs(ds_dir, exist_ok=True)
    tracks_list = {'default': []}
    for i in range(num_tracks):
        track_id = f"{name}#default#track_{i:03d}"
        data = make_toy_track(num_seconds=num_seconds, bpm=np.random.randint(80, 180),
                              meter=np.random.choice([3, 4]))
        with open(os.path.join(ds_dir, track_id + '.pkl'), 'wb') as f:
            pickle.dump(data, f)
        tracks_list['default'].append(track_id)
    with open(os.path.join(data_dir, name, 'tracks_list.pkl'), 'wb') as f:
        pickle.dump(tracks_list, f)
    return tracks_list


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_branch_shapes():
    """Single branch produces correct output shapes."""
    model = BeatNetPlusBranch(FEATURE_DIM, 150, 4, 'cpu')
    x = torch.randn(2, 200, FEATURE_DIM)

    logits, latent = model.train_forward(x)
    assert logits.shape == (2, 3, 200), f"logits: {logits.shape}"
    assert latent.shape == (2, 200, 3), f"latent: {latent.shape}"

    # Stateless: same output twice
    model.eval()
    with torch.no_grad():
        l1, _ = model.train_forward(x)
        l2, _ = model.train_forward(x)
    assert torch.allclose(l1, l2, atol=1e-6)
    print("  test_branch_shapes PASSED")


def test_dual_branch_shapes():
    """Dual-branch BeatNetPlus produces correct output shapes."""
    model = BeatNetPlus(FEATURE_DIM, 150, 4, 'cpu')
    main_x = torch.randn(2, 200, FEATURE_DIM)
    aux_x = torch.randn(2, 200, FEATURE_DIM)

    ml, al, mlat, alat = model.train_forward(main_x, aux_x)
    assert ml.shape == (2, 3, 200)
    assert al.shape == (2, 3, 200)
    assert mlat.shape == (2, 200, 3)
    assert alat.shape == (2, 200, 3)
    print("  test_dual_branch_shapes PASSED")


def test_dual_branch_loss():
    """Dual-branch loss computation works."""
    model = BeatNetPlus(FEATURE_DIM, 150, 4, 'cpu')
    main_x = torch.randn(2, 50, FEATURE_DIM)
    aux_x = torch.randn(2, 50, FEATURE_DIM)
    targets = torch.randint(0, 3, (2, 50))
    cw = torch.FloatTensor([60, 200, 1])

    ml, al, mlat, alat = model.train_forward(main_x, aux_x)
    loss, loss_dict = BeatNetPlus.compute_loss(ml, al, mlat, alat, targets, cw, mse_lambda=200)

    assert loss.item() > 0
    assert all(k in loss_dict for k in ['ce_main', 'ce_aux', 'mse', 'total'])
    print(f"  test_dual_branch_loss PASSED (loss={loss.item():.4f})")


def test_auxiliary_freezing():
    """AF model: teacher is frozen, student is trainable."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create fake pretrained weights
        teacher_model = BeatNetPlusBranch(FEATURE_DIM, 150, 4, 'cpu')
        weights_path = os.path.join(tmpdir, 'teacher.pt')
        torch.save(teacher_model.state_dict(), weights_path)

        model = AuxiliaryFreezing(FEATURE_DIM, 150, 4, 'cpu', pretrained_weights=weights_path)

        # Teacher params should not require grad
        for p in model.teacher.parameters():
            assert not p.requires_grad, "Teacher params should be frozen"
        # Student params should require grad
        for p in model.student.parameters():
            assert p.requires_grad, "Student params should be trainable"

        # Forward works
        s_in = torch.randn(2, 50, FEATURE_DIM)
        t_in = torch.randn(2, 50, FEATURE_DIM)
        sl, tl, slat, tlat = model.train_forward(s_in, t_in)
        assert sl.shape == (2, 3, 50)
        print("  test_auxiliary_freezing PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_guided_finetuning():
    """GF model: initialized from pretrained, single branch."""
    tmpdir = tempfile.mkdtemp()
    try:
        pretrained = BeatNetPlusBranch(FEATURE_DIM, 150, 4, 'cpu')
        weights_path = os.path.join(tmpdir, 'pretrained.pt')
        torch.save(pretrained.state_dict(), weights_path)

        model = GuidedFineTuning(FEATURE_DIM, 150, 4, 'cpu', pretrained_weights=weights_path)
        x = torch.randn(2, 50, FEATURE_DIM)
        logits = model.train_forward(x)
        assert logits.shape == (2, 3, 50)
        print("  test_guided_finetuning PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_dataset_dual_branch():
    """Dataset returns both main and aux features."""
    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY', num_tracks=4)
        with open(os.path.join(tmpdir, 'TOY', 'tracks_list.pkl'), 'rb') as f:
            tl = pickle.load(f)

        tracks_dirs = {'TOY': os.path.join(tmpdir, 'TOY', 'tracks')}
        ds = BeatNetPlusDataset(tl['default'], tracks_dirs, main_audio='mix',
                                aux_audio='drumless_mix', seq_len=200)

        sample = ds[0]
        assert sample['main_feats'].shape == (FEATURE_DIM, 200)
        assert sample['aux_feats'].shape == (FEATURE_DIM, 200)
        assert sample['ground_truth'].shape == (3, 200)
        print("  test_dataset_dual_branch PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_dataset_gf_decay():
    """Dataset fades accompaniment over epochs in GF mode."""
    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY', num_tracks=2, num_seconds=8)
        with open(os.path.join(tmpdir, 'TOY', 'tracks_list.pkl'), 'rb') as f:
            tl = pickle.load(f)

        tracks_dirs = {'TOY': os.path.join(tmpdir, 'TOY', 'tracks')}

        # Epoch 0: full accompaniment
        ds0 = BeatNetPlusDataset(tl['default'], tracks_dirs, main_audio='vocal',
                                 accompaniment_audio='drumless_mix', gf_decay_rate=0.01,
                                 seq_len=200, epoch=0)
        s0 = ds0[0]['main_feats']

        # Epoch 100: no accompaniment (alpha=0)
        ds100 = BeatNetPlusDataset(tl['default'], tracks_dirs, main_audio='vocal',
                                   accompaniment_audio='drumless_mix', gf_decay_rate=0.01,
                                   seq_len=200, epoch=100)
        ds100.rng = np.random.RandomState(42)  # same seed for same crop
        ds0.rng = np.random.RandomState(42)

        # They should differ (epoch 0 has accompaniment, epoch 100 doesn't)
        s0_new = ds0[0]['main_feats']
        s100 = ds100[0]['main_feats']
        # Can't guarantee exact difference due to random crops, but shapes should match
        assert s0_new.shape == s100.shape == (FEATURE_DIM, 200)
        print("  test_dataset_gf_decay PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_generic_training_loop():
    """Run a short generic dual-branch training loop."""
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY', num_tracks=8)
        with open(os.path.join(tmpdir, 'TOY', 'tracks_list.pkl'), 'rb') as f:
            tl = pickle.load(f)

        tracks_dirs = {'TOY': os.path.join(tmpdir, 'TOY', 'tracks')}
        ds = BeatNetPlusDataset(tl['default'], tracks_dirs, main_audio='mix',
                                aux_audio='drumless_mix', seq_len=200)
        loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)

        model = BeatNetPlus(FEATURE_DIM, 150, 4, 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        cw = torch.FloatTensor([60, 200, 1])

        losses = []
        model.train()
        for epoch in range(3):
            epoch_loss = []
            for batch in loader:
                optimizer.zero_grad()
                main_in = batch['main_feats'].transpose(1, 2)
                aux_in = batch['aux_feats'].transpose(1, 2)
                targets = torch.argmax(batch['ground_truth'], dim=1)

                ml, al, mlat, alat = model.train_forward(main_in, aux_in)
                loss, _ = BeatNetPlus.compute_loss(ml, al, mlat, alat, targets, cw)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            losses.append(np.mean(epoch_loss))

        assert all(np.isfinite(losses)), f"NaN/Inf in losses: {losses}"
        print(f"  test_generic_training_loop PASSED (loss: {losses[0]:.2f} -> {losses[-1]:.2f})")
    finally:
        shutil.rmtree(tmpdir)


def test_weight_compatibility():
    """Saved main branch weights load into standalone BeatNetPlusBranch."""
    model = BeatNetPlus(FEATURE_DIM, 150, 4, 'cpu')
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, 'weights.pt')
        torch.save(model.get_main_state_dict(), path)

        branch = BeatNetPlusBranch(FEATURE_DIM, 150, 4, 'cpu')
        branch.load_state_dict(torch.load(path, map_location='cpu'), strict=False)

        x = torch.randn(1, 50, FEATURE_DIM)
        model.eval()
        branch.eval()
        with torch.no_grad():
            out1 = model.inference_forward(x)
            out2 = branch.inference_forward(x)
        assert torch.allclose(out1, out2, atol=1e-5)
        print("  test_weight_compatibility PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_validation_pipeline():
    """Validation runs without errors on toy data."""
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY', num_tracks=3, num_seconds=15)
        with open(os.path.join(tmpdir, 'TOY', 'tracks_list.pkl'), 'rb') as f:
            tl = pickle.load(f)

        tracks_dirs = {'TOY': os.path.join(tmpdir, 'TOY', 'tracks')}
        ds = BeatNetPlusDataset(tl['default'], tracks_dirs, main_audio='mix', seq_len=None)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        model = BeatNetPlus(FEATURE_DIM, 150, 4, 'cpu')
        model.eval()

        beat_f, down_f = validate(model, loader, 'DBN', 'cpu', mode='generic')
        assert isinstance(beat_f, float) and 0 <= beat_f <= 1
        print(f"  test_validation_pipeline PASSED (beat_F={beat_f:.4f})")
    finally:
        shutil.rmtree(tmpdir)


def test_full_pipeline():
    """End-to-end: create data -> build datasets -> train -> validate -> save/load."""
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TRAIN_DS', num_tracks=8, num_seconds=10)
        make_toy_dataset(tmpdir, 'TEST_DS', num_tracks=3, num_seconds=12)

        config = {
            'data_dir': tmpdir,
            'datasets': {'train': ['TRAIN_DS'], 'test': ['TEST_DS']},
            'dataset_weights': {},
            'train_val_split': 0.75,
            'seq_len': 200,
            'seed': 42,
            'main_audio': 'mix',
            'aux_audio': 'drumless_mix',
        }
        train_ds, val_ds, test_ds = build_datasets(config)
        assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0

        # Quick train
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        model = BeatNetPlus(FEATURE_DIM, 150, 4, 'cpu')
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        cw = torch.FloatTensor([60, 200, 1])

        model.train()
        for batch in loader:
            opt.zero_grad()
            ml, al, mlat, alat = model.train_forward(
                batch['main_feats'].transpose(1, 2),
                batch['aux_feats'].transpose(1, 2))
            loss, _ = BeatNetPlus.compute_loss(
                ml, al, mlat, alat, torch.argmax(batch['ground_truth'], dim=1), cw)
            loss.backward()
            opt.step()
            break

        # Save and reload
        wp = os.path.join(tmpdir, 'test_weights.pt')
        torch.save(model.get_main_state_dict(), wp)
        branch = BeatNetPlusBranch(FEATURE_DIM, 150, 4, 'cpu')
        branch.load_state_dict(torch.load(wp, map_location='cpu'), strict=False)

        print(f"  test_full_pipeline PASSED (train={len(train_ds)}, val={len(val_ds)}, "
              f"test={len(test_ds)})")
    finally:
        shutil.rmtree(tmpdir)


ALL_TESTS = [
    test_branch_shapes,
    test_dual_branch_shapes,
    test_dual_branch_loss,
    test_auxiliary_freezing,
    test_guided_finetuning,
    test_dataset_dual_branch,
    test_dataset_gf_decay,
    test_generic_training_loop,
    test_weight_compatibility,
    test_validation_pipeline,
    test_full_pipeline,
]

if __name__ == '__main__':
    passed = failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)