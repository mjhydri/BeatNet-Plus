# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# Main training script for BeatNet+.
# Supports three training modes:
#   1. Generic dual-branch training (main + auxiliary with MSE latent matching)
#   2. Auxiliary Freezing (AF) adaptation (frozen teacher + student)
#   3. Guided Fine-Tuning (GF) adaptation (single branch with fading accompaniment)
#
# Usage:
#   python -m BeatNetPlus.train --config src/BeatNetPlus/configs/generic.yaml
#   python -m BeatNetPlus.train --config src/BeatNetPlus/configs/auxiliary_freezing.yaml \
#       pretrained_weights=output/generic/best_model_weights.pt
#   python -m BeatNetPlus.train --config src/BeatNetPlus/configs/guided_finetuning.yaml \
#       pretrained_weights=output/generic/best_model_weights.pt

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from BeatNetPlus.dataset import build_datasets
from BeatNetPlus.model import (
    BeatNetPlusBranch, BeatNetPlus, AuxiliaryFreezing, GuidedFineTuning
)

logger = logging.getLogger('BeatNetPlus.train')


def load_config(config_path, overrides=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if overrides:
        for ov in overrides:
            if '=' not in ov:
                continue
            key, val = ov.split('=', 1)
            for parser in (int, float):
                try:
                    val = parser(val)
                    break
                except ValueError:
                    continue
            else:
                if val.lower() in ('true', 'false'):
                    val = val.lower() == 'true'
            config[key] = val
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Validation (shared across all modes)
# ---------------------------------------------------------------------------

def validate(model, val_loader, inference_type, device, mode='generic'):
    """Run validation. Returns average beat and downbeat F-measures."""
    from madmom.evaluation import BeatEvaluation
    from madmom.features import DBNDownBeatTrackingProcessor

    model.eval()
    beat_fmeasures, down_fmeasures = [], []

    with torch.no_grad():
        for batch in val_loader:
            main_feats = batch['main_feats'].transpose(1, 2).to(device)
            gt = batch['ground_truth']

            if mode == 'generic':
                preds = model.inference_forward(main_feats)[0]
            elif mode == 'auxiliary_freezing':
                preds = model.inference_forward(main_feats)[0]
            elif mode == 'guided_finetuning':
                preds = model.inference_forward(main_feats)[0]
            else:
                preds = model.inference_forward(main_feats)[0]

            preds = F.softmax(preds, dim=0)
            preds = preds.cpu().numpy()
            preds = np.transpose(preds[:2, :])

            gt_np = gt[0].numpy()
            beats_g = np.argwhere(gt_np[0] == 1) * 0.02
            downs_g = np.argwhere(gt_np[1] == 1) * 0.02
            beats_g = np.sort(np.append(downs_g, beats_g)).flatten()
            downs_g = downs_g.flatten()

            if len(beats_g) == 0:
                continue

            try:
                if inference_type == 'DBN':
                    meter = max(2, round(len(beats_g) / max(1, len(downs_g))))
                    meter = min(meter, 4)
                    dbn = DBNDownBeatTrackingProcessor(
                        beats_per_bar=[meter], fps=50, observation_lambda=16)
                    decoded = dbn(preds)
                    if len(decoded) == 0:
                        continue
                    pred_downs = decoded[:, 0][decoded[:, 1] == 1]
                    pred_beats = decoded[:, 0]
                elif inference_type == 'PF':
                    from BeatNetPlus.particle_filtering_cascade import particle_filter_cascade
                    pf = particle_filter_cascade(beats_per_bar=[], fps=50, plot=[], mode='online')
                    output = pf.process(preds)
                    if output is None or len(output) == 0:
                        continue
                    pred_beats = output[:, 0]
                    pred_downs = output[:, 0][output[:, 1] == 1]
                else:
                    continue
            except (ValueError, IndexError):
                continue

            if len(pred_beats) > 0 and len(beats_g) > 0:
                beat_fmeasures.append(BeatEvaluation(pred_beats, beats_g, skip=5).fmeasure)
            if len(pred_downs) > 0 and len(downs_g) > 0:
                down_fmeasures.append(BeatEvaluation(pred_downs, downs_g, skip=5).fmeasure)

    return (np.mean(beat_fmeasures) if beat_fmeasures else 0.0,
            np.mean(down_fmeasures) if down_fmeasures else 0.0)


# ---------------------------------------------------------------------------
# Training: Generic dual-branch
# ---------------------------------------------------------------------------

def train_generic(config):
    """Train the generic BeatNet+ dual-branch model."""
    device = torch.device(config.get('device', 'cpu'))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    train_ds, val_ds, test_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 40),
                              shuffle=True, num_workers=config.get('num_workers', 4),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=min(config.get('num_workers', 4), 2))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=min(config.get('num_workers', 4), 2))

    model = BeatNetPlus(
        config.get('feature_dim', 288), config.get('num_cells', 150),
        config.get('num_layers', 4), device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 5e-4))
    cw = torch.FloatTensor(config.get('class_weights', [60, 200, 1])).to(device)
    mse_lambda = config.get('mse_lambda', 200.0)

    best_val_f, patience_counter = 0.0, 0
    max_epochs = config.get('max_epochs', 10000)
    patience = config.get('patience', 20)
    checkpoint_every = config.get('checkpoint_every', 10)

    logger.info(f"Generic dual-branch training: epochs={max_epochs}, batch={config.get('batch_size', 40)}")

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            main_in = batch['main_feats'].transpose(1, 2).to(device)
            aux_in = batch['aux_feats'].transpose(1, 2).to(device)
            targets = torch.argmax(batch['ground_truth'], dim=1).to(device)

            m_log, a_log, m_lat, a_lat = model.train_forward(main_in, aux_in)
            loss, loss_dict = BeatNetPlus.compute_loss(
                m_log, a_log, m_lat, a_lat, targets, cw, mse_lambda)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_dict['total'])

        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('train/loss', avg_loss, epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{max_epochs} | loss={avg_loss:.4f} | "
                        f"time={time.time()-t0:.1f}s")

        if (epoch + 1) % checkpoint_every == 0:
            beat_f, down_f = validate(model, val_loader, config.get('val_inference', 'DBN'),
                                      device, mode='generic')
            writer.add_scalar('val/beat_f', beat_f, epoch + 1)
            writer.add_scalar('val/down_f', down_f, epoch + 1)
            logger.info(f"  Val: beat_F={beat_f:.4f}, down_F={down_f:.4f}")

            if len(test_ds) > 0:
                tb, td = validate(model, test_loader, config.get('val_inference', 'DBN'),
                                  device, mode='generic')
                writer.add_scalar('test/beat_f', tb, epoch + 1)
                writer.add_scalar('test/down_f', td, epoch + 1)
                logger.info(f"  Test: beat_F={tb:.4f}, down_F={td:.4f}")

            # Save checkpoint
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'best_val_f': best_val_f, 'config': config},
                        os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            torch.save(model.get_main_state_dict(),
                        os.path.join(output_dir, f'model_weights_epoch_{epoch+1}.pt'))

            if beat_f > best_val_f:
                best_val_f = beat_f
                patience_counter = 0
                torch.save(model.get_main_state_dict(),
                            os.path.join(output_dir, 'best_model_weights.pt'))
                logger.info(f"  New best model (beat_F={best_val_f:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

    torch.save(model.get_main_state_dict(), os.path.join(output_dir, 'final_model_weights.pt'))
    writer.close()
    logger.info(f"Training complete. Best val beat_F={best_val_f:.4f}")
    return model


# ---------------------------------------------------------------------------
# Training: Auxiliary Freezing
# ---------------------------------------------------------------------------

def train_auxiliary_freezing(config):
    """Train with Auxiliary Freezing (AF) adaptation."""
    device = torch.device(config.get('device', 'cpu'))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    pretrained = config.get('pretrained_weights')
    if not pretrained:
        raise ValueError("pretrained_weights is required for auxiliary_freezing mode")

    train_ds, val_ds, test_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 40),
                              shuffle=True, num_workers=config.get('num_workers', 4),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=min(config.get('num_workers', 4), 2))

    model = AuxiliaryFreezing(
        config.get('feature_dim', 288), config.get('num_cells', 150),
        config.get('num_layers', 4), device, pretrained_weights=pretrained)
    model.to(device)

    # Only optimize student parameters (teacher is frozen)
    optimizer = torch.optim.Adam(model.student.parameters(),
                                  lr=config.get('learning_rate', 5e-4))
    cw = torch.FloatTensor(config.get('class_weights', [60, 200, 1])).to(device)
    mse_lambda = config.get('mse_lambda', 200.0)

    best_val_f, patience_counter = 0.0, 0
    max_epochs = config.get('max_epochs', 5000)
    patience = config.get('patience', 20)
    checkpoint_every = config.get('checkpoint_every', 10)

    logger.info(f"Auxiliary Freezing training: teacher={pretrained}")

    for epoch in range(max_epochs):
        model.student.train()
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            student_in = batch['main_feats'].transpose(1, 2).to(device)
            teacher_in = batch['aux_feats'].transpose(1, 2).to(device)
            targets = torch.argmax(batch['ground_truth'], dim=1).to(device)

            s_log, t_log, s_lat, t_lat = model.train_forward(student_in, teacher_in)
            loss, loss_dict = AuxiliaryFreezing.compute_loss(
                s_log, s_lat, t_lat, targets, cw, mse_lambda)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_dict['total'])

        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('train/loss', avg_loss, epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{max_epochs} | loss={avg_loss:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            beat_f, down_f = validate(model, val_loader, config.get('val_inference', 'DBN'),
                                      device, mode='auxiliary_freezing')
            writer.add_scalar('val/beat_f', beat_f, epoch + 1)
            writer.add_scalar('val/down_f', down_f, epoch + 1)
            logger.info(f"  Val: beat_F={beat_f:.4f}, down_F={down_f:.4f}")

            torch.save(model.get_student_state_dict(),
                        os.path.join(output_dir, f'model_weights_epoch_{epoch+1}.pt'))

            if beat_f > best_val_f:
                best_val_f = beat_f
                patience_counter = 0
                torch.save(model.get_student_state_dict(),
                            os.path.join(output_dir, 'best_model_weights.pt'))
                logger.info(f"  New best (beat_F={best_val_f:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

    torch.save(model.get_student_state_dict(),
                os.path.join(output_dir, 'final_model_weights.pt'))
    writer.close()
    return model


# ---------------------------------------------------------------------------
# Training: Guided Fine-Tuning
# ---------------------------------------------------------------------------

def train_guided_finetuning(config):
    """Train with Guided Fine-Tuning (GF) adaptation."""
    device = torch.device(config.get('device', 'cpu'))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    pretrained = config.get('pretrained_weights')
    if not pretrained:
        raise ValueError("pretrained_weights is required for guided_finetuning mode")

    train_ds, val_ds, test_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 40),
                              shuffle=True, num_workers=config.get('num_workers', 4),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=min(config.get('num_workers', 4), 2))

    model = GuidedFineTuning(
        config.get('feature_dim', 288), config.get('num_cells', 150),
        config.get('num_layers', 4), device, pretrained_weights=pretrained)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 5e-4))
    cw = torch.FloatTensor(config.get('class_weights', [60, 200, 1])).to(device)
    gf_decay_rate = config.get('gf_decay_rate', 0.01)

    best_val_f, patience_counter = 0.0, 0
    max_epochs = config.get('max_epochs', 5000)
    patience = config.get('patience', 20)
    checkpoint_every = config.get('checkpoint_every', 10)

    logger.info(f"Guided Fine-Tuning: decay_rate={gf_decay_rate}")

    for epoch in range(max_epochs):
        model.train()
        # Update the dataset's epoch for accompaniment decay scheduling
        train_loader.dataset.epoch = epoch
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            feats = batch['main_feats'].transpose(1, 2).to(device)
            targets = torch.argmax(batch['ground_truth'], dim=1).to(device)

            logits = model.train_forward(feats)
            loss, loss_dict = GuidedFineTuning.compute_loss(logits, targets, cw)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_dict['total'])

        acc_scale = max(0.0, 1.0 - epoch * gf_decay_rate)
        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('train/loss', avg_loss, epoch + 1)
        writer.add_scalar('train/accompaniment_scale', acc_scale, epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{max_epochs} | loss={avg_loss:.4f} | "
                        f"accomp_scale={acc_scale:.3f}")

        if (epoch + 1) % checkpoint_every == 0:
            beat_f, down_f = validate(model, val_loader, config.get('val_inference', 'DBN'),
                                      device, mode='guided_finetuning')
            writer.add_scalar('val/beat_f', beat_f, epoch + 1)
            writer.add_scalar('val/down_f', down_f, epoch + 1)
            logger.info(f"  Val: beat_F={beat_f:.4f}, down_F={down_f:.4f}")

            torch.save(model.get_state_dict(),
                        os.path.join(output_dir, f'model_weights_epoch_{epoch+1}.pt'))

            if beat_f > best_val_f:
                best_val_f = beat_f
                patience_counter = 0
                torch.save(model.get_state_dict(),
                            os.path.join(output_dir, 'best_model_weights.pt'))
                logger.info(f"  New best (beat_F={best_val_f:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

    torch.save(model.get_state_dict(), os.path.join(output_dir, 'final_model_weights.pt'))
    writer.close()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TRAINING_MODES = {
    'generic': train_generic,
    'auxiliary_freezing': train_auxiliary_freezing,
    'guided_finetuning': train_guided_finetuning,
}


def train(config):
    set_seed(config.get('seed', 42))
    mode = config.get('training_mode', 'generic')
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training_mode: {mode}. Must be one of {list(TRAINING_MODES)}")
    logger.info(f"Training mode: {mode}")
    return TRAINING_MODES[mode](config)


def main():
    parser = argparse.ArgumentParser(description='Train BeatNet+ model')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('overrides', nargs='*', help='Config overrides (key=value)')
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    if args.resume:
        config['resume'] = args.resume

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s',
                        datefmt='%H:%M:%S')
    train(config)


if __name__ == '__main__':
    main()