# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# Evaluation script for BeatNet+.
# Evaluates beat/downbeat tracking with multiple inference methods and tolerance windows.
#
# Usage:
#   python -m BeatNetPlus.evaluate --config src/BeatNetPlus/configs/generic.yaml \
#       --weights output/generic/best_model_weights.pt
#   python -m BeatNetPlus.evaluate --weights model.pt --data_dir ./data --test_datasets GTZAN

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from BeatNetPlus.dataset import build_datasets
from BeatNetPlus.model import BeatNetPlusBranch

logger = logging.getLogger('BeatNetPlus.evaluate')


def evaluate_track(preds, gt_np, inference_type='DBN', tolerance=0.07):
    """Evaluate a single track. Returns (beat_f, down_f) or (None, None) on failure."""
    from madmom.evaluation import BeatEvaluation
    from madmom.features import DBNDownBeatTrackingProcessor

    beats_g = np.argwhere(gt_np[0] == 1) * 0.02
    downs_g = np.argwhere(gt_np[1] == 1) * 0.02
    beats_g = np.sort(np.append(downs_g, beats_g)).flatten()
    downs_g = downs_g.flatten()

    if len(beats_g) == 0:
        return None, None

    try:
        if inference_type == 'DBN':
            meter = max(2, round(len(beats_g) / max(1, len(downs_g))))
            meter = min(meter, 4)
            dbn = DBNDownBeatTrackingProcessor(
                beats_per_bar=[meter], fps=50, observation_lambda=16)
            decoded = dbn(preds)
            if len(decoded) == 0:
                return None, None
            pred_beats = decoded[:, 0]
            pred_downs = decoded[:, 0][decoded[:, 1] == 1]
        elif inference_type == 'PF':
            from BeatNetPlus.particle_filtering_cascade import particle_filter_cascade
            pf = particle_filter_cascade(beats_per_bar=[], fps=50, plot=[], mode='online')
            output = pf.process(preds)
            if output is None or len(output) == 0:
                return None, None
            pred_beats = output[:, 0]
            pred_downs = output[:, 0][output[:, 1] == 1]
        else:
            return None, None
    except (ValueError, IndexError):
        return None, None

    beat_f = BeatEvaluation(pred_beats, beats_g, skip=5).fmeasure if len(pred_beats) > 0 else 0.0
    down_f = BeatEvaluation(pred_downs, downs_g, skip=5).fmeasure if len(pred_downs) > 0 else 0.0
    return beat_f, down_f


def evaluate(model, data_loader, device, inference_types=('DBN', 'PF'),
             tolerances=(0.07, 0.2)):
    """Evaluate model on a dataset with multiple inference methods and tolerances.

    Returns
    -------
    results : dict
        Nested dict: results[inference_type][tolerance] = {'beat_f': float, 'down_f': float}
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            feats = batch['main_feats'].transpose(1, 2).to(device)
            gt = batch['ground_truth']

            logits = model.inference_forward(feats)[0]
            probs = F.softmax(logits, dim=0).cpu().numpy()
            preds = np.transpose(probs[:2, :])

            all_preds.append((preds, gt[0].numpy()))

    results = {}
    for inf_type in inference_types:
        results[inf_type] = {}
        for tol in tolerances:
            beat_fs, down_fs = [], []
            for preds, gt_np in all_preds:
                bf, df = evaluate_track(preds, gt_np, inf_type, tol)
                if bf is not None:
                    beat_fs.append(bf)
                if df is not None:
                    down_fs.append(df)

            results[inf_type][tol] = {
                'beat_f': np.mean(beat_fs) if beat_fs else 0.0,
                'down_f': np.mean(down_fs) if down_fs else 0.0,
                'n_tracks': len(beat_fs),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate BeatNet+ model')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--test_datasets', type=str, nargs='+', default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--inference', type=str, nargs='+', default=['DBN', 'PF'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s',
                        datefmt='%H:%M:%S')

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.test_datasets:
        config['datasets'] = {'train': [], 'test': args.test_datasets}
    config['device'] = args.device

    device = torch.device(args.device)
    model = BeatNetPlusBranch(
        config.get('feature_dim', 288), config.get('num_cells', 150),
        config.get('num_layers', 4), device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.to(device)

    _, _, test_ds = build_datasets(config)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    logger.info(f"Evaluating {args.weights} on {len(test_ds)} tracks")
    results = evaluate(model, test_loader, device, inference_types=args.inference)

    for inf_type, tol_results in results.items():
        for tol, metrics in tol_results.items():
            logger.info(f"  {inf_type} @ {int(tol*1000)}ms: "
                        f"beat_F={metrics['beat_f']:.4f}, down_F={metrics['down_f']:.4f} "
                        f"({metrics['n_tracks']} tracks)")


if __name__ == '__main__':
    main()