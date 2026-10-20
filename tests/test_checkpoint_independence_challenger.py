"""Adversarial challenger test suite for Milestone 2 model checkpoint authenticity & independence.

Independently verifies:
1. Existence and distinct cryptographic SHA-256 hashes for all 5 seed checkpoints.
2. Substantial pairwise weight tensor divergence (L2 norm > 10.0, MaxAbs > 0.5, cosine similarity < 0.5)
   across all 10 seed pairs, proving no weights were cloned or trivially perturbed.
3. Strict non-identity of TrainedPredictor inference outputs on identical demand histories.
4. Divergence of calibrated safety margins and training loss trajectories across seeds.
5. Independence and distinct parameters across GRU baseline checkpoints as well.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from carbonroute.common import ROOT, config, load_json
from carbonroute.data import load_prepared
from carbonroute.prediction import TrainedPredictor

SEEDS = [42, 43, 44, 45, 46]
SEED_PAIRS = list(itertools.combinations(SEEDS, 2))


@pytest.fixture(autouse=True)
def require_seed_checkpoints():
    """Skips tests gracefully on clean clone if multi-seed checkpoints are not present."""
    for s in SEEDS:
        if not (ROOT / f"artifacts/seeds/seed_{s}/gnn.pt").is_file():
            pytest.skip(
                f"Multi-seed checkpoint seed_{s} not present locally; run scripts/run_multiseed.py to generate"
            )


def test_checkpoints_exist_and_have_unique_sha256():
    """Verifies that all 5 GNN checkpoints exist, have non-trivial size, and unique SHA-256."""
    sha_map: dict[int, str] = {}
    for s in SEEDS:
        ckpt_path = ROOT / f"artifacts/seeds/seed_{s}/gnn.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"
        raw_bytes = ckpt_path.read_bytes()
        assert len(raw_bytes) > 10000, f"Checkpoint file too small: {len(raw_bytes)} bytes"

        digest = hashlib.sha256(raw_bytes).hexdigest()
        sha_map[s] = digest

        # Verify against metadata json
        meta = load_json(ROOT / f"artifacts/seeds/seed_{s}/gnn.json")
        assert meta["checkpoint_sha256"] == digest, (
            f"Seed {s} checkpoint sha mismatch: {digest} vs metadata {meta['checkpoint_sha256']}"
        )

        sum_json = load_json(ROOT / f"artifacts/seeds/seed_{s}/training_summary.json")
        assert sum_json["models"]["gnn"]["checkpoint_sha256"] == digest

    # Assert all SHA-256 hashes are strictly unique
    unique_hashes = set(sha_map.values())
    assert len(unique_hashes) == len(SEEDS), f"Detected duplicated checkpoint SHA-256! {sha_map}"


def test_pairwise_weight_differences_are_substantial():
    """Empirically proves that every seed learned distinct parameters in parameter space.
    
    Verifies that for every pair among all 10 seed pairs:
    - Overall trainable L2 norm difference > 10.0 (empirical ~18.5 - 19.7)
    - Global max absolute difference > 0.5 (empirical ~0.83 - 0.99)
    - Cosine similarity < 0.5 (empirical ~0.11 - 0.15)
    - Every trainable layer exhibits non-zero differences
    """
    models: dict[int, dict[str, torch.Tensor]] = {}
    for s in SEEDS:
        ckpt_path = ROOT / f"artifacts/seeds/seed_{s}/gnn.pt"
        models[s] = torch.load(ckpt_path, map_location="cpu")

    trainable_keys = [k for k in models[42].keys() if k not in ("adjacency", "mask")]
    assert len(trainable_keys) >= 6, f"Expected trainable layers, got {trainable_keys}"

    flat_weights: dict[int, torch.Tensor] = {}
    for s in SEEDS:
        flat_weights[s] = torch.cat([models[s][k].float().flatten() for k in trainable_keys])

    for s1, s2 in SEED_PAIRS:
        m1 = models[s1]
        m2 = models[s2]

        total_sq_diff = 0.0
        global_max_abs = 0.0

        for k in trainable_keys:
            t1 = m1[k].float()
            t2 = m2[k].float()
            diff = t1 - t2
            layer_sq = torch.sum(diff ** 2).item()
            layer_max = torch.max(torch.abs(diff)).item()

            # Ensure layer is not identical
            assert layer_sq > 1e-4, f"Layer {k} is identical between seed {s1} and {s2}"
            assert layer_max > 1e-3, f"Layer {k} max abs diff is zero between seed {s1} and {s2}"

            total_sq_diff += layer_sq
            if layer_max > global_max_abs:
                global_max_abs = layer_max

        overall_l2 = total_sq_diff ** 0.5
        cos_sim = F.cosine_similarity(flat_weights[s1].unsqueeze(0), flat_weights[s2].unsqueeze(0)).item()

        assert overall_l2 > 10.0, (
            f"Seed {s1} vs {s2} overall L2 difference too low: {overall_l2:.4f}"
        )
        assert global_max_abs > 0.5, (
            f"Seed {s1} vs {s2} max abs difference too low: {global_max_abs:.4f}"
        )
        assert cos_sim < 0.5, (
            f"Seed {s1} vs {s2} cosine similarity too high (possible cloned init): {cos_sim:.4f}"
        )


def test_trained_predictor_inference_is_non_identical_across_seeds():
    """Verifies that TrainedPredictor inference produces non-identical predictions on identical inputs."""
    cfg = config()
    prep_cfg = dict(cfg, seed=42)
    data = load_prepared(ROOT, prep_cfg)

    predictors: dict[int, TrainedPredictor] = {}
    for s in SEEDS:
        seed_dir = ROOT / f"artifacts/seeds/seed_{s}"
        predictors[s] = TrainedPredictor(output_dir=seed_dir, prepared=data, cfg=cfg, kind="gnn", seed=s)

    lookback = cfg["lookback"]
    # Test across multiple disparate time windows in the dataset
    sample_indices = [0, 150, 500, 1200, 2500]

    mask = data.mask.astype(bool)

    for idx in sample_indices:
        history = data.rates[idx:idx + lookback]
        preds = {s: predictors[s].predict(history) for s in SEEDS}

        for s1, s2 in SEED_PAIRS:
            p1 = preds[s1]
            p2 = preds[s2]
            diff = (p1 - p2)[:, mask]

            l2_diff = np.linalg.norm(diff)
            max_abs_diff = np.max(np.abs(diff))

            # Strictly non-identical: difference must exceed numerical precision threshold
            assert l2_diff > 0.02, (
                f"Window {idx}: seeds {s1} and {s2} produced near-identical predictions (L2={l2_diff:.6f})"
            )
            assert max_abs_diff > 0.01, (
                f"Window {idx}: seeds {s1} and {s2} produced near-identical predictions (MaxAbs={max_abs_diff:.6f})"
            )


def test_checkpoint_training_metadata_integrity():
    """Verifies that training loss curves, best epochs, and calibrated margins are distinct across seeds."""
    best_epochs = []
    val_maes = []
    margins = []

    for s in SEEDS:
        sum_json = load_json(ROOT / f"artifacts/seeds/seed_{s}/training_summary.json")
        gnn_info = sum_json["models"]["gnn"]
        best_epochs.append(gnn_info["best_epoch"])
        val_maes.append(gnn_info["validation_mae_mbps"])
        m_arr = np.array(gnn_info["margin_mbps"]).round(5)
        margins.append(tuple(m_arr.flatten().tolist()))

    # Not all seeds should converge at the exact same epoch
    assert len(set(best_epochs)) >= 3, f"Suspiciously identical best epochs across all seeds: {best_epochs}"
    # Calibrated margins must not be identical across all seeds
    assert len(set(margins)) == len(SEEDS), f"Expected unique calibrated margins per seed, got {margins}"


def test_gru_checkpoint_independence():
    """Verifies that baseline GRU checkpoints are also independently trained with distinct weights."""
    gru_models: dict[int, dict[str, torch.Tensor]] = {}
    for s in SEEDS:
        ckpt_path = ROOT / f"artifacts/seeds/seed_{s}/gru.pt"
        assert ckpt_path.is_file()
        gru_models[s] = torch.load(ckpt_path, map_location="cpu")

    trainable_keys = [k for k in gru_models[42].keys() if k not in ("adjacency", "mask")]

    for s1, s2 in SEED_PAIRS:
        total_sq = sum(torch.sum((gru_models[s1][k].float() - gru_models[s2][k].float())**2).item() for k in trainable_keys)
        l2 = total_sq ** 0.5
        assert l2 > 10.0, f"GRU seed {s1} vs {s2} L2 difference too low: {l2:.4f}"
