"""Small dense GCN + GRU, matched no-graph GRU, and persistence comparison."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import fingerprint, load_json, save_json, sha256
from .data import Prepared, load_prepared, window_indices


class Predictor(Protocol):
    margin: np.ndarray

    def predict(self, history: np.ndarray) -> np.ndarray:
        """History only, ending before the target; output [horizon, node, destination]."""
        ...


class DemandNet(nn.Module):
    adjacency: torch.Tensor
    mask: torch.Tensor
    def __init__(self, adjacency: np.ndarray, mask: np.ndarray, hidden: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("adjacency", torch.tensor(adjacency, dtype=torch.float32))
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        # Preserve local OD information alongside neighbor-aggregated information.
        self.spatial = nn.Linear(16, hidden)
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * 8)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape
        neighbors = torch.einsum("ij,btjf->btif", self.adjacency, x)
        embedded = torch.relu(self.spatial(torch.cat((x, neighbors), dim=-1)))
        sequence = embedded.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        _, hidden = self.temporal(sequence)
        correction = self.head(hidden[-1]).reshape(batch, nodes, self.horizon, 8).permute(0, 2, 1, 3)
        # Residual forecast starts from persistence; learned corrections can go either way.
        return torch.clamp(x[:, -1:, :, :] + correction, min=0) * self.mask


class Persistence:
    def __init__(self, horizon: int, mask: np.ndarray, margin: np.ndarray | None = None):
        self.horizon, self.mask = horizon, mask
        self.margin = np.zeros((horizon, 8, 8)) if margin is None else margin

    def predict(self, history: np.ndarray) -> np.ndarray:
        if len(history) == 0:
            raise ValueError("Persistence needs at least one observation")
        return np.repeat(history[-1:]*self.mask, self.horizon, axis=0)


class TrainedPredictor:
    def __init__(
        self,
        folder: Path | None = None,
        prepared: Prepared | None = None,
        cfg: dict[str, Any] | None = None,
        kind: str = "gnn",
        seed: int | None = None,
        output_dir: Path | None = None,
    ):
        if output_dir is not None:
            folder = Path(output_dir)
        elif folder is None:
            folder = Path("artifacts/seeds") / f"seed_{seed}" if seed is not None else Path("artifacts")
        else:
            folder = Path(folder)

        if prepared is None:
            raise ValueError("TrainedPredictor requires prepared dataset for checksum and graph verification")
        if cfg is None:
            raise ValueError("TrainedPredictor requires configuration dict")

        cfg = copy.deepcopy(cfg)
        if seed is not None:
            cfg["seed"] = seed

        meta = load_json(folder / f"{kind}.json")
        if meta["config_hash"] != fingerprint(cfg) or meta["traffic_sha256"] != prepared.manifest["traffic_sha256"]:
            raise ValueError("Checkpoint configuration or training data mismatch; retrain")
        if meta["graph_hash"] != prepared.network.graph_hash:
            raise ValueError("Checkpoint graph mismatch")
        if sha256(folder / f"{kind}.pt") != meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        torch.set_num_threads(cfg["cpu_threads"])
        adjacency = prepared.network.adjacency() if kind == "gnn" else np.eye(8, dtype=np.float32)
        mask = prepared.mask
        self.model = DemandNet(adjacency, mask, cfg["hidden_size"], cfg["horizon"])
        self.model.load_state_dict(torch.load(folder / f"{kind}.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.scale = prepared.scale
        self.lookback = cfg["lookback"]
        self.margin = np.array(meta["margin_mbps"], dtype=np.float32)

    def predict(self, history: np.ndarray) -> np.ndarray:
        if history.shape != (self.lookback, 8, 8):
            raise ValueError(f"Expected exactly {self.lookback} historical matrices")
        with torch.inference_mode():
            x = torch.tensor(history[None] / self.scale, dtype=torch.float32)
            return self.model(x)[0].numpy() * self.scale


def make_windows(data: Prepared, cfg: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = data.manifest["splits"][split]
    indices = window_indices(data.times, lo, hi, cfg["lookback"], cfg["horizon"])
    x = np.stack([data.rates[k-cfg["lookback"]:k] for k in indices])
    y = np.stack([data.rates[k:k+cfg["horizon"]] for k in indices])
    return x, y, indices


def batch_prediction(model: DemandNet, x: np.ndarray, scale: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            chunks.append(model(torch.tensor(x[start:start+batch_size] / scale,
                                             dtype=torch.float32)).numpy() * scale)
    return np.concatenate(chunks)


def calibrate_margin(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray, q: float) -> np.ndarray:
    """Per-horizon positive validation residual quantile, pooled across active pairs."""
    residual = np.maximum(actual-predicted, 0)[:, :, mask.astype(bool)]
    quantiles = np.quantile(residual, q, axis=(0, 2))
    return quantiles[:, None, None] * mask[None]


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray, margin: np.ndarray,
                     mask: np.ndarray, high_threshold: float) -> list[dict[str, Any]]:
    rows = []
    for h in range(actual.shape[1]):
        y, p = actual[:, h, mask.astype(bool)], predicted[:, h, mask.astype(bool)]
        error = np.abs(y-p)
        high = y >= high_threshold
        rows.append({"horizon_bins": h+1, "mae_mbps": float(error.mean()),
                     "rmse_mbps": float(np.sqrt(np.mean((y-p)**2))),
                     "high_load_mae_mbps": float(error[high].mean()) if high.any() else None,
                     "high_load_threshold_mbps": high_threshold,
                     "upper_margin_coverage": float(np.mean(y <= p+margin[h, mask.astype(bool)])),
                     "points": int(y.size)})
    return rows


def train(
    root: Path,
    cfg: dict[str, Any],
    seed: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if seed is not None:
        cfg["seed"] = seed
    prep_manifest_path = root / "inputs/processed/manifest.json"
    if prep_manifest_path.exists():
        prep_seed = load_json(prep_manifest_path).get("config", {}).get("seed", cfg["seed"])
        prep_cfg = dict(cfg, seed=prep_seed)
    else:
        prep_cfg = cfg
    data = load_prepared(root, prep_cfg)
    folder = Path(output_dir) if output_dir is not None else (root / "artifacts")
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    xtrain, ytrain, _ = make_windows(data, cfg, "train")
    xval, yval, _ = make_windows(data, cfg, "validation")
    xtest, ytest, test_indices = make_windows(data, cfg, "test")
    high_threshold = float(np.quantile(ytrain[:, :, data.mask.astype(bool)], .9))
    logs: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for kind in ("gnn", "gru"):
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        adjacency = data.network.adjacency() if kind == "gnn" else np.eye(8, dtype=np.float32)
        model = DemandNet(adjacency, data.mask, cfg["hidden_size"], cfg["horizon"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        loader = DataLoader(TensorDataset(torch.tensor(xtrain / data.scale, dtype=torch.float32),
                                         torch.tensor(ytrain / data.scale, dtype=torch.float32)),
                            batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(cfg["seed"]))
        active = torch.tensor(data.mask.astype(bool))
        scale_t = torch.tensor(data.scale)
        best, best_epoch, stale = float("inf"), 0, 0
        best_state = copy.deepcopy(model.state_dict())
        started = time.perf_counter()
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            loss_sum = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                prediction = model(x)
                # Optimize equal-weight active OD MAE in the reported Mbps units.
                loss = torch.abs((prediction-y)*scale_t)[:, :, active].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(x)
            val = batch_prediction(model, xval, data.scale, cfg["batch_size"])
            score = float(np.abs(val-yval)[:, :, data.mask.astype(bool)].mean())
            logs.append({"model": kind, "epoch": epoch, "train_mae_mbps": loss_sum/len(xtrain),
                         "validation_mae_mbps": score})
            if score < best - 1e-6:
                best, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0 or stale >= cfg["patience"]:
                print(f"{kind} epoch {epoch}: validation MAE {score:.4f} Mbps", flush=True)
            if stale >= cfg["patience"]:
                break
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), folder / f"{kind}.pt")
        val = batch_prediction(model, xval, data.scale, cfg["batch_size"])
        margin = calibrate_margin(yval, val, data.mask, cfg["margin_quantile"])
        info = {"model": kind, "architecture": "local + normalized adjacency GCN, shared node GRU, residual head"
                if kind == "gnn" else "same parameters with identity adjacency; no neighbor communication",
                "parameters": sum(p.numel() for p in model.parameters()),
                "config_hash": fingerprint(cfg), "graph_hash": data.network.graph_hash,
                "traffic_sha256": data.manifest["traffic_sha256"],
                "checkpoint_sha256": sha256(folder / f"{kind}.pt"), "seed": cfg["seed"],
                "best_epoch": best_epoch, "validation_mae_mbps": best,
                "training_seconds": time.perf_counter()-started,
                "margin_mbps": margin.tolist(), "node_ids": data.manifest["graph"]["node_ids"],
                "pairs": data.manifest["pairs"], "scale": data.scale.tolist()}
        save_json(folder / f"{kind}.json", info)
        metadata[kind] = info
        for split, x, y in (("validation", xval, yval), ("test", xtest, ytest)):
            p = batch_prediction(model, x, data.scale, cfg["batch_size"])
            metrics.extend({"model": kind, "split": split, **row} for row in
                           forecast_metrics(y, p, margin, data.mask, high_threshold))
            if split == "test":
                np.savez_compressed(folder / f"{kind}_test_forecasts.npz", indices=test_indices,
                                    actual=y, predicted=p)
    val_p = np.repeat(xval[:, -1:, :, :], cfg["horizon"], axis=1)
    persistence_margin = calibrate_margin(yval, val_p, data.mask, cfg["margin_quantile"])
    save_json(folder / "persistence.json", {"margin_mbps": persistence_margin.tolist(),
                                           "config_hash": fingerprint(cfg),
                                           "traffic_sha256": data.manifest["traffic_sha256"]})
    for split, x, y in (("validation", xval, yval), ("test", xtest, ytest)):
        p = np.repeat(x[:, -1:, :, :], cfg["horizon"], axis=1)
        metrics.extend({"model": "persistence", "split": split, **row} for row in
                       forecast_metrics(y, p, persistence_margin, data.mask, high_threshold))
    pd.DataFrame(logs).to_csv(folder / "training.csv", index=False)
    pd.DataFrame(metrics).to_csv(folder / "forecast_metrics.csv", index=False)
    save_json(folder / "training_summary.json", {"config": cfg, "models": metadata,
                                                "torch_version": torch.__version__})
    return metadata
