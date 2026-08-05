#!/usr/bin/env python3
"""

random  : uniform random sample (only when explicitly requested)
coreset : PatchCore ApproximateGreedyCoresetSampler
none    : use every nominal training patch

outputs:
- metrics.csv: distance of every defect type to the nominal reference,
  ratio to normal, AUROC, and nominal-threshold overlap
- blending.csv: percentage of each defect inside/outside the nominal region
- distances.npz: raw normal and per-defect nearest-neighbour distances
- distance_distribution.png
- fitted reducer/checkpoint when applicable

When two or more methods are selected, comparison.csv is also produced.

The nominal training bank is subsampled exactly ONCE. The same selected patch
indices are reused by every method, including the Transformer latent bank.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import KernelPCA, PCA, SparsePCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

METHODS = (
    "original",
    "pca",
    "umap",
    "autoencoder",
    "kernel_pca",
    "sparse_pca",
    "transformer",
)

ALIASES = {
    "raw": "original",
    "orig": "original",
    "ae": "autoencoder",
    "kpca": "kernel_pca",
    "kernelpca": "kernel_pca",
    "spca": "sparse_pca",
    "sparsepca": "sparse_pca",
    "transformer_ae": "transformer",
}

MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)


@dataclass
class MethodResult:
    name: str
    metrics: pd.DataFrame
    blending: pd.DataFrame
    normal_distances: np.ndarray
    defect_distances: Dict[str, np.ndarray]
    fit_seconds: float
    transform_seconds: float
    reference_size: int
    output_dim: int
    extra: Dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_methods(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(x.strip().lower() for x in value.split(",") if x.strip())
    if "all" in tokens:
        return list(METHODS)
    result: List[str] = []
    for token in tokens:
        token = ALIASES.get(token, token)
        if token not in METHODS:
            raise ValueError(f"Unknown method '{token}'. Valid: all, {', '.join(METHODS)}")
        if token not in result:
            result.append(token)
    if not result:
        raise ValueError("No methods selected.")
    return result


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def add_repo_src(repo_root: Path) -> None:
    src = repo_root / "src"
    if not src.exists():
        raise FileNotFoundError(f"PatchCore src directory not found: {src}")
    sys.path.insert(0, str(src))


def build_patchcore_model(device: torch.device):
    import patchcore.backbones as backbones
    import patchcore.common as common
    import patchcore.patchcore as patchcore_module
    import patchcore.sampler as sampler

    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"

    model = patchcore_module.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(
            on_gpu=device.type == "cuda",
            num_workers=4,
        ),
    )
    return model


def build_loaders(data_root: Path, category: str, batch_size: int, workers: int, seed: int):
    from patchcore.datasets import mvtec

    train_dataset = mvtec.MVTecDataset(
        source=str(data_root), classname=category, resize=256,
        train_val_split=1, imagesize=224,
        split=mvtec.DatasetSplit.TRAIN, seed=seed, augment=False,
    )
    test_dataset = mvtec.MVTecDataset(
        source=str(data_root), classname=category, resize=256,
        imagesize=224, split=mvtec.DatasetSplit.TEST, seed=seed,
    )
    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        train_dataset,
        test_dataset,
        DataLoader(train_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def extract_patch_embeddings(model, images: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
    images = images.to(model.device)
    with torch.no_grad():
        embeddings, patch_shapes = model._embed(
            images, detach=True, provide_patch_shapes=True
        )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    h, w = patch_shapes[0]
    embeddings = embeddings.reshape(images.shape[0], h * w, -1)
    return embeddings, (h, w)


def masks_to_patch_labels(
    masks: torch.Tensor,
    patch_grid: Tuple[int, int],
    anomaly_fraction_threshold: float,
) -> np.ndarray:
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)
    masks = masks.float()
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    fractions = F.adaptive_avg_pool2d(masks, output_size=patch_grid).squeeze(1)
    return (fractions >= anomaly_fraction_threshold).cpu().numpy()


def collect_training_embeddings(model, loader: DataLoader, keep_sequences: bool):
    flat: List[np.ndarray] = []
    sequences: List[np.ndarray] = []
    patch_grid: Optional[Tuple[int, int]] = None
    for i, batch in enumerate(loader):
        embeddings, patch_grid = extract_patch_embeddings(model, batch["image"])
        flat.append(embeddings.reshape(-1, embeddings.shape[-1]))
        if keep_sequences:
            sequences.append(embeddings.astype(np.float16))
        if (i + 1) % 10 == 0 or i + 1 == len(loader):
            print(f"[features] train {i + 1}/{len(loader)}")
    if patch_grid is None:
        raise RuntimeError("No training features extracted.")
    return (
        np.concatenate(flat).astype(np.float32),
        np.concatenate(sequences) if keep_sequences else None,
        patch_grid,
    )


def safe_name(value) -> str:
    if torch.is_tensor(value):
        value = value.item()
    return str(value)


def random_cap(values: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if maximum > 0 and len(values) > maximum:
        indices = rng.choice(len(values), maximum, replace=False)
        return values[indices]
    return values


def collect_test_embeddings(
    model,
    loader: DataLoader,
    anomaly_fraction_threshold: float,
    max_normal: int,
    max_defect: int,
    seed: int,
):
    normal_batches: List[np.ndarray] = []
    defects: MutableMapping[str, List[np.ndarray]] = defaultdict(list)

    for i, batch in enumerate(loader):
        embeddings, patch_grid = extract_patch_embeddings(model, batch["image"])
        labels = masks_to_patch_labels(
            batch["mask"], patch_grid, anomaly_fraction_threshold
        )
        for image_index in range(embeddings.shape[0]):
            patch_labels = labels[image_index].reshape(-1)
            defect_type = safe_name(batch["anomaly"][image_index])
            normal_batches.append(embeddings[image_index][~patch_labels])
            if defect_type != "good" and patch_labels.any():
                defects[defect_type].append(embeddings[image_index][patch_labels])
        if (i + 1) % 10 == 0 or i + 1 == len(loader):
            print(f"[features] test {i + 1}/{len(loader)}")

    rng = np.random.default_rng(seed)
    normal = random_cap(np.concatenate(normal_batches).astype(np.float32), max_normal, rng)
    processed = {
        name: random_cap(np.concatenate(parts).astype(np.float32), max_defect, rng)
        for name, parts in sorted(defects.items())
    }
    if not processed:
        raise RuntimeError("No defective patches found for this category.")
    return normal, processed


def subsample_reference(
    features: np.ndarray,
    mode: str,
    max_patches: int,
    percentage: float,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(features)
    if mode == "none":
        return features, np.arange(n, dtype=np.int64)

    if mode == "random":
        size = n if max_patches <= 0 else min(max_patches, n)
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=size, replace=False)
        return features[indices], indices.astype(np.int64)

    if mode == "coreset":
        import patchcore.sampler as sampler

        effective = percentage
        if max_patches > 0:
            effective = min(effective, max_patches / n)
        effective = min(max(effective, 1.0 / n), 1.0)
        selector = sampler.ApproximateGreedyCoresetSampler(
            percentage=effective,
            device=device,
        )
        selected = np.asarray(selector.run(features), dtype=np.float32)
        stored = getattr(selector, "last_selected_indices", None)
        if stored is not None:
            indices = np.asarray(stored, dtype=np.int64)
        else:
            # Upstream PatchCore does not always expose the selected row indices.
            # Recover them by matching every selected coreset vector to its exact
            # nearest row in the original nominal feature matrix. This lets every
            # dimensionality-reduction method reuse the SAME nominal patches.
            index_finder = NearestNeighbors(
                n_neighbors=1,
                metric="euclidean",
                n_jobs=-1,
            )
            index_finder.fit(features)
            recovery_distances, recovered = index_finder.kneighbors(
                selected,
                return_distance=True,
            )
            if not np.allclose(recovery_distances[:, 0], 0.0, atol=1e-6):
                raise RuntimeError(
                    "Could not recover exact PatchCore coreset indices. "
                    "Modify ApproximateGreedyCoresetSampler to expose "
                    "last_selected_indices."
                )
            indices = recovered[:, 0].astype(np.int64)
        return selected, indices

    raise ValueError(f"Unknown subsampling mode: {mode}")


def transform_batches(function, values: np.ndarray, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [function(values[start:start + batch_size]) for start in range(0, len(values), batch_size)],
        axis=0,
    )


def evaluate(
    method: str,
    reference: np.ndarray,
    normal: np.ndarray,
    defects: Mapping[str, np.ndarray],
    nominal_percentile: float,
    fit_seconds: float,
    transform_seconds: float,
    extra: Optional[Dict[str, float]] = None,
) -> MethodResult:
    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn_model.fit(reference)
    normal_distances = nn_model.kneighbors(normal, return_distance=True)[0][:, 0]
    normal_mean = float(normal_distances.mean())
    threshold = float(np.percentile(normal_distances, nominal_percentile))

    rows, blend_rows = [], []
    distance_map: Dict[str, np.ndarray] = {}
    for defect_type, values in sorted(defects.items()):
        distances = nn_model.kneighbors(values, return_distance=True)[0][:, 0]
        distance_map[defect_type] = distances
        labels = np.concatenate([np.zeros(len(normal_distances)), np.ones(len(distances))])
        scores = np.concatenate([normal_distances, distances])
        inside = float(100 * np.mean(distances <= threshold))

        rows.append({
            "Method": method,
            "Defect Type": defect_type,
            "Patch Count": len(distances),
            "Normal Mean Distance": normal_mean,
            "Normal Median Distance": float(np.median(normal_distances)),
            "Defect Mean Distance": float(distances.mean()),
            "Defect Median Distance": float(np.median(distances)),
            "Defect Std Distance": float(distances.std()),
            "Ratio to Normal": float(distances.mean() / max(normal_mean, 1e-12)),
            "AUROC vs Normal": float(roc_auc_score(labels, scores)),
            "Nominal Threshold": threshold,
            "Inside Nominal Region (%)": inside,
            "Outside Nominal Region (%)": 100 - inside,
            "Reference Size": len(reference),
            "Output Dimension": reference.shape[1],
            "Fit Seconds": fit_seconds,
            "Transform Seconds": transform_seconds,
        })
        blend_rows.append({
            "Method": method,
            "Defect Type": defect_type,
            "Nominal Percentile": nominal_percentile,
            "Nominal Threshold": threshold,
            "Inside Nominal Region (%)": inside,
            "Outside Nominal Region (%)": 100 - inside,
        })

    return MethodResult(
        name=method,
        metrics=pd.DataFrame(rows),
        blending=pd.DataFrame(blend_rows),
        normal_distances=normal_distances,
        defect_distances=distance_map,
        fit_seconds=fit_seconds,
        transform_seconds=transform_seconds,
        reference_size=len(reference),
        output_dim=reference.shape[1],
        extra=extra or {},
    )


def save_result(result: MethodResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(directory / "metrics.csv", index=False)
    result.blending.to_csv(directory / "blending.csv", index=False)

    arrays = {"normal": result.normal_distances}
    arrays.update({f"defect__{k}": v for k, v in result.defect_distances.items()})
    np.savez_compressed(directory / "distances.npz", **arrays)

    all_distances = np.concatenate([result.normal_distances, *result.defect_distances.values()])
    bins = np.linspace(all_distances.min(), np.percentile(all_distances, 99.5), 70)
    plt.figure(figsize=(10, 6))
    plt.hist(result.normal_distances, bins=bins, density=True, histtype="step", linewidth=2, label="normal")
    for name, distances in sorted(result.defect_distances.items()):
        plt.hist(distances, bins=bins, density=True, histtype="step", linewidth=2, label=name)
    plt.xlabel("Distance to nearest nominal patch")
    plt.ylabel("Density")
    plt.title(f"{result.name}: nearest-nominal distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "distance_distribution.png", dpi=180)
    plt.close()

    summary = {
        "method": result.name,
        "reference_size": result.reference_size,
        "output_dimension": result.output_dim,
        "fit_seconds": result.fit_seconds,
        "transform_seconds": result.transform_seconds,
        **result.extra,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))


class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decoder(z), z


def encode_ae(model, values, mean, std, device, batch_size):
    scaled = ((values - mean) / std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(scaled)), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model.encode(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def run_autoencoder(train, normal, defects, args, directory, device):
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train))
    split = max(1, int(0.9 * len(indices)))
    train_idx, val_idx = indices[:split], indices[split:]
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]

    mean = train[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(train[train_idx].std(axis=0, keepdims=True), 1e-8)
    train_scaled = ((train[train_idx] - mean) / std).astype(np.float32)
    val_scaled = ((train[val_idx] - mean) / std).astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_scaled)), batch_size=args.ae_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_scaled)), batch_size=args.ae_batch_size, shuffle=False)

    model = EmbeddingAutoencoder(train.shape[1], args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ae_lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_loss, best_state, bad_epochs = math.inf, None, 0
    train_history, val_history = [], []

    start = time.perf_counter()
    for epoch in range(args.ae_epochs):
        model.train(); total = count = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward(); optimizer.step()
            total += loss.item() * len(batch); count += len(batch)
        train_loss = total / count

        model.eval(); total = count = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction, _ = model(batch)
                loss = criterion(reconstruction, batch)
                total += loss.item() * len(batch); count += len(batch)
        val_loss = total / count
        train_history.append(train_loss); val_history.append(val_loss)
        print(f"[autoencoder] epoch {epoch + 1:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss; best_state = copy.deepcopy(model.state_dict()); bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.ae_patience:
                print("[autoencoder] early stopping")
                break

    fit_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Autoencoder produced no checkpoint.")
    model.load_state_dict(best_state)

    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "input_dim": train.shape[1], "latent_dim": args.latent_dim}, directory / "autoencoder.pt")
    plt.figure(figsize=(8, 5)); plt.plot(train_history, label="Training"); plt.plot(val_history, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Embedding autoencoder"); plt.legend(); plt.tight_layout()
    plt.savefig(directory / "training_curve.png", dpi=180); plt.close()

    start = time.perf_counter()
    train_z = encode_ae(model, train, mean, std, device, args.ae_encode_batch_size)
    normal_z = encode_ae(model, normal, mean, std, device, args.ae_encode_batch_size)
    defects_z = {k: encode_ae(model, v, mean, std, device, args.ae_encode_batch_size) for k, v in defects.items()}
    transform_seconds = time.perf_counter() - start
    return train_z, normal_z, defects_z, fit_seconds, transform_seconds, {"best_validation_loss": float(best_loss)}


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.sequences, self.mean, self.std = sequences, mean, std
    def __len__(self): return len(self.sequences)
    def __getitem__(self, index):
        x = self.sequences[index].astype(np.float32)
        return torch.from_numpy((x - self.mean) / self.std)


class SpatialTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, patches, latent_dim, model_dim, heads, encoder_layers, decoder_layers, dropout):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, patches, model_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=model_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=model_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.to_latent = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim))
        self.from_latent = nn.Linear(latent_dim, model_dim)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)
        self.output_projection = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, input_dim))

    def encode(self, x):
        h = self.input_projection(x) + self.position_embedding
        return self.to_latent(self.encoder(h))

    def forward(self, x):
        z = self.encode(x)
        h = self.from_latent(z) + self.position_embedding
        return self.output_projection(self.decoder(h)), z


def encode_transformer_sequences(model, sequences, mean, std, device, batch_size):
    loader = DataLoader(SequenceDataset(sequences, mean, std), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs.append(model.encode(batch.to(device)).cpu().numpy())
    latent = np.concatenate(outputs)
    return latent.reshape(-1, latent.shape[-1])


def collect_transformer_test(model_pc, model_t, loader, mean, std, device, threshold):
    normal, defects = [], defaultdict(list)
    model_t.eval()
    for i, batch in enumerate(loader):
        embeddings, patch_grid = extract_patch_embeddings(model_pc, batch["image"])
        labels = masks_to_patch_labels(batch["mask"], patch_grid, threshold)
        normalized = ((embeddings.astype(np.float32) - mean) / std).astype(np.float32)
        with torch.no_grad():
            latent = model_t.encode(torch.from_numpy(normalized).to(device)).cpu().numpy()
        for image_index in range(latent.shape[0]):
            mask = labels[image_index].reshape(-1)
            defect_type = safe_name(batch["anomaly"][image_index])
            normal.append(latent[image_index][~mask])
            if defect_type != "good" and mask.any():
                defects[defect_type].append(latent[image_index][mask])
        if (i + 1) % 10 == 0 or i + 1 == len(loader):
            print(f"[transformer] test {i + 1}/{len(loader)}")
    return np.concatenate(normal), {k: np.concatenate(v) for k, v in sorted(defects.items())}


def run_transformer(sequences, patch_grid, patchcore_model, test_loader, selected_indices, args, directory, device):
    if sequences is None:
        raise RuntimeError("Transformer requires intact training sequences.")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(sequences))
    split = max(1, int(0.85 * len(indices)))
    train_idx, val_idx = indices[:split], indices[split:]
    if len(val_idx) == 0: val_idx = train_idx[-1:]
    train_seq, val_seq = sequences[train_idx], sequences[val_idx]
    mean = train_seq.astype(np.float32).mean(axis=(0, 1))
    std = np.maximum(train_seq.astype(np.float32).std(axis=(0, 1)), 1e-6)

    train_loader = DataLoader(SequenceDataset(train_seq, mean, std), batch_size=args.transformer_batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_seq, mean, std), batch_size=args.transformer_batch_size, shuffle=False)

    model = SpatialTransformerAutoencoder(
        input_dim=sequences.shape[-1], patches=patch_grid[0] * patch_grid[1],
        latent_dim=args.latent_dim, model_dim=args.transformer_model_dim,
        heads=args.transformer_heads, encoder_layers=args.transformer_encoder_layers,
        decoder_layers=args.transformer_decoder_layers, dropout=args.transformer_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.transformer_lr, weight_decay=args.transformer_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.transformer_epochs, 1), eta_min=1e-6)
    criterion = nn.SmoothL1Loss()
    best_loss, best_state, bad_epochs = math.inf, None, 0
    train_history, val_history = [], []

    start = time.perf_counter()
    for epoch in range(args.transformer_epochs):
        model.train(); total = count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * len(batch); count += len(batch)
        train_loss = total / count

        model.eval(); total = count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstruction, _ = model(batch)
                loss = criterion(reconstruction, batch)
                total += loss.item() * len(batch); count += len(batch)
        val_loss = total / count
        train_history.append(train_loss); val_history.append(val_loss)
        print(f"[transformer] epoch {epoch + 1:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss; best_state = copy.deepcopy(model.state_dict()); bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.transformer_patience:
                print("[transformer] early stopping")
                break
        scheduler.step()

    fit_seconds = time.perf_counter() - start
    if best_state is None: raise RuntimeError("Transformer produced no checkpoint.")
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "patch_grid": patch_grid, "input_dim": sequences.shape[-1],
                "latent_dim": args.latent_dim}, directory / "transformer_autoencoder.pt")
    plt.figure(figsize=(8, 5)); plt.plot(train_history, label="Training"); plt.plot(val_history, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Smooth L1"); plt.title("Spatial Transformer autoencoder"); plt.legend(); plt.tight_layout()
    plt.savefig(directory / "training_curve.png", dpi=180); plt.close()

    start = time.perf_counter()
    all_train_z = encode_transformer_sequences(model, sequences, mean, std, device, args.transformer_batch_size)
    normal_z, defects_z = collect_transformer_test(
        patchcore_model, model, test_loader, mean, std, device, args.anomaly_fraction_threshold
    )
    transform_seconds = time.perf_counter() - start

    # The Transformer must train on intact image sequences, but its nominal
    # nearest-neighbour reference must use the exact SAME patches selected once
    # before any method runs. Flattening the sequence latents preserves the same
    # patch order as collect_training_embeddings().
    if len(all_train_z) <= int(np.max(selected_indices)):
        raise RuntimeError(
            "Selected nominal indices do not match the flattened Transformer "
            "latent bank."
        )
    train_z = all_train_z[selected_indices]
    return train_z, normal_z, defects_z, fit_seconds, transform_seconds, {"best_validation_loss": float(best_loss)}


def run_method(method, sampled_train, selected_indices, normal, defects, sequences, patch_grid, patchcore_model, test_loader, args, directory, device):
    print(f"\n{'=' * 72}\nMETHOD: {method}\n{'=' * 72}")

    if method == "original":
        train_z, normal_z, defects_z = sampled_train, normal, dict(defects)
        fit_seconds = transform_seconds = 0.0; extra = {}

    elif method == "pca":
        reducer = PCA(n_components=args.latent_dim, random_state=args.seed)
        start = time.perf_counter(); train_z = reducer.fit_transform(sampled_train); fit_seconds = time.perf_counter() - start
        start = time.perf_counter(); normal_z = reducer.transform(normal); defects_z = {k: reducer.transform(v) for k, v in defects.items()}; transform_seconds = time.perf_counter() - start
        joblib.dump(reducer, directory / "pca.joblib")
        extra = {"explained_variance": float(reducer.explained_variance_ratio_.sum())}

    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("Install UMAP with: pip install umap-learn") from exc
        reducer = umap.UMAP(
            n_components=args.umap_dim, n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist, metric="euclidean", random_state=args.seed,
        )
        start = time.perf_counter(); train_z = reducer.fit_transform(sampled_train); fit_seconds = time.perf_counter() - start
        start = time.perf_counter(); normal_z = reducer.transform(normal); defects_z = {k: reducer.transform(v) for k, v in defects.items()}; transform_seconds = time.perf_counter() - start
        joblib.dump(reducer, directory / "umap.joblib"); extra = {}

    elif method == "autoencoder":
        train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra = run_autoencoder(sampled_train, normal, defects, args, directory, device)

    elif method == "kernel_pca":
        # No second sampling: Kernel PCA receives the exact nominal reference
        # selected once in main(), just like every other method.
        train_input = sampled_train
        scaler = StandardScaler(); scaled = scaler.fit_transform(train_input)
        gamma = args.kpca_gamma if args.kpca_gamma > 0 else 1.0 / scaled.shape[1]
        reducer = KernelPCA(
            n_components=args.latent_dim, kernel="rbf", gamma=gamma,
            eigen_solver="randomized", random_state=args.seed,
            remove_zero_eig=True, n_jobs=-1,
        )
        start = time.perf_counter(); train_z = reducer.fit_transform(scaled); fit_seconds = time.perf_counter() - start
        transform = lambda x: reducer.transform(scaler.transform(x))
        start = time.perf_counter(); normal_z = transform_batches(transform, normal, args.transform_batch_size); defects_z = {k: transform_batches(transform, v, args.transform_batch_size) for k, v in defects.items()}; transform_seconds = time.perf_counter() - start
        joblib.dump({"scaler": scaler, "reducer": reducer}, directory / "kernel_pca.joblib"); extra = {"gamma": float(gamma)}

    elif method == "sparse_pca":
        # No second sampling: Sparse PCA receives the exact nominal reference
        # selected once in main(), just like every other method.
        train_input = sampled_train
        scaler = StandardScaler(); scaled = scaler.fit_transform(train_input)
        reducer = SparsePCA(
            n_components=args.latent_dim, alpha=args.spca_alpha,
            ridge_alpha=args.spca_ridge_alpha, max_iter=args.spca_max_iter,
            tol=args.spca_tol, method="lars", random_state=args.seed, n_jobs=-1,
        )
        start = time.perf_counter(); train_z = reducer.fit_transform(scaled); fit_seconds = time.perf_counter() - start
        transform = lambda x: reducer.transform(scaler.transform(x))
        start = time.perf_counter(); normal_z = transform_batches(transform, normal, args.transform_batch_size); defects_z = {k: transform_batches(transform, v, args.transform_batch_size) for k, v in defects.items()}; transform_seconds = time.perf_counter() - start
        joblib.dump({"scaler": scaler, "reducer": reducer}, directory / "sparse_pca.joblib"); extra = {}

    elif method == "transformer":
        train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra = run_transformer(
            sequences, patch_grid, patchcore_model, test_loader,
            selected_indices, args, directory, device
        )
    else:
        raise ValueError(method)

    return evaluate(
        method,
        np.asarray(train_z, dtype=np.float32),
        np.asarray(normal_z, dtype=np.float32),
        {k: np.asarray(v, dtype=np.float32) for k, v in defects_z.items()},
        args.nominal_percentile,
        fit_seconds,
        transform_seconds,
        extra,
    )


def make_comparison(results: Sequence[MethodResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_table = pd.concat([r.metrics for r in results], ignore_index=True)
    metrics = [
        "Normal Mean Distance", "Defect Mean Distance", "Defect Median Distance",
        "Ratio to Normal", "AUROC vs Normal", "Inside Nominal Region (%)",
        "Outside Nominal Region (%)", "Reference Size", "Output Dimension",
        "Fit Seconds", "Transform Seconds",
    ]
    wide = long_table.pivot(index="Defect Type", columns="Method", values=metrics)
    wide.columns = [f"{method} | {metric}" for metric, method in wide.columns]
    return long_table, wide.reset_index()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PatchCore dimensionality-reduction pipeline")
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("dim_reduction_results"))
    p.add_argument("--category", choices=MVTEC_CATEGORIES, required=True)
    p.add_argument("--methods", nargs="+", default=["all"], help="all or method names; comma-separated also works")
    p.add_argument("--subsampling", choices=("random", "coreset", "none"), default="coreset")
    p.add_argument("--subsample-percentage", type=float, default=0.1)
    p.add_argument("--max-train-patches", type=int, default=30000)
    p.add_argument("--max-normal-test-patches", type=int, default=15000)
    p.add_argument("--max-defect-test-patches", type=int, default=15000)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--anomaly-fraction-threshold", type=float, default=0.10)
    p.add_argument("--nominal-percentile", type=float, default=95.0)
    p.add_argument("--transform-batch-size", type=int, default=1000)

    p.add_argument("--umap-dim", type=int, default=16)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.0)

    p.add_argument("--ae-epochs", type=int, default=100)
    p.add_argument("--ae-patience", type=int, default=10)
    p.add_argument("--ae-batch-size", type=int, default=1024)
    p.add_argument("--ae-encode-batch-size", type=int, default=2048)
    p.add_argument("--ae-lr", type=float, default=1e-3)

    p.add_argument("--kpca-gamma", type=float, default=-1.0)

    p.add_argument("--spca-alpha", type=float, default=1.0)
    p.add_argument("--spca-ridge-alpha", type=float, default=0.01)
    p.add_argument("--spca-max-iter", type=int, default=300)
    p.add_argument("--spca-tol", type=float, default=1e-4)

    p.add_argument("--transformer-epochs", type=int, default=100)
    p.add_argument("--transformer-patience", type=int, default=15)
    p.add_argument("--transformer-batch-size", type=int, default=2)
    p.add_argument("--transformer-lr", type=float, default=2e-4)
    p.add_argument("--transformer-weight-decay", type=float, default=1e-4)
    p.add_argument("--transformer-model-dim", type=int, default=256)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-encoder-layers", type=int, default=4)
    p.add_argument("--transformer-decoder-layers", type=int, default=4)
    p.add_argument("--transformer-dropout", type=float, default=0.1)
    return p


def validate(args) -> None:
    if not 0 < args.subsample_percentage <= 1: raise ValueError("--subsample-percentage must be in (0,1]")
    if not 0 < args.anomaly_fraction_threshold <= 1: raise ValueError("--anomaly-fraction-threshold must be in (0,1]")
    if not 0 < args.nominal_percentile < 100: raise ValueError("--nominal-percentile must be in (0,100)")
    if args.latent_dim <= 0: raise ValueError("--latent-dim must be positive")


def main() -> None:
    args = parser().parse_args()
    validate(args)
    methods = parse_methods(args.methods)
    device = resolve_device(args.device)
    set_seed(args.seed)

    repo_root = args.repo_root.resolve(); data_root = args.data_root.resolve()
    add_repo_src(repo_root)
    run_dir = args.output_dir.resolve() / args.category / args.subsampling
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({"repo_root": str(repo_root), "data_root": str(data_root), "output_dir": str(run_dir), "methods": methods, "device": str(device)})
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(json.dumps(config, indent=2, default=str))

    model = build_patchcore_model(device)
    train_dataset, test_dataset, train_loader, test_loader = build_loaders(
        data_root, args.category, args.batch_size, args.num_workers, args.seed
    )
    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    all_train, sequences, patch_grid = collect_training_embeddings(
        model, train_loader, keep_sequences="transformer" in methods
    )
    sampled_train, selected_indices = subsample_reference(
        all_train, args.subsampling, args.max_train_patches,
        args.subsample_percentage, args.seed, device,
    )
    np.save(run_dir / "selected_reference_indices.npy", selected_indices)
    print(f"All nominal patches: {all_train.shape}")
    print(f"Selected nominal reference: {sampled_train.shape}")

    normal, defects = collect_test_embeddings(
        model, test_loader, args.anomaly_fraction_threshold,
        args.max_normal_test_patches, args.max_defect_test_patches, args.seed,
    )
    print(f"Normal test patches: {normal.shape}")
    for name, values in defects.items(): print(f"{name}: {values.shape}")

    results: List[MethodResult] = []
    for method in methods:
        method_dir = run_dir / method; method_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_method(
                method, sampled_train, selected_indices, normal, defects,
                sequences, patch_grid, model, test_loader, args, method_dir, device,
            )
            save_result(result, method_dir)
            results.append(result)
            print(result.metrics.round(4).to_string(index=False))
        except Exception as exc:
            error = {"method": method, "error_type": type(exc).__name__, "message": str(exc)}
            (method_dir / "error.json").write_text(json.dumps(error, indent=2))
            print(f"[ERROR] {method}: {type(exc).__name__}: {exc}")

    if not results:
        raise RuntimeError("Every selected method failed. Check per-method error.json files.")

    long_table, comparison = make_comparison(results)
    long_table.to_csv(run_dir / "all_method_results_long.csv", index=False)
    if len(results) > 1:
        comparison.to_csv(run_dir / "comparison.csv", index=False)
        print("\nCOMPARISON\n")
        print(comparison.round(4).to_string(index=False))

    print(f"\nFinished. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
