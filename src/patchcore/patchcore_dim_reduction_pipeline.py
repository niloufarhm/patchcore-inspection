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
- distance_cdf.png
- representation_2d_pca.png
- nearest_neighbor_feature_retrieval.png
- blending_comparison.png (run-level)
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
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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

from svdd.patch_feature_svdd import (
    PatchFeatureSVDDNet,
    fit_deep_svdd,
    transform_deep_svdd,
)

METHODS = (
    "original",
    "pca",
    "umap",
    "autoencoder",
    "kernel_pca",
    #"sparse_pca",
    "transformer",
)

ALIASES = {
    "raw": "original",
    "orig": "original",
    "ae": "autoencoder",
    "kpca": "kernel_pca",
    "kernelpca": "kernel_pca",
    "spca": "sparse_pca",
    #"sparsepca": "sparse_pca",
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


def parse_categories(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(x.strip().lower() for x in value.split(",") if x.strip())
    if "all" in tokens:
        return list(MVTEC_CATEGORIES)
    result: List[str] = []
    for token in tokens:
        if token not in MVTEC_CATEGORIES:
            raise ValueError(f"Unknown category '{token}'. Valid: all, {', '.join(MVTEC_CATEGORIES)}")
        if token not in result:
            result.append(token)
    if not result:
        raise ValueError("No categories selected.")
    return result


def parse_subsamplings(values: Sequence[str]) -> List[str]:
    valid = ("random", "coreset", "none")
    tokens: List[str] = []
    for value in values:
        tokens.extend(x.strip().lower() for x in value.split(",") if x.strip())
    if "all" in tokens:
        return list(valid)
    result: List[str] = []
    for token in tokens:
        if token not in valid:
            raise ValueError(f"Unknown subsampling '{token}'. Valid: all, {', '.join(valid)}")
        if token not in result:
            result.append(token)
    if not result:
        raise ValueError("No subsampling methods selected.")
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
    preselected_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(features)
    if mode == "none":
        return features, np.arange(n, dtype=np.int64)

    target_size = max(1, int(round(n * percentage)))
    if max_patches > 0:
        target_size = min(target_size, max_patches)
    target_size = min(target_size, n)

    if mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=target_size, replace=False).astype(np.int64)
        return features[indices], indices

    if mode == "coreset":
        # When a feature cache already contains exactly the requested coreset,
        # reuse those saved indices instead of recomputing it.
        if preselected_indices is not None:
            cached = np.asarray(preselected_indices, dtype=np.int64)
            if len(cached) > 0:
                if cached.min() < 0 or cached.max() >= n:
                    raise ValueError("Cached coreset indices are out of bounds.")
                # PatchCore samplers may round floor/ceil differently by one or a
                # few samples.  If the saved cache is effectively the requested
                # percentage, reuse it instead of running greedy coreset again.
                cached_fraction = len(cached) / float(n)
                tolerance = max(2.0 / float(n), 1e-3)
                max_ok = max_patches <= 0 or len(cached) <= max_patches
                if max_ok and abs(cached_fraction - percentage) <= tolerance:
                    print(
                        f"[subsampling] reusing cached coreset indices: "
                        f"{len(cached)}/{n} ({cached_fraction:.4%})"
                    )
                    return np.asarray(features[cached], dtype=np.float32), cached

        import patchcore.sampler as sampler

        effective = target_size / n
        selector = sampler.ApproximateGreedyCoresetSampler(
            percentage=effective,
            device=device,
        )
        selected = np.asarray(selector.run(np.array(features, dtype=np.float32, copy=True)), dtype=np.float32)
        stored = getattr(selector, "last_selected_indices", None)
        if stored is not None:
            indices = np.asarray(stored, dtype=np.int64)
        else:
            index_finder = NearestNeighbors(
                n_neighbors=1, metric="euclidean", n_jobs=-1,
            )
            index_finder.fit(features)
            recovery_distances, recovered = index_finder.kneighbors(
                selected, return_distance=True,
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


def load_cached_category(cache_root: Path, category: str, args) -> Dict[str, Any]:
    category_dir = cache_root / category
    required = [
        "train_features_full.npy", "coreset_indices.npy", "test_features.npy",
        "test_image_offsets.npy", "test_metadata.json", "test_masks_uint8.npz",
        "config.json",
    ]
    missing = [name for name in required if not (category_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete feature cache for {category}: missing {', '.join(missing)} in {category_dir}"
        )

    all_train = np.asarray(np.load(category_dir / "train_features_full.npy", mmap_mode="r"), dtype=np.float32)
    cached_coreset_indices = np.asarray(np.load(category_dir / "coreset_indices.npy"), dtype=np.int64)
    test_features = np.asarray(np.load(category_dir / "test_features.npy", mmap_mode="r"), dtype=np.float32)
    test_offsets = np.asarray(np.load(category_dir / "test_image_offsets.npy"), dtype=np.int64)
    masks = np.load(category_dir / "test_masks_uint8.npz")["masks"]
    metadata = json.loads((category_dir / "test_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((category_dir / "config.json").read_text(encoding="utf-8"))

    patch_grid = tuple(config.get("train_patch_grid", config.get("test_patch_grid", [])))
    if len(patch_grid) != 2:
        raise ValueError(f"Invalid patch grid in {category_dir / 'config.json'}")
    patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
    patches_per_image = patch_grid[0] * patch_grid[1]

    if len(all_train) % patches_per_image != 0:
        raise ValueError(
            f"Training feature count {len(all_train)} cannot be reshaped into {patch_grid} sequences."
        )
    sequences = all_train.reshape(-1, patches_per_image, all_train.shape[1])

    normal_parts: List[np.ndarray] = []
    defect_parts: MutableMapping[str, List[np.ndarray]] = defaultdict(list)
    test_sequences: List[np.ndarray] = []
    test_labels: List[np.ndarray] = []
    test_types: List[str] = []

    if len(metadata) + 1 != len(test_offsets) or len(metadata) != len(masks):
        raise ValueError("Cached metadata, offsets, and masks have inconsistent lengths.")

    for i, meta in enumerate(metadata):
        start, end = int(test_offsets[i]), int(test_offsets[i + 1])
        image_features = test_features[start:end]
        if len(image_features) != patches_per_image:
            raise ValueError(
                f"Test image {i} has {len(image_features)} patches; expected {patches_per_image}."
            )
        mask_tensor = torch.as_tensor(masks[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        fractions = F.adaptive_avg_pool2d(mask_tensor, output_size=patch_grid).squeeze().cpu().numpy()
        labels = (fractions >= args.anomaly_fraction_threshold).reshape(-1)
        defect_type = str(meta.get("anomaly_type", "good"))

        normal_parts.append(image_features[~labels])
        if defect_type != "good" and labels.any():
            defect_parts[defect_type].append(image_features[labels])

        test_sequences.append(image_features)
        test_labels.append(labels)
        test_types.append(defect_type)

    rng = np.random.default_rng(args.seed)
    normal = random_cap(np.concatenate(normal_parts).astype(np.float32), args.max_normal_test_patches, rng)
    defects = {
        name: random_cap(np.concatenate(parts).astype(np.float32), args.max_defect_test_patches, rng)
        for name, parts in sorted(defect_parts.items())
    }
    if not defects:
        raise RuntimeError(f"No defective patches found in cached category {category}.")

    return {
        "all_train": all_train,
        "sequences": sequences.astype(np.float16) if ("transformer" in parse_methods(args.methods) or any(m == "transformer" for m, _ in getattr(args, "reduction_chain_parsed", []))) else None,
        "patch_grid": patch_grid,
        "normal": normal,
        "defects": defects,
        "cached_coreset_indices": cached_coreset_indices,
        "cached_test": {
            "sequences": np.stack(test_sequences).astype(np.float32),
            "labels": test_labels,
            "types": test_types,
        },
        "cache_config": config,
    }


def collect_transformer_test_cached(model_t, cached_test, mean, std, device, batch_size):
    sequences = cached_test["sequences"]
    labels = cached_test["labels"]
    defect_types = cached_test["types"]
    normal, defects = [], defaultdict(list)
    model_t.eval()

    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        batch = ((sequences[start:end] - mean) / std).astype(np.float32)
        with torch.no_grad():
            latent = model_t.encode(torch.from_numpy(batch).to(device)).cpu().numpy()
        for local, global_index in enumerate(range(start, end)):
            mask = np.asarray(labels[global_index], dtype=bool).reshape(-1)
            defect_type = defect_types[global_index]
            normal.append(latent[local][~mask])
            if defect_type != "good" and mask.any():
                defects[defect_type].append(latent[local][mask])

    return np.concatenate(normal), {k: np.concatenate(v) for k, v in sorted(defects.items())}


def transform_batches(function, values: np.ndarray, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [function(values[start:start + batch_size]) for start in range(0, len(values), batch_size)],
        axis=0,
    )


def nearest_nominal_distances(
    reference: np.ndarray,
    query: np.ndarray,
    batch_size: int = 4096,
    backend: str = "auto",
) -> np.ndarray:
    """Exact 1-NN Euclidean distances, using FAISS when available."""
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)

    use_faiss = backend in ("auto", "faiss")
    if use_faiss:
        try:
            import faiss
        except Exception:
            if backend == "faiss":
                raise
        else:
            index = faiss.IndexFlatL2(reference.shape[1])
            index.add(reference)
            parts = []
            for start in range(0, len(query), batch_size):
                d2, _ = index.search(query[start:start + batch_size], 1)
                parts.append(np.sqrt(np.maximum(d2[:, 0], 0.0)))
            return np.concatenate(parts).astype(np.float32, copy=False)

    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn_model.fit(reference)
    parts = []
    for start in range(0, len(query), batch_size):
        part = nn_model.kneighbors(
            query[start:start + batch_size], return_distance=True
        )[0][:, 0]
        parts.append(part)
    return np.concatenate(parts).astype(np.float32, copy=False)


def evaluate(
    method: str,
    reference: np.ndarray,
    normal: np.ndarray,
    defects: Mapping[str, np.ndarray],
    nominal_percentile: float,
    fit_seconds: float,
    transform_seconds: float,
    extra: Optional[Dict[str, float]] = None,
    nn_query_batch_size: int = 4096,
    nn_backend: str = "auto",
) -> MethodResult:
    normal_distances = nearest_nominal_distances(
        reference, normal, batch_size=nn_query_batch_size, backend=nn_backend
    )
    normal_mean = float(normal_distances.mean())
    threshold = float(np.percentile(normal_distances, nominal_percentile))

    rows, blend_rows = [], []
    distance_map: Dict[str, np.ndarray] = {}
    for defect_type, values in sorted(defects.items()):
        distances = nearest_nominal_distances(
            reference, values, batch_size=nn_query_batch_size, backend=nn_backend
        )
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


def save_cdf_plot(result: MethodResult, directory: Path) -> None:
    """Save empirical CDF curves for normal and every defect type."""
    plt.figure(figsize=(10, 6))

    def plot_ecdf(values: np.ndarray, label: str, linewidth: float = 2.0) -> None:
        ordered = np.sort(np.asarray(values))
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        plt.plot(ordered, cumulative, linewidth=linewidth, label=label)

    plot_ecdf(result.normal_distances, "normal", linewidth=2.5)
    for defect_type, distances in sorted(result.defect_distances.items()):
        plot_ecdf(distances, defect_type)

    threshold = float(result.metrics["Nominal Threshold"].iloc[0])
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"nominal threshold ({threshold:.3f})",
    )
    plt.xlabel("Distance to nearest nominal patch")
    plt.ylabel("Cumulative fraction")
    plt.title(f"{result.name}: cumulative nearest-nominal distances")
    plt.ylim(0.0, 1.01)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "distance_cdf.png", dpi=180)
    plt.close()


def save_2d_projection(
    method: str,
    reference: np.ndarray,
    normal: np.ndarray,
    defects: Mapping[str, np.ndarray],
    directory: Path,
    seed: int,
    max_points_per_class: int,
) -> None:
    """Create a comparable 2-D PCA view of each method's own representation."""
    rng = np.random.default_rng(seed)

    def sample(values: np.ndarray) -> np.ndarray:
        if max_points_per_class <= 0 or len(values) <= max_points_per_class:
            return values
        chosen = rng.choice(len(values), max_points_per_class, replace=False)
        return values[chosen]

    reference_sample = sample(reference)
    normal_sample = sample(normal)
    defect_samples = {name: sample(values) for name, values in defects.items()}

    projector = PCA(n_components=2, random_state=seed)
    projector.fit(reference_sample)
    normal_2d = projector.transform(normal_sample)
    defects_2d = {
        name: projector.transform(values)
        for name, values in defect_samples.items()
    }

    plt.figure(figsize=(10, 8))
    plt.scatter(
        normal_2d[:, 0], normal_2d[:, 1],
        s=9, alpha=0.18, label="normal",
    )
    for defect_type, points in sorted(defects_2d.items()):
        plt.scatter(
            points[:, 0], points[:, 1],
            s=12, alpha=0.42,
            label=defect_type.replace("_", " "),
        )
    explained = float(projector.explained_variance_ratio_.sum())
    plt.xlabel("Visualization component 1")
    plt.ylabel("Visualization component 2")
    plt.title(
        f"{method}: 2-D visualization of the evaluated representation\n"
        f"PCA used only for plotting; explained variance={explained:.2%}"
    )
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "representation_2d_pca.png", dpi=180)
    plt.close()


def save_nearest_neighbor_feature_examples(
    method: str,
    reference: np.ndarray,
    defects: Mapping[str, np.ndarray],
    directory: Path,
    examples_per_defect: int,
) -> None:
    """
    Save nearest-neighbour retrieval diagnostics in feature space.

    The current pipeline does not retain source image/patch coordinates, so this
    figure visualizes query and retrieved nominal vectors rather than RGB crops.
    It still reveals whether the retrieved nominal representation closely follows
    the defective query. Exact patch-image retrieval requires retaining image IDs
    and patch-grid coordinates during feature collection.
    """
    if examples_per_defect <= 0:
        return

    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn_model.fit(reference)
    rows = []

    for defect_type, values in sorted(defects.items()):
        distances, indices = nn_model.kneighbors(values, return_distance=True)
        distances = distances[:, 0]
        indices = indices[:, 0]
        if len(values) == 0:
            continue
        ranks = np.linspace(0, len(values) - 1, min(examples_per_defect, len(values))).astype(int)
        ordered = np.argsort(distances)
        for rank in ranks:
            query_index = int(ordered[rank])
            rows.append((defect_type, values[query_index], reference[indices[query_index]], distances[query_index]))

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(3.0, 2.4 * len(rows))), squeeze=False)
    for ax, (defect_type, query, neighbour, distance) in zip(axes[:, 0], rows):
        dimensions = np.arange(min(len(query), 128))
        ax.plot(dimensions, query[:len(dimensions)], linewidth=1.2, label="defect query")
        ax.plot(dimensions, neighbour[:len(dimensions)], linewidth=1.2, label="nearest nominal")
        ax.set_title(f"{defect_type} | nearest distance={distance:.4f}")
        ax.set_xlabel("Feature dimension (first 128)")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.15)
        ax.legend()
    fig.suptitle(f"{method}: nearest-neighbour feature retrieval examples", y=1.002)
    fig.tight_layout()
    fig.savefig(directory / "nearest_neighbor_feature_retrieval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_blending_comparison_plot(long_table: pd.DataFrame, run_dir: Path) -> None:
    """Compare the percentage of defects that blend into the nominal region."""
    plot_table = long_table.pivot_table(
        index="Method",
        columns="Defect Type",
        values="Inside Nominal Region (%)",
        aggfunc="first",
    )
    if plot_table.empty:
        return

    ax = plot_table.plot(kind="bar", figsize=(12, 7))
    ax.set_ylabel("Defective patches inside nominal region (%)")
    ax.set_xlabel("Method")
    ax.set_title("Blending comparison across dimensionality-reduction methods")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Defect type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(run_dir / "blending_comparison.png", dpi=180)
    plt.close()

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

    save_cdf_plot(result, directory)

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


def run_transformer(sequences, patch_grid, patchcore_model, test_loader, selected_indices, args, directory, device, cached_test=None):
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
    if cached_test is not None:
        normal_z, defects_z = collect_transformer_test_cached(
            model, cached_test, mean, std, device, args.transformer_batch_size
        )
    else:
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



def parse_projection_methods(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(x.strip().lower() for x in value.split(",") if x.strip())
    if not tokens or "none" in tokens:
        return []
    if "all" in tokens:
        return ["pca", "umap"]
    valid = {"pca", "umap"}
    result: List[str] = []
    for token in tokens:
        if token not in valid:
            raise ValueError(
                f"Unknown projection method '{token}'. Valid: none, all, pca, umap"
            )
        if token not in result:
            result.append(token)
    return result


def save_projection_stage(
    representation_method: str,
    train_z: np.ndarray,
    normal_z: np.ndarray,
    defects_z: Mapping[str, np.ndarray],
    directory: Path,
    args,
) -> None:
    """Project an ALREADY reduced representation to 2-D/3-D for visualization.

    Important: this stage is intentionally disabled for ``original`` because the
    requested workflow is representation reduction first (e.g. 1024->64) and
    projection second (e.g. 64->2/3). The projector is fitted on nominal
    reference features only; test normal/defect features are transform-only.
    """
    projection_methods = getattr(args, "projection_methods_parsed", [])
    if not projection_methods:
        return

    if representation_method == "original":
        print(
            "[projection] skipping method=original: projection is only allowed "
            "after a dimensionality-reduction/representation method."
        )
        return

    dims = sorted(set(int(d) for d in args.projection_dims))
    projection_root = directory / "projections"
    projection_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    def sample(values: np.ndarray) -> np.ndarray:
        limit = args.projection_points_per_class
        if limit <= 0 or len(values) <= limit:
            return values
        idx = rng.choice(len(values), limit, replace=False)
        return values[idx]

    # Fit uses nominal reference only. Sampling here is visualization/projector
    # fitting control and never introduces test defects into the fit.
    train_fit = sample(train_z)
    normal_plot = sample(normal_z)
    defects_plot = {name: sample(values) for name, values in defects_z.items()}

    for projection_method in projection_methods:
        for dim in dims:
            if dim not in (2, 3):
                raise ValueError("--projection-dims currently supports only 2 and 3")
            if train_z.shape[1] <= dim:
                raise ValueError(
                    f"Cannot project {representation_method} output with dimension "
                    f"{train_z.shape[1]} to {dim}D; projection input must have more "
                    "dimensions than its output."
                )

            out_dir = projection_root / f"{projection_method}_{dim}d"
            out_dir.mkdir(parents=True, exist_ok=True)

            if projection_method == "pca":
                projector = PCA(n_components=dim, random_state=args.seed)
                train_projected = projector.fit_transform(train_fit)
                normal_projected = projector.transform(normal_plot)
                defects_projected = {
                    name: projector.transform(values)
                    for name, values in defects_plot.items()
                }
                joblib.dump(projector, out_dir / "projector.joblib")
                projection_extra = {
                    "explained_variance": float(
                        projector.explained_variance_ratio_.sum()
                    )
                }

            elif projection_method == "umap":
                try:
                    import umap
                except ImportError as exc:
                    raise ImportError(
                        "Projection with UMAP requires: pip install umap-learn"
                    ) from exc
                projector = umap.UMAP(
                    n_components=dim,
                    n_neighbors=args.projection_umap_neighbors,
                    min_dist=args.projection_umap_min_dist,
                    metric="euclidean",
                    random_state=args.seed,
                )
                train_projected = projector.fit_transform(train_fit)
                normal_projected = projector.transform(normal_plot)
                defects_projected = {
                    name: projector.transform(values)
                    for name, values in defects_plot.items()
                }
                joblib.dump(projector, out_dir / "projector.joblib")
                projection_extra = {}

            else:
                raise ValueError(projection_method)

            # Store projected coordinates so later plotting/analysis never needs
            # to refit the projection.
            np.save(out_dir / "nominal_reference_projected.npy", train_projected)
            np.save(out_dir / "normal_test_projected.npy", normal_projected)
            np.savez_compressed(
                out_dir / "defects_projected.npz",
                **{f"defect__{k}": v for k, v in defects_projected.items()},
            )

            config = {
                "representation_method": representation_method,
                "representation_input_dimension": int(train_z.shape[1]),
                "projection_method": projection_method,
                "projection_output_dimension": dim,
                "fit_on": "nominal_reference_only",
                "nominal_reference_points_used_for_fit": int(len(train_fit)),
                "normal_test_points_plotted": int(len(normal_plot)),
                "projection_points_per_class": int(args.projection_points_per_class),
                **projection_extra,
            }
            (out_dir / "projection_config.json").write_text(
                json.dumps(config, indent=2)
            )

            # Static plot.
            if dim == 2:
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    normal_projected[:, 0], normal_projected[:, 1],
                    s=9, alpha=0.18, label="normal",
                )
                for defect_type, points in sorted(defects_projected.items()):
                    plt.scatter(
                        points[:, 0], points[:, 1], s=12, alpha=0.42,
                        label=defect_type.replace("_", " "),
                    )
                plt.xlabel("Projection dimension 1")
                plt.ylabel("Projection dimension 2")
                plt.title(
                    f"{representation_method} {train_z.shape[1]}D -> "
                    f"{projection_method.upper()} 2D"
                )
                plt.grid(alpha=0.15)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "projection_2d.png", dpi=180)
                plt.close()
            else:
                fig = plt.figure(figsize=(11, 9))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    normal_projected[:, 0], normal_projected[:, 1],
                    normal_projected[:, 2], s=9, alpha=0.18, label="normal",
                )
                for defect_type, points in sorted(defects_projected.items()):
                    ax.scatter(
                        points[:, 0], points[:, 1], points[:, 2],
                        s=12, alpha=0.42, label=defect_type.replace("_", " "),
                    )
                ax.set_xlabel("Projection dimension 1")
                ax.set_ylabel("Projection dimension 2")
                ax.set_zlabel("Projection dimension 3")
                ax.set_title(
                    f"{representation_method} {train_z.shape[1]}D -> "
                    f"{projection_method.upper()} 3D"
                )
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "projection_3d.png", dpi=180)
                plt.close(fig)

                if args.projection_html:
                    try:
                        import plotly.graph_objects as go
                    except ImportError:
                        warnings.warn(
                            "plotly is not installed; skipping interactive HTML. "
                            "Install it with: pip install plotly"
                        )
                    else:
                        traces = [
                            go.Scatter3d(
                                x=normal_projected[:, 0],
                                y=normal_projected[:, 1],
                                z=normal_projected[:, 2],
                                mode="markers",
                                name="normal",
                                marker={"size": 2, "opacity": 0.25},
                            )
                        ]
                        for defect_type, points in sorted(defects_projected.items()):
                            traces.append(
                                go.Scatter3d(
                                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                    mode="markers",
                                    name=defect_type.replace("_", " "),
                                    marker={"size": 3, "opacity": 0.55},
                                )
                            )
                        fig_html = go.Figure(data=traces)
                        fig_html.update_layout(
                            title=(
                                f"{representation_method} {train_z.shape[1]}D -> "
                                f"{projection_method.upper()} 3D"
                            ),
                            scene={
                                "xaxis_title": "Projection dimension 1",
                                "yaxis_title": "Projection dimension 2",
                                "zaxis_title": "Projection dimension 3",
                            },
                        )
                        fig_html.write_html(
                            out_dir / "projection_3d.html",
                            include_plotlyjs="cdn",
                        )

            print(
                f"[projection] {representation_method} {train_z.shape[1]}D -> "
                f"{projection_method} {dim}D saved in {out_dir}"
            )


def checkpoint_filename(method: str) -> Optional[str]:
    return {
        "pca": "pca.joblib",
        "umap": "umap.joblib",
        "autoencoder": "autoencoder.pt",
        "kernel_pca": "kernel_pca.joblib",
        "transformer": "transformer_autoencoder.pt",
        "deep_svdd": "deep_svdd.pt"
    }.get(method)


def resolve_existing_checkpoint(
    method: str,
    args,
    category: str,
    subsampling: str,
    current_method_dir: Path,
) -> Optional[Path]:
    if args.force_refit:
        return None
    filename = checkpoint_filename(method)
    if filename is None:
        return None

    candidates = []
    if args.checkpoint_root is not None:
        candidates.append(
            args.checkpoint_root.resolve() / category / subsampling / method / filename
        )
    candidates.append(current_method_dir / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_autoencoder_representation(checkpoint: Path, train, normal, defects, args, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim", train.shape[1]))
    latent_dim = int(ckpt.get("latent_dim", args.latent_dim))
    if input_dim != train.shape[1]:
        raise ValueError(
            f"Autoencoder checkpoint expects {input_dim}D input, got {train.shape[1]}D."
        )
    model = EmbeddingAutoencoder(input_dim, latent_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    start = time.perf_counter()
    train_z = encode_ae(model, train, mean, std, device, args.ae_encode_batch_size)
    normal_z = encode_ae(model, normal, mean, std, device, args.ae_encode_batch_size)
    defects_z = {
        k: encode_ae(model, v, mean, std, device, args.ae_encode_batch_size)
        for k, v in defects.items()
    }
    elapsed = time.perf_counter() - start
    return train_z, normal_z, defects_z, 0.0, elapsed, {
        "checkpoint_reused": 1.0,
        "checkpoint_latent_dim": float(latent_dim),
    }


def load_transformer_representation(
    checkpoint: Path, sequences, selected_indices, cached_test, args, device
):
    if sequences is None:
        raise RuntimeError("Transformer checkpoint reuse requires intact nominal sequences.")
    if cached_test is None:
        raise RuntimeError(
            "Transformer checkpoint reuse from --source data is not implemented; "
            "use --source features or --force-refit."
        )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    patch_grid = tuple(ckpt.get("patch_grid", (int(np.sqrt(sequences.shape[1])),) * 2))
    input_dim = int(ckpt.get("input_dim", sequences.shape[-1]))
    latent_dim = int(ckpt.get("latent_dim", args.latent_dim))

    # Infer architecture from state dict where possible, falling back to CLI values.
    state = ckpt["state_dict"]
    model_dim = int(state["input_projection.weight"].shape[0])
    heads = args.transformer_heads
    enc_layers = len({k.split('.')[2] for k in state if k.startswith('encoder.layers.')}) or args.transformer_encoder_layers
    dec_layers = len({k.split('.')[2] for k in state if k.startswith('decoder.layers.')}) or args.transformer_decoder_layers
    model = SpatialTransformerAutoencoder(
        input_dim=input_dim,
        patches=int(patch_grid[0] * patch_grid[1]),
        latent_dim=latent_dim,
        model_dim=model_dim,
        heads=heads,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        dropout=args.transformer_dropout,
    ).to(device)
    model.load_state_dict(state)
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    start = time.perf_counter()
    all_train_z = encode_transformer_sequences(
        model, sequences, mean, std, device, args.transformer_batch_size
    )
    train_z = all_train_z[selected_indices]
    normal_z, defects_z = collect_transformer_test_cached(
        model, cached_test, mean, std, device, args.transformer_batch_size
    )
    elapsed = time.perf_counter() - start
    return train_z, normal_z, defects_z, 0.0, elapsed, {
        "checkpoint_reused": 1.0,
        "checkpoint_latent_dim": float(latent_dim),
    }


def run_method(
    method, sampled_train, selected_indices, normal, defects, sequences,
    patch_grid, patchcore_model, test_loader, args, directory, device,
    cached_test=None, category=None, subsampling=None,
):
    print(f"\n{'=' * 72}\nMETHOD: {method}\n{'=' * 72}")

    checkpoint = resolve_existing_checkpoint(
        method, args, category, subsampling, directory
    ) if category is not None and subsampling is not None else None

    if method == "original":
        train_z, normal_z, defects_z = sampled_train, normal, dict(defects)
        fit_seconds = transform_seconds = 0.0
        extra = {"checkpoint_reused": 0.0}

    elif checkpoint is not None and method == "pca":
        print(f"[checkpoint] loading PCA: {checkpoint}")
        reducer = joblib.load(checkpoint)
        start = time.perf_counter()
        train_z = reducer.transform(sampled_train)
        normal_z = reducer.transform(normal)
        defects_z = {k: reducer.transform(v) for k, v in defects.items()}
        transform_seconds = time.perf_counter() - start
        fit_seconds = 0.0
        extra = {
            "checkpoint_reused": 1.0,
            "explained_variance": float(reducer.explained_variance_ratio_.sum()),
        }

    elif checkpoint is not None and method == "umap":
        print(f"[checkpoint] loading UMAP: {checkpoint}")
        reducer = joblib.load(checkpoint)
        start = time.perf_counter()
        train_z = reducer.transform(sampled_train)
        normal_z = reducer.transform(normal)
        defects_z = {k: reducer.transform(v) for k, v in defects.items()}
        transform_seconds = time.perf_counter() - start
        fit_seconds = 0.0
        extra = {"checkpoint_reused": 1.0}

    elif checkpoint is not None and method == "autoencoder":
        print(f"[checkpoint] loading Autoencoder: {checkpoint}")
        train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra = (
            load_autoencoder_representation(
                checkpoint, sampled_train, normal, defects, args, device
            )
        )

    elif checkpoint is not None and method == "kernel_pca":
        print(f"[checkpoint] loading Kernel PCA: {checkpoint}")
        payload = joblib.load(checkpoint)
        scaler, reducer = payload["scaler"], payload["reducer"]
        transform = lambda x: reducer.transform(scaler.transform(x))
        start = time.perf_counter()
        train_z = transform_batches(transform, sampled_train, args.transform_batch_size)
        normal_z = transform_batches(transform, normal, args.transform_batch_size)
        defects_z = {
            k: transform_batches(transform, v, args.transform_batch_size)
            for k, v in defects.items()
        }
        transform_seconds = time.perf_counter() - start
        fit_seconds = 0.0
        extra = {
            "checkpoint_reused": 1.0,
            "gamma": float(getattr(reducer, "gamma", np.nan)),
        }

    elif checkpoint is not None and method == "transformer":
        print(f"[checkpoint] loading Transformer: {checkpoint}")
        train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra = (
            load_transformer_representation(
                checkpoint, sequences, selected_indices, cached_test, args, device
            )
        )

    elif method == "pca":
        print("[checkpoint] PCA checkpoint not found; fitting from scratch")
        reducer = PCA(n_components=args.latent_dim, random_state=args.seed)
        start = time.perf_counter()
        train_z = reducer.fit_transform(sampled_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        normal_z = reducer.transform(normal)
        defects_z = {k: reducer.transform(v) for k, v in defects.items()}
        transform_seconds = time.perf_counter() - start
        joblib.dump(reducer, directory / "pca.joblib")
        extra = {
            "checkpoint_reused": 0.0,
            "explained_variance": float(reducer.explained_variance_ratio_.sum()),
        }

    elif method == "umap":
        print("[checkpoint] UMAP checkpoint not found; fitting from scratch")
        try:
            import umap
        except ImportError as exc:
            raise ImportError("Install UMAP with: pip install umap-learn") from exc
        reducer = umap.UMAP(
            n_components=args.umap_dim, n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist, metric="euclidean", random_state=args.seed,
        )
        start = time.perf_counter()
        train_z = reducer.fit_transform(sampled_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        normal_z = reducer.transform(normal)
        defects_z = {k: reducer.transform(v) for k, v in defects.items()}
        transform_seconds = time.perf_counter() - start
        joblib.dump(reducer, directory / "umap.joblib")
        extra = {"checkpoint_reused": 0.0}

    elif method == "autoencoder":
        print("[checkpoint] Autoencoder checkpoint not found; fitting from scratch")
        train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra = run_autoencoder(
            sampled_train, normal, defects, args, directory, device
        )
        extra["checkpoint_reused"] = 0.0

    elif method == "kernel_pca":
        print("[checkpoint] Kernel PCA checkpoint not found; fitting from scratch")
        train_input = sampled_train
        scaler = StandardScaler()
        scaled = scaler.fit_transform(train_input)
        gamma = args.kpca_gamma if args.kpca_gamma > 0 else 1.0 / scaled.shape[1]
        reducer = KernelPCA(
            n_components=args.latent_dim, kernel="rbf", gamma=gamma,
            eigen_solver="randomized", random_state=args.seed,
            remove_zero_eig=True, n_jobs=-1,
        )
        start = time.perf_counter()
        train_z = reducer.fit_transform(scaled)
        fit_seconds = time.perf_counter() - start
        transform = lambda x: reducer.transform(scaler.transform(x))
        start = time.perf_counter()
        normal_z = transform_batches(transform, normal, args.transform_batch_size)
        defects_z = {
            k: transform_batches(transform, v, args.transform_batch_size)
            for k, v in defects.items()
        }
        transform_seconds = time.perf_counter() - start
        joblib.dump({"scaler": scaler, "reducer": reducer}, directory / "kernel_pca.joblib")
        extra = {"checkpoint_reused": 0.0, "gamma": float(gamma)}

    elif method == "transformer":
        print("[checkpoint] Transformer checkpoint not found; fitting from scratch")
        train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra = run_transformer(
            sequences, patch_grid, patchcore_model, test_loader,
            selected_indices, args, directory, device, cached_test=cached_test
        )
        extra["checkpoint_reused"] = 0.0
    else:
        raise ValueError(method)

    train_z = np.asarray(train_z, dtype=np.float32)
    normal_z = np.asarray(normal_z, dtype=np.float32)
    defects_z = {k: np.asarray(v, dtype=np.float32) for k, v in defects_z.items()}

    save_2d_projection(
        method, train_z, normal_z, defects_z, directory,
        args.seed, args.visualization_points_per_class,
    )
    save_nearest_neighbor_feature_examples(
        method, train_z, defects_z, directory,
        args.retrieval_examples_per_defect,
    )

    save_projection_stage(method, train_z, normal_z, defects_z, directory, args)

    return evaluate(
        method, train_z, normal_z, defects_z, args.nominal_percentile,
        fit_seconds, transform_seconds, extra,
        nn_query_batch_size=args.nn_query_batch_size,
        nn_backend=args.nn_backend,
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



# -----------------------------------------------------------------------------
# Sequential dimensionality-reduction chains
# -----------------------------------------------------------------------------

def parse_reduction_chain(values: Optional[Sequence[str]]) -> List[Tuple[str, int]]:
    """Parse e.g. ``pca:64 umap:32 pca:8 umap:3``.

    Each stage is fitted ONLY on the current nominal training/reference features.
    Normal-test and defect-test features are transform-only at every stage.
    """
    if not values:
        return []

    tokens: List[str] = []
    for value in values:
        tokens.extend(x.strip().lower() for x in value.split(",") if x.strip())

    stages: List[Tuple[str, int]] = []
    valid = {"pca", "umap", "autoencoder", "kernel_pca", "transformer", "deep_svdd"}
    for token in tokens:
        if ":" not in token:
            raise ValueError(
                f"Invalid chain stage '{token}'. Use METHOD:DIM, e.g. pca:64 or umap:3."
            )
        method_token, dim_token = token.rsplit(":", 1)
        method = ALIASES.get(method_token.strip(), method_token.strip())
        if method not in valid:
            raise ValueError(
                f"Unknown chain method '{method}'. Valid: {', '.join(sorted(valid))}"
            )
        try:
            dim = int(dim_token)
        except ValueError as exc:
            raise ValueError(f"Invalid output dimension in chain stage '{token}'.") from exc
        if dim <= 0:
            raise ValueError(f"Chain output dimensions must be positive; got {dim} in '{token}'.")
        stages.append((method, dim))

    for index, (method, _) in enumerate(stages):
        if method == "transformer" and index != 0:
            raise ValueError(
                "Transformer is currently supported only as the FIRST stage of a sequential "
                "chain because it requires intact image patch sequences. Example: "
                "transformer:64 pca:16 umap:3."
            )
    return stages


def chain_name(stages: Sequence[Tuple[str, int]]) -> str:
    return "__".join(f"{method}{dim}" for method, dim in stages)


def _checkpoint_dims_ok(method: str, checkpoint: Path, input_dim: int, output_dim: int) -> bool:
    """Best-effort dimensionality check before reusing a stage checkpoint."""
    try:
        if method == "pca":
            reducer = joblib.load(checkpoint)
            in_dim = int(getattr(reducer, "n_features_in_", input_dim))
            out_dim = int(getattr(reducer, "n_components_", getattr(reducer, "n_components", output_dim)))
            return in_dim == input_dim and out_dim == output_dim

        if method == "umap":
            reducer = joblib.load(checkpoint)
            out_dim = int(getattr(reducer, "n_components", output_dim))
            raw = getattr(reducer, "_raw_data", None)
            in_dim = int(raw.shape[1]) if raw is not None and getattr(raw, "ndim", 0) == 2 else input_dim
            return in_dim == input_dim and out_dim == output_dim

        if method == "kernel_pca":
            payload = joblib.load(checkpoint)
            scaler, reducer = payload["scaler"], payload["reducer"]
            in_dim = int(getattr(scaler, "n_features_in_", input_dim))
            out_dim = int(getattr(reducer, "n_components", output_dim))
            return in_dim == input_dim and out_dim == output_dim

        if method in {"autoencoder", "transformer"}:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            in_dim = int(ckpt.get("input_dim", input_dim))
            out_dim = int(ckpt.get("latent_dim", output_dim))
            return in_dim == input_dim and out_dim == output_dim

        if method == "deep_svdd":
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            in_dim = int(ckpt["input_dim"])
            out_dim = int(ckpt["output_dim"])
            return in_dim == input_dim and out_dim == output_dim
          
    except Exception as exc:
        print(f"[chain checkpoint] could not validate {checkpoint}: {type(exc).__name__}: {exc}")
        return False
    return False


def resolve_chain_checkpoint(
    stage_index: int,
    method: str,
    input_dim: int,
    output_dim: int,
    stage_dir: Path,
    args,
    category: str,
    subsampling: str,
) -> Optional[Path]:
    """Find a reusable checkpoint for one chain stage.

    - Any stage may reuse a checkpoint already saved in its own chain stage folder.
    - Stage 1 may additionally reuse the legacy first-stage checkpoint tree supplied
      via --checkpoint-root, because its input is still the original PatchCore space.
    - Later stages never reuse a legacy checkpoint trained on a different prefix.
    """
    if args.force_refit:
        return None

    filename = checkpoint_filename(method)
    if filename is None:
        return None

    candidates: List[Path] = [stage_dir / filename]
    if stage_index == 0 and args.checkpoint_root is not None:
        candidates.insert(
            0,
            args.checkpoint_root.resolve() / category / subsampling / method / filename,
        )

    for candidate in candidates:
        if candidate.exists():
            if _checkpoint_dims_ok(method, candidate, input_dim, output_dim):
                return candidate
            print(
                f"[chain checkpoint] ignoring incompatible checkpoint: {candidate} "
                f"(wanted {input_dim}D -> {output_dim}D)"
            )
    return None


def _run_chain_autoencoder(
    train: np.ndarray,
    normal: np.ndarray,
    defects: Mapping[str, np.ndarray],
    output_dim: int,
    args,
    directory: Path,
    device: torch.device,
):
    old_latent = args.latent_dim
    args.latent_dim = output_dim
    try:
        return run_autoencoder(train, normal, defects, args, directory, device)
    finally:
        args.latent_dim = old_latent


def _run_chain_transformer_first_stage(
    sequences,
    patch_grid,
    patchcore_model,
    test_loader,
    selected_indices,
    cached_test,
    output_dim: int,
    args,
    directory: Path,
    device: torch.device,
):
    old_latent = args.latent_dim
    args.latent_dim = output_dim
    try:
        return run_transformer(
            sequences, patch_grid, patchcore_model, test_loader,
            selected_indices, args, directory, device, cached_test=cached_test,
        )
    finally:
        args.latent_dim = old_latent


def apply_chain_stage(
    stage_index: int,
    method: str,
    output_dim: int,
    train: np.ndarray,
    normal: np.ndarray,
    defects: Mapping[str, np.ndarray],
    args,
    stage_dir: Path,
    device: torch.device,
    category: str,
    subsampling: str,
    *,
    sequences=None,
    patch_grid=None,
    patchcore_model=None,
    test_loader=None,
    selected_indices=None,
    cached_test=None,
):
    """Fit/load one sequential stage on nominal train and transform test data."""
    input_dim = int(train.shape[1])
    if output_dim >= input_dim:
        raise ValueError(
            f"Sequential reduction must reduce dimension at every stage; "
            f"stage {stage_index + 1} requested {input_dim}D -> {output_dim}D."
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_chain_checkpoint(
        stage_index, method, input_dim, output_dim, stage_dir,
        args, category, subsampling,
    )

    fit_seconds = 0.0
    transform_seconds = 0.0
    extra: Dict[str, Any] = {}
    checkpoint_reused_from: Optional[str] = None

    if checkpoint is not None:
        checkpoint_reused_from = str(checkpoint)
        print(
            f"[chain stage {stage_index + 1}] loading {method} checkpoint: {checkpoint}"
        )

        if method == "pca":
            reducer = joblib.load(checkpoint)
            start = time.perf_counter()
            train_z = reducer.transform(train)
            normal_z = reducer.transform(normal)
            defects_z = {k: reducer.transform(v) for k, v in defects.items()}
            transform_seconds = time.perf_counter() - start
            extra["explained_variance"] = float(reducer.explained_variance_ratio_.sum())

        elif method == "umap":
            reducer = joblib.load(checkpoint)
            start = time.perf_counter()
            train_z = reducer.transform(train)
            normal_z = reducer.transform(normal)
            defects_z = {k: reducer.transform(v) for k, v in defects.items()}
            transform_seconds = time.perf_counter() - start

        elif method == "kernel_pca":
            payload = joblib.load(checkpoint)
            scaler, reducer = payload["scaler"], payload["reducer"]
            transform = lambda x: reducer.transform(scaler.transform(x))
            start = time.perf_counter()
            train_z = transform_batches(transform, train, args.transform_batch_size)
            normal_z = transform_batches(transform, normal, args.transform_batch_size)
            defects_z = {
                k: transform_batches(transform, v, args.transform_batch_size)
                for k, v in defects.items()
            }
            transform_seconds = time.perf_counter() - start
            extra["gamma"] = float(getattr(reducer, "gamma", np.nan))

        elif method == "autoencoder":
            train_z, normal_z, defects_z, _, transform_seconds, loaded_extra = (
                load_autoencoder_representation(
                    checkpoint, train, normal, defects, args, device
                )
            )
            extra.update(loaded_extra)

        elif method == "transformer":
            if stage_index != 0:
                raise ValueError("Transformer can only be the first chain stage.")
            train_z, normal_z, defects_z, _, transform_seconds, loaded_extra = (
                load_transformer_representation(
                    checkpoint, sequences, selected_indices, cached_test, args, device
                )
            )
            extra.update(loaded_extra)


        elif method == "deep_svdd":

            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            net = PatchFeatureSVDDNet(input_dim=int(ckpt["input_dim"]), rep_dim=int(ckpt["output_dim"]),).to(device)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
        
            c = ckpt["center"].to(device)
            start = time.perf_counter()
            train_z = transform_deep_svdd(net, train, device,)
            normal_z = transform_deep_svdd(net, normal, device)
            defects_z = {name: transform_deep_svdd(net, values, device) for name, values in defects.items() }
            transform_seconds = time.perf_counter() - start
        
            extra["svdd_center_norm"] = float(torch.linalg.vector_norm(c).item())

        else:
            raise ValueError(method)

        extra["checkpoint_reused"] = 1.0

    else:
        print(
            f"[chain stage {stage_index + 1}] fitting {method}: "
            f"{input_dim}D -> {output_dim}D on NOMINAL reference only"
        )

        if method == "pca":
            reducer = PCA(n_components=output_dim, random_state=args.seed)
            start = time.perf_counter()
            train_z = reducer.fit_transform(train)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            normal_z = reducer.transform(normal)
            defects_z = {k: reducer.transform(v) for k, v in defects.items()}
            transform_seconds = time.perf_counter() - start
            joblib.dump(reducer, stage_dir / "pca.joblib")
            extra["explained_variance"] = float(reducer.explained_variance_ratio_.sum())

        elif method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError("Install UMAP with: pip install umap-learn") from exc
            reducer = umap.UMAP(
                n_components=output_dim,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric="euclidean",
                random_state=args.seed,
            )
            start = time.perf_counter()
            train_z = reducer.fit_transform(train)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            normal_z = reducer.transform(normal)
            defects_z = {k: reducer.transform(v) for k, v in defects.items()}
            transform_seconds = time.perf_counter() - start
            joblib.dump(reducer, stage_dir / "umap.joblib")

        elif method == "autoencoder":
            train_z, normal_z, defects_z, fit_seconds, transform_seconds, ae_extra = (
                _run_chain_autoencoder(
                    train, normal, defects, output_dim, args, stage_dir, device
                )
            )
            extra.update(ae_extra)

        elif method == "kernel_pca":
            scaler = StandardScaler()
            scaled = scaler.fit_transform(train)
            gamma = args.kpca_gamma if args.kpca_gamma > 0 else 1.0 / scaled.shape[1]
            reducer = KernelPCA(
                n_components=output_dim,
                kernel="rbf",
                gamma=gamma,
                eigen_solver="randomized",
                random_state=args.seed,
                remove_zero_eig=True,
                n_jobs=-1,
            )
            start = time.perf_counter()
            train_z = reducer.fit_transform(scaled)
            fit_seconds = time.perf_counter() - start
            transform = lambda x: reducer.transform(scaler.transform(x))
            start = time.perf_counter()
            normal_z = transform_batches(transform, normal, args.transform_batch_size)
            defects_z = {
                k: transform_batches(transform, v, args.transform_batch_size)
                for k, v in defects.items()
            }
            transform_seconds = time.perf_counter() - start
            joblib.dump({"scaler": scaler, "reducer": reducer}, stage_dir / "kernel_pca.joblib")
            extra["gamma"] = float(gamma)

        elif method == "transformer":
            if stage_index != 0:
                raise ValueError("Transformer can only be the first chain stage.")
            train_z, normal_z, defects_z, fit_seconds, transform_seconds, tr_extra = (
                _run_chain_transformer_first_stage(
                    sequences, patch_grid, patchcore_model, test_loader,
                    selected_indices, cached_test, output_dim,
                    args, stage_dir, device,
                )
            )
            extra.update(tr_extra)


        elif method == "deep_svdd":
        
            start = time.perf_counter()
        
            net, c = fit_deep_svdd(train,
                output_dim=output_dim,
                device=device,
                epochs=args.deep_svdd_epochs,
                lr=args.deep_svdd_lr,
                batch_size=args.deep_svdd_batch_size,
                weight_decay=args.deep_svdd_weight_decay,
            )
        
            fit_seconds = time.perf_counter() - start
        
            start = time.perf_counter()
          
            train_z = transform_deep_svdd(net, train, device)
            normal_z = transform_deep_svdd(net, normal, device)
            defects_z = {
                name: transform_deep_svdd(net, values,device)
                for name, values in defects.items()
            }
            transform_seconds = time.perf_counter() - start
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "center": c.detach().cpu(),
                    "input_dim": int(train.shape[1]),
                    "output_dim": int(output_dim),
                },
                stage_dir / "deep_svdd.pt",
            )
            extra["svdd_center_norm"] = float(torch.linalg.vector_norm(c).item())

        else:
            raise ValueError(method)

        extra["checkpoint_reused"] = 0.0

    train_z = np.asarray(train_z, dtype=np.float32)
    normal_z = np.asarray(normal_z, dtype=np.float32)
    defects_z = {k: np.asarray(v, dtype=np.float32) for k, v in defects_z.items()}

    if train_z.shape[1] != output_dim:
        raise RuntimeError(
            f"Chain stage {stage_index + 1} ({method}) returned {train_z.shape[1]}D, "
            f"expected {output_dim}D."
        )

    np.save(stage_dir / "nominal_reference.npy", train_z)
    np.save(stage_dir / "normal_test.npy", normal_z)
    np.savez_compressed(
        stage_dir / "defects_test.npz",
        **{f"defect__{k}": v for k, v in defects_z.items()},
    )

    stage_config = {
        "stage": stage_index + 1,
        "method": method,
        "input_dimension": input_dim,
        "output_dimension": output_dim,
        "fit_on": "nominal_reference_only",
        "normal_test_policy": "transform_only",
        "defect_test_policy": "transform_only",
        "checkpoint_reused_from": checkpoint_reused_from,
        "fit_seconds": fit_seconds,
        "transform_seconds": transform_seconds,
        **extra,
    }
    (stage_dir / "stage_config.json").write_text(
        json.dumps(stage_config, indent=2, default=str), encoding="utf-8"
    )

    return train_z, normal_z, defects_z, fit_seconds, transform_seconds, extra


def save_chain_lowdim_plot(
    chain_label: str,
    train: np.ndarray,
    normal: np.ndarray,
    defects: Mapping[str, np.ndarray],
    directory: Path,
    args,
) -> None:
    """Plot a final 2-D/3-D chain output without fitting anything else."""
    dim = int(train.shape[1])
    if dim not in (2, 3):
        return

    rng = np.random.default_rng(args.seed)

    def sample(values: np.ndarray) -> np.ndarray:
        limit = args.projection_points_per_class
        if limit <= 0 or len(values) <= limit:
            return values
        idx = rng.choice(len(values), limit, replace=False)
        return values[idx]

    normal_plot = sample(normal)
    defects_plot = {k: sample(v) for k, v in defects.items()}

    if dim == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(normal_plot[:, 0], normal_plot[:, 1], s=9, alpha=0.18, label="normal")
        for defect_type, points in sorted(defects_plot.items()):
            plt.scatter(
                points[:, 0], points[:, 1], s=12, alpha=0.42,
                label=defect_type.replace("_", " "),
            )
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(chain_label)
        plt.grid(alpha=0.15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(directory / "chain_2d.png", dpi=180)
        plt.close()
        return

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        normal_plot[:, 0], normal_plot[:, 1], normal_plot[:, 2],
        s=9, alpha=0.18, label="normal",
    )
    for defect_type, points in sorted(defects_plot.items()):
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=12, alpha=0.42, label=defect_type.replace("_", " "),
        )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title(chain_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "chain_3d.png", dpi=180)
    plt.close(fig)

    if args.projection_html:
        try:
            import plotly.graph_objects as go
        except ImportError:
            warnings.warn("plotly is not installed; skipping chain_3d.html")
        else:
            traces = [
                go.Scatter3d(
                    x=normal_plot[:, 0], y=normal_plot[:, 1], z=normal_plot[:, 2],
                    mode="markers", name="normal",
                    marker={"size": 2, "opacity": 0.25},
                )
            ]
            for defect_type, points in sorted(defects_plot.items()):
                traces.append(
                    go.Scatter3d(
                        x=points[:, 0], y=points[:, 1], z=points[:, 2],
                        mode="markers", name=defect_type.replace("_", " "),
                        marker={"size": 3, "opacity": 0.55},
                    )
                )
            fig_html = go.Figure(data=traces)
            fig_html.update_layout(
                title=chain_label,
                scene={
                    "xaxis_title": "Dimension 1",
                    "yaxis_title": "Dimension 2",
                    "zaxis_title": "Dimension 3",
                },
            )
            fig_html.write_html(directory / "chain_3d.html", include_plotlyjs="cdn")


def run_reduction_chain(
    args,
    stages: Sequence[Tuple[str, int]],
    category: str,
    subsampling: str,
    device: torch.device,
    prepared: Dict[str, Any],
    sampled_train: np.ndarray,
    selected_indices: np.ndarray,
    run_dir: Path,
) -> Path:
    """Run an ordered sequential reduction chain.

    Example: 1024 -> PCA64 -> UMAP32 -> PCA8 -> UMAP3.
    Every stage fits on the CURRENT nominal reference only. Defects are never used
    in fitting or model selection inside this function.
    """
    slug = chain_name(stages)
    chain_dir = run_dir / "chains" / slug
    chain_dir.mkdir(parents=True, exist_ok=True)

    train_current = np.asarray(sampled_train, dtype=np.float32)
    normal_current = np.asarray(prepared["normal"], dtype=np.float32)
    defects_current = {
        k: np.asarray(v, dtype=np.float32) for k, v in prepared["defects"].items()
    }

    config = {
        "category": category,
        "subsampling": subsampling,
        "source": args.source,
        "chain": [
            {"stage": i + 1, "method": method, "output_dim": dim}
            for i, (method, dim) in enumerate(stages)
        ],
        "chain_name": slug,
        "input_dimension": int(train_current.shape[1]),
        "fit_policy": "nominal_reference_only_at_every_stage",
        "test_policy": "normal_and_defects_transform_only",
        "selected_reference_count": int(len(train_current)),
    }
    (chain_dir / "chain_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"SEQUENTIAL CHAIN: {slug}")
    print("  " + " -> ".join([f"{train_current.shape[1]}D"] + [f"{m.upper()}{d}" for m, d in stages]))
    print("  FIT POLICY: nominal reference ONLY at every stage")
    print("  TEST POLICY: normal + defects are transform-only")

    stage_summaries = []
    for stage_index, (method, output_dim) in enumerate(stages):
        input_dim = int(train_current.shape[1])
        stage_dir = chain_dir / f"stage_{stage_index + 1:02d}_{method}_{output_dim}d"

        train_current, normal_current, defects_current, fit_s, transform_s, extra = (
            apply_chain_stage(
                stage_index, method, output_dim,
                train_current, normal_current, defects_current,
                args, stage_dir, device, category, subsampling,
                sequences=prepared.get("sequences"),
                patch_grid=prepared.get("patch_grid"),
                patchcore_model=prepared.get("patchcore_model"),
                test_loader=prepared.get("test_loader"),
                selected_indices=selected_indices,
                cached_test=prepared.get("cached_test"),
            )
        )

        stage_summaries.append({
            "stage": stage_index + 1,
            "method": method,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "fit_seconds": fit_s,
            "transform_seconds": transform_s,
            **extra,
        })

        if args.chain_evaluate_each_stage:
            result = evaluate(
                f"stage_{stage_index + 1:02d}_{method}_{output_dim}d",
                train_current, normal_current, defects_current,
                args.nominal_percentile, fit_s, transform_s, extra,
                nn_query_batch_size=args.nn_query_batch_size,
                nn_backend=args.nn_backend,
            )
            save_result(result, stage_dir / "evaluation")

        if output_dim in (2, 3):
            prefix = " -> ".join(
                f"{m.upper()}{d}" for m, d in stages[: stage_index + 1]
            )
            save_chain_lowdim_plot(
                prefix, train_current, normal_current, defects_current,
                stage_dir, args,
            )

    # Always evaluate the final representation.
    final_method, final_dim = stages[-1]
    final_extra = {
        "chain_length": float(len(stages)),
        "final_dimension": float(final_dim),
    }
    final_result = evaluate(
        slug, train_current, normal_current, defects_current,
        args.nominal_percentile,
        float(sum(x["fit_seconds"] for x in stage_summaries)),
        float(sum(x["transform_seconds"] for x in stage_summaries)),
        final_extra,
        nn_query_batch_size=args.nn_query_batch_size,
        nn_backend=args.nn_backend,
    )
    save_result(final_result, chain_dir / "final_evaluation")

    (chain_dir / "stage_summary.json").write_text(
        json.dumps(stage_summaries, indent=2, default=str), encoding="utf-8"
    )
    print(f"[chain] finished: {chain_dir}")
    return chain_dir

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PatchCore dimensionality-reduction pipeline")
    p.add_argument("--source", choices=("data", "features"), default="features",
                   help="data: extract PatchCore features from MVTec images; features: load the saved feature cache")
    p.add_argument("--repo-root", type=Path, required=True,
                   help="PatchCore repository root; still needed for coreset sampling and --source data")
    p.add_argument("--data-root", type=Path, default=None,
                   help="MVTec root; required only when --source data")
    p.add_argument("--feature-cache-root", type=Path, default=None,
                   help="Root containing category feature-cache folders; required when --source features")
    p.add_argument("--output-dir", type=Path, default=Path("dim_reduction_results"))
    p.add_argument(
        "--checkpoint-root", type=Path, default=None,
        help=(
            "Existing dimensionality-reduction result root containing "
            "<category>/<subsampling>/<method>/<checkpoint>. If a checkpoint "
            "exists it is loaded instead of refitting."
        ),
    )
    p.add_argument(
        "--force-refit", action="store_true",
        help="Ignore existing reducer/checkpoint files and train the first-stage method again.",
    )
    p.add_argument("--categories", nargs="+", default=["bottle"],
                   help="One or more categories, comma-separated values, or 'all'")
    p.add_argument("--methods", nargs="+", default=["all"], help="all or method names; comma-separated also works")
    p.add_argument(
        "--reduction-chain", nargs="+", default=None,
        help=(
            "Optional sequential reduction chain. Example: "
            "--reduction-chain pca:64 umap:32 pca:8 umap:3. "
            "When supplied, chain mode runs instead of the legacy --methods/--projection-methods path."
        ),
    )
    p.add_argument(
        "--chain-evaluate-each-stage", action="store_true",
        help="Also compute/save nearest-nominal evaluation metrics after every intermediate chain stage. Final stage is always evaluated.",
    )
    p.add_argument("--subsampling", nargs="+", default=["coreset"],
                   help="One or more of: random coreset none all")
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
    p.add_argument("--nn-query-batch-size", type=int, default=4096)
    p.add_argument(
        "--nn-backend", choices=("auto", "faiss", "sklearn"), default="auto",
        help="1-NN evaluation backend. auto prefers exact FAISS IndexFlatL2 when installed.",
    )
    p.add_argument("--visualization-points-per-class", type=int, default=2000)
    p.add_argument("--retrieval-examples-per-defect", type=int, default=3)

    # Optional second-stage projection. This is applied only to outputs of
    # reduced representation methods, never directly to the original 1024-D space.
    p.add_argument(
        "--projection-methods", nargs="+", default=["none"],
        help="Second-stage projection: none, pca, umap, all. Applied after each reduced method.",
    )
    p.add_argument(
        "--projection-dims", nargs="+", type=int, default=[2, 3],
        help="Projection output dimensions; currently 2 and/or 3.",
    )
    p.add_argument("--projection-points-per-class", type=int, default=3000)
    p.add_argument("--projection-umap-neighbors", type=int, default=30)
    p.add_argument("--projection-umap-min-dist", type=float, default=0.0)
    p.add_argument(
        "--projection-html", action="store_true",
        help="Also save an interactive Plotly HTML for 3-D projections.",
    )

    p.add_argument("--umap-dim", type=int, default=16)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.0)

    p.add_argument("--ae-epochs", type=int, default=100)
    p.add_argument("--ae-patience", type=int, default=10)
    p.add_argument("--ae-batch-size", type=int, default=1024)
    p.add_argument("--ae-encode-batch-size", type=int, default=2048)
    p.add_argument("--ae-lr", type=float, default=1e-3)

    p.add_argument("--kpca-gamma", type=float, default=-1.0)

    # p.add_argument("--spca-alpha", type=float, default=1.0)
    # p.add_argument("--spca-ridge-alpha", type=float, default=0.01)
    # p.add_argument("--spca-max-iter", type=int, default=300)
    # p.add_argument("--spca-tol", type=float, default=1e-4)

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

    p.add_argument("--deep-svdd-epochs", type=int, default=150)
    p.add_argument( "--deep-svdd-lr", type=float, default=1e-4)
    p.add_argument("--deep-svdd-batch-size", type=int, default=256)
    p.add_argument("--deep-svdd-weight-decay", type=float, default=5e-7)
    
    return p


def validate(args) -> None:
    if args.source == "data" and args.data_root is None:
        raise ValueError("--data-root is required when --source data")
    if args.source == "features" and args.feature_cache_root is None:
        raise ValueError("--feature-cache-root is required when --source features")
    if not 0 < args.subsample_percentage <= 1: raise ValueError("--subsample-percentage must be in (0,1]")
    if not 0 < args.anomaly_fraction_threshold <= 1: raise ValueError("--anomaly-fraction-threshold must be in (0,1]")
    if not 0 < args.nominal_percentile < 100: raise ValueError("--nominal-percentile must be in (0,100)")
    if args.latent_dim <= 0: raise ValueError("--latent-dim must be positive")
    if any(dim not in (2, 3) for dim in args.projection_dims):
        raise ValueError("--projection-dims currently accepts only 2 and/or 3")
    if args.projection_points_per_class == 0:
        raise ValueError("--projection-points-per-class must be positive or negative for unlimited")
    chain = getattr(args, "reduction_chain_parsed", [])
    if chain:
        previous_dim = 1024  # PatchCore cache/model output in this pipeline
        for index, (method, dim) in enumerate(chain):
            if dim >= previous_dim:
                raise ValueError(
                    f"--reduction-chain must strictly decrease dimensions; "
                    f"stage {index + 1} requests {previous_dim}D -> {dim}D."
                )
            previous_dim = dim
            if method == "transformer" and index != 0:
                raise ValueError("Transformer is supported only as the first chain stage.")


def run_one_configuration(
    args, methods, category, subsampling, device, prepared: Dict[str, Any]
) -> Path:
    run_dir = args.output_dir.resolve() / category / subsampling
    run_dir.mkdir(parents=True, exist_ok=True)

    all_train = prepared["all_train"]
    sequences = prepared.get("sequences")
    patch_grid = prepared["patch_grid"]
    normal = prepared["normal"]
    defects = prepared["defects"]
    cached_indices = prepared.get("cached_coreset_indices") if subsampling == "coreset" else None

    sampled_train, selected_indices = subsample_reference(
        all_train, subsampling, args.max_train_patches,
        args.subsample_percentage, args.seed, device,
        preselected_indices=cached_indices,
    )
    np.save(run_dir / "selected_reference_indices.npy", selected_indices)

    config = vars(args).copy()
    config.update({
        "repo_root": str(args.repo_root.resolve()),
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
        "feature_cache_root": str(args.feature_cache_root.resolve()) if args.feature_cache_root else None,
        "output_dir": str(run_dir),
        "category": category,
        "subsampling_current": subsampling,
        "methods": methods,
        "device": str(device),
        "all_nominal_patch_count": int(len(all_train)),
        "selected_reference_count": int(len(sampled_train)),
    })
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    print(f"\n{'#' * 80}")
    print(f"CATEGORY={category} | SOURCE={args.source} | SUBSAMPLING={subsampling}")
    print(f"All nominal patches: {all_train.shape}")
    print(f"Selected nominal reference: {sampled_train.shape}")
    print(f"Normal test patches: {normal.shape}")
    for name, values in defects.items():
        print(f"{name}: {values.shape}")

    chain = getattr(args, "reduction_chain_parsed", [])
    if chain:
        run_reduction_chain(
            args, chain, category, subsampling, device, prepared,
            sampled_train, selected_indices, run_dir,
        )
        print(f"Finished chain configuration: {run_dir}")
        return run_dir

    results: List[MethodResult] = []
    for method in methods:
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_method(
                method, sampled_train, selected_indices, normal, defects,
                sequences, patch_grid, prepared.get("patchcore_model"),
                prepared.get("test_loader"), args, method_dir, device,
                cached_test=prepared.get("cached_test"),
                category=category, subsampling=subsampling,
            )
            save_result(result, method_dir)
            results.append(result)
            print(result.metrics.round(4).to_string(index=False))
        except Exception as exc:
            error = {"method": method, "error_type": type(exc).__name__, "message": str(exc)}
            (method_dir / "error.json").write_text(json.dumps(error, indent=2))
            print(f"[ERROR] {method}: {type(exc).__name__}: {exc}")

    if not results:
        raise RuntimeError(
            f"Every selected method failed for {category}/{subsampling}. "
            "Check per-method error.json files."
        )

    long_table, comparison = make_comparison(results)
    long_table.to_csv(run_dir / "all_method_results_long.csv", index=False)
    save_blending_comparison_plot(long_table, run_dir)
    if len(results) > 1:
        comparison.to_csv(run_dir / "comparison.csv", index=False)

    print(f"Finished: {run_dir}")
    return run_dir


def prepare_category_from_data(args, category, methods, device) -> Dict[str, Any]:
    data_root = args.data_root.resolve()
    model = build_patchcore_model(device)
    train_dataset, test_dataset, train_loader, test_loader = build_loaders(
        data_root, category, args.batch_size, args.num_workers, args.seed
    )
    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    all_train, sequences, patch_grid = collect_training_embeddings(
        model, train_loader, keep_sequences=("transformer" in methods or any(m == "transformer" for m, _ in getattr(args, "reduction_chain_parsed", [])))
    )
    normal, defects = collect_test_embeddings(
        model, test_loader, args.anomaly_fraction_threshold,
        args.max_normal_test_patches, args.max_defect_test_patches, args.seed,
    )
    return {
        "all_train": all_train,
        "sequences": sequences,
        "patch_grid": patch_grid,
        "normal": normal,
        "defects": defects,
        "patchcore_model": model,
        "test_loader": test_loader,
        "cached_test": None,
        "cached_coreset_indices": None,
    }


def main() -> None:
    args = parser().parse_args()
    args.reduction_chain_parsed = parse_reduction_chain(args.reduction_chain)
    validate(args)
    methods = parse_methods(args.methods)
    args.projection_methods_parsed = parse_projection_methods(args.projection_methods)
    categories = parse_categories(args.categories)
    subsamplings = parse_subsamplings(args.subsampling)
    device = resolve_device(args.device)
    set_seed(args.seed)

    args.repo_root = args.repo_root.resolve()
    add_repo_src(args.repo_root)
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint_root is not None:
        args.checkpoint_root = args.checkpoint_root.resolve()

    print("Selected configuration:")
    print(f"  source:       {args.source}")
    print(f"  categories:   {categories}")
    print(f"  subsampling:  {subsamplings}")
    if args.reduction_chain_parsed:
        print(f"  mode:         sequential chain")
        print(f"  chain:        {args.reduction_chain_parsed}")
        print("  fit policy:   NOMINAL reference only at every stage")
        print("  test policy:  normal + defects transform-only")
    else:
        print(f"  methods:      {methods}")
        print(f"  projections:  {args.projection_methods_parsed or ['none']} -> {args.projection_dims}")
    print(f"  checkpoint:   {args.checkpoint_root if args.checkpoint_root else 'current output folders'}")
    print(f"  force refit:  {args.force_refit}")
    print(f"  NN backend:   {args.nn_backend}")
    print(f"  device:       {device}")

    completed = []
    failures = []

    for category in categories:
        try:
            if args.source == "features":
                prepared = load_cached_category(args.feature_cache_root.resolve(), category, args)
            else:
                prepared = prepare_category_from_data(args, category, methods, device)

            for subsampling in subsamplings:
                try:
                    out = run_one_configuration(
                        args, methods, category, subsampling, device, prepared
                    )
                    completed.append(str(out))
                except Exception as exc:
                    failures.append({
                        "category": category, "subsampling": subsampling,
                        "error_type": type(exc).__name__, "message": str(exc),
                    })
                    print(f"[RUN FAILED] {category}/{subsampling}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            failures.append({
                "category": category, "subsampling": None,
                "error_type": type(exc).__name__, "message": str(exc),
            })
            print(f"[CATEGORY FAILED] {category}: {type(exc).__name__}: {exc}")

    summary = {
        "source": args.source,
        "categories": categories,
        "subsampling": subsamplings,
        "methods": methods if not args.reduction_chain_parsed else None,
        "reduction_chain": args.reduction_chain_parsed,
        "completed": completed,
        "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "batch_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"Completed configurations: {len(completed)}")
    print(f"Failed configurations:    {len(failures)}")
    print(f"Summary: {args.output_dir / 'batch_run_summary.json'}")
    if failures:
        print("Some configurations failed; see batch_run_summary.json and per-method error.json files.")


if __name__ == "__main__":
    main()
