"""CPATK-native contrastive latent embedding backend.

This module implements a supervised contrastive embedding model for Cell
Painting profiles.  It is deliberately independent of the external CLIPn
package.  The goal is not to reproduce CLIPn, but to provide a defensible,
well-logged, testable latent representation for CPATK workflows.
"""

from __future__ import annotations

import copy
import logging
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cpatk.threading_utils import configure_torch_threads, normalise_thread_count

try:  # pragma: no cover - exercised indirectly in environments with torch
    import torch
    from torch import nn
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - broken torch environments
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc


@dataclass
class NativeContrastiveConfig:
    """Configuration for the CPATK-native contrastive backend."""

    latent_dim: int = 20
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    activation: str = "gelu"
    normalisation: str = "layernorm"
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 256
    positives_per_label: int = 2
    temperature: float = 0.10
    validation_fraction: float = 0.15
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    eval_batches: int = 4
    steps_per_epoch: Optional[int] = None
    random_state: int = 42
    device: str = "auto"
    positive_column: str = "cpd_id"
    normalise_latent: bool = True
    encode_chunk_size: int = 32768
    n_threads: int = 1


@dataclass
class NativeContrastiveResult:
    """Result container returned by the native contrastive backend."""

    latent_table: pd.DataFrame
    model: object
    training_loss: pd.DataFrame
    training_summary: pd.DataFrame
    positive_label_report: pd.DataFrame
    split_report: pd.DataFrame
    backend_status: pd.DataFrame


def require_torch_stack(*, purpose: str) -> None:
    """Raise a clear error if PyTorch is unavailable."""
    if TORCH_IMPORT_ERROR is not None or torch is None or nn is None:
        raise ImportError(
            "PyTorch could not be imported for "
            f"{purpose}. Install a CPU or CUDA PyTorch build in the CPATK "
            "environment before using --backend_module cpatk_contrastive."
        ) from TORCH_IMPORT_ERROR


def get_native_contrastive_status() -> pd.DataFrame:
    """Return a one-row status table for the native contrastive backend."""
    available = TORCH_IMPORT_ERROR is None and torch is not None
    version = str(getattr(torch, "__version__", "not_imported")) if torch is not None else "not_imported"
    cuda_available = False
    cuda_count = 0
    cuda_names = ""
    if available:
        try:
            cuda_available = bool(torch.cuda.is_available())
            cuda_count = int(torch.cuda.device_count())
            cuda_names = ";".join(
                str(torch.cuda.get_device_name(index)) for index in range(cuda_count)
            )
        except Exception:
            cuda_available = False
            cuda_count = 0
            cuda_names = "unavailable"
    return pd.DataFrame.from_records(
        [
            {
                "backend_module": "cpatk_contrastive",
                "available": bool(available),
                "module_file": __file__,
                "message": (
                    "CPATK-native contrastive backend is available."
                    if available
                    else str(TORCH_IMPORT_ERROR)
                ),
                "torch_version": version,
                "torch_cuda_available": cuda_available,
                "torch_cuda_device_count": cuda_count,
                "torch_cuda_device_names": cuda_names,
                "python_version": sys.version.replace("\n", " "),
                "platform": platform.platform(),
            }
        ]
    )


class ContrastiveEncoder(nn.Module):
    """Simple MLP encoder for Cell Painting contrastive learning."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "gelu",
        dropout: float = 0.10,
        normalisation: str = "layernorm",
    ) -> None:
        """Initialise the encoder."""
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        activation_layer = _activation_layer(name=activation)
        norm_name = str(normalisation).lower()
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                continue
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if norm_name == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_name == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(copy.deepcopy(activation_layer))
            if float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, int(latent_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        """Encode input values and L2-normalise latent vectors."""
        latent = self.network(values)
        return torch.nn.functional.normalize(latent, p=2, dim=1)


def _activation_layer(*, name: str) -> "nn.Module":
    """Return an activation layer from a user-facing name."""
    require_torch_stack(purpose="native contrastive activation construction")
    value = str(name).lower()
    if value == "relu":
        return nn.ReLU()
    if value == "elu":
        return nn.ELU()
    if value == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if value == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported native contrastive activation: {name}")


def resolve_torch_device(*, requested: str = "auto") -> "torch.device":
    """Resolve a requested device string into a torch device."""
    require_torch_stack(purpose="native contrastive device resolution")
    value = str(requested or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def supervised_contrastive_loss(
    *,
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float = 0.10,
) -> "torch.Tensor":
    """Calculate supervised contrastive loss for a mini-batch.

    Anchors that have no positive partner in the current mini-batch are ignored.
    A clear error is raised when the whole batch has no positive pairs, because
    such a batch cannot train a supervised contrastive objective.
    """
    require_torch_stack(purpose="native contrastive loss")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional tensor.")
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional tensor.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels must have the same row count.")
    if embeddings.shape[0] < 2:
        raise ValueError("At least two samples are required for contrastive loss.")

    labels = labels.reshape(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    logits_mask = torch.ones_like(positive_mask) - torch.eye(
        positive_mask.shape[0], device=embeddings.device
    )
    positive_mask = positive_mask * logits_mask
    positives_per_anchor = positive_mask.sum(dim=1)
    valid_anchor_mask = positives_per_anchor > 0
    if not bool(valid_anchor_mask.any()):
        raise ValueError("The mini-batch contains no positive label pairs.")

    logits = torch.matmul(embeddings, embeddings.T) / max(float(temperature), 1e-8)
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(
        positives_per_anchor,
        min=1.0,
    )
    return -mean_log_prob_pos[valid_anchor_mask].mean()


def combine_cleaned_matrices(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    config: NativeContrastiveConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Combine cleaned per-dataset matrices and construct labels."""
    frames = []
    meta_frames = []
    feature_counts = []
    for dataset_name, matrix in cleaned.items():
        if matrix.empty:
            continue
        matrix_frame = matrix.reset_index(drop=True).copy()
        matrix_frame.insert(0, "Dataset", str(dataset_name))
        matrix_frame.insert(1, "Sample", np.arange(matrix_frame.shape[0], dtype=int))
        frames.append(matrix_frame)
        meta = metadata.loc[metadata["Dataset"].astype(str) == str(dataset_name)].copy()
        meta = meta.sort_values("Sample").reset_index(drop=True)
        meta_frames.append(meta)
        feature_counts.append(int(matrix.shape[1]))
    if not frames:
        raise ValueError("No non-empty matrices were supplied to the native contrastive backend.")
    if len(set(feature_counts)) != 1:
        raise ValueError("All native contrastive matrices must have the same number of features.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined_metadata = pd.concat(meta_frames, ignore_index=True, sort=False)
    combined_metadata = combined_metadata.merge(
        combined.loc[:, ["Dataset", "Sample"]],
        on=["Dataset", "Sample"],
        how="right",
    )
    feature_cols = [column for column in combined.columns if column not in {"Dataset", "Sample"}]
    positive_column = config.positive_column
    if positive_column not in combined_metadata.columns:
        raise ValueError(
            "The configured native contrastive positive column is missing from "
            f"metadata: {positive_column}"
        )
    positive_values = combined_metadata[positive_column].fillna("missing").astype(str)
    valid_mask = positive_values.str.len() > 0
    if not bool(valid_mask.all()):
        positive_values.loc[~valid_mask] = "missing"
    codes, uniques = pd.factorize(positive_values, sort=True)
    label_report = _make_positive_label_report(
        positive_values=positive_values,
        codes=codes,
        positive_column=positive_column,
    )
    X = combined.loc[:, feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Native contrastive input contains non-finite values after preprocessing.")
    return X, codes.astype(np.int64), combined_metadata.reset_index(drop=True), label_report


def _make_positive_label_report(
    *,
    positive_values: pd.Series,
    codes: np.ndarray,
    positive_column: str,
) -> pd.DataFrame:
    """Summarise positive-label replication for contrastive learning."""
    counts = positive_values.value_counts(dropna=False).rename_axis("positive_label")
    report = counts.reset_index(name="n_profiles")
    report.insert(0, "positive_column", positive_column)
    report["usable_for_contrastive_training"] = report["n_profiles"] >= 2
    report["is_singleton"] = report["n_profiles"] == 1
    report["encoded_label"] = report["positive_label"].map(
        {label: int(code) for label, code in zip(positive_values, codes)}
    )
    return report


def make_stratified_validation_split(
    *,
    labels: np.ndarray,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create a row-level train/validation split without breaking small labels."""
    rng = np.random.default_rng(seed=int(random_state))
    labels = np.asarray(labels, dtype=np.int64)
    train_mask = np.ones(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    records = []
    if validation_fraction <= 0:
        records.append(
            {
                "split_status": "validation_disabled",
                "n_train": int(train_mask.sum()),
                "n_validation": 0,
            }
        )
        return train_mask, val_mask, pd.DataFrame.from_records(records)

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        n_label = int(indices.shape[0])
        if n_label < 4:
            records.append(
                {
                    "encoded_label": int(label),
                    "n_profiles": n_label,
                    "n_train": n_label,
                    "n_validation": 0,
                    "reason": "kept_in_train_too_few_profiles_for_validation_pairs",
                }
            )
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = int(round(n_label * float(validation_fraction)))
        n_val = min(max(2, n_val), n_label - 2)
        val_indices = shuffled[:n_val]
        train_mask[val_indices] = False
        val_mask[val_indices] = True
        records.append(
            {
                "encoded_label": int(label),
                "n_profiles": n_label,
                "n_train": int(n_label - n_val),
                "n_validation": int(n_val),
                "reason": "split_with_positive_pairs_in_both_sets",
            }
        )
    split_report = pd.DataFrame.from_records(records)
    return train_mask, val_mask, split_report


def _usable_positive_groups(*, labels: np.ndarray, indices: np.ndarray) -> dict[int, np.ndarray]:
    """Return label groups with at least two rows in the supplied index pool."""
    groups = {}
    labels = np.asarray(labels, dtype=np.int64)
    indices = np.asarray(indices, dtype=int)
    for label in np.unique(labels[indices]):
        label_indices = indices[labels[indices] == label]
        if label_indices.shape[0] >= 2:
            groups[int(label)] = label_indices
    return groups


def sample_positive_batch(
    *,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    positives_per_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a mini-batch that contains positive label pairs."""
    groups = _usable_positive_groups(labels=labels, indices=indices)
    if not groups:
        raise ValueError("No repeated positive labels are available for batch sampling.")
    label_values = np.asarray(sorted(groups), dtype=int)
    per_label = int(max(2, positives_per_label))
    n_labels = int(max(1, batch_size // per_label))
    replace_labels = n_labels > label_values.shape[0]
    chosen_labels = rng.choice(label_values, size=n_labels, replace=replace_labels)
    batch_parts = []
    for label in chosen_labels:
        label_indices = groups[int(label)]
        replace_samples = per_label > label_indices.shape[0]
        batch_parts.append(
            rng.choice(label_indices, size=per_label, replace=replace_samples)
        )
    batch = np.concatenate(batch_parts)
    if batch.shape[0] > int(batch_size):
        batch = batch[: int(batch_size)]
    rng.shuffle(batch)
    return batch.astype(int)


def _evaluate_sampled_loss(
    *,
    model: ContrastiveEncoder,
    X_tensor: "torch.Tensor",
    y_tensor: "torch.Tensor",
    labels_np: np.ndarray,
    indices: np.ndarray,
    config: NativeContrastiveConfig,
    rng: np.random.Generator,
    device: "torch.device",
) -> float:
    """Evaluate sampled supervised contrastive loss."""
    usable = _usable_positive_groups(labels=labels_np, indices=indices)
    if not usable:
        return float("nan")
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(max(1, config.eval_batches))):
            batch = sample_positive_batch(
                labels=labels_np,
                indices=indices,
                batch_size=int(config.batch_size),
                positives_per_label=int(config.positives_per_label),
                rng=rng,
            )
            embeddings = model(X_tensor[batch].to(device))
            loss = supervised_contrastive_loss(
                embeddings=embeddings,
                labels=y_tensor[batch].to(device),
                temperature=float(config.temperature),
            )
            losses.append(float(loss.detach().cpu().item()))
    model.train()
    return float(np.nanmean(losses)) if losses else float("nan")


def _same_label_neighbour_rate(
    *,
    latent: np.ndarray,
    labels: np.ndarray,
    n_jobs: int = 1,
) -> float:
    """Calculate nearest-neighbour same-label rate for diagnostics."""
    if latent.shape[0] < 2:
        return float("nan")
    try:
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(
            n_neighbors=2,
            metric="cosine",
            n_jobs=max(1, int(n_jobs)),
        )
        nn_model.fit(latent)
        _, indices = nn_model.kneighbors(latent, return_distance=True)
        neighbour_indices = indices[:, 1]
        return float(np.mean(labels[neighbour_indices] == labels))
    except Exception:
        return float("nan")


def encode_matrix(
    *,
    model: ContrastiveEncoder,
    X: np.ndarray,
    device: "torch.device",
    chunk_size: int,
) -> np.ndarray:
    """Encode a full matrix in row chunks."""
    model.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, X.shape[0], max(1, int(chunk_size))):
            end = min(start + int(chunk_size), X.shape[0])
            batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
            latent = model(batch).detach().cpu().numpy()
            encoded.append(latent)
    if not encoded:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(encoded, axis=0)


def fit_native_contrastive_backend(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    config: NativeContrastiveConfig,
    logger: Optional[logging.Logger] = None,
) -> NativeContrastiveResult:
    """Fit and project the CPATK-native contrastive backend."""
    require_torch_stack(purpose="CPATK-native contrastive backend")
    if logger is not None:
        logger.info("Starting CPATK-native contrastive backend with config: %s", asdict(config))
    threads = configure_torch_threads(n_threads=config.n_threads, logger=logger)
    config.n_threads = normalise_thread_count(value=threads, default=1)
    _set_reproducible_seeds(seed=int(config.random_state))
    device = resolve_torch_device(requested=config.device)
    X, labels, metadata_aligned, label_report = combine_cleaned_matrices(
        cleaned=cleaned,
        metadata=metadata,
        config=config,
    )
    usable_label_count = int((label_report["n_profiles"] >= 2).sum())
    usable_profile_count = int(label_report.loc[label_report["n_profiles"] >= 2, "n_profiles"].sum())
    if usable_label_count < 1:
        raise ValueError(
            "CPATK-native contrastive learning needs at least one positive label "
            "with two or more profiles. Use a less granular positive column or run PCA."
        )
    if usable_profile_count < 2:
        raise ValueError("Fewer than two profiles are available for contrastive training.")

    train_mask, val_mask, split_report = make_stratified_validation_split(
        labels=labels,
        validation_fraction=float(config.validation_fraction),
        random_state=int(config.random_state),
    )
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    if not _usable_positive_groups(labels=labels, indices=train_indices):
        raise ValueError(
            "The training split contains no repeated positive labels. Reduce the "
            "validation fraction or use a less granular positive column."
        )
    validation_has_pairs = bool(_usable_positive_groups(labels=labels, indices=val_indices))
    if logger is not None and not validation_has_pairs:
        logger.warning(
            "Native contrastive validation split has no repeated positive labels; "
            "early stopping will monitor sampled training loss."
        )

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(labels, dtype=torch.long)
    model = ContrastiveEncoder(
        input_dim=int(X.shape[1]),
        latent_dim=int(config.latent_dim),
        hidden_dims=list(config.hidden_dims),
        activation=str(config.activation),
        dropout=float(config.dropout),
        normalisation=str(config.normalisation),
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng(seed=int(config.random_state))
    best_state = copy.deepcopy(model.state_dict())
    best_monitor = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    records = []
    stopping_reason = "completed_max_epochs"
    monitor_name = "validation_loss" if validation_has_pairs else "train_loss"

    if config.steps_per_epoch is None or int(config.steps_per_epoch) <= 0:
        steps_per_epoch = int(max(1, math.ceil(train_indices.shape[0] / max(1, int(config.batch_size)))))
    else:
        steps_per_epoch = int(config.steps_per_epoch)
    if logger is not None:
        logger.info(
            "Native contrastive training will use %d sampled mini-batches per epoch.",
            steps_per_epoch,
        )

    for epoch in range(1, int(max(1, config.epochs)) + 1):
        step_losses = []
        sampled_rows = 0
        model.train()
        for _step in range(steps_per_epoch):
            batch_indices = sample_positive_batch(
                labels=labels,
                indices=train_indices,
                batch_size=int(config.batch_size),
                positives_per_label=int(config.positives_per_label),
                rng=rng,
            )
            optimiser.zero_grad(set_to_none=True)
            embeddings = model(X_tensor[batch_indices].to(device))
            loss = supervised_contrastive_loss(
                embeddings=embeddings,
                labels=y_tensor[batch_indices].to(device),
                temperature=float(config.temperature),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            step_losses.append(float(loss.detach().cpu().item()))
            sampled_rows += int(batch_indices.shape[0])
        train_loss = float(np.nanmean(step_losses)) if step_losses else float("nan")
        train_loss_min = float(np.nanmin(step_losses)) if step_losses else float("nan")
        train_loss_max = float(np.nanmax(step_losses)) if step_losses else float("nan")
        validation_loss = _evaluate_sampled_loss(
            model=model,
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            labels_np=labels,
            indices=val_indices if validation_has_pairs else train_indices,
            config=config,
            rng=rng,
            device=device,
        )
        monitor_loss = validation_loss if validation_has_pairs else train_loss
        if np.isfinite(monitor_loss) and (best_monitor - monitor_loss) > float(config.early_stopping_min_delta):
            best_monitor = float(monitor_loss)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        records.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "monitor_loss": monitor_loss,
                "monitor_name": monitor_name,
                "batch_size": int(config.batch_size),
                "n_train_steps": int(steps_per_epoch),
                "n_training_examples_sampled": int(sampled_rows),
                "train_loss_min": train_loss_min,
                "train_loss_max": train_loss_max,
                "learning_rate": float(config.learning_rate),
                "temperature": float(config.temperature),
            }
        )
        if logger is not None and (epoch == 1 or epoch % 10 == 0):
            logger.info(
                "Native contrastive epoch %d/%d: train_loss=%.5f, %s=%.5f",
                epoch,
                int(config.epochs),
                train_loss,
                monitor_name,
                monitor_loss,
            )
        if epochs_without_improvement >= int(max(1, config.early_stopping_patience)):
            stopping_reason = "early_stopping_monitor_plateau"
            break
    model.load_state_dict(best_state)
    latent = encode_matrix(
        model=model,
        X=X,
        device=device,
        chunk_size=int(config.encode_chunk_size),
    )
    if config.normalise_latent:
        latent = _normalise_numpy_rows(values=latent)
    latent_cols = [f"latent_{idx + 1}" for idx in range(latent.shape[1])]
    latent_table = pd.DataFrame(latent, columns=latent_cols)
    latent_table.insert(0, "Dataset", metadata_aligned["Dataset"].to_numpy())
    latent_table.insert(1, "Sample", metadata_aligned["Sample"].to_numpy())
    for column in metadata_aligned.columns:
        if column not in latent_table.columns:
            latent_table[column] = metadata_aligned[column].to_numpy()

    nn_rate = _same_label_neighbour_rate(
        latent=latent,
        labels=labels,
        n_jobs=int(config.n_threads),
    )
    loss_table = pd.DataFrame.from_records(records)
    summary = pd.DataFrame.from_records(
        [
            {
                "backend": "cpatk_contrastive",
                "training_policy": "supervised_contrastive_with_validation_early_stopping",
                "positive_column": str(config.positive_column),
                "n_profiles": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_positive_labels": int(label_report.shape[0]),
                "n_repeated_positive_labels": usable_label_count,
                "n_profiles_in_repeated_positive_labels": usable_profile_count,
                "latent_dim": int(config.latent_dim),
                "hidden_dims": ";".join(map(str, config.hidden_dims)),
                "activation": str(config.activation),
                "normalisation": str(config.normalisation),
                "dropout": float(config.dropout),
                "batch_size": int(config.batch_size),
                "steps_per_epoch": int(steps_per_epoch),
                "positives_per_label": int(config.positives_per_label),
                "temperature": float(config.temperature),
                "requested_epochs": int(config.epochs),
                "epochs_completed": int(loss_table.shape[0]),
                "best_epoch": int(best_epoch),
                "best_monitor_loss": float(best_monitor) if np.isfinite(best_monitor) else np.nan,
                "monitor_name": monitor_name,
                "stopping_reason": stopping_reason,
                "validation_fraction": float(config.validation_fraction),
                "validation_has_positive_pairs": validation_has_pairs,
                "nearest_neighbour_same_positive_label_rate": nn_rate,
                "device": str(device),
                "n_threads": int(config.n_threads),
                "torch_version": str(torch.__version__),
            }
        ]
    )
    if logger is not None:
        logger.info("CPATK-native contrastive backend complete: %s", summary.to_dict(orient="records"))
    return NativeContrastiveResult(
        latent_table=latent_table,
        model=model,
        training_loss=loss_table,
        training_summary=summary,
        positive_label_report=label_report,
        split_report=split_report,
        backend_status=get_native_contrastive_status(),
    )


def _normalise_numpy_rows(*, values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2-normalise a NumPy matrix."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return values / norms


def _set_reproducible_seeds(*, seed: int) -> None:
    """Set NumPy and torch seeds for reproducibility."""
    require_torch_stack(purpose="native contrastive reproducibility")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
