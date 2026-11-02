from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOF_ARCHIVE = PROJECT_ROOT / "112 OOF.npz"
ALPHAS = [30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0]
BATCH_SIZE = 50_000


def rank01(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") / len(values)).astype(np.float32)


def fold_sufficient_statistics(
    matrix: np.ndarray,
    target: np.ndarray,
    fold_indices: np.ndarray,
) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray]:
    feature_count = matrix.shape[1]
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    target_sum = 0.0
    cross_features = np.zeros((feature_count, feature_count), dtype=np.float64)
    cross_target = np.zeros(feature_count, dtype=np.float64)

    for start in range(0, len(fold_indices), BATCH_SIZE):
        batch_indices = fold_indices[start : start + BATCH_SIZE]
        x_batch = matrix[batch_indices].astype(np.float64, copy=False)
        y_batch = target[batch_indices].astype(np.float64, copy=False)
        feature_sum += x_batch.sum(axis=0)
        target_sum += y_batch.sum()
        cross_features += x_batch.T @ x_batch
        cross_target += x_batch.T @ y_batch

    return len(fold_indices), feature_sum, target_sum, cross_features, cross_target


target = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "train.csv",
    usecols=["satisfaction"],
)["satisfaction"].to_numpy(dtype=np.float64)
folds = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "train_with_folds.parquet",
    columns=["fold"],
)["fold"].to_numpy()
stack_oof = np.load(
    PROJECT_ROOT / "artifacts" / "oof" / "stack247.npy"
).reshape(-1)

archive = np.load(OOF_ARCHIVE, allow_pickle=True)
member_names = archive["members"].tolist()
row_count = len(target)
feature_count = 1 + len(member_names)

print(f"Building rank matrix: {row_count:,} rows x {feature_count} models", flush=True)
matrix = np.empty((row_count, feature_count), dtype=np.float32)
matrix[:, 0] = rank01(stack_oof)

for member_index, member_name in enumerate(member_names, start=1):
    matrix[:, member_index] = rank01(archive[f"{member_name}__oof"])
    if member_index % 10 == 0 or member_index == len(member_names):
        print(f"ranked {member_index}/{len(member_names)} new models", flush=True)

del archive

baseline_rank = matrix[:, 0].copy()
matrix[:, 1:] -= baseline_rank[:, None]
matrix[:, 0] = 0.0
residual_target = target - baseline_rank

fold_stats = []
for fold in range(5):
    indices = np.flatnonzero(folds == fold)
    fold_stats.append(fold_sufficient_statistics(matrix, residual_target, indices))
    print(f"prepared statistics for fold {fold}", flush=True)

total_n = sum(item[0] for item in fold_stats)
total_x_sum = sum((item[1] for item in fold_stats), np.zeros(feature_count))
total_y_sum = sum(item[2] for item in fold_stats)
total_xtx = sum(
    (item[3] for item in fold_stats),
    np.zeros((feature_count, feature_count)),
)
total_xty = sum((item[4] for item in fold_stats), np.zeros(feature_count))

stack_auc = roc_auc_score(target, stack_oof)
stack_fold_auc = [
    roc_auc_score(target[folds == fold], stack_oof[folds == fold])
    for fold in range(5)
]

print(f"stack247 OOF AUC: {stack_auc:.9f}", flush=True)
print("\nHonest five-fold anchored ridge correction", flush=True)

for alpha in ALPHAS:
    meta_oof = np.empty(row_count, dtype=np.float64)
    fold_deltas = []

    for fold, valid_stats in enumerate(fold_stats):
        valid_n, valid_x_sum, valid_y_sum, valid_xtx, valid_xty = valid_stats
        train_n = total_n - valid_n
        train_x_sum = total_x_sum - valid_x_sum
        train_y_sum = total_y_sum - valid_y_sum
        train_xtx = total_xtx - valid_xtx
        train_xty = total_xty - valid_xty

        x_mean = train_x_sum / train_n
        y_mean = train_y_sum / train_n
        centered_xtx = train_xtx - np.outer(train_x_sum, train_x_sum) / train_n
        centered_xty = train_xty - train_x_sum * train_y_sum / train_n

        system = centered_xtx.copy()
        system.flat[:: feature_count + 1] += alpha
        weights = np.linalg.solve(system, centered_xty)
        intercept = y_mean - x_mean @ weights

        valid_indices = np.flatnonzero(folds == fold)
        for start in range(0, len(valid_indices), BATCH_SIZE):
            batch_indices = valid_indices[start : start + BATCH_SIZE]
            meta_oof[batch_indices] = (
                baseline_rank[batch_indices]
                + matrix[batch_indices] @ weights
                + intercept
            )

        fold_auc = roc_auc_score(target[valid_indices], meta_oof[valid_indices])
        fold_deltas.append(fold_auc - stack_fold_auc[fold])

    meta_auc = roc_auc_score(target, meta_oof)
    print(
        f"alpha={alpha:8.0f}  auc={meta_auc:.9f}  "
        f"delta={meta_auc - stack_auc:+.9f}  "
        f"positive={sum(delta > 0 for delta in fold_deltas)}/5  "
        f"min={min(fold_deltas):+.9f}  "
        f"folds={[f'{delta:+.9f}' for delta in fold_deltas]}",
        flush=True,
    )
