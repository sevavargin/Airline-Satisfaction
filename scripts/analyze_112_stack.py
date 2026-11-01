from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOF_ARCHIVE = PROJECT_ROOT / "112 OOF.npz"
CONSENSUS_SIZES = [5, 10, 20, 40, 80, 112]
BLEND_WEIGHTS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


target = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "train.csv",
    usecols=["satisfaction"],
)["satisfaction"].to_numpy()
folds = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "train_with_folds.parquet",
    columns=["fold"],
)["fold"].to_numpy()
stack_oof = np.load(
    PROJECT_ROOT / "artifacts" / "oof" / "stack247.npy"
).reshape(-1)

archive = np.load(OOF_ARCHIVE, allow_pickle=True)
member_names = archive["members"].tolist()

scored_members = []
for member_name in member_names:
    member_oof = archive[f"{member_name}__oof"]
    scored_members.append((roc_auc_score(target, member_oof), member_name))

scored_members.sort(reverse=True)

row_count = len(target)
rank_sum = np.zeros(row_count, dtype=np.float64)
consensus_by_size = {}

for member_number, (_, member_name) in enumerate(scored_members, start=1):
    member_oof = archive[f"{member_name}__oof"]
    rank_sum += rankdata(member_oof, method="average") / row_count

    if member_number in CONSENSUS_SIZES:
        consensus_by_size[member_number] = (rank_sum / member_number).copy()
        print(f"prepared top-{member_number} consensus", flush=True)

stack_rank = rankdata(stack_oof, method="average") / row_count
stack_auc = roc_auc_score(target, stack_oof)
stack_fold_auc = {
    fold: roc_auc_score(target[folds == fold], stack_oof[folds == fold])
    for fold in range(5)
}

result_rows = []
for consensus_size, consensus_oof in consensus_by_size.items():
    consensus_auc = roc_auc_score(target, consensus_oof)
    correlation = np.corrcoef(stack_rank, consensus_oof)[0, 1]

    for weight in BLEND_WEIGHTS:
        blend_oof = (1.0 - weight) * stack_rank + weight * consensus_oof
        blend_auc = roc_auc_score(target, blend_oof)
        fold_differences = []

        for fold in range(5):
            valid_mask = folds == fold
            fold_auc = roc_auc_score(target[valid_mask], blend_oof[valid_mask])
            fold_differences.append(fold_auc - stack_fold_auc[fold])

        result_rows.append(
            {
                "top_k": consensus_size,
                "consensus_auc": consensus_auc,
                "correlation_stack": correlation,
                "weight": weight,
                "overall_difference": blend_auc - stack_auc,
                "positive_folds": int(np.sum(np.asarray(fold_differences) > 0)),
                "minimum_fold_difference": float(np.min(fold_differences)),
                "mean_fold_difference": float(np.mean(fold_differences)),
            }
        )

results = pd.DataFrame(result_rows)
best_rows = results.loc[
    results.groupby("top_k")["overall_difference"].idxmax()
].sort_values("overall_difference", ascending=False)

print("\nBest blend for each consensus size")
print(
    best_rows.to_string(
        index=False,
        float_format=lambda value: f"{value:.9f}",
    )
)
print("\nTop 15 individual members")
for auc, member_name in scored_members[:15]:
    print(f"{auc:.9f}  {member_name}")
