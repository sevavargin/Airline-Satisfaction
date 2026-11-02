from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_LIBRARY_PATH = PROJECT_ROOT / "112 OOF.npz"
NEW_SIGNAL_WEIGHTS = [0.05, 0.10]

TOP_FIVE_MEMBERS = [
    "combo_p22_gbdt_xgb_auxev_s1234",
    "r6_tabpfn_classctxbag_oof",
    "combo_p22_gbdt_lgb_auxall_s2024",
    "combo_p22_gbdt_lgb_auxall_s1234",
    "combo_p22_gbdt_xgb_auxall_s31337",
]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    return rankdata(values, method="average") / len(values)


sample_submission = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
)
v54_submission = pd.read_csv(
    PROJECT_ROOT / "submissions" / "rabari_v54_submission.csv"
)
stack247_test = np.load(
    PROJECT_ROOT / "artifacts" / "test_predictions" / "stack247.npy"
).reshape(-1)
new_library = np.load(NEW_LIBRARY_PATH, allow_pickle=True)

assert v54_submission["id"].equals(sample_submission["id"])
assert len(stack247_test) == len(sample_submission)

new_rank_sum = np.zeros(len(sample_submission), dtype=np.float64)
for member_name in TOP_FIVE_MEMBERS:
    member_test = new_library[f"{member_name}__test"]
    assert len(member_test) == len(sample_submission)
    assert np.isfinite(member_test).all()
    new_rank_sum += percentile_rank(member_test)

new_consensus_rank = new_rank_sum / len(TOP_FIVE_MEMBERS)
stack247_rank = percentile_rank(stack247_test)
v54_rank = percentile_rank(v54_submission["satisfaction"].to_numpy())

print("members:")
for member_name in TOP_FIVE_MEMBERS:
    print(" -", member_name)

for new_signal_weight in NEW_SIGNAL_WEIGHTS:
    new_signal = new_signal_weight * (new_consensus_rank - stack247_rank)
    corrected_score = v54_rank + new_signal
    corrected_rank = percentile_rank(corrected_score)

    submission = sample_submission.copy()
    submission["satisfaction"] = corrected_rank
    weight_label = int(round(new_signal_weight * 100))
    output_path = (
        PROJECT_ROOT
        / "submissions"
        / f"v54_plus_new112_top5_delta{weight_label:02d}.csv"
    )
    submission.to_csv(output_path, index=False)

    print(f"\nnew signal weight: {new_signal_weight:.2f}")
    print(f"correlation with V54: {np.corrcoef(corrected_rank, v54_rank)[0, 1]:.9f}")
    print(f"applied signal RMS: {new_signal.std():.9f}")
    print(f"rows: {len(submission):,}")
    print("saved:", output_path)
