from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_LIBRARY_PATH = PROJECT_ROOT / "112 OOF.npz"
SIGNAL_WEIGHTS = [0.05, 0.10]

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


sample = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "sample_submission.csv")
xwc = pd.read_csv(PROJECT_ROOT / "submissions" / "xwc_v10_exact_submission.csv")
stack_test = np.load(
    PROJECT_ROOT / "artifacts" / "test_predictions" / "stack247.npy"
).reshape(-1)
library = np.load(NEW_LIBRARY_PATH, allow_pickle=True)

assert xwc["id"].equals(sample["id"])
assert len(stack_test) == len(sample)

consensus = np.zeros(len(sample), dtype=np.float64)
for member_name in TOP_FIVE_MEMBERS:
    member_test = library[f"{member_name}__test"]
    assert len(member_test) == len(sample)
    assert np.isfinite(member_test).all()
    consensus += percentile_rank(member_test)
consensus /= len(TOP_FIVE_MEMBERS)

xwc_rank = percentile_rank(xwc["satisfaction"].to_numpy())
stack_rank = percentile_rank(stack_test)
validated_direction = consensus - stack_rank

for weight in SIGNAL_WEIGHTS:
    candidate_rank = percentile_rank(xwc_rank + weight * validated_direction)
    submission = sample.copy()
    submission["satisfaction"] = candidate_rank

    label = int(round(weight * 100))
    output_path = (
        PROJECT_ROOT
        / "submissions"
        / f"xwc_v10_plus_new112_delta{label:02d}.csv"
    )
    submission.to_csv(output_path, index=False)

    print(f"weight={weight:.2f}")
    print(f"correlation_xwc={np.corrcoef(candidate_rank, xwc_rank)[0, 1]:.9f}")
    print(f"signal_rms={(weight * validated_direction).std():.9f}")
    print(f"saved={output_path}")
