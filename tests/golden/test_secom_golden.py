import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.secom.main import RAW_COLUMNS, SensorRequest, predict


def run_golden():
    demo = pd.read_csv(ROOT / "services" / "secom" / "data" / "demo_rows.csv")[RAW_COLUMNS]
    expected = pd.read_csv(ROOT / "tests" / "golden" / "secom_final_wafer_risk_scores.csv").head(len(demo))

    max_diff = 0.0
    for i in range(len(demo)):
        vals = [None if pd.isna(v) else float(v) for v in demo.iloc[i].tolist()]
        res = predict(SensorRequest(values=vals, columns=RAW_COLUMNS, explain=(i == 0)))
        diff = abs(res["scores"]["fused_telemetry_risk"] - expected.loc[i, "fused_telemetry_risk"])
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-4, f"Golden test mismatch: max_diff={max_diff}"
    print(f"SECOM GOLDEN TEST PASSED! Verified {len(demo)} rows | Max diff: {max_diff:.6f}")


if __name__ == "__main__":
    run_golden()