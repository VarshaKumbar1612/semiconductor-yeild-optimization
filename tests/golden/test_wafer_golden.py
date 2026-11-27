import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.wafer.main import CLASSES, WaferRequest, predict


def run_golden():
    data = np.load(ROOT / "services" / "wafer" / "data" / "demo_samples.npz", allow_pickle=True)
    maps = data["maps"]
    expected_map = {
        "wafer_bar": data["bar_probs"],
        "wafer_b1": data["b1_probs"],
        "wafer_spp": data["spp_probs"],
    }

    max_diffs = {"wafer_bar": 0.0, "wafer_b1": 0.0, "wafer_spp": 0.0}

    for i, m in enumerate(maps):
        res = predict(WaferRequest(matrix=m.tolist(), explain=(i == 0)))
        for item in res["results"]:
            mid = item["model_id"]
            got = np.array([item["scores"][c] for c in CLASSES], dtype=np.float32)
            diff = float(np.max(np.abs(got - expected_map[mid][i])))
            max_diffs[mid] = max(max_diffs[mid], diff)
            assert np.allclose(got, expected_map[mid][i], atol=1e-6), f"{mid} sample {i} diff={diff}"

    print(f"WM-811K GOLDEN TEST PASSED! Verified {len(maps)} wafers | Max diffs: {max_diffs}")


if __name__ == "__main__":
    run_golden()
