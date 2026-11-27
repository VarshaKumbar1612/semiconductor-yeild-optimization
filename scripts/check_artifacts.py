import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAFER_DIR = ROOT / "services" / "wafer"
SECOM_DIR = ROOT / "services" / "secom"

WAFER_PATH_MAP = {
    "data/demo_samples.npz": WAFER_DIR / "data" / "demo_samples.npz",
    "data/validator_limits.json": WAFER_DIR / "data" / "validator_limits.json",
    "wafer_b1/b1_lib.py": WAFER_DIR / "libs" / "b1_lib.py",
    "wafer_b1/b1_s0_last.weights.h5": WAFER_DIR / "artifacts" / "wafer_b1" / "b1_s0_last.weights.h5",
    "wafer_bar/bar_model.joblib": WAFER_DIR / "artifacts" / "wafer_bar" / "bar_model.joblib",
    "wafer_spp/spp_lib.py": WAFER_DIR / "libs" / "spp_lib.py",
    "wafer_spp/spp_s0_last.weights.h5": WAFER_DIR / "artifacts" / "wafer_spp" / "spp_s0_last.weights.h5",
}

EXTRA_HASHES = {
    "services/wafer/data/cause_table_v2.csv": (
        WAFER_DIR / "data" / "cause_table_v2.csv",
        "ff378b47e460c814619ce25cf5acd4b9",
    ),
}

REQUIRED_SECOM_DATA = [
    SECOM_DIR / "data" / "demo_rows.csv",
    SECOM_DIR / "data" / "feature_order.json",
    SECOM_DIR / "data" / "table1_protocol_a_summary.csv",
    SECOM_DIR / "data" / "table2_protocol_b_summary.csv",
    SECOM_DIR / "data" / "table3_shap_sensor_stability.csv",
]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all() -> bool:
    ok = True
    manifest_path = WAFER_DIR / "artifacts" / "MANIFEST.csv"

    print("=== Verifying WM-811K Bundle Manifest ===")
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row["file"]
            expected_md5 = row["md5"].strip().lower()
            target = WAFER_PATH_MAP.get(rel, WAFER_DIR / rel)
            if not target.exists():
                print(f"[MISSING] {rel} -> {target}")
                ok = False
                continue
            actual_md5 = md5_of(target)
            if actual_md5 != expected_md5:
                print(f"[HASH MISMATCH] {rel}: expected {expected_md5}, got {actual_md5}")
                ok = False
            else:
                print(f"[OK] {rel} ({actual_md5})")

    print("\n=== Verifying Audited Literature Table ===")
    for label, (path, expected_md5) in EXTRA_HASHES.items():
        if not path.exists():
            print(f"[MISSING] {label}")
            ok = False
            continue
        actual_md5 = md5_of(path)
        if actual_md5 != expected_md5:
            print(f"[HASH MISMATCH] {label}: expected {expected_md5}, got {actual_md5}")
            ok = False
        else:
            print(f"[OK] {label} ({actual_md5})")

    print("\n=== Verifying SECOM Artifacts ===")
    secom_joblibs = sorted((SECOM_DIR / "artifacts").glob("*.joblib"))
    if not secom_joblibs:
        print("[MISSING] No .joblib model artifact found in services/secom/artifacts/")
        ok = False
    else:
        for jb in secom_joblibs:
            print(f"[OK] {jb.relative_to(ROOT)} ({jb.stat().st_size:,} bytes | md5: {md5_of(jb)})")

    for path in REQUIRED_SECOM_DATA:
        if not path.exists():
            print(f"[MISSING] {path.relative_to(ROOT)}")
            ok = False
        else:
            print(f"[OK] {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")

    return ok


if __name__ == "__main__":
    if verify_all():
        print("\nALL ARTIFACTS VERIFIED SUCCESSFULLY.")
        sys.exit(0)
    print("\nARTIFACT VERIFICATION FAILED.")
    sys.exit(1)