"""
================================================================================
TITLE: Final Leak-Free Multi-Criterion Feature Selection & Hardness-Aware Ensemble
       Framework for Semiconductor Wafer Defect Detection (SECOM)
DESCRIPTION:
    Complete, end-to-end SECOM pipeline with full step-by-step console logging:
    - Step 1: Chronological Ordering & Timestamp Integrity Verification
    - Step 2: In-Fold Missingness Metadata, Distance-KNN Imputation & Robust Scaling
    - Step 3: Collinearity Filtering (|r| > 0.90) & Consensus-RFECV Selection
    - Step 4: Strictly Nested Two-Protocol Evaluation:
              * Protocol A: 10 Repeated Stratified 80/20 Splits (Literature Benchmark)
              * Protocol B: 3 Expanding-Window Chronological Splits (Fab Deployment)
    - Step 5: Multi-Split TreeSHAP Associated Sensor Stability & Wafer Diagnostics
================================================================================
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
from imblearn.ensemble import BalancedRandomForestClassifier
from imbens.ensemble import SelfPacedEnsembleClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def banner(step_title: str):
    print("\n" + "=" * 92)
    print(f"  {step_title}")
    print("=" * 92)


# ==============================================================================
# STEP 1: DATA LOADING, TIMESTAMP VERIFICATION & CHRONOLOGICAL SORTING
# ==============================================================================
banner("STEP 1: DATA LOADING, TIMESTAMP VERIFICATION & CHRONOLOGICAL ORDERING")

df = pd.read_csv("uci-secom.csv")
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
print(f"[INPUT]  Raw File Loaded         : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"[CHECK]  Invalid Timestamps      : {df['Time'].isna().sum()}")
print(f"[CHECK]  Raw File Pre-Sorted?    : {df['Time'].is_monotonic_increasing} -> Sorting by 'Time'...")

df = df.sort_values("Time").reset_index(drop=True)
df["time_rank"] = np.arange(len(df)) / len(df)

X_all = df.drop(columns=["Time", "Pass/Fail", "time_rank"])
y_all = df["Pass/Fail"].map({-1: 0, 1: 1})  # 0 = Pass, 1 = Fail
time_all = df[["time_rank"]]

pass_cnt = int((y_all == 0).sum())
fail_cnt = int((y_all == 1).sum())
print(f"[OUTPUT] Feature Matrix (X_all)  : {X_all.shape[0]} wafers x {X_all.shape[1]} raw sensors")
print(f"[OUTPUT] Target Vector (y_all)   : Pass (0) = {pass_cnt} ({pass_cnt/len(y_all)*100:.2f}%) | "
      f"Fail (1) = {fail_cnt} ({fail_cnt/len(y_all)*100:.2f}%) | Ratio = {pass_cnt/fail_cnt:.2f}:1")
print(f"[INFO]   Production Date Span    : {df['Time'].min()}  -->  {df['Time'].max()}")

# Pre-compute absolute correlation of each sensor with time_rank for regime-shift flagging
time_corrs = {
    col: float(abs(np.corrcoef(df["time_rank"], df[col].fillna(df[col].median()))[0, 1]))
    for col in X_all.columns if df[col].nunique(dropna=True) > 1
}


# ==============================================================================
# MODULAR IN-SPLIT PREPROCESSING & CONSENSUS-RFECV PIPELINE CLASS
# ==============================================================================
class SECOMTelemetryPipeline:
    """
    Self-contained, zero-leakage SECOM telemetry feature engineering and modeling pipeline.
    All imputers, scalers, collinearity filters, and RFECV selectors are fit strictly
    on training splits only.
    """

    def __init__(self, random_state: int = 42, optuna_trials: int = 15, verbose: bool = False):
        self.random_state = random_state
        self.optuna_trials = optuna_trials
        self.verbose = verbose
        self.drop_cols_ = []
        self.collinear_cols_ = []
        self.selected_features_ = []
        self.imputer_ = KNNImputer(n_neighbors=5, weights="distance")
        self.scaler_ = RobustScaler()
        self.models_ = {}

    def preprocess_transform(self, X_raw: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Extracts missingness metadata, drops dead/sparse sensors, KNN-imputes, and Robust-scales."""
        nan_cnt = X_raw.isnull().sum(axis=1).values.reshape(-1, 1)

        if fit:
            zero_var = X_raw.columns[X_raw.nunique(dropna=True) <= 1].tolist()
            high_miss = X_raw.columns[X_raw.isnull().mean() > 0.40].tolist()
            self.drop_cols_ = list(set(zero_var + high_miss))

            if self.verbose:
                print(f"[INPUT]  Training Split Shape    : {X_raw.shape[0]} wafers x {X_raw.shape[1]} sensors")
                print(f"[INFO]   Zero-Variance Dropped   : {len(zero_var)} columns (nunique <= 1 on Train)")
                print(f"[INFO]   High-Missing Dropped    : {len(high_miss)} columns (> 40% NaN on Train)")
                print(f"[INFO]   Total Redundant Dropped : {len(self.drop_cols_)} columns")

        X_c = X_raw.drop(columns=self.drop_cols_)
        if fit:
            X_scaled = self.scaler_.fit_transform(self.imputer_.fit_transform(X_c))
        else:
            X_scaled = self.scaler_.transform(self.imputer_.transform(X_c))

        df_out = pd.DataFrame(X_scaled, columns=X_c.columns, index=X_raw.index)
        df_out["missing_sensor_count"] = nan_cnt

        if fit and self.verbose:
            print(f"[INFO]   Imputation & Scaling    : KNNImputer(k=5, distance) + RobustScaler (Train-Fit)")
            print(f"[INFO]   Metadata Engineered     : Added 'missing_sensor_count' (wafer-level NaN count)")
            print(f"[OUTPUT] Clean Preprocessed Data : {df_out.shape[0]} wafers x {df_out.shape[1]} features")
            print(f"[CHECK]  Remaining NaNs in Train : {int(df_out.isnull().sum().sum())}")

        return df_out

    def fit_feature_selection(self, X_tr_raw: pd.DataFrame, y_tr: pd.Series):
        """Performs in-fold collinearity filtering (|r| > 0.90) and Multi-Criterion Consensus-RFECV."""
        X_tr = self.preprocess_transform(X_tr_raw, fit=True)

        # Collinearity filter |r| > 0.90 on training data only
        corr = X_tr.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.collinear_cols_ = [c for c in upper.columns if any(upper[c] > 0.90)]
        X_tr_u = X_tr.drop(columns=self.collinear_cols_)

        # Multi-criterion consensus ranking (Random Forest + LightGBM + Mutual Information)
        rf_r = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=self.random_state, n_jobs=-1
        ).fit(X_tr_u, y_tr)
        lgb_r = LGBMClassifier(
            n_estimators=100, is_unbalance=True, random_state=self.random_state, verbose=-1, n_jobs=-1
        ).fit(X_tr_u, y_tr)
        mi_r = pd.Series(mutual_info_classif(X_tr_u, y_tr, random_state=self.random_state), index=X_tr_u.columns)

        imp_rf = pd.Series(rf_r.feature_importances_, index=X_tr_u.columns)
        imp_lgb = pd.Series(lgb_r.feature_importances_, index=X_tr_u.columns)
        consensus = (
            (imp_rf / (imp_rf.max() + 1e-9)) +
            (imp_lgb / (imp_lgb.max() + 1e-9)) +
            (mi_r / (mi_r.max() + 1e-9))
        ) / 3.0

        candidates = consensus.nlargest(75).index.tolist()

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rfecv = RFECV(
            estimator=LGBMClassifier(n_estimators=80, is_unbalance=True, random_state=self.random_state, verbose=-1),
            step=3,
            cv=inner_cv,
            scoring="average_precision",
            min_features_to_select=15,
            n_jobs=-1
        ).fit(X_tr_u[candidates], y_tr)

        self.selected_features_ = np.array(candidates)[rfecv.support_].tolist()

        if self.verbose:
            print(f"[INPUT]  Preprocessed Features   : {X_tr.shape[1]} features")
            print(f"[INFO]   Collinear Dropped       : {len(self.collinear_cols_)} columns (|r| > 0.90) -> {X_tr_u.shape[1]} remaining")
            print(f"[INFO]   Consensus Shortlist     : Top 75 features (RF + LightGBM + Mutual Information)")
            print(f"[INFO]   Best Inner CV PR-AUC    : {rfecv.cv_results_['mean_test_score'].max():.4f}")
            print(f"[OUTPUT] Optimal RFECV Features  : {len(self.selected_features_)} features selected")
            print(f"[OUTPUT] Top 10 Selected Sensors : {self.selected_features_[:10]}")

        return X_tr_u[self.selected_features_]

    def transform_features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X_proc = self.preprocess_transform(X_raw, fit=False)
        X_uncorr = X_proc.drop(columns=self.collinear_cols_)
        return X_uncorr[self.selected_features_]

    def fit_models(self, X_tr_sel: pd.DataFrame, y_tr: pd.Series):
        """Tunes SPE-DT via nested Optuna and trains all benchmark + proposed models."""
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        imb_ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        def objective(trial):
            clf = SelfPacedEnsembleClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=trial.suggest_int("max_depth", 4, 16),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
                    random_state=self.random_state
                ),
                n_estimators=trial.suggest_int("n_estimators", 60, 200),
                random_state=self.random_state
            )
            oof = cross_val_predict(clf, X_tr_sel, y_tr, cv=inner_cv, method="predict_proba")[:, 1]
            return average_precision_score(y_tr, oof)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.optuna_trials)
        bp = study.best_params

        self.models_ = {
            "L1 Logistic Regression": LogisticRegression(
                penalty="l1", solver="liblinear", class_weight="balanced", C=0.5, random_state=self.random_state
            ),
            "Plain Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
            "Balanced Random Forest": BalancedRandomForestClassifier(
                n_estimators=250, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
            "SPE-DT (Optuna Tuned)": SelfPacedEnsembleClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=bp["max_depth"], min_samples_leaf=bp["min_samples_leaf"], random_state=self.random_state
                ),
                n_estimators=bp["n_estimators"],
                random_state=self.random_state
            ),
            "SPE-LightGBM": SelfPacedEnsembleClassifier(
                estimator=LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.05, random_state=self.random_state, verbose=-1
                ),
                n_estimators=100,
                random_state=self.random_state
            ),
            "Cost-Sensitive LightGBM": LGBMClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=6,
                scale_pos_weight=imb_ratio, random_state=self.random_state, verbose=-1
            ),
            "Cost-Sensitive XGBoost": XGBClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=5,
                scale_pos_weight=imb_ratio, eval_metric="logloss", random_state=self.random_state
            )
        }

        for clf in self.models_.values():
            clf.fit(X_tr_sel, y_tr)
        return self


# ==============================================================================
# STEP 2 & STEP 3 DEMONSTRATION ON REFERENCE 80/20 SPLIT (TRANSPARENT LOGS)
# ==============================================================================
X_ref_tr, X_ref_te, y_ref_tr, y_ref_te = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

banner("STEP 2: IN-FOLD CLEANING, KNN IMPUTATION & ROBUST SCALING (REFERENCE SPLIT)")
ref_pipe = SECOMTelemetryPipeline(random_state=42, optuna_trials=15, verbose=True)
X_ref_tr_proc = ref_pipe.preprocess_transform(X_ref_tr, fit=True)
X_ref_te_proc = ref_pipe.preprocess_transform(X_ref_te, fit=False)
print(f"[OUTPUT] Clean Test Matrix       : {X_ref_te_proc.shape[0]} wafers x {X_ref_te_proc.shape[1]} features (NaNs: {int(X_ref_te_proc.isnull().sum().sum())})")

banner("STEP 3: COLLINEARITY FILTERING & CONSENSUS-RFECV SELECTION (REFERENCE SPLIT)")
X_ref_tr_sel = ref_pipe.fit_feature_selection(X_ref_tr, y_ref_tr)
X_ref_te_sel = ref_pipe.transform_features(X_ref_te)
print(f"[OUTPUT] Final Selected Test Set : {X_ref_te_sel.shape[0]} wafers x {X_ref_te_sel.shape[1]} features")


# ==============================================================================
# EVALUATION HELPER FUNCTIONS
# ==============================================================================
def compute_alarm_metrics(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, base_rate: float) -> dict:
    """Computes ROC-AUC, PR-AUC, and Recall/Lift at fixed 5%, 10%, and 20% fab inspection budgets."""
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    rec_at, lift_at = {}, {}
    for pct in [5, 10, 20]:
        k = max(1, int(np.ceil(len(y_true) * (pct / 100.0))))
        top_idx = np.argsort(y_prob)[::-1][:k]
        preds = np.zeros_like(y_true)
        preds[top_idx] = 1
        rec_at[pct] = recall_score(y_true, preds, zero_division=0) * 100.0
        prec = precision_score(y_true, preds, zero_division=0)
        lift_at[pct] = prec / max(base_rate, 1e-9)

    return {
        "Model": model_name,
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Rec@5% (%)": rec_at[5],
        "Rec@10% (%)": rec_at[10],
        "Rec@20% (%)": rec_at[20],
        "Lift@10%": lift_at[10],
        "Lift@20%": lift_at[20]
    }


def summarize_with_bootstrap(df_folds: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Aggregates Mean +/- SD and 95% Bootstrap Confidence Intervals across evaluation splits."""
    summary = []
    rng = np.random.default_rng(42)
    for model, grp in df_folds.groupby("Model", sort=False):
        roc_vals = grp["ROC-AUC"].values
        pr_vals = grp["PR-AUC"].values
        boot_roc = [np.mean(rng.choice(roc_vals, size=len(roc_vals), replace=True)) for _ in range(n_boot)]
        ci_low, ci_high = np.percentile(boot_roc, [2.5, 97.5])

        summary.append({
            "Model": model,
            "ROC-AUC (Mean±SD)": f"{np.mean(roc_vals):.4f} ± {np.std(roc_vals):.4f}",
            "ROC-AUC 95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "PR-AUC (Mean±SD)": f"{np.mean(pr_vals):.4f} ± {np.std(pr_vals):.4f}",
            "Rec@5% (%)": f"{grp['Rec@5% (%)'].mean():.1f}%",
            "Rec@10% (%)": f"{grp['Rec@10% (%)'].mean():.1f}%",
            "Rec@20% (%)": f"{grp['Rec@20% (%)'].mean():.1f}%",
            "Lift@10%": f"{grp['Lift@10%'].mean():.2f}x",
            "Lift@20%": f"{grp['Lift@20%'].mean():.2f}x"
        })
    return pd.DataFrame(summary)


def extract_brf_shap(brf_model, X_sel: pd.DataFrame):
    """Computes Class-1 (Fail) TreeSHAP values for Balanced Random Forest."""
    explainer = shap.TreeExplainer(brf_model)
    sv = explainer.shap_values(X_sel)
    if isinstance(sv, list):
        sv_fail = sv[1]
    elif sv.ndim == 3:
        sv_fail = sv[:, :, 1]
    else:
        sv_fail = sv
    return explainer, sv_fail


# ==============================================================================
# STEP 4A: PROTOCOL A — 10 REPEATED STRATIFIED 80/20 SPLITS + SHAP STABILITY
# ==============================================================================
banner("STEP 4A: PROTOCOL A — 10 REPEATED STRATIFIED SPLITS (NESTED PIPELINE)")

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
proto_a_rows = []
top10_counts = {}
shap_scores_accum = {}

for fold_id, (tr_idx, te_idx) in enumerate(sss.split(X_all, y_all), 1):
    pipe = SECOMTelemetryPipeline(random_state=42 + fold_id, optuna_trials=15, verbose=False)
    X_tr_sel = pipe.fit_feature_selection(X_all.iloc[tr_idx], y_all.iloc[tr_idx])
    X_te_sel = pipe.transform_features(X_all.iloc[te_idx])
    pipe.fit_models(X_tr_sel, y_all.iloc[tr_idx])

    y_tr_fold = y_all.iloc[tr_idx]
    y_te_fold = y_all.iloc[te_idx]
    base_rate = y_te_fold.mean()

    t_clf = LogisticRegression().fit(time_all.iloc[tr_idx], y_tr_fold)
    t_probs = t_clf.predict_proba(time_all.iloc[te_idx])[:, 1]
    rec_t = compute_alarm_metrics("Time-Only Baseline", y_te_fold.values, t_probs, base_rate)
    rec_t["Split"] = fold_id
    proto_a_rows.append(rec_t)

    for name, clf in pipe.models_.items():
        probs = clf.predict_proba(X_te_sel)[:, 1]
        rec = compute_alarm_metrics(name, y_te_fold.values, probs, base_rate)
        rec["Split"] = fold_id
        proto_a_rows.append(rec)

    _, sv_fail = extract_brf_shap(pipe.models_["Balanced Random Forest"], X_te_sel)
    mean_imp = pd.Series(np.abs(sv_fail).mean(axis=0), index=X_te_sel.columns)
    for feat in mean_imp.nlargest(10).index:
        top10_counts[feat] = top10_counts.get(feat, 0) + 1
    for feat, val in mean_imp.items():
        shap_scores_accum.setdefault(feat, []).append(float(val))

    print(f"[Protocol A] Split {fold_id:02d}/10 Complete | Selected Features: {len(pipe.selected_features_)}")

df_proto_a = pd.DataFrame(proto_a_rows)
summary_a = summarize_with_bootstrap(df_proto_a)
print("\n--- TABLE 1: PROTOCOL A BENCHMARK (10 REPEATED STRATIFIED 80/20 SPLITS) ---")
print(summary_a.to_string(index=False))
df_proto_a.to_csv("table1_protocol_a_raw_folds.csv", index=False)
summary_a.to_csv("table1_protocol_a_summary.csv", index=False)


# ==============================================================================
# STEP 4B: PROTOCOL B — EXPANDING-WINDOW CHRONOLOGICAL DEPLOYMENT SPLITS
# ==============================================================================
banner("STEP 4B: PROTOCOL B — EXPANDING-WINDOW CHRONOLOGICAL FORECASTING")

n = len(X_all)
windows = [
    (slice(0, int(0.50 * n)), slice(int(0.50 * n), int(0.65 * n))),
    (slice(0, int(0.65 * n)), slice(int(0.65 * n), int(0.80 * n))),
    (slice(0, int(0.80 * n)), slice(int(0.80 * n), n))
]

proto_b_rows = []
final_chron_pipe = None
final_X_future_sel = None
final_y_future = None

for w_id, (tr_sl, te_sl) in enumerate(windows, 1):
    pipe = SECOMTelemetryPipeline(random_state=100 + w_id, optuna_trials=15, verbose=False)
    X_tr_sel = pipe.fit_feature_selection(X_all.iloc[tr_sl], y_all.iloc[tr_sl])
    X_te_sel = pipe.transform_features(X_all.iloc[te_sl])
    pipe.fit_models(X_tr_sel, y_all.iloc[tr_sl])

    y_tr_w = y_all.iloc[tr_sl]
    y_te_w = y_all.iloc[te_sl]
    base_rate = y_te_w.mean()

    t_clf = LogisticRegression().fit(time_all.iloc[tr_sl], y_tr_w)
    t_probs = t_clf.predict_proba(time_all.iloc[te_sl])[:, 1]
    rec_t = compute_alarm_metrics("Time-Only Baseline", y_te_w.values, t_probs, base_rate)
    rec_t["Window"] = w_id
    proto_b_rows.append(rec_t)

    for name, clf in pipe.models_.items():
        probs = clf.predict_proba(X_te_sel)[:, 1]
        rec = compute_alarm_metrics(name, y_te_w.values, probs, base_rate)
        rec["Window"] = w_id
        proto_b_rows.append(rec)

    if w_id == 3:
        final_chron_pipe = pipe
        final_X_future_sel = X_te_sel
        final_y_future = y_te_w

    print(f"[Protocol B] Window {w_id}/3 Complete | Train={len(X_tr_sel)}, Test={len(X_te_sel)} (Fails={y_te_w.sum()})")

df_proto_b = pd.DataFrame(proto_b_rows)
summary_b = summarize_with_bootstrap(df_proto_b)
print("\n--- TABLE 2: PROTOCOL B BENCHMARK (EXPANDING CHRONOLOGICAL DEPLOYMENT) ---")
print(summary_b.to_string(index=False))
df_proto_b.to_csv("table2_protocol_b_raw_windows.csv", index=False)
summary_b.to_csv("table2_protocol_b_summary.csv", index=False)


# ==============================================================================
# STEP 5: MULTI-SPLIT SHAP STABILITY, FIGURES & INTEGRATION OUTPUTS
# ==============================================================================
banner("STEP 5: MULTI-SPLIT SHAP ASSOCIATED SENSOR STABILITY & DIAGNOSTICS")

stability_rows = []
for feat, count in top10_counts.items():
    t_corr = time_corrs.get(feat, 0.0)
    if feat == "missing_sensor_count":
        diag_label = "Wafer Telemetry Completeness Metadata"
    elif t_corr > 0.15:
        diag_label = "Regime-Shift Sensitive Sensor (|r_time| > 0.15)"
    else:
        diag_label = "Stationary Process Sensor"

    stability_rows.append({
        "Associated Sensor / Feature": feat,
        "Top-10 Stability (10 Splits)": f"{count}/10 ({count*10}%)",
        "Mean |SHAP| Impact": round(float(np.mean(shap_scores_accum[feat])), 5),
        "Time Correlation |r|": round(t_corr, 4),
        "Diagnostic Classification": diag_label
    })

stability_df = pd.DataFrame(stability_rows).sort_values(
    by=["Top-10 Stability (10 Splits)", "Mean |SHAP| Impact"], ascending=False
)
print("--- TABLE 3: MULTI-SPLIT SHAP ASSOCIATED SENSOR STABILITY (BALANCED RF) ---")
print(stability_df.head(12).to_string(index=False))
stability_df.to_csv("table3_shap_sensor_stability.csv", index=False)

brf_final = final_chron_pipe.models_["Balanced Random Forest"]
explainer, sv_fail = extract_brf_shap(brf_final, final_X_future_sel)

plt.figure(figsize=(10, 6))
shap.summary_plot(sv_fail, final_X_future_sel, max_display=12, show=False)
plt.title("Associated Sensor Contributions on Future Chronological Holdout (Balanced RF)", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("fig1_final_shap_global_summary.png", dpi=300)
plt.close()

future_probs_brf = brf_final.predict_proba(final_X_future_sel)[:, 1]
future_probs_spe = final_chron_pipe.models_["SPE-DT (Optuna Tuned)"].predict_proba(final_X_future_sel)[:, 1]
fused_risk = 0.5 * future_probs_brf + 0.5 * future_probs_spe

fail_positions = np.where(final_y_future.values == 1)[0]
if len(fail_positions) > 0:
    top_fail_pos = fail_positions[np.argmax(fused_risk[fail_positions])]
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    exp_obj = shap.Explanation(
        values=sv_fail[top_fail_pos],
        base_values=base_val,
        data=final_X_future_sel.iloc[top_fail_pos],
        feature_names=final_chron_pipe.selected_features_
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp_obj, max_display=10, show=False)
    plt.title(f"Wafer-Level Associated Sensor Diagnosis (Future Test Index #{final_X_future_sel.index[top_fail_pos]})", fontsize=11)
    plt.tight_layout()
    plt.savefig("fig2_final_shap_local_waterfall.png", dpi=300)
    plt.close()

n_fut = len(final_X_future_sel)
flag_10 = np.zeros(n_fut, dtype=int)
flag_20 = np.zeros(n_fut, dtype=int)
flag_10[np.argsort(fused_risk)[::-1][:max(1, int(np.ceil(0.10 * n_fut)))]] = 1
flag_20[np.argsort(fused_risk)[::-1][:max(1, int(np.ceil(0.20 * n_fut)))]] = 1

risk_export = pd.DataFrame({
    "wafer_chronological_index": final_X_future_sel.index,
    "actual_fail": final_y_future.values,
    "prob_balanced_rf": np.round(future_probs_brf, 4),
    "prob_spe_dt": np.round(future_probs_spe, 4),
    "fused_telemetry_risk": np.round(fused_risk, 4),
    "alarm_flag_top10pct": flag_10,
    "alarm_flag_top20pct": flag_20
})
risk_export.to_csv("secom_final_wafer_risk_scores.csv", index=False)

summary_json = {
    "pipeline_version": "SECOM_Final_Publication_Pipeline",
    "feature_engineering": "KNNImputer(k=5, distance) + RobustScaler + missing_sensor_count + Consensus-RFECV",
    "protocol_A_benchmark": "10 Repeated Stratified 80/20 Splits (table1_protocol_a_summary.csv)",
    "protocol_B_deployment": "3 Expanding-Window Chronological Splits (table2_protocol_b_summary.csv)",
    "top_stable_associated_sensors": stability_df.head(10)["Associated Sensor / Feature"].tolist(),
    "exported_files": [
        "table1_protocol_a_summary.csv",
        "table2_protocol_b_summary.csv",
        "table3_shap_sensor_stability.csv",
        "secom_final_wafer_risk_scores.csv",
        "fig1_final_shap_global_summary.png",
        "fig2_final_shap_local_waterfall.png"
    ]
}
with open("secom_pipeline_metadata.json", "w") as f:
    json.dump(summary_json, f, indent=4)

print("\n[SAVED] All tables, 300-DPI SHAP figures, and risk score CSVs exported successfully!")
print("=" * 92)