import os
import sys
import joblib

# Allow importing secom_main from the same folder
sys.path.append(os.path.dirname(__file__))
import secom_main  # This runs your original script ONCE (takes ~20 mins)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Save the already-trained pipeline and SHAP explainer
bundle = {
    "pipe": secom_main.final_chron_pipe,
    "explainer": secom_main.explainer,
    "raw_columns": secom_main.X_all.columns.tolist(),
    "selected_features": secom_main.final_chron_pipe.selected_features_,
}
joblib.dump(bundle, "models/secom_model.joblib")

# 2. Save 25 unseen test rows to upload later in your frontend
demo_df = secom_main.df.iloc[int(0.80 * len(secom_main.df)):].head(25)
demo_df.to_csv("data/demo_test_wafers.csv", index=False)

print("\nSUCCESS! Saved 'models/secom_model.joblib' and 'data/demo_test_wafers.csv'")