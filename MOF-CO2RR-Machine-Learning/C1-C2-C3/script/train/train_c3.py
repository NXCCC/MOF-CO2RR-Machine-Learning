"""Train and evaluate the C3 photocatalytic CO2 reduction product classifier.

Uses the original notebook's 'All' feature group, preprocessing, SMOTE,
XGBoost hyperparameters, and ten stratified 80/20 splits (seeds 42–51).
No feature selection, plots, or SHAP analysis is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PRODUCTS = ("CO", "CH4", "CH3OH", "HCOOH")
FEATURE_GROUPS = {
    "Metal": ["M_Electron_Affinity", "M_Hardness", "M_Ionization", "M_MeltingPoint", "M_Polarizability", "M_Zeff", "M_d_electrons", "M_EN_Pauling"],
    "Linker": ["L_MW", "L_TPSA", "L_NumHDonors", "L_NumHAcceptors", "L_LOGP", "L_NumAromaticRings", "L_NumRotatableBonds"],
    "Geometry": ["ML_band_gap", "LCD", "PLD", "LFPD", "Density", "VSA", "GSA", "VF", "PV"],
    "Conditions": ["Additive_Role", "Rxn_Time", "Rxn_Temp", "Catalyst_Mass", "P_CO2_bar", "Photosensitizer_code", "Phase_code", "Light_Type_code", "mix_density_g_ml", "mix_bp_C"],
}


def load_data(csv_path: Path):
    try:
        df = pd.read_csv(csv_path, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")
    if "MAJORgas_std" not in df.columns:
        raise ValueError("Missing target column: MAJORgas_std")
    df = df.loc[df["MAJORgas_std"].isin(PRODUCTS)].copy()
    features = [col for cols in FEATURE_GROUPS.values() for col in cols if col in df.columns]
    if not features or df.empty:
        raise ValueError("No usable rows or input features found in the CSV.")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["MAJORgas_std"])
    if len(label_encoder.classes_) != 4:
        raise ValueError(f"Expected all four product classes, found: {list(label_encoder.classes_)}")
    return df[features].copy(), y, label_encoder


def build_pipeline(X: pd.DataFrame, seed: int) -> ImbPipeline:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    unsupported = set(X.columns) - set(numeric) - set(categorical)
    if unsupported:
        raise ValueError(f"Unsupported input column dtypes: {sorted(unsupported)}")
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("std", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=seed)),
        ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                               random_state=seed, n_jobs=-1)),
    ])


def evaluate(X: pd.DataFrame, y: np.ndarray, runs: int) -> pd.DataFrame:
    results = []
    for run in range(runs):
        seed = 42 + run
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        model = build_pipeline(X_train, seed)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        # All four classes must be present in the test split for multiclass ROC-AUC.
        if len(np.unique(y_test)) != 4:
            raise ValueError("Each test split must contain all four classes; increase class counts.")
        one_hot = np.eye(4, dtype=int)[y_test]
        results.append({
            "Run": run + 1, "Seed": seed, "N_Test": len(y_test),
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro_F1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "Weighted_F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, predictions),
            "MCC": matthews_corrcoef(y_test, predictions),
            "Macro_Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Macro_Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "Brier_Multiclass": np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)),
            "ROC_AUC_Macro_OVR": roc_auc_score(y_test, probabilities, labels=np.arange(4),
                                                 multi_class="ovr", average="macro"),
        })
        print(f"Run {run+1}/{runs} | Seed {seed} | Macro-F1: {results[-1]['Macro_F1']:.4f}")
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate C3 product selectivity classification.")
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    X, y, encoder = load_data(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate(X, y, args.runs)
    results.to_csv(args.output_dir / "C3_Metrics_All_Runs.csv", index=False)
    summary = results.drop(columns=["Run", "Seed", "N_Test"]).agg(["mean", "std"]).T
    summary.to_csv(args.output_dir / "C3_Metrics_Summary.csv", index_label="Metric")
    print("\nEvaluation (mean ± standard deviation):")
    print(summary.to_string(float_format=lambda n: f"{n:.4f}"))

    # Fit a separate final model on the complete dataset for downstream inference.
    # This full-data fit is not used to compute holdout metrics above.
    final_model = build_pipeline(X, seed=42)
    final_model.fit(X, y)
    joblib.dump(final_model, args.output_dir / "C3_model.pkl")
    pd.DataFrame({"encoded_label": np.arange(len(encoder.classes_)),
                  "product": encoder.classes_}).to_csv(args.output_dir / "C3_Label_Mapping.csv", index=False)
    print(f"\nSaved model and evaluation files to {args.output_dir}")
    print(f"Product label mapping: {dict(enumerate(encoder.classes_))}")


if __name__ == "__main__":
    main()
