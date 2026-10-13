"""Train and evaluate the C2 high-performance classifier for photocatalytic MOFs.

Uses a fixed descriptor set (no automated feature selection), and produces no plots.
The target is defined by the 60th percentile of the filtered input data, as in
its source notebook. The evaluation uses ten stratified 80/20 splits (seeds 0–9).
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
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_FEATURES = [
    "M_EN_Pauling", "M_Electron_Affinity", "M_Hardness", "M_Ionization",
    "M_MeltingPoint", "M_Polarizability", "L_MW", "L_LOGP", "L_TPSA",
    "L_NumHDonors", "L_NumHAcceptors", "L_NumAromaticRings",
    "L_NumRotatableBonds", "FG_NH2", "ML_band_gap", "LCD", "PLD",
    "Density", "GSA", "VSA", "PV", "VF", "Additive_Role", "Rxn_Time",
    "Rxn_Temp", "Catalyst_Mass", "P_CO2_bar", "Photosensitizer_code",
    "Phase_code", "Light_Type_code", "mix_density_g_ml", "mix_bp_C",
    "mix_dielectric", "mix_KT_beta",
]
PRODUCTS = {"CO", "CH4", "CH3OH", "HCOOH"}


def load_data(path: Path):
    try:
        df = pd.read_csv(path, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")
    df = df.loc[df["MAJORgas_std"].isin(PRODUCTS)].copy()
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df = df.dropna(subset=["total"])
    if df.empty:
        raise ValueError("No records remain after product/total filtering.")
    threshold = df["total"].quantile(0.6)
    y = (df["total"] >= threshold).astype(int)
    features = [c for c in BASE_FEATURES if c in df.columns]
    features += [c for c in df.columns if "MACCSFP" in c and c not in features]
    if not features:
        raise ValueError("No expected descriptor columns found in the input CSV.")
    if y.nunique() < 2:
        raise ValueError("Both High and Low classes are required.")
    return df[features].copy(), y, threshold, features


def build_pipeline(X: pd.DataFrame, seed: int, sampling: str):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    unsupported = set(X.columns) - set(numeric) - set(categorical)
    if unsupported:
        raise ValueError(f"Unsupported input dtypes: {sorted(unsupported)}")
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("std", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    prep = ColumnTransformer(transformers)
    sampler = SMOTE(random_state=seed) if sampling == "smote" else "passthrough"
    return ImbPipeline([
        ("prep", prep),
        ("smote", sampler),
        ("classifier", XGBClassifier(
            n_estimators=200, eval_metric="logloss", random_state=seed,
        )),
    ])


def evaluate(y_true, y_pred, p):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, p),
        "PR_AUC_AP_High": average_precision_score(y_true, p),
        "PR_AUC_AP_Low": average_precision_score(1 - y_true, 1 - p),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Precision_High": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall_High": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision_Low": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall_Low": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Brier_Raw": brier_score_loss(y_true, p),
    }


def train(input_csv: Path, output_dir: Path, sampling: str, save_model: bool):
    X, y, threshold, features = load_data(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Feature": features}).to_csv(output_dir / "Input_Features.csv", index=False)
    print(f"Records: {len(X)} | Input features: {len(features)} | High threshold: {threshold:.6g}")
    print(f"Sampling: {sampling} | Evaluation: 10 stratified splits, 80/20")

    runs = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        model = build_pipeline(X, seed, sampling)
        model.fit(X_train, y_train)
        labels = model.predict(X_test)
        classes = list(model.classes_)
        probs = model.predict_proba(X_test)[:, classes.index(1)]
        runs.append({"Seed": seed, **evaluate(y_test, labels, probs)})
        print(f"Seed {seed}: F1={runs[-1]['F1']:.4f}, ROC-AUC={runs[-1]['AUC']:.4f}")

    results = pd.DataFrame(runs)
    results.to_csv(output_dir / "Performance_Metrics_10Runs.csv", index=False)
    summary = results.drop(columns="Seed").agg(["mean", "std"]).T
    summary.to_csv(output_dir / "Performance_Metrics_Final.csv")
    print("\nEvaluation mean ± SD:\n", summary.to_string(float_format=lambda v: f"{v:.4f}"))

    if save_model:
        # Final deployment fit uses the full dataset; it is not used for test metrics.
        final_model = build_pipeline(X, seed=42, sampling=sampling)
        final_model.fit(X, y)
        joblib.dump(final_model, output_dir / "C2_model.pkl")
        print(f"Saved fitted model: {output_dir / 'C2_model.pkl'}")
    print(f"Saved metrics to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="C2 MOF high-performance model training (no plots/feature selection)")
    parser.add_argument("--input", type=Path, default=Path("data/Data_reg_fixed_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sampling", choices=["smote", "none"], default="smote")
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()
    train(args.input, args.output_dir, args.sampling, not args.no_save_model)

if __name__ == "__main__":
    main()
