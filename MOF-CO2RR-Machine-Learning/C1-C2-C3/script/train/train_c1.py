
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

METAL_FEATURES = [
    "M_Electron_Affinity", "M_Hardness", "M_Ionization", "M_MeltingPoint",
    "M_Polarizability", "M_Zeff", "M_d_electrons",
]
LINKER_FEATURES = [
    "L_MW", "L_TPSA", "L_NumHDonors", "L_NumHAcceptors",
    "Anchor_carboxylate_or_acid", "Anchor_imidazole_like_N",
    "Anchor_phosphonate", "Anchor_pyridyl_like_ArN", "Anchor_sulfonate",
    "FG_ArN", "FG_COOH", "FG_C_O", "FG_NH2", "FG_OH", "FG_amide",
    "FG_ether", "FG_halogen",
]
GEOMETRY_FEATURES = [
    "ML_band_gap", "LCD", "PLD", "LFPD", "Density", "GSA", "VF", "PV",
]
CONDITION_FEATURES = [
    "Additive_Role", "Rxn_Time", "Rxn_Temp", "Catalyst_Mass",
    "P_CO2_bar", "Photosensitizer_code", "Phase_code", "Light_Type_code",
    "mix_bp_C", "mix_viscosity_cP", "mix_KT_pi_star", "mix_KT_alpha",
]
POSITIVE_PRODUCTS = {"CO", "CH4", "CH3OH", "HCOOH", "CH3COOH", "(CH3)2CO"}


def load_training_data(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, encoding="latin1").dropna(subset=["MAJORgas_std"])
    if df.empty:
        raise ValueError("No rows remain after removing missing target values.")
  
    y = df["MAJORgas_std"].isin(POSITIVE_PRODUCTS).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("C1 training requires both active and inactive examples.")
    fingerprint_cols = [c for c in df.columns if "MACCSFP" in c]
    configured = METAL_FEATURES + LINKER_FEATURES + GEOMETRY_FEATURES + CONDITION_FEATURES + fingerprint_cols
    features = list(dict.fromkeys(c for c in configured if c in df.columns))
    if not features:
        raise ValueError("None of the predefined training features were found in the CSV.")
    print(f"Training records: {len(df)}; predefined input features: {len(features)}")
    print(f"Active: {int(y.sum())}; inactive: {len(y) - int(y.sum())}")
    return df.loc[:, features].copy(), y


def build_pipeline(X: pd.DataFrame, use_smote: bool, seed: int) -> ImbPipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    preprocessing = ColumnTransformer(transformers=transformers)

    return ImbPipeline([
        ("preprocessor", preprocessing),
        ("smote", SMOTE(random_state=seed) if use_smote else "passthrough"),
        ("classifier", XGBClassifier(random_state=seed, eval_metric="logloss")),
    ])


def evaluate(model: ImbPipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    pred = model.predict(X_test)
    classes = list(model.classes_)
    prob = model.predict_proba(X_test)[:, classes.index(1)]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "PR_AUC_AP": average_precision_score(y_test, prob),
        "Macro_F1": f1_score(y_test, pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
        "Brier": brier_score_loss(y_test, prob),
        "Precision_Active": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "Recall_Active": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "Precision_Inactive": precision_score(y_test, pred, pos_label=0, zero_division=0),
        "Recall_Inactive": recall_score(y_test, pred, pos_label=0, zero_division=0),
    }


def train(input_csv: Path, output_dir: Path, seed: int, use_smote: bool) -> None:
    X, y = load_training_data(input_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model = build_pipeline(X_train, use_smote, seed)
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(
        model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    metrics = evaluate(search.best_estimator_, X_test, y_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, output_dir / "C1_model.pkl")
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame([search.best_params_]).to_csv(output_dir / "best_parameters.csv", index=False)
    pd.DataFrame({"Feature": X.columns}).to_csv(output_dir / "input_features.csv", index=False)
    print("Best parameters:", search.best_params_)
    print("Holdout test metrics:\n", pd.Series(metrics).round(4).to_string())
    print(f"Saved model and CSV outputs in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the C1 CO2RR classifier without feature selection or plots.")
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=["smote", "none"], default="smote")
    args = parser.parse_args()
    train(args.input, args.output_dir, args.seed, args.sampling == "smote")


if __name__ == "__main__":
    main()
