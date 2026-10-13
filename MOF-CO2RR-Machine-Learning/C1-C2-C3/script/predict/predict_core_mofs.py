

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Inference settings from the original screening workflow.
C1_CONDITIONS = {
    "Gas": {
        "Rxn_Temp": 25.0, "P_CO2_bar": 1.01325, "Rxn_Time": 10.0,
        "Catalyst_Mass": 5.0, "Reactor_Volume": 50.0,
        "Additive_Role": 0, "Photosensitizer_code": 0,
        "Phase_code": 1, "Light_Type_code": 8.0,
        "mix_density_g_ml": 0.001, "mix_bp_C": -78.0,
        "mix_viscosity_cP": 0.01, "mix_AN": 0.0, "mix_DN": 0.0,
        "mix_KT_pi_star": 0.0, "mix_KT_alpha": 0.0,
        "mix_KT_beta": 0.0, "mix_dielectric": 1.0,
    },
    "Liquid": {
        "Rxn_Temp": 25.0, "P_CO2_bar": 1.01325, "Rxn_Time": 10.0,
        "Catalyst_Mass": 5.0, "Reactor_Volume": 50.0,
        "Additive_Role": 1, "Photosensitizer_code": 0,
        "Phase_code": 0, "Light_Type_code": 8.0,
        "mix_density_g_ml": 0.998, "mix_bp_C": 100.0,
        "mix_viscosity_cP": 1.0, "mix_AN": 54.8, "mix_DN": 18.0,
        "mix_KT_pi_star": 1.09, "mix_KT_alpha": 1.17,
        "mix_KT_beta": 0.47, "mix_dielectric": 80.1,
    },
}

# C2/C3 settings intentionally preserve the original model-specific values.
C2_CONDITIONS = {
    "Gas": {key: val for key, val in C1_CONDITIONS["Gas"].items()
            if key != "Reactor_Volume"},
    "Liquid": {
        **{key: val for key, val in C1_CONDITIONS["Liquid"].items()
           if key != "Reactor_Volume"},
        "Photosensitizer_code": 1,
    },
}

PRODUCT_LABELS = {0: "CH3OH", 1: "CH4", 2: "CO", 3: "HCOOH"}


def select_column(dataframe: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Find the first available descriptor column by its recognized name."""
    for column in candidates:
        if column in dataframe.columns:
            return column
    raise KeyError(f"Missing {label} column; expected one of {candidates}")


def physical_prefilter(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the original PLD, band-gap, and EA/IP screening criteria."""
    pld = select_column(dataframe, ("PLD", "pld", "Pore Limiting Diameter"), "PLD")
    gap = select_column(dataframe, (
        "ML_band_gap", "prediction", "pred_gap", "HSE_band_gap", "band_gap"
    ), "band gap")
    ea = select_column(dataframe, ("pred_EA", "EA", "electron_affinity", "ML_EA"), "EA")
    ip = select_column(dataframe, ("pred_IP", "IP", "ionization_potential", "ML_IP"), "IP")

    data = dataframe.copy()
    for column in {pld, gap, ea, ip}:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    initial = len(data)
    data = data.loc[data[pld] >= 3.3].copy()
    n_pld = len(data)
    data = data.loc[data[gap].between(1.6, 3.1, inclusive="both")].copy()
    n_gap = len(data)

    if not data.empty:
        ea_median, ip_median = data[ea].median(), data[ip].median()
        if pd.isna(ea_median) or pd.isna(ip_median):
            raise ValueError("Cannot identify EA/IP convention: insufficient numeric data.")
        if ea_median > 0 and ip_median > 0:
            # Positive EA/IP convention: CBM = -EA, VBM = -IP.
            data = data.loc[(data[ea] <= 4.44) & (data[ip] >= 5.67)].copy()
            convention = "positive EA/IP"
        else:
            # Negative vacuum-level convention; retain original inference rule.
            data = data.loc[(data[ea] >= -4.44) & (data[ip] <= -5.67)].copy()
            convention = "negative vacuum levels"
        print(f"EA/IP convention (median-based): {convention}")

    counts = {"Initial structures": initial, "PLD >= 3.3": n_pld,
              "1.6 <= band gap <= 3.1": n_gap, "EA/IP qualified": len(data)}
    for label, number in counts.items():
        print(f"{label}: {number}")
    return data, counts


def prepare_features(model, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Align the input with saved training-time feature names."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError(
            "The loaded model has no feature_names_in_. Provide a fitted pipeline "
            "with saved feature names to ensure reproducible feature alignment."
        )
    missing = [name for name in feature_names if name not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} model features: {missing[:20]}")
    return dataframe.loc[:, list(feature_names)].copy()


def apply_conditions(dataframe: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    result = dataframe.copy()
    for column, value in conditions.items():
        result[column] = value
    return result


def predict_binary(model, dataframe: pd.DataFrame, conditions: dict):
    """Return positive-class labels and probabilities for one phase."""
    features = prepare_features(model, apply_conditions(dataframe, conditions))
    labels = np.asarray(model.predict(features)) == 1
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Expected positive class 1; got {classes}")
    probabilities = model.predict_proba(features)[:, classes.index(1)]
    return labels, probabilities


def run_predictions(input_csv: Path, c1_path: Path, c2_path: Path,
                    c3_path: Path, output_csv: Path, physical_output: Path | None = None) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    if data.empty:
        raise ValueError("The input CSV is empty.")

    data, _ = physical_prefilter(data)
    if physical_output is not None:
        physical_output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(physical_output, index=False)
        print(f"Saved physically qualified MOFs to: {physical_output}")
    if data.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_csv, index=False)
        print("No structures passed physical prefiltering.")
        return data

    c1_model = joblib.load(c1_path)
    c2_model = joblib.load(c2_path)
    c3_model = joblib.load(c3_path)

    # C1: retain structures predicted active in at least one phase.
    gas_active, gas_activity_prob = predict_binary(c1_model, data, C1_CONDITIONS["Gas"])
    liq_active, liq_activity_prob = predict_binary(c1_model, data, C1_CONDITIONS["Liquid"])
    data["C1_active_gas"] = gas_active.astype(int)
    data["C1_active_liq"] = liq_active.astype(int)
    data["C1_Prob_Gas"] = gas_activity_prob
    data["C1_Prob_Liquid"] = liq_activity_prob
    active = data.loc[gas_active | liq_active].copy()
    print(f"C1 active: {len(active)} / {len(data)}")
    if active.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        active.to_csv(output_csv, index=False)
        return active

    # C2: retain structures predicted high-performing in either phase.
    gas_hp, gas_perf_prob = predict_binary(c2_model, active, C2_CONDITIONS["Gas"])
    liq_hp, liq_perf_prob = predict_binary(c2_model, active, C2_CONDITIONS["Liquid"])
    active["C2_highperf_gas"] = gas_hp.astype(int)
    active["C2_highperf_liq"] = liq_hp.astype(int)
    active["High_Perf_Prob_Gas"] = gas_perf_prob
    active["High_Perf_Prob_Liquid"] = liq_perf_prob
    elite = active.loc[gas_hp | liq_hp].copy()
    print(f"C2 high-performing: {len(elite)} / {len(active)}")
    if elite.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        elite.to_csv(output_csv, index=False)
        return elite

    elite["High_Perf_Prob"] = np.maximum(
        elite["High_Perf_Prob_Gas"], elite["High_Perf_Prob_Liquid"]
    )
    # As in the original workflow, ties favor the liquid setting.
    elite["Best_Phase_Target"] = np.where(
        elite["High_Perf_Prob_Gas"] > elite["High_Perf_Prob_Liquid"],
        "Gas", "Liquid"
    )

    # C3: predict products under the best C2 phase-specific conditions.
    c3_input = elite.copy()
    for phase, conditions in C2_CONDITIONS.items():
        mask = elite["Best_Phase_Target"] == phase
        for column, value in conditions.items():
            c3_input.loc[mask, column] = value

    features = prepare_features(c3_model, c3_input)
    labels = c3_model.predict(features)
    proba = c3_model.predict_proba(features)
    elite["Predicted_Product"] = [
        PRODUCT_LABELS.get(int(label), f"Unknown_{label}") for label in labels
    ]
    elite["Product_Confidence"] = proba.max(axis=1)

    elite = elite.sort_values("High_Perf_Prob", ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    elite.to_csv(output_csv, index=False)
    print(f"Saved {len(elite)} predicted candidates to: {output_csv}")
    print(elite["Predicted_Product"].value_counts().to_string())
    return elite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physical prefiltering and C1/C2/C3 predictions for CoRE MOFs.")
    parser.add_argument("--input", type=Path, default=Path("data/CoRE_MOF_all_descriptor.csv"))
    parser.add_argument("--c1-model", type=Path, default=Path("models/C1_model.pkl"))
    parser.add_argument("--c2-model", type=Path, default=Path("models/C2_model.pkl"))
    parser.add_argument("--c3-model", type=Path, default=Path("models/C3_model.pkl"))
    parser.add_argument("--output", type=Path, default=Path("results/CoRE_MOF_Predictions.csv"))
    parser.add_argument("--physical-output", type=Path,
                        default=Path("results/CoRE_MOF_Physically_Qualified.csv"))
    args = parser.parse_args()
    run_predictions(args.input, args.c1_model, args.c2_model, args.c3_model, args.output, args.physical_output)


if __name__ == "__main__":
    main()
