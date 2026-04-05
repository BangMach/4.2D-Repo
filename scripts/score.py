import json
import logging
import os
import pickle

import joblib
import numpy as np
import pandas as pd

model = None
scaler = None
tool_wear_max = 253.0

RAW_REQUIRED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

DEFAULT_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_delta",
    "power_proxy",
    "tool_wear_pct",
    "torque_speed_ratio",
]

feature_columns = DEFAULT_FEATURE_COLUMNS


def _find_model_path(model_dir):
    preferred = os.path.join(model_dir, "best-performance-model.pkl")
    if os.path.isfile(preferred):
        return preferred

    candidates = [
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(".pkl") or name.endswith(".joblib")
    ]
    if not candidates:
        raise FileNotFoundError(f"No .pkl or .joblib model file found in {model_dir}")
    return candidates[0]


def _load_object(path):
    # Support both joblib and pickle artifacts.
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _load_optional_scaler(model_dir):
    scaler_candidates = [
        os.path.join(model_dir, "scaler.pkl"),
        os.path.join(model_dir, "scaler.joblib"),
    ]
    for path in scaler_candidates:
        if os.path.isfile(path):
            return _load_object(path)
    return None


def engineer_features(df):
    global tool_wear_max
    df = df.copy()

    df["temp_delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_proxy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] / 1000.0

    # Use a fixed training reference so single-row requests don't collapse to 100%.
    max_wear = float(tool_wear_max) if tool_wear_max else 253.0
    if max_wear <= 0:
        df["tool_wear_pct"] = 0.0
    else:
        df["tool_wear_pct"] = df["Tool wear [min]"] / max_wear * 100.0

    df["torque_speed_ratio"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1.0)
    return df


def init():
    global model, scaler, feature_columns, tool_wear_max

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = _find_model_path(model_dir)
    loaded = _load_object(model_path)

    # Accept either plain classifier or a dict-like bundle.
    if isinstance(loaded, dict):
        model = loaded.get("model")
        scaler = loaded.get("scaler")
        feature_columns = loaded.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
        training_stats = loaded.get("training_stats", {})
        tool_wear_max = float(training_stats.get("tool_wear_max", 253.0))
    else:
        model = loaded
        scaler = _load_optional_scaler(model_dir)
        feature_columns = DEFAULT_FEATURE_COLUMNS

    if model is None:
        raise ValueError("Loaded artifact does not contain a valid model")

    if scaler is None:
        raise ValueError(
            "Scaler is missing. Deploy a model bundle with model, scaler, and feature_columns "
            "to match training preprocessing."
        )

    logging.info(f"Loaded model from: {model_path}")


def _to_dataframe(payload):
    data = payload.get("data", payload) if isinstance(payload, dict) else payload

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return pd.DataFrame(data)

    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], list)):
        return pd.DataFrame(data, columns=RAW_REQUIRED_COLUMNS)

    raise ValueError("Invalid input format. Use list of dicts or list of lists.")


def run(raw_data):
    try:
        payload = json.loads(raw_data)
        df = _to_dataframe(payload)

        missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return {"error": f"Missing required columns: {missing}"}

        df = engineer_features(df)
        X = df[feature_columns]

        if scaler is not None:
            X_input = scaler.transform(X)
        else:
            X_input = np.asarray(X)

        preds = model.predict(X_input).tolist()
        result = {"predictions": preds}

        if hasattr(model, "predict_proba"):
            result["probabilities"] = model.predict_proba(X_input).tolist()

        return result

    except Exception as e:
        logging.exception("Scoring failed")
        return {"error": str(e)}
