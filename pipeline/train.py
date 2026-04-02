"""
pipeline/train.py
────────────────────────────────────────────────────────────
Entrena el clasificador de regímenes de mercado.

Modelo: Random Forest con class_weight='balanced'
  - Robusto frente a outliers (picos de Binance)
  - Maneja desbalance de clases automáticamente
  - Feature importance incluida en metadata

Validación: TimeSeriesSplit (no shuffle)
  - Crítico: los datos financieros no son i.i.d.
  - No se puede mezclar futuro con pasado en validación

Serialización: joblib (.pkl) + metadata JSON
  - El JSON incluye versión de sklearn, fecha, accuracy y features
  - El bot en vivo lee el JSON para validar compatibilidad
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import sklearn

from labels.labeler import get_feature_columns

logger = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────
MODELS_DIR          = Path(os.getenv("MODELS_DIR", "models"))
MODEL_PATH          = MODELS_DIR / "regime_model.pkl"
META_PATH           = MODELS_DIR / "regime_model_meta.json"
MIN_ACCURACY        = float(os.getenv("MIN_ACCURACY", "0.60"))  # umbral para guardar
N_ESTIMATORS        = int(os.getenv("N_ESTIMATORS", "300"))
MAX_DEPTH           = int(os.getenv("MAX_DEPTH", "8"))
N_CV_SPLITS         = int(os.getenv("N_CV_SPLITS", "5"))
TRAIN_RATIO         = float(os.getenv("TRAIN_RATIO", "0.80"))   # 80% train, 20% test final


def train(df: pd.DataFrame) -> dict:
    """
    Entrena el modelo y lo guarda si supera el umbral de accuracy.

    Args:
        df: DataFrame con features y columna `target_regime` (salida del labeler)

    Returns:
        dict con métricas: accuracy, report, confusion_matrix, feature_importance
    
    Raises:
        ValueError si el accuracy no supera MIN_ACCURACY (no guarda el modelo)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_columns()

    # Verificar que todas las features existen
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes en el DataFrame: {missing}")

    X = df[feature_cols].values
    y = df["target_regime"].values

    logger.info("Dataset: %d muestras | Features: %d | Clases: %s",
                len(X), len(feature_cols), np.unique(y).tolist())

    # ── Validación cruzada temporal (TimeSeriesSplit) ──────
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores: list[float] = []

    logger.info("Validación cruzada temporal (%d splits)...", N_CV_SPLITS)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        score = accuracy_score(y_val, clf.predict(X_val))
        cv_scores.append(score)
        logger.info("  Fold %d: accuracy = %.3f", fold, score)

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    logger.info("CV accuracy: %.3f ± %.3f", cv_mean, cv_std)

    # ── Entrenamiento final en 80% de los datos ────────────
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("Entrenando modelo final: %d train / %d test", len(X_train), len(X_test))
    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    # ── Métricas en test set ───────────────────────────────
    y_pred    = final_model.predict(X_test)
    test_acc  = float(accuracy_score(y_test, y_pred))
    report    = classification_report(y_test, y_pred,
                                      target_names=["Lateral(0)", "Tendencia(1)", "AltaVol(2)"],
                                      output_dict=True)
    conf_mat  = confusion_matrix(y_test, y_pred).tolist()

    logger.info("Test accuracy: %.3f", test_acc)
    logger.info("\n%s", classification_report(y_test, y_pred,
                target_names=["Lateral(0)", "Tendencia(1)", "AltaVol(2)"]))

    # ── Feature Importance ────────────────────────────────
    importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    logger.info("Feature Importance:")
    for feat, imp in importance_sorted.items():
        logger.info("  %-40s %.4f", feat, imp)

    # ── Guardar modelo solo si supera el umbral ────────────
    if test_acc < MIN_ACCURACY:
        msg = (f"Accuracy {test_acc:.3f} < umbral {MIN_ACCURACY:.3f}. "
               f"Modelo NO guardado. El bot sigue con el .pkl anterior.")
        logger.warning(msg)
        raise ValueError(msg)

    # Serializar modelo
    joblib.dump(final_model, MODEL_PATH)
    logger.info("OK Modelo guardado: %s", MODEL_PATH)

    # Metadata
    meta = {
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "sklearn_version":  sklearn.__version__,
        "n_samples":        int(len(X)),
        "n_features":       int(len(feature_cols)),
        "feature_columns":  feature_cols,
        "cv_accuracy_mean": round(cv_mean, 4),
        "cv_accuracy_std":  round(cv_std, 4),
        "test_accuracy":    round(test_acc, 4),
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "feature_importance": {k: round(v, 4) for k, v in importance_sorted.items()},
        "hyperparams": {
            "n_estimators": N_ESTIMATORS,
            "max_depth":    MAX_DEPTH,
            "class_weight": "balanced",
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("OK Metadata guardada: %s", META_PATH)

    return meta
