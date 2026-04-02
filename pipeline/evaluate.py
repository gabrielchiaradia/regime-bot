"""
pipeline/evaluate.py
────────────────────────────────────────────────────────────
Evaluación del modelo entrenado.
Genera reporte visual en consola y opcionalmente exporta PNG/CSV.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
META_PATH  = MODELS_DIR / "regime_model_meta.json"

REGIME_NAMES = {0: "Lateral", 1: "Tendencia", 2: "Alta Volatilidad"}
REGIME_EMOJIS = {0: "📊", 1: "📈", 2: "⚡"}


def print_evaluation_report(meta: dict | None = None) -> None:
    """
    Imprime el reporte de evaluación en consola.
    Si `meta` es None, lo lee del archivo JSON.
    """
    if meta is None:
        if not META_PATH.exists():
            logger.error("No existe metadata en %s. Entrená el modelo primero.", META_PATH)
            return
        meta = json.loads(META_PATH.read_text())

    _separator = "─" * 60

    print(f"\n{'═' * 60}")
    print(f"  EVALUACIÓN DEL CLASIFICADOR DE REGÍMENES")
    print(f"{'═' * 60}")
    print(f"  Entrenado:   {meta.get('trained_at', 'N/A')[:19]}")
    print(f"  Sklearn:     {meta.get('sklearn_version', 'N/A')}")
    print(f"  Muestras:    {meta.get('n_samples', 'N/A'):,}")
    print(f"  Features:    {meta.get('n_features', 'N/A')}")
    print(_separator)

    # Accuracy
    cv_mean = meta.get("cv_accuracy_mean", 0)
    cv_std  = meta.get("cv_accuracy_std", 0)
    test_acc = meta.get("test_accuracy", 0)
    print(f"\n  📊 ACCURACY")
    print(f"  CV (TimeSeriesSplit):  {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"  Test final:            {test_acc:.3f}")

    # Por clase
    report = meta.get("classification_report", {})
    print(f"\n  📋 POR RÉGIMEN")
    print(f"  {'Régimen':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Soporte':>10}")
    print(f"  {_separator}")
    class_map = {"Lateral(0)": 0, "Tendencia(1)": 1, "AltaVol(2)": 2}
    for class_name, regime_id in class_map.items():
        if class_name in report:
            r = report[class_name]
            emoji = REGIME_EMOJIS.get(regime_id, "")
            name  = f"{emoji} {REGIME_NAMES.get(regime_id, class_name)}"
            print(f"  {name:<20} {r['precision']:>10.3f} {r['recall']:>10.3f} "
                  f"{r['f1-score']:>10.3f} {int(r['support']):>10,}")

    # Confusion matrix
    conf_mat = meta.get("confusion_matrix", [])
    if conf_mat:
        print(f"\n  🔢 MATRIZ DE CONFUSIÓN")
        print(f"  {'':15} {'Pred 0':>10} {'Pred 1':>10} {'Pred 2':>10}")
        for i, row in enumerate(conf_mat):
            name = f"Real {i} ({REGIME_NAMES.get(i, '')[:8]})"
            print(f"  {name:<15}", end="")
            for val in row:
                print(f"  {val:>8,}", end="")
            print()

    # Feature Importance
    fi = meta.get("feature_importance", {})
    if fi:
        print(f"\n  🏆 FEATURE IMPORTANCE (Top 8)")
        max_imp = max(fi.values()) if fi else 1
        for i, (feat, imp) in enumerate(list(fi.items())[:8], 1):
            bar_len = int((imp / max_imp) * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            short_name = feat.replace("feature_", "")
            print(f"  {i}. {short_name:<30} {bar}  {imp:.4f}")

    print(f"\n{'═' * 60}\n")


def export_predictions_csv(
    df: pd.DataFrame,
    output_path: str = "models/predictions_sample.csv",
) -> None:
    """
    Exporta una muestra del dataset con las predicciones para inspección manual.
    Útil para verificar que el labeler está etiquetando correctamente.
    """
    cols = (
        ["close", "feature_adx", "feature_volatility_ratio",
         "feature_vwap_distance", "feature_volume_zscore",
         "atr_current", "atr_prev_day_mean", "_atr_ratio",
         "target_regime"]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    available = [c for c in cols if c in df.columns]
    sample = df[available].tail(500)
    sample.to_csv(output_path, encoding="utf-8-sig")  # utf-8-sig para Excel en Windows
    logger.info("Muestra exportada: %s (%d filas)", output_path, len(sample))
