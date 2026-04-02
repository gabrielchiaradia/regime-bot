"""
inference/classifier.py
────────────────────────────────────────────────────────────
Inferencia en tiempo real del régimen de mercado.

HISTÉRESIS (anti-péndulo):
  El régimen solo cambia si el modelo lo predice N velas consecutivas.
  Esto evita que el orchestrator abra/cierre posiciones erráticamente
  cuando el mercado está en zona de transición.

USO:
    classifier = RegimeClassifier()
    classifier.load_model()

    # En cada vela nueva:
    regime = classifier.predict(df_3m, df_15m)
    # regime: 0 (Lateral), 1 (Tendencia), 2 (Alta Volatilidad)

COMPATIBILIDAD:
    Al cargar el modelo, verifica que la versión de sklearn coincida
    con la del entrenamiento. Si no, lanza warning pero sigue funcionando
    (sklearn suele ser compatible hacia atrás en minor versions).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from features.builder import build_features
from labels.labeler import get_feature_columns

logger = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────
MODELS_DIR           = Path(os.getenv("MODELS_DIR", "models"))
MODEL_PATH           = MODELS_DIR / "regime_model.pkl"
META_PATH            = MODELS_DIR / "regime_model_meta.json"
REGIME_CONFIRMATION  = int(os.getenv("REGIME_CONFIRMATION", "3"))   # velas para confirmar cambio
WARMUP_CANDLES_LIVE  = int(os.getenv("WARMUP_CANDLES_LIVE", "250")) # warmup para inferencia

REGIME_NAMES = {
    0: "LATERAL",
    1: "TENDENCIA",
    2: "ALTA_VOLATILIDAD",
}


class RegimeClassifier:
    """
    Clasificador de régimen de mercado con histéresis.
    
    Estado interno:
        _current_regime     : régimen confirmado actualmente activo
        _candidate_regime   : régimen que el modelo predice pero no ha confirmado
        _candidate_count    : cuántas velas consecutivas el modelo predice `_candidate`
        _model              : RandomForestClassifier cargado
        _meta               : metadata del entrenamiento
    """

    def __init__(self) -> None:
        self._model              = None
        self._meta:  dict        = {}
        self._feature_cols: list = get_feature_columns()

        self._current_regime:   int = 0   # arrancamos asumiendo lateral (conservador)
        self._candidate_regime: int = 0
        self._candidate_count:  int = 0

        self._prediction_history: list[int] = []  # últimas N predicciones crudas

    # ── Carga del modelo ───────────────────────────────────

    def load_model(self) -> None:
        """
        Carga el modelo y la metadata desde disco.
        Verifica compatibilidad de sklearn y features.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {MODEL_PATH}. "
                "Ejecutá run_pipeline.py primero."
            )

        self._model = joblib.load(MODEL_PATH)
        logger.info("✅ Modelo cargado: %s", MODEL_PATH)

        if META_PATH.exists():
            self._meta = json.loads(META_PATH.read_text())
            self._validate_compatibility()
        else:
            logger.warning("No se encontró metadata (%s). Continuando sin validación.", META_PATH)

    def _validate_compatibility(self) -> None:
        """Verifica sklearn version y features."""
        trained_sklearn = self._meta.get("sklearn_version", "")
        current_sklearn = sklearn.__version__

        if trained_sklearn != current_sklearn:
            logger.warning(
                "⚠️  sklearn version mismatch: entrenado con %s, corriendo con %s. "
                "Si hay problemas de predicción, re-entrená el modelo.",
                trained_sklearn, current_sklearn
            )

        trained_features = self._meta.get("feature_columns", [])
        if trained_features and trained_features != self._feature_cols:
            raise ValueError(
                f"Features del modelo no coinciden con las del classifier.\n"
                f"  Modelo:     {trained_features}\n"
                f"  Classifier: {self._feature_cols}\n"
                "Re-entrená el modelo con la versión actual del código."
            )

        test_acc = self._meta.get("test_accuracy", 0)
        trained_at = self._meta.get("trained_at", "N/A")[:19]
        logger.info(
            "Modelo: accuracy=%.3f | entrenado=%s | sklearn=%s",
            test_acc, trained_at, trained_sklearn
        )

    # ── Predicción ─────────────────────────────────────────

    def predict(
        self,
        df_3m:  pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> int:
        """
        Calcula el régimen actual con histéresis.
        
        Args:
            df_3m:  últimas ~500 velas de 3m (ya descargadas)
            df_15m: últimas ~500 velas de 15m
        
        Returns:
            int: régimen confirmado (0, 1 o 2)
        
        El régimen solo cambia si el modelo predice el mismo valor
        durante REGIME_CONFIRMATION velas consecutivas.
        """
        if self._model is None:
            raise RuntimeError("Modelo no cargado. Llamá load_model() primero.")

        # Construir features de la última vela
        raw_prediction = self._compute_raw_prediction(df_3m, df_15m)
        self._prediction_history.append(raw_prediction)
        if len(self._prediction_history) > 10:
            self._prediction_history.pop(0)

        # ── Histéresis ────────────────────────────────────
        previous_regime = self._current_regime

        if raw_prediction == self._current_regime:
            # Predicción coincide con régimen actual → resetear candidato
            self._candidate_regime = self._current_regime
            self._candidate_count  = 0

        elif raw_prediction == self._candidate_regime:
            # Predicción coincide con el candidato → incrementar contador
            self._candidate_count += 1

            if self._candidate_count >= REGIME_CONFIRMATION:
                # ¡Cambio confirmado!
                self._current_regime   = self._candidate_regime
                self._candidate_count  = 0
                logger.info(
                    "🔄 CAMBIO DE RÉGIMEN: %s → %s (confirmado en %d velas)",
                    REGIME_NAMES.get(previous_regime, previous_regime),
                    REGIME_NAMES.get(self._current_regime, self._current_regime),
                    REGIME_CONFIRMATION,
                )
        else:
            # Nuevo candidato diferente al anterior → reiniciar
            self._candidate_regime = raw_prediction
            self._candidate_count  = 1

        return self._current_regime

    def _compute_raw_prediction(
        self,
        df_3m:  pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> int:
        """
        Calcula los features y obtiene la predicción cruda del modelo
        (sin histéresis) para la última vela disponible.
        """
        df_features = build_features(df_3m, df_15m, warmup_candles=WARMUP_CANDLES_LIVE)

        # Tomar solo la última fila (última vela completada)
        last_row = df_features[self._feature_cols].iloc[[-1]]

        # Verificar NaN
        if last_row.isnull().any().any():
            nan_cols = last_row.columns[last_row.isnull().any()].tolist()
            logger.warning("NaN en features: %s → manteniendo régimen actual", nan_cols)
            return self._current_regime

        prediction = int(self._model.predict(last_row)[0])
        proba      = self._model.predict_proba(last_row)[0]

        logger.debug(
            "Raw prediction: %s | proba=[L:%.2f T:%.2f V:%.2f] | candidato=%s×%d",
            REGIME_NAMES.get(prediction, prediction),
            proba[0], proba[1], proba[2],
            REGIME_NAMES.get(self._candidate_regime, self._candidate_regime),
            self._candidate_count,
        )

        return prediction

    # ── Estado público ─────────────────────────────────────

    @property
    def current_regime(self) -> int:
        return self._current_regime

    @property
    def current_regime_name(self) -> str:
        return REGIME_NAMES.get(self._current_regime, "DESCONOCIDO")

    @property
    def is_confirming_change(self) -> bool:
        """True si hay un cambio de régimen en proceso de confirmación."""
        return self._candidate_regime != self._current_regime and self._candidate_count > 0

    @property
    def confirmation_progress(self) -> str:
        """String legible del progreso de confirmación."""
        if not self.is_confirming_change:
            return ""
        return (
            f"Confirmando {REGIME_NAMES.get(self._candidate_regime, '?')}: "
            f"{self._candidate_count}/{REGIME_CONFIRMATION}"
        )

    def status(self) -> dict:
        """Retorna estado completo para logging / Telegram."""
        return {
            "current_regime":      self._current_regime,
            "current_regime_name": self.current_regime_name,
            "is_confirming":       self.is_confirming_change,
            "confirmation":        self.confirmation_progress,
            "recent_predictions":  self._prediction_history[-5:],
        }
