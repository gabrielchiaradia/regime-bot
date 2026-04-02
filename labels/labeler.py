"""
labels/labeler.py
────────────────────────────────────────────────────────────
Etiquetado automático de regímenes de mercado.

REGÍMENES:
  0 — Lateral/Rango   : ADX < 20 Y precio dentro de Bollinger Bands (2σ)
  1 — Tendencia       : ADX > 25 Y precio por encima/debajo de EMA(50)
  2 — Alta Volatilidad: ATR actual > 1.5× ATR medio del día anterior

PRIORIDAD: 2 → 1 → 0
  Si una vela califica para múltiples regímenes, el de mayor prioridad gana.
  Esto evita ambigüedad: una explosión de volatilidad es siempre Régimen 2
  independientemente de si el ADX es alto.

ZONA GRIS:
  Velas que no califican para ningún régimen (ADX entre 20-25, precio
  dentro de BB pero ADX algo elevado) se etiquetan como 0 (conservador).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Umbrales (configurables vía .env) ─────────────────────
ADX_RANGE_THRESHOLD  = float(20)   # ADX < este valor → candidato a Régimen 0
ADX_TREND_THRESHOLD  = float(25)   # ADX > este valor → candidato a Régimen 1
ATR_SPIKE_MULTIPLIER = float(1.5)  # ATR > N × ATR_prev_day → Régimen 2


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna `target_regime` al DataFrame.
    
    Requiere columnas (producidas por builder.py):
      feature_adx, bb_upper, bb_lower, ema_50,
      atr_current, atr_prev_day_mean, close
    
    Retorna el DataFrame con columna `target_regime` (int: 0, 1, 2)
    y columnas auxiliares de diagnóstico.
    """
    df = df.copy()

    close    = df["close"]
    adx      = df["feature_adx"]
    bb_upper = df["bb_upper"]
    bb_lower = df["bb_lower"]
    ema_50   = df["ema_50"]
    atr_curr = df["atr_current"]
    atr_prev = df["atr_prev_day_mean"]

    # ── Condiciones individuales ───────────────────────────

    # Régimen 2: spike de volatilidad (prioridad máxima)
    # Condición doble: ATR spike + estructura de vela caótica
    #   1. ATR actual > N× ATR promedio del día anterior
    #   2. Mecha > 50% del cuerpo de la vela (barrido de liquidez, no vela tendencial)
    #      mecha_total = high - low
    #      cuerpo      = |close - open|
    #      condición   : mecha > 0.5 × cuerpo  (si cuerpo = 0 usamos ATR como proxy)
    atr_ratio  = atr_curr / atr_prev.replace(0, np.nan)
    high       = df["high"]
    low        = df["low"]
    open_price = df["open"]
    mecha      = high - low
    cuerpo     = (close - open_price).abs().replace(0, atr_curr)  # evitar div/0
    cond_mecha_caotica = mecha > (0.5 * cuerpo)
    cond_vol_spike = (atr_ratio > ATR_SPIKE_MULTIPLIER) & cond_mecha_caotica

    # Régimen 1: tendencia fuerte
    precio_sobre_ema  = close > ema_50
    precio_bajo_ema   = close < ema_50
    cond_trend = (adx > ADX_TREND_THRESHOLD) & (precio_sobre_ema | precio_bajo_ema)

    # Régimen 0: lateral (precio dentro de BB y ADX bajo)
    precio_dentro_bb = (close <= bb_upper) & (close >= bb_lower)
    cond_range = (adx < ADX_RANGE_THRESHOLD) & precio_dentro_bb

    # ── Etiquetado con prioridad 2 → 1 → 0 ───────────────
    regime = pd.Series(0, index=df.index, dtype=int)  # default: 0 (zona gris → conservador)
    regime = regime.where(~cond_range,   0)  # Régimen 0
    regime = regime.where(~cond_trend,   1)  # Régimen 1 (sobreescribe 0)
    regime = regime.where(~cond_vol_spike, 2)  # Régimen 2 (sobreescribe todo)

    df["target_regime"]    = regime
    df["_atr_ratio"]       = atr_ratio       # diagnóstico
    df["_precio_vs_ema"]   = close - ema_50  # diagnóstico
    # Exponer como feature explícita para el modelo
    df["feature_atr_ratio"] = atr_ratio

    # ── Estadísticas de distribución ──────────────────────
    counts = regime.value_counts().sort_index()
    total  = len(regime)
    logger.info("Distribución de regímenes:")
    names = {0: "Lateral (0)", 1: "Tendencia (1)", 2: "Alta Vol (2)"}
    for r, count in counts.items():
        pct = count / total * 100
        logger.info("  %s: %d velas (%.1f%%)", names.get(r, r), count, pct)

    # Advertencia si algún régimen es muy escaso (< 5%)
    for r in [0, 1, 2]:
        pct = counts.get(r, 0) / total * 100
        if pct < 5:
            logger.warning(
                "WARN Régimen %d tiene solo %.1f%% de los datos — "
                "el modelo puede tener dificultades para aprenderlo. "
                "Considerar ajustar umbrales ADX/ATR.",
                r, pct
            )

    return df


def get_feature_columns() -> list[str]:
    """
    Retorna la lista canónica de features usadas por el modelo.
    Centralizada acá para que train.py y classifier.py usen exactamente
    las mismas columnas (evita desincronización).
    """
    return [
        "feature_log_return",
        "feature_volatility_ratio",
        "feature_vwap_distance",
        "feature_rsi_slope",
        "feature_adx",
        "feature_adx_slope",          # aceleración del ADX — diferencia pánico de tendencia
        "feature_volume_zscore",
        # Features de micro-estructura (3m resampleado)
        "feature_volatility_ratio_micro",
        "feature_volume_zscore_micro",
    ]
