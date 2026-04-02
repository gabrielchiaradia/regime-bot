"""
features/builder.py
────────────────────────────────────────────────────────────
Calcula los 6 features del clasificador de regímenes.

ANTI DATA-LEAKAGE:
  - Todas las ventanas rolling usan .shift(1) donde corresponde
  - El VWAP se resetea diariamente (anclado a 00:00 UTC)
  - El Volume Z-Score usa media/std del pasado, nunca del período actual
  - Los features de la vela `t` solo usan información de velas <= t

FLUJO:
  df_3m  → resample a 15m → merge con df_15m → DataFrame con features
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Parámetros (configurables) ─────────────────────────────
ATR_SHORT_PERIOD    = int(5)    # Para Volatility Ratio
ATR_LONG_PERIOD     = int(20)   # Para Volatility Ratio
RSI_PERIOD          = int(14)
RSI_SLOPE_PERIOD    = int(5)    # Ventana de regresión lineal del RSI
ADX_PERIOD          = int(14)
VOLUME_ZSCORE_WINDOW = int(20)  # Ventana para μ y σ del volumen
VWAP_BAND_MULT      = float(2.0)


# ══════════════════════════════════════════════════════════
#  INDICADORES BASE
# ══════════════════════════════════════════════════════════

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI estándar (Wilder smoothing)."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX estándar (14 períodos).
    Mide fuerza de tendencia independientemente de su dirección.
    """
    high, low, close = df["high"], df["low"], df["close"]

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_val   = _atr(df, period)
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_val
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    """
    VWAP anclado diariamente (reset 00:00 UTC).
    Anti-leakage: usa acumulado desde inicio del día hasta vela t.
    """
    typ_price = (df["high"] + df["low"] + df["close"]) / 3
    typ_vol   = typ_price * df["volume"]

    date_key  = df.index.normalize()  # trunca a 00:00 UTC

    cum_typ_vol = typ_vol.groupby(date_key).cumsum()
    cum_vol     = df["volume"].groupby(date_key).cumsum()

    return cum_typ_vol / cum_vol.replace(0, np.nan)


def _bollinger_bands(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Bandas de Bollinger. Retorna (upper, lower)."""
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    return sma + std_mult * std, sma - std_mult * std


def _rsi_slope(rsi: pd.Series, window: int = 5) -> pd.Series:
    """
    Pendiente de la regresión lineal del RSI en ventana `window`.
    Normalizada por el rango del RSI (0-100) para ser escala-independiente.
    Anti-leakage: rolling estricto (solo pasado).
    """
    x = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        # Regresión lineal simple: slope = cov(x,y)/var(x)
        return float(np.polyfit(x, y, 1)[0])

    raw_slope = rsi.rolling(window).apply(_slope, raw=True)
    # Normalizar: slope máximo teórico sería 100/(window-1) si RSI va de 0 a 100
    max_slope = 100.0 / max(window - 1, 1)
    return raw_slope / max_slope  # rango aproximado [-1, 1]


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-Score del volumen.
    ANTI-LEAKAGE: usa .shift(1) para que la media/std de la vela t
    no incluya el volumen de la propia vela t.
    """
    vol_shifted = volume.shift(1)  # ← clave anti-leakage
    mean = vol_shifted.rolling(window).mean()
    std  = vol_shifted.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)


# ══════════════════════════════════════════════════════════
#  RESAMPLE 3m → 15m
# ══════════════════════════════════════════════════════════

def _resample_3m_to_15m(df_3m: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega velas de 3m a 15m para extraer features de micro-estructura.
    Usa label='left' y closed='left' para que cada vela 15m agregue
    las 5 velas de 3m que la componen (sin incluir la siguiente).
    """
    resampled = df_3m.resample("15min", label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    return resampled


# ══════════════════════════════════════════════════════════
#  BUILDER PRINCIPAL
# ══════════════════════════════════════════════════════════

def build_features(
    df_3m:  pd.DataFrame,
    df_15m: pd.DataFrame,
    warmup_candles: int = 250,
) -> pd.DataFrame:
    """
    Construye el DataFrame de features listo para el labeler y el modelo.

    Features producidas:
      feature_log_return    : ln(Close_t / Close_{t-1})
      feature_volatility_ratio : ATR(5) / ATR(20)  — de 3m resampleado
      feature_vwap_distance : (Close - VWAP) / VWAP  — VWAP anclado diario
      feature_rsi_slope     : pendiente normalizada del RSI(14) en 5 períodos
      feature_adx           : ADX(14) — fuerza de tendencia
      feature_volume_zscore : Z-Score del volumen (ventana 20, anti-leakage)

    Columnas auxiliares para el labeler:
      atr_current           : ATR(14) actual (para Régimen 2)
      atr_prev_day_mean     : media del ATR del día anterior (para Régimen 2)
      bb_upper, bb_lower    : Bandas de Bollinger (para Régimen 0)
      ema_50                : EMA(50) (para Régimen 1)
      close                 : precio de cierre
      volume                : volumen
    """
    logger.info("Construyendo features...")

    # ── 1. Micro-estructura desde 3m ──────────────────────
    df_micro = _resample_3m_to_15m(df_3m)

    atr_short_3m = _atr(df_micro, ATR_SHORT_PERIOD)
    atr_long_3m  = _atr(df_micro, ATR_LONG_PERIOD)
    vol_ratio_3m = atr_short_3m / atr_long_3m.replace(0, np.nan)
    vol_zscore_3m = _volume_zscore(df_micro["volume"], VOLUME_ZSCORE_WINDOW)

    micro_features = pd.DataFrame({
        "feature_volatility_ratio_micro": vol_ratio_3m,
        "feature_volume_zscore_micro":    vol_zscore_3m,
    })

    # ── 2. Features sobre 15m ─────────────────────────────
    df = df_15m.copy()

    # Log return
    df["feature_log_return"] = np.log(df["close"] / df["close"].shift(1))

    # VWAP distance (anclado diario)
    vwap = _vwap_daily(df)
    df["feature_vwap_distance"] = (df["close"] - vwap) / vwap.replace(0, np.nan)
    df["_vwap"] = vwap  # auxiliar para el labeler

    # RSI Slope
    rsi = _rsi(df["close"], RSI_PERIOD)
    df["feature_rsi_slope"] = _rsi_slope(rsi, RSI_SLOPE_PERIOD)
    df["_rsi"] = rsi  # auxiliar

    # ADX
    df["feature_adx"] = _adx(df, ADX_PERIOD)

    # Volatility ratio y Volume Z-Score (de 15m para consistencia operacional)
    atr5_15m  = _atr(df, ATR_SHORT_PERIOD)
    atr20_15m = _atr(df, ATR_LONG_PERIOD)
    df["feature_volatility_ratio"] = atr5_15m / atr20_15m.replace(0, np.nan)
    df["feature_volume_zscore"]    = _volume_zscore(df["volume"], VOLUME_ZSCORE_WINDOW)

    # ── 3. Auxiliares para el labeler ─────────────────────
    atr14 = _atr(df, ADX_PERIOD)
    df["atr_current"] = atr14

    # ATR medio del día ANTERIOR (anti-leakage: shift para no ver el día actual)
    # Agrupamos por fecha y calculamos la media, luego shift(1) al nivel de día
    df["_date"] = df.index.normalize()
    daily_atr_mean = atr14.groupby(df["_date"]).transform("mean").shift(1)
    df["atr_prev_day_mean"] = daily_atr_mean

    bb_upper, bb_lower = _bollinger_bands(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["ema_50"]   = df["close"].ewm(span=50, adjust=False).mean()

    # ── 4. Merge con micro-estructura de 3m ───────────────
    df = df.join(micro_features, how="left", rsuffix="_3m")

    # ── 5. Eliminar período de warmup ─────────────────────
    df = df.iloc[warmup_candles:].copy()

    # ── 6. Eliminar filas con NaN en features clave ───────
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Eliminadas %d filas con NaN en features", dropped)

    logger.info("Features construidas: %d filas, %d features | %s → %s",
                len(df), len(feature_cols),
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"))

    return df
