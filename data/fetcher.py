"""
data/fetcher.py
────────────────────────────────────────────────────────────
Descarga velas históricas de Binance Futures (o testnet).
Retorna DataFrames limpios de 3m y 15m listos para el builder.

Sin data leakage: solo retorna datos hasta `now`. El labeler
y builder son responsables de no mirar el futuro.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────
BINANCE_BASE_URL  = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
SYMBOL            = os.getenv("SYMBOL", "ETHUSDT")
WARMUP_CANDLES    = int(os.getenv("WARMUP_CANDLES", "250"))   # velas descartadas para warmup EMA
MAX_CANDLES_LIMIT = 1500                                       # límite de Binance por request


def _fetch_klines(
    symbol: str,
    interval: str,
    total_candles: int,
    end_time_ms: int | None = None,
) -> pd.DataFrame:
    """
    Descarga `total_candles` velas de Binance usando paginación si es necesario.
    Siempre descarga hasta `end_time_ms` (o ahora si es None).
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    all_rows: list[list] = []

    # Binance devuelve hasta 1500 velas por request — paginamos hacia atrás
    remaining = total_candles
    current_end = end_time_ms  # None = ahora

    while remaining > 0:
        limit = min(remaining, MAX_CANDLES_LIMIT)
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if current_end:
            params["endTime"] = current_end

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("Error descargando %s %s: %s", symbol, interval, e)
            raise

        if not data:
            break

        # Guardar el open_time ANTES del prepend (después data[0] sería el acumulado)
        first_open_time = int(data[0][0])

        all_rows = data + all_rows  # prepend: los más antiguos primero
        remaining -= len(data)

        # Siguiente página: terminar justo antes del open de la primera vela recibida
        current_end = first_open_time - 1

        if len(data) < limit:
            break  # Binance no tiene más historia

        time.sleep(0.1)  # rate limit conservador

    if not all_rows:
        raise ValueError(f"No se obtuvieron datos para {symbol} {interval}")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])

    # Tipos
    df["open_time"]  = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Descartar la vela actual (incompleta) — no leakage
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df.index < now_utc]

    logger.info("Descargadas %d velas %s %s (desde %s hasta %s)",
                len(df), symbol, interval,
                df.index[0].strftime("%Y-%m-%d %H:%M"),
                df.index[-1].strftime("%Y-%m-%d %H:%M"))

    return df


def fetch_training_data(
    symbol: str = SYMBOL,
    days: int = 360,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descarga datos históricos para entrenamiento.
    
    Retorna:
        df_3m  : velas de 3 minutos (features de micro-estructura)
        df_15m : velas de 15 minutos (features operacionales + labels)
    
    Incluye WARMUP_CANDLES extra para que EMA/ADX converjan correctamente
    antes de la primera etiqueta válida.
    """
    logger.info("Iniciando descarga de datos de entrenamiento: %s, %d días", symbol, days)

    # Velas necesarias (incluyendo warmup)
    candles_15m = days * 24 * 4 + WARMUP_CANDLES   # 4 velas de 15m por hora
    candles_3m  = days * 24 * 20 + WARMUP_CANDLES  # 20 velas de 3m por hora

    df_15m = _fetch_klines(symbol, "15m", candles_15m)
    df_3m  = _fetch_klines(symbol, "3m",  candles_3m)

    return df_3m, df_15m


def fetch_live_data(
    symbol: str = SYMBOL,
    candles_15m: int = 500,
    candles_3m: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descarga datos para inferencia en tiempo real.
    Menos historia — solo lo necesario para que los indicadores converjan.
    """
    logger.info("Descarga live: %s | 15m×%d 3m×%d", symbol, candles_15m, candles_3m)
    df_15m = _fetch_klines(symbol, "15m", candles_15m)
    df_3m  = _fetch_klines(symbol, "3m",  candles_3m)
    return df_3m, df_15m
