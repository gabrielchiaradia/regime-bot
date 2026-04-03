"""
scripts/run_bot.py
────────────────────────────────────────────────────────────
Entry point del contenedor Bot (24/7).
Ciclo de inferencia:
  1. Carga el modelo (.pkl)
  2. En cada vela de 15m:
     a. Descarga últimas 500 velas (3m + 15m)
     b. Calcula features
     c. Predice régimen (con histéresis)
     d. Pasa al router → ejecuta estrategia correspondiente
     e. Loguea y notifica por Telegram si el régimen cambia

El bot espera al cierre de cada vela de 15m (sincronización con el reloj UTC).
Si el modelo cambia en disco (re-entrenamiento semanal), lo recarga automáticamente.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import fetch_live_data
from inference.classifier import RegimeClassifier
from orchestrator.router import build_default_router, REGIME_NAMES

# ── Logging ────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
logger = logging.getLogger("run_bot")

# ── Config ──────────────────────────────────────────────────
SYMBOL          = os.getenv("SYMBOL", "ETHUSDT")
CANDLES_15M     = int(os.getenv("CANDLES_LIVE_15M", "500"))
CANDLES_3M      = int(os.getenv("CANDLES_LIVE_3M",  "500"))
MODEL_PATH      = Path(os.getenv("MODELS_DIR", "models")) / f"regime_model_{os.getenv('SYMBOL', 'ETHUSDT')}.pkl"
CYCLE_SECONDS   = int(os.getenv("CYCLE_SECONDS", "900"))  # 15 min = 900s

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def _send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Telegram error: %s", e)


def _wait_for_next_candle() -> None:
    """Espera hasta el cierre de la próxima vela de 15m (sincronizado con UTC)."""
    now = datetime.now(timezone.utc)
    minutes = now.minute
    # Próximo múltiplo de 15
    next_15 = (minutes // 15 + 1) * 15
    if next_15 >= 60:
        wait_minutes = 60 - minutes
    else:
        wait_minutes = next_15 - minutes
    wait_seconds = wait_minutes * 60 - now.second + 2  # +2s de margen para que cierre
    logger.info("⏱️  Próxima vela en %dm %ds", wait_minutes, now.second)
    time.sleep(max(wait_seconds, 1))


def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("REGIME BOT ARRANCANDO | %s", SYMBOL)
    logger.info("=" * 60)

    # ── Cargar modelo ──────────────────────────────────────
    classifier = RegimeClassifier(symbol=SYMBOL)
    try:
        classifier.load_model()
    except FileNotFoundError as e:
        logger.error("%s\nEjecutá run_pipeline.py primero.", e)
        sys.exit(1)

    router = build_default_router()

    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0

    _send_telegram(
        f"🤖 *Regime Bot iniciado*\n"
        f"  Símbolo: `{SYMBOL}`\n"
        f"  Régimen inicial: {classifier.current_regime_name}\n"
        f"  Confirmación: {os.getenv('REGIME_CONFIRMATION', '3')} velas"
    )

    # ── Ciclo principal ────────────────────────────────────
    while True:
        try:
            # Verificar si el modelo fue re-entrenado
            if MODEL_PATH.exists():
                new_mtime = MODEL_PATH.stat().st_mtime
                if new_mtime > model_mtime:
                    logger.info("🔄 Modelo actualizado en disco — recargando...")
                    classifier.load_model()
                    model_mtime = new_mtime
                    _send_telegram("🔄 Modelo recargado (re-entrenamiento semanal)")

            # ── Datos live ─────────────────────────────────
            df_3m, df_15m = fetch_live_data(
                symbol=SYMBOL,
                candles_15m=CANDLES_15M,
                candles_3m=CANDLES_3M,
            )

            # ── Predicción con histéresis ──────────────────
            previous_regime = classifier.current_regime
            new_regime = classifier.predict(df_3m, df_15m)

            # ── Router ────────────────────────────────────
            # Por ahora sin posiciones reales (stub)
            result = router.on_new_regime(
                new_regime=new_regime,
                open_positions=[],  # TODO: integrar con el bot real
                current_price=float(df_15m["close"].iloc[-1]),
            )

            # Log de estado
            status = classifier.status()
            logger.info(
                "Régimen: %s | %s",
                status["current_regime_name"],
                status["confirmation"] or "estable",
            )

            if result["execute_strategy"]:
                router.run_strategy(new_regime)

            # Notificación Telegram solo en cambio de régimen confirmado
            if new_regime != previous_regime:
                _send_telegram(
                    f"🔄 *Cambio de régimen confirmado*\n"
                    f"  `{REGIME_NAMES.get(previous_regime, str(previous_regime))}` → "
                    f"`{REGIME_NAMES.get(new_regime, str(new_regime))}`\n"
                    f"  Par: `{SYMBOL}`"
                )

        except KeyboardInterrupt:
            logger.info("Bot detenido manualmente.")
            _send_telegram("🛑 Regime Bot detenido manualmente.")
            break
        except Exception as e:
            logger.error("Error en ciclo principal: %s", e, exc_info=True)
            _send_telegram(f"⚠️ Error en Regime Bot: `{e}`")
            time.sleep(30)  # pausa antes de reintentar
            continue

        # Esperar próxima vela
        _wait_for_next_candle()


if __name__ == "__main__":
    main()
