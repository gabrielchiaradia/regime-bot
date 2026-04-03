"""
scripts/run_bot.py
────────────────────────────────────────────────────────────
Bot multi-par de clasificación de regímenes.
En cada vela de 15m escanea todos los pares configurados,
clasifica el régimen de cada uno y notifica por Telegram
con un resumen cuando alguno cambia.

Configuración en .env:
  SYMBOLS=ETHUSDT,BTCUSDT,SOLUSDT   (separados por coma)
  O bien SYMBOL=ETHUSDT              (un solo par, compatibilidad)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # no instalado, usar variables del entorno

from data.fetcher import fetch_live_data
from inference.classifier import RegimeClassifier
from orchestrator.router import build_default_router, REGIME_NAMES

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

_symbols_env = os.getenv("SYMBOLS", "") or os.getenv("SYMBOL", "ETHUSDT")
SYMBOLS       = [s.strip() for s in _symbols_env.split(",") if s.strip()]
CANDLES_15M   = int(os.getenv("CANDLES_LIVE_15M", "500"))
CANDLES_3M    = int(os.getenv("CANDLES_LIVE_3M",  "500"))
MODELS_DIR    = Path(os.getenv("MODELS_DIR", "models"))
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
REGIME_EMOJIS = {0: "Lateral", 1: "Tendencia", 2: "AltaVol"}


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
    now          = datetime.now(timezone.utc)
    next_15      = (now.minute // 15 + 1) * 15
    wait_minutes = (60 - now.minute) if next_15 >= 60 else (next_15 - now.minute)
    wait_seconds = wait_minutes * 60 - now.second + 2
    logger.info("Proxima vela en %dm %ds", wait_minutes, now.second)
    time.sleep(max(wait_seconds, 1))


def _process_symbol(symbol, classifier, router, prev_regimes):
    df_3m, df_15m = fetch_live_data(
        symbol=symbol,
        candles_15m=CANDLES_15M,
        candles_3m=CANDLES_3M,
    )
    previous   = prev_regimes.get(symbol, 0)
    new_regime = classifier.predict(df_3m, df_15m)
    status     = classifier.status()

    result = router.on_new_regime(
        new_regime=new_regime,
        open_positions=[],
        current_price=float(df_15m["close"].iloc[-1]),
    )

    logger.info("[%s] Regimen: %s | %s",
        symbol,
        status["current_regime_name"],
        status["confirmation"] or "estable",
    )

    if result["execute_strategy"]:
        router.run_strategy(new_regime)

    return new_regime, (new_regime != previous)


def main() -> None:
    logger.info("=" * 60)
    logger.info("REGIME BOT MULTI-PAR | %s", ", ".join(SYMBOLS))
    logger.info("=" * 60)

    classifiers:  dict = {}
    routers:      dict = {}
    model_mtimes: dict = {}
    prev_regimes: dict = {}

    for symbol in SYMBOLS:
        clf = RegimeClassifier(symbol=symbol)
        try:
            clf.load_model()
            classifiers[symbol]  = clf
            routers[symbol]      = build_default_router()
            mp = MODELS_DIR / f"regime_model_{symbol}.pkl"
            model_mtimes[symbol] = mp.stat().st_mtime if mp.exists() else 0
            prev_regimes[symbol] = 0
        except FileNotFoundError as e:
            logger.error("%s -- omitiendo par.", e)

    if not classifiers:
        logger.error("Ningun modelo disponible. Ejecuta run_pipeline.py primero.")
        sys.exit(1)

    _send_telegram(
        "*Regime Bot iniciado*\n"
        f"Pares: `{'`, `'.join(classifiers.keys())}`\n"
        f"Confirmacion: {os.getenv('REGIME_CONFIRMATION', '3')} velas"
    )

    while True:
        try:
            cambios: list = []
            resumen: list = []

            for symbol, clf in classifiers.items():
                mp = MODELS_DIR / f"regime_model_{symbol}.pkl"
                if mp.exists():
                    new_mtime = mp.stat().st_mtime
                    if new_mtime > model_mtimes.get(symbol, 0):
                        logger.info("[%s] Modelo actualizado -- recargando...", symbol)
                        clf.load_model()
                        model_mtimes[symbol] = new_mtime

                try:
                    new_regime, hubo_cambio = _process_symbol(
                        symbol, clf, routers[symbol], prev_regimes
                    )
                except Exception as e:
                    logger.error("[%s] Error: %s", symbol, e, exc_info=True)
                    resumen.append(f"`{symbol}` ERROR")
                    continue

                label = REGIME_EMOJIS.get(new_regime, str(new_regime))
                resumen.append(f"`{symbol}` {label}")

                if hubo_cambio:
                    prev_label = REGIME_EMOJIS.get(prev_regimes[symbol], str(prev_regimes[symbol]))
                    cambios.append(f"`{symbol}`: {prev_label} -> {label}")
                    prev_regimes[symbol] = new_regime

            if cambios:
                msg  = "*Cambio de regimen*\n" + "\n".join(cambios)
                msg += "\n\n*Estado actual*\n" + "\n".join(resumen)
                _send_telegram(msg)

        except KeyboardInterrupt:
            logger.info("Bot detenido manualmente.")
            _send_telegram("Bot detenido manualmente.")
            break
        except Exception as e:
            logger.error("Error ciclo principal: %s", e, exc_info=True)
            _send_telegram(f"Error: `{e}`")
            time.sleep(30)
            continue

        _wait_for_next_candle()


if __name__ == "__main__":
    main()
