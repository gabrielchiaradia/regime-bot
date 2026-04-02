"""
scripts/run_pipeline.py
────────────────────────────────────────────────────────────
Entry point del contenedor Trainer.
Ejecuta el pipeline completo:
  1. Descarga datos históricos de Binance (3m + 15m)
  2. Calcula features (builder)
  3. Etiqueta regímenes (labeler)
  4. Entrena el modelo (train)
  5. Muestra evaluación (evaluate)
  6. (Opcional) Exporta CSV de muestra para inspección

Si el accuracy no supera el umbral mínimo (MIN_ACCURACY en .env),
el modelo anterior NO se sobreescribe y se envía alerta por Telegram.

USO:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --symbol BTCUSDT --days 180
    python scripts/run_pipeline.py --export-csv  # exporta muestra para debug
    python scripts/run_pipeline.py --dry-run     # solo descarga y features, sin entrenar
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Agregar raíz al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import fetch_training_data
from features.builder import build_features
from labels.labeler import label_regimes
from pipeline.train import train
from pipeline.evaluate import print_evaluation_report, export_predictions_csv

# ── Logging ────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("run_pipeline")

# ── Telegram (opcional) ────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def _send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🤖 *Regime Classifier*\n{msg}",
            "parse_mode": "Markdown",
        }, timeout=10)
    except Exception as e:
        logger.warning("Telegram error: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento del clasificador")
    parser.add_argument("--symbol",     default=os.getenv("SYMBOL", "ETHUSDT"))
    parser.add_argument("--days",       type=int, default=int(os.getenv("TRAINING_DAYS", "360")))
    parser.add_argument("--export-csv", action="store_true", help="Exporta muestra del dataset")
    parser.add_argument("--dry-run",    action="store_true", help="Solo descarga y features, sin entrenar")
    args = parser.parse_args()

    # Crear directorio de logs
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("PIPELINE DE ENTRENAMIENTO — %s | %d días", args.symbol, args.days)
    logger.info("=" * 60)

    # ── 1. Descarga de datos ───────────────────────────────
    logger.info("[1/5] Descargando datos históricos...")
    try:
        df_3m, df_15m = fetch_training_data(symbol=args.symbol, days=args.days)
    except Exception as e:
        msg = f"❌ Error descargando datos: {e}"
        logger.error(msg)
        _send_telegram(msg)
        sys.exit(1)

    # ── 2. Features ────────────────────────────────────────
    logger.info("[2/5] Calculando features...")
    try:
        df_features = build_features(df_3m, df_15m)
    except Exception as e:
        msg = f"❌ Error calculando features: {e}"
        logger.error(msg)
        _send_telegram(msg)
        sys.exit(1)

    # ── 3. Etiquetado ──────────────────────────────────────
    logger.info("[3/5] Etiquetando regímenes...")
    try:
        df_labeled = label_regimes(df_features)
    except Exception as e:
        msg = f"❌ Error en el labeler: {e}"
        logger.error(msg)
        _send_telegram(msg)
        sys.exit(1)

    if args.export_csv:
        export_predictions_csv(df_labeled)

    if args.dry_run:
        logger.info("--dry-run activado: pipeline completo hasta etiquetado. Sin entrenamiento.")
        logger.info("Dataset listo: %d filas | Régimen 0: %d | 1: %d | 2: %d",
                    len(df_labeled),
                    (df_labeled["target_regime"] == 0).sum(),
                    (df_labeled["target_regime"] == 1).sum(),
                    (df_labeled["target_regime"] == 2).sum())
        return

    # ── 4. Entrenamiento ───────────────────────────────────
    logger.info("[4/5] Entrenando modelo...")
    try:
        meta = train(df_labeled)
    except ValueError as e:
        # Accuracy por debajo del umbral — modelo NO guardado
        msg = f"⚠️ Modelo no guardado: {e}"
        logger.warning(msg)
        _send_telegram(msg)
        sys.exit(0)  # No es un error fatal — el bot sigue con el .pkl anterior
    except Exception as e:
        msg = f"❌ Error en entrenamiento: {e}"
        logger.error(msg, exc_info=True)
        _send_telegram(msg)
        sys.exit(1)

    # ── 5. Evaluación ──────────────────────────────────────
    logger.info("[5/5] Evaluación del modelo...")
    print_evaluation_report(meta)

    # Notificación de éxito
    acc = meta.get("test_accuracy", 0)
    cv  = meta.get("cv_accuracy_mean", 0)
    msg = (
        f"✅ Modelo re-entrenado exitosamente\n"
        f"  Símbolo: {args.symbol} | {args.days} días\n"
        f"  Test accuracy: {acc:.3f}\n"
        f"  CV accuracy:   {cv:.3f}"
    )
    logger.info(msg)
    _send_telegram(msg)


if __name__ == "__main__":
    main()
