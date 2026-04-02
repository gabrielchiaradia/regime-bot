"""
orchestrator/router.py
────────────────────────────────────────────────────────────
Orquestador de estrategias según el régimen de mercado.

REGÍMENES → ESTRATEGIAS:
  0 (Lateral)          → VWAP Mean Reversion  (bot actual)
  1 (Tendencia)        → VWAP Trend Following (bot con cruce de VWAP + EMA)
  2 (Alta Volatilidad) → STOP — no operar

MANEJO DE TRANSICIONES:
  Cuando el régimen cambia, el router evalúa las posiciones abiertas
  del régimen anterior y decide si cerrarlas o heredarlas.

  Régimen X → Régimen 2 (Alta Vol):
    Si PnL > -1%: cerrar inmediatamente
    Si PnL < -1%: heredar con SL original (no crystallizar pérdida)

  Régimen 0 → Régimen 1 (o viceversa):
    Si PnL > 0: cerrar y reubicar en nueva estrategia
    Si PnL < 0: heredar con SL ajustado al nuevo régimen

NOTA: Las estrategias reales (lógica VWAP, etc.) se implementan
cuando se integren los bots existentes. Este stub define la interfaz
y la lógica de transición.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Regime(IntEnum):
    LATERAL     = 0
    TENDENCIA   = 1
    ALTA_VOL    = 2


REGIME_NAMES = {
    Regime.LATERAL:   "📊 LATERAL (VWAP Reversion)",
    Regime.TENDENCIA: "📈 TENDENCIA (VWAP Trend)",
    Regime.ALTA_VOL:  "⚡ ALTA VOLATILIDAD (STOP)",
}

# Umbral de PnL para decidir si cerrar o heredar posición en transición
PNL_CLOSE_THRESHOLD = float(-0.01)  # -1%


@dataclass
class Position:
    """Representa una posición abierta del régimen anterior."""
    symbol:        str
    side:          str         # "LONG" | "SHORT"
    entry_price:   float
    current_price: float
    stop_loss:     float
    take_profit:   float
    regime_origin: int         # régimen en que se abrió
    size:          float = 0.0

    @property
    def pnl_pct(self) -> float:
        """PnL como porcentaje del precio de entrada."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class RouterState:
    """Estado interno del router."""
    current_regime:  int = Regime.LATERAL
    previous_regime: int = Regime.LATERAL
    open_positions:  list[Position] = field(default_factory=list)
    trades_skipped:  int = 0   # velas donde Régimen 2 frenó la operativa


class StrategyRouter:
    """
    Orquestador de estrategias.
    
    Uso:
        router = StrategyRouter()
        router.register_strategy(Regime.LATERAL,   my_vwap_reversion_fn)
        router.register_strategy(Regime.TENDENCIA, my_vwap_trend_fn)

        # En cada ciclo del bot:
        action = router.on_new_regime(new_regime, current_positions, current_price)
    """

    def __init__(self) -> None:
        self._state = RouterState()
        self._strategies: dict[int, Callable] = {}

    def register_strategy(self, regime: int, strategy_fn: Callable) -> None:
        """Registra una función de estrategia para un régimen."""
        self._strategies[regime] = strategy_fn
        logger.info("Estrategia registrada para régimen %d: %s",
                    regime, strategy_fn.__name__)

    def on_new_regime(
        self,
        new_regime: int,
        open_positions: list[Position],
        current_price: float,
    ) -> dict:
        """
        Procesa un nuevo régimen y retorna las acciones a tomar.
        
        Returns:
            dict con:
                "regime":          régimen actual
                "regime_name":     nombre legible
                "actions":         lista de acciones ["CLOSE_POSITION", "HOLD", "STOP"]
                "execute_strategy": bool — si debe correr la estrategia del régimen
                "log":             mensajes de log
        """
        previous = self._state.current_regime
        self._state.previous_regime = previous
        self._state.current_regime  = new_regime
        self._state.open_positions  = open_positions

        actions = []
        log_msgs = []

        # ── Sin cambio de régimen ──────────────────────────
        if new_regime == previous:
            if new_regime == Regime.ALTA_VOL:
                self._state.trades_skipped += 1
                actions.append("STOP")
                log_msgs.append(f"⚡ Alta volatilidad activa. Operativa pausada "
                                f"({self._state.trades_skipped} velas).")
            else:
                actions.append("CONTINUE")
        else:
            # ── Cambio de régimen ──────────────────────────
            log_msgs.append(
                f"🔄 Régimen: {REGIME_NAMES[previous]} → {REGIME_NAMES[new_regime]}"
            )
            self._state.trades_skipped = 0

            # Evaluar posiciones abiertas del régimen anterior
            for pos in open_positions:
                action, reason = self._evaluate_position_on_transition(
                    pos, new_regime
                )
                actions.append(action)
                log_msgs.append(f"  Posición {pos.side} {pos.symbol}: {action} ({reason})")

            if not open_positions:
                actions.append("SWITCH_STRATEGY")
                log_msgs.append("  Sin posiciones abiertas → cambio limpio de estrategia.")

        # ── Decidir si ejecutar estrategia ────────────────
        execute = new_regime != Regime.ALTA_VOL

        result = {
            "regime":           new_regime,
            "regime_name":      REGIME_NAMES.get(new_regime, str(new_regime)),
            "actions":          actions,
            "execute_strategy": execute,
            "log":              log_msgs,
        }

        for msg in log_msgs:
            logger.info(msg)

        return result

    def _evaluate_position_on_transition(
        self,
        pos: Position,
        new_regime: int,
    ) -> tuple[str, str]:
        """
        Decide qué hacer con una posición abierta al cambiar de régimen.
        
        Returns:
            (acción, razón)
            Acciones: "CLOSE", "HOLD", "ADJUST_SL"
        """
        pnl = pos.pnl_pct

        # Transición a Alta Volatilidad
        if new_regime == Regime.ALTA_VOL:
            if pnl > PNL_CLOSE_THRESHOLD:
                return "CLOSE", f"Régimen 2: PnL {pnl:.1%} > umbral → cerrar"
            else:
                return "HOLD", f"Régimen 2: PnL {pnl:.1%} < umbral → heredar con SL original"

        # Transición entre Lateral ↔ Tendencia
        if pnl > 0:
            return "CLOSE", f"PnL positivo ({pnl:.1%}) → cerrar y reubicar en nueva estrategia"
        else:
            return "HOLD", f"PnL negativo ({pnl:.1%}) → heredar con SL ajustado"

    def run_strategy(self, regime: int, *args, **kwargs):
        """
        Ejecuta la estrategia registrada para el régimen dado.
        Si no hay estrategia registrada, loguea un warning.
        """
        if regime == Regime.ALTA_VOL:
            logger.info("⚡ Régimen 2: no se ejecuta ninguna estrategia.")
            return None

        strategy_fn = self._strategies.get(regime)
        if strategy_fn is None:
            logger.warning(
                "No hay estrategia registrada para régimen %d (%s). "
                "Registrá una con router.register_strategy().",
                regime, REGIME_NAMES.get(regime, "?")
            )
            return None

        return strategy_fn(*args, **kwargs)

    @property
    def state(self) -> RouterState:
        return self._state


# ══════════════════════════════════════════════════════════
#  STUBS DE ESTRATEGIAS (reemplazar con los bots reales)
# ══════════════════════════════════════════════════════════

def strategy_vwap_mean_reversion(*args, **kwargs) -> dict:
    """
    STUB: Lógica de reversión a la media VWAP.
    → Se reemplaza con la lógica del bot VWAP actual.
    """
    logger.info("📊 [STUB] Ejecutando VWAP Mean Reversion")
    return {"action": "STUB", "strategy": "vwap_mean_reversion"}


def strategy_vwap_trend_following(*args, **kwargs) -> dict:
    """
    STUB: Lógica de seguimiento de tendencia con cruce VWAP + EMA.
    → Se reemplaza con el bot 2 (VWAP trend).
    """
    logger.info("📈 [STUB] Ejecutando VWAP Trend Following")
    return {"action": "STUB", "strategy": "vwap_trend_following"}


def build_default_router() -> StrategyRouter:
    """Construye un router con los stubs por defecto."""
    router = StrategyRouter()
    router.register_strategy(Regime.LATERAL,   strategy_vwap_mean_reversion)
    router.register_strategy(Regime.TENDENCIA, strategy_vwap_trend_following)
    # Régimen 2 no tiene estrategia registrada → router.run_strategy() lo maneja
    return router
