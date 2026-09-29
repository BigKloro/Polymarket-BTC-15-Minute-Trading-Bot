"""
15-Minute BTC Trading Strategy
Main strategy that coordinates signal processing and trading decisions
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import deque
from loguru import logger
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine, FusedSignal
from core.strategy_brain.signal_processors.base_processor import SignalDirection
from execution.execution_engine import get_execution_engine


class BTCStrategy15Min:
    """15-minute BTC trading strategy."""

    def __init__(
        self,
        max_position_size: Decimal = Decimal("10.0"),
        stop_loss_pct: float = 0.30,
        take_profit_pct: float = 0.20,
        max_positions: int = 2,
    ):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions

        # Signal processors
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=0.15,
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25,
            extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=0.05,
        )
        self.orderbook_processor = OrderBookImbalanceProcessor()
        self.tick_velocity_processor = TickVelocityProcessor()
        self.deribit_pcr_processor = DeribitPCRProcessor()

        # Fusion engine
        self.fusion_engine = get_fusion_engine()

        # Execution engine (handles risk, orders, position lifecycle)
        self.execution_engine = get_execution_engine()
        self.execution_engine.on_position_closed = self._on_position_closed

        # Price history for signal processors
        self.price_history: deque = deque(maxlen=100)

        # Tick buffer for velocity processor (~2 min of ticks at 1s cadence)
        self._tick_buffer: deque = deque(maxlen=120)

        # Current market data
        self._current_price: Optional[Decimal] = None
        self._spot_price_consensus: Optional[Decimal] = None
        self._sentiment_score: Optional[float] = None

        # Polymarket market metadata
        self._yes_token_id: Optional[str] = None
        self._market_expiry: Optional[datetime] = None

        # Strategy state
        self._is_running = False
        self._last_decision_time: Optional[datetime] = None

        # Statistics
        self._signals_processed = 0
        self._trades_executed = 0
        self._total_pnl = Decimal("0")

        logger.info(
            f"Initialized 15-Min BTC Strategy: "
            f"max_position=${max_position_size}, "
            f"SL={stop_loss_pct:.0%}, TP={take_profit_pct:.0%}"
        )

    def set_market_info(
        self,
        yes_token_id: Optional[str] = None,
        market_expiry: Optional[datetime] = None,
    ) -> None:
        """Set Polymarket market metadata (call after market discovery)."""
        if yes_token_id:
            self._yes_token_id = yes_token_id
            logger.info(f"Set YES token ID: {yes_token_id[:16]}…")
        if market_expiry:
            self._market_expiry = market_expiry
            logger.info(f"Set market expiry: {market_expiry}")

    async def start(self) -> None:
        """Start the strategy."""
        if self._is_running:
            logger.warning("Strategy already running")
            return

        self._is_running = True
        logger.info("Strategy started")

        asyncio.create_task(self._decision_loop())
        asyncio.create_task(self._position_monitor_loop())
        asyncio.create_task(self._daily_reset_loop())

    async def stop(self) -> None:
        """Stop the strategy."""
        self._is_running = False
        logger.info("Strategy stopped")

    def update_market_data(
        self,
        price: Decimal,
        spot_consensus: Optional[Decimal] = None,
        sentiment: Optional[float] = None,
    ) -> None:
        """Update market data and feed tick buffer for velocity processor."""
        self._current_price = price
        self.price_history.append(price)
        self._tick_buffer.append({'ts': datetime.now(), 'price': price})

        if spot_consensus:
            self._spot_price_consensus = spot_consensus

        if sentiment is not None:
            self._sentiment_score = sentiment

    async def _decision_loop(self) -> None:
        """Main decision loop — runs every 15 minutes."""
        while self._is_running:
            try:
                await self._wait_for_next_interval()
                await self._make_decision()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(60)

    async def _position_monitor_loop(self) -> None:
        """Check stop-loss / take-profit every 30 seconds (not just on 15-min ticks)."""
        while self._is_running:
            await asyncio.sleep(30)
            if self._current_price:
                try:
                    await self.execution_engine.update_positions(self._current_price)
                except Exception as e:
                    logger.error(f"Error in position monitor: {e}")

    async def _daily_reset_loop(self) -> None:
        """Reset daily risk limits at midnight."""
        while self._is_running:
            now = datetime.now()
            next_midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            await asyncio.sleep((next_midnight - now).total_seconds())
            if self._is_running:
                self.execution_engine.risk_engine.reset_daily_stats()
                logger.info("Daily stats reset at midnight")

    async def _wait_for_next_interval(self) -> None:
        """Wait until next 15-minute mark."""
        now = datetime.now()
        minutes_past = now.minute % 15
        wait_minutes = 15 if minutes_past == 0 else 15 - minutes_past
        next_time = (now + timedelta(minutes=wait_minutes)).replace(second=0, microsecond=0)
        wait_seconds = (next_time - now).total_seconds()
        logger.info(f"Waiting {wait_seconds:.0f}s until next decision ({next_time.strftime('%H:%M')})")
        await asyncio.sleep(wait_seconds)

    async def _make_decision(self) -> None:
        """Make trading decision based on fused signals."""
        logger.info("=" * 60)
        logger.info("MAKING TRADING DECISION")
        logger.info("=" * 60)

        if not self._current_price:
            logger.warning("No current price data available")
            return

        # Skip new entries near market expiry (signal noise spikes near resolution)
        if self._market_expiry:
            time_to_expiry = (self._market_expiry - datetime.now()).total_seconds()
            if time_to_expiry < 3600:
                logger.warning(f"Market expires in {time_to_expiry/60:.0f}min — skipping new positions")
                return

        signals = self._process_signals()

        if not signals:
            logger.info("No signals generated")
            return

        logger.info(f"Generated {len(signals)} signals")
        for sig in signals:
            logger.info(
                f"  [{sig.source}] {sig.direction.value}: "
                f"score={sig.score:.1f}, confidence={sig.confidence:.2%}"
            )

        fused = self.fusion_engine.fuse_signals(
            signals,
            min_signals=1,
            min_score=60.0,
        )

        if not fused:
            logger.info("No actionable fused signal")
            return

        logger.info(
            f"FUSED SIGNAL: {fused.direction.value} "
            f"(score={fused.score:.1f}, confidence={fused.confidence:.2%})"
        )

        if not fused.is_actionable:
            logger.info("Signal not strong enough to trade")
            return

        if len(self.execution_engine.get_open_positions()) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return

        await self._execute_trade(fused)
        self._last_decision_time = datetime.now()

    def _process_signals(self) -> List:
        """Run all signal processors and return non-None results."""
        signals = []

        if len(self.price_history) < 20:
            logger.debug("Not enough price history yet")
            return signals

        metadata: Dict[str, Any] = {}
        if self._spot_price_consensus:
            metadata['spot_price'] = float(self._spot_price_consensus)
        if self._sentiment_score is not None:
            metadata['sentiment_score'] = self._sentiment_score
        if self._yes_token_id:
            metadata['yes_token_id'] = self._yes_token_id
        metadata['tick_buffer'] = list(self._tick_buffer)

        price_list = list(self.price_history)

        # Processors with no extra preconditions
        for processor in [
            self.spike_detector,
            self.sentiment_processor,
            self.tick_velocity_processor,
            self.deribit_pcr_processor,
        ]:
            sig = processor.process(self._current_price, price_list, metadata)
            if sig:
                signals.append(sig)

        # Divergence requires a spot price benchmark
        if self._spot_price_consensus:
            sig = self.divergence_processor.process(self._current_price, price_list, metadata)
            if sig:
                signals.append(sig)

        # Order book requires the YES token ID to hit the CLOB API
        if self._yes_token_id:
            sig = self.orderbook_processor.process(self._current_price, price_list, metadata)
            if sig:
                signals.append(sig)

        self._signals_processed += len(signals)
        return signals

    async def _execute_trade(self, signal: FusedSignal) -> None:
        """Delegate trade execution to ExecutionEngine (handles sizing and risk)."""
        if signal.direction == SignalDirection.BULLISH:
            stop_loss = self._current_price * Decimal(str(1 - self.stop_loss_pct))
            take_profit = self._current_price * Decimal(str(1 + self.take_profit_pct))
        else:
            stop_loss = self._current_price * Decimal(str(1 + self.stop_loss_pct))
            take_profit = self._current_price * Decimal(str(1 - self.take_profit_pct))

        signal_sources = [s.source for s in signal.signals]

        order = await self.execution_engine.execute_signal(
            signal_direction=signal.direction,
            signal_confidence=signal.confidence,
            signal_score=signal.score,
            current_price=self._current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={"signal_sources": signal_sources},
        )

        if order:
            self._trades_executed += 1
            logger.info(
                f"Trade executed: {signal.direction.value} "
                f"(score={signal.score:.1f}, sources={signal_sources})"
            )

    async def _on_position_closed(self, position: Dict[str, Any]) -> None:
        """Callback from ExecutionEngine — track cumulative P&L."""
        pnl = position.get("pnl") or Decimal("0")
        self._total_pnl += pnl

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "is_running": self._is_running,
            "signals_processed": self._signals_processed,
            "trades_executed": self._trades_executed,
            "open_positions": len(self.execution_engine.get_open_positions()),
            "total_pnl": float(self._total_pnl),
            "last_decision": self._last_decision_time.isoformat() if self._last_decision_time else None,
            "processors": {
                "spike_detector": self.spike_detector.get_stats(),
                "sentiment": self.sentiment_processor.get_stats(),
                "divergence": self.divergence_processor.get_stats(),
                "orderbook": self.orderbook_processor.get_stats(),
                "tick_velocity": self.tick_velocity_processor.get_stats(),
                "deribit_pcr": self.deribit_pcr_processor.get_stats(),
            },
            "fusion_engine": self.fusion_engine.get_statistics(),
        }


# Singleton instance
_strategy_instance = None

def get_btc_strategy() -> BTCStrategy15Min:
    """Get singleton strategy instance."""
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = BTCStrategy15Min()
    return _strategy_instance
