"""
Patches PolymarketExecutionClient._parse_trades_response_object to auto-load
instruments missing from cache (e.g. fills from resolved past markets).
"""
import requests
import msgspec
from loguru import logger

from nautilus_trader.adapters.polymarket.execution import PolymarketExecutionClient
from nautilus_trader.adapters.polymarket.common.gamma_markets import normalize_gamma_market_to_clob_format
from nautilus_trader.adapters.polymarket.common.parsing import parse_polymarket_instrument
from nautilus_trader.adapters.polymarket.common.symbol import get_polymarket_instrument_id

GAMMA_BASE = "https://gamma-api.polymarket.com"

_original_parse = PolymarketExecutionClient._parse_trades_response_object
# market_ids we already tried and failed — skip without HTTP call next time
_failed_market_ids: set = set()


def _patched_parse(self, command, json_obj, parsed_fill_keys, reports):
    raw = msgspec.json.encode(json_obj)
    polymarket_trade = self._decoder_trade_report.decode(raw)
    market_id = polymarket_trade.market

    if market_id not in _failed_market_ids:
        try:
            filled_ids = polymarket_trade.get_filled_user_order_ids(self._wallet_address, self._api_key)
            for oid in filled_ids:
                asset_id = polymarket_trade.get_asset_id(oid)
                instrument_id = get_polymarket_instrument_id(market_id, asset_id)
                if self._cache.instrument(instrument_id) is None:
                    _load_instrument_sync(self, market_id, asset_id, instrument_id)
        except Exception as e:
            logger.warning(f"Trade reconciliation pre-load check failed: {e}")

    return _original_parse(self, command, json_obj, parsed_fill_keys, reports)


def _load_instrument_sync(exec_client, market_id, asset_id, instrument_id):
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"condition_ids": market_id, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            # Market fully purged from Gamma — remember and skip next time
            _failed_market_ids.add(market_id)
            logger.debug(f"Gamma API: market {market_id[:16]}... not found (resolved/purged), skipping future lookups")
            return
        market = data[0] if isinstance(data, list) else data.get("data", [None])[0]
        if not market:
            _failed_market_ids.add(market_id)
            return
        normalized = normalize_gamma_market_to_clob_format(market)
        for token_info in normalized.get("tokens", []):
            if token_info["token_id"] != asset_id:
                continue
            instrument = parse_polymarket_instrument(
                market_info=normalized,
                token_id=asset_id,
                outcome=token_info["outcome"],
                ts_init=exec_client._clock.timestamp_ns(),
            )
            exec_client._cache.add_instrument(instrument)
            logger.info(f"Auto-loaded missing instrument {instrument_id} from Gamma API")
            return
        logger.warning(f"asset_id {asset_id} not found in tokens for market {market_id}")
    except Exception as e:
        logger.error(f"Failed to auto-load instrument {instrument_id}: {e}")


def apply_patch():
    PolymarketExecutionClient._parse_trades_response_object = _patched_parse
    logger.info("Patched _parse_trades_response_object (auto-load missing instruments)")
