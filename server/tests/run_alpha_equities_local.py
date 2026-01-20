import os, json
from src.context import set_api_key
from src.tools.alpha_equities import macro_snapshot, symbol_snapshot, discover_equities

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") or ""
if not API_KEY:
    raise SystemExit("Missing ALPHA_VANTAGE_API_KEY")

set_api_key(API_KEY)

print("=== macro_snapshot (defaults: SPY,^VIX,UUP) ===")
res_macro = macro_snapshot(includeSector="false", includeMovers="false")
print(json.dumps(res_macro, indent=2))

print("\n=== symbol_snapshot(AAPL) ===")
res_symbol = symbol_snapshot(symbol="AAPL", outputsize="compact")
print(json.dumps(res_symbol, indent=2))

print("\n=== symbol_snapshot(^VIX) ===")
res_vix = symbol_snapshot(symbol="^VIX", outputsize="compact")
print(json.dumps(res_vix, indent=2))

print("\n=== symbol_snapshot(VIX) ===")
res_vix_alias = symbol_snapshot(symbol="VIX", outputsize="compact")
print(json.dumps(res_vix_alias, indent=2))

print("\n=== discover_equities (small) ===")
res_disc = discover_equities(source="combined", topN="5", resultCount="3", maxSymbols="3", computeVolatility="false", outputsize="compact")
print(json.dumps(res_disc, indent=2))
