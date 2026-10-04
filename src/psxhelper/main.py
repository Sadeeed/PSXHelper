"""Checkpoint 1: aggregate research API. No background provider polling."""

from decimal import Decimal, localcontext

from fastapi import FastAPI, HTTPException

from .data import SUPPORTED_SYMBOLS, DataUnavailable, ResearchService, UnknownSymbol
from .schemas import TickerBundle


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="PSXHelper", version="0.1.0")
    app.state.research = service or ResearchService()

    def get_bundle(symbol):
        try:
            return app.state.research.get(symbol)
        except UnknownSymbol as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/")
    def index():
        return {"checkpoint": "1: aggregate data API", "docs": "/docs",
                "notice": "Partial provider observations; rights unverified. Research UI not implemented."}

    @app.get("/health")
    def health():
        return {"status": "ok", "data_mode": "remote_unverified",
                "provider_status": "not probed by health endpoint",
                "local_model": "not connected"}

    @app.get("/api/v1/tickers")
    def tickers():
        return {"symbols": list(SUPPORTED_SYMBOLS), "coverage": "Quote and close/volume adapter only; availability varies by ticker."}

    # requests/pandas are blocking: regular def routes use FastAPI's thread pool.
    @app.get("/api/v1/stocks/{symbol}", response_model=TickerBundle)
    def stock(symbol: str):
        return get_bundle(symbol)

    @app.get("/api/v1/stocks/{symbol}/ohlcv")
    def history(symbol: str):
        bundle = get_bundle(symbol)
        return {"symbol": bundle.symbol, "bundle_hash": bundle.bundle_hash,
                "ohlcv": [bar.model_dump(mode="json") for bar in bundle.ohlcv],
                "coverage": bundle.coverage["history"].model_dump(),
                "source": bundle.source_refs["price_history"].model_dump(mode="json")}

    @app.get("/api/v1/stocks/{symbol}/metrics")
    def metrics(symbol: str):
        bundle = get_bundle(symbol)
        results = {}
        with localcontext() as context:
            context.prec = 28
            for key, value, unit in (
                ("price", bundle.quote.price, f"{bundle.currency} per unit"),
                ("volume", bundle.quote.volume, "provider trading units"),
            ):
                results[key] = {
                    "value": str(value) if value is not None else None,
                    "state": "available" if value is not None else "missing",
                    "label": f"Provider {bundle.quote.price_kind} {key}",
                    "unit": unit, "formula": "sourced observation",
                    "input_refs": [f"quote.{key}"],
                    "market_date": str(bundle.quote.market_date) if bundle.quote.market_date else None,
                    "source_ref": bundle.quote.source_ref,
                }
            for key, eps_key, label in (
                ("pe_ttm", "reported_ttm", "P/E TTM"),
                ("pe_annual", "reported_annual", "P/E annual"),
            ):
                eps = bundle.eps.get(eps_key)
                price = bundle.quote.price
                state, value, warnings = "missing", None, []
                if eps is None or price is None:
                    warnings.append("Price or source-reported EPS unavailable; no annual/TTM substitution.")
                elif price <= 0 or eps.value <= 0:
                    state = "not_meaningful"
                    warnings.append("Price and EPS must both be positive.")
                elif (eps.basis not in {"consolidated", "unconsolidated"}
                      or not eps.period or not eps.period.strip()
                      or eps.unit != f"{bundle.currency} per unit"
                      or eps.source_ref not in bundle.source_refs):
                    warnings.append("EPS requires an identified period, accounting basis, compatible unit and source.")
                else:
                    state, value = "available", str(price / eps.value)
                results[key] = {
                    "value": value, "state": state, "label": label,
                    "unit": "ratio", "formula": f"quote.price / eps.{eps_key}.value",
                    "input_refs": ["quote.price", f"eps.{eps_key}"],
                    "period": eps.period if eps else None,
                    "basis": eps.basis if eps else None,
                    "source_ref": eps.source_ref if eps else None,
                    "warnings": warnings,
                }
            bars = [bar for bar in bundle.ohlcv
                    if bundle.quote.market_date is None or bar.date <= bundle.quote.market_date][-20:]
            observed = [bar for bar in bars if bar.volume is not None]
            results["average_volume"] = {
                "value": str(sum((bar.volume for bar in observed), Decimal(0)) / len(observed)) if observed else None,
                "state": "partial" if observed else "missing",
                "label": f"Average volume across {len(observed)} observed sessions",
                "unit": "provider trading units", "formula": "sum(observed volumes) / observed count",
                "input_refs": [f"ohlcv[{bar.date}].volume" for bar in observed],
                "observed_count": len(observed), "window_bar_count": len(bars),
                "warnings": ["Market calendar and session completeness unverified; not a full 20-session average."],
            }
        return {"symbol": bundle.symbol, "bundle_hash": bundle.bundle_hash, "metrics": results}

    return app


app = create_app()
