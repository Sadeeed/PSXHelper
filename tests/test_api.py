import unittest
from datetime import date, timedelta
from decimal import Decimal, getcontext, localcontext
from unittest.mock import Mock

from fastapi.testclient import TestClient
from test_data import sample_bundle

from psxhelper.data import DataUnavailable, ResearchService
from psxhelper.main import create_app
from psxhelper.schemas import DailyBar, Fact


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.adapter = Mock()
        self.adapter.fetch.return_value = sample_bundle()
        self.client = TestClient(create_app(ResearchService(self.adapter)))
        self.addCleanup(self.client.close)

    def test_supported_symbol_returns_aggregate_decimal_strings(self):
        response = self.client.get("/api/v1/stocks/dcr")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "DCR")
        self.assertEqual(data["quote"]["price"], "40.125")
        self.assertIn("manifest", data)
        self.assertEqual(data["financials"], {"income": [], "balance": [], "cashflow": []})

    def test_unknown_symbol_is_404_without_provider_call(self):
        self.assertEqual(self.client.get("/api/v1/stocks/UNKNOWN").status_code, 404)
        self.adapter.fetch.assert_not_called()

    def test_provider_failure_is_503(self):
        self.adapter.fetch.side_effect = DataUnavailable("unavailable")
        self.assertEqual(self.client.get("/api/v1/stocks/DCR").status_code, 503)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_history_uses_same_bundle_and_hash(self):
        bundle = self.client.get("/api/v1/stocks/DCR").json()
        bars = self.client.get("/api/v1/stocks/DCR/ohlcv").json()
        self.assertEqual(bars["bundle_hash"], bundle["bundle_hash"])
        self.assertEqual(bars["ohlcv"], bundle["ohlcv"])
        self.adapter.fetch.assert_called_once()

    def test_health_and_discovery_do_not_fetch(self):
        self.assertEqual(self.client.get("/health").json()["local_model"], "not connected")
        self.assertEqual(self.client.get("/api/v1/tickers").json()["symbols"], ["DCR", "AHCL"])
        self.assertEqual(self.client.get("/").status_code, 200)
        self.adapter.fetch.assert_not_called()

    def test_metrics_preserve_source_observations_and_partial_volume(self):
        data = self.client.get("/api/v1/stocks/DCR/metrics").json()
        self.assertEqual(data["bundle_hash"], self.adapter.fetch.return_value.bundle_hash)
        results = data["metrics"]
        self.assertEqual(results["price"]["value"], "40.125")
        self.assertEqual(results["price"]["market_date"], "2026-10-02")
        self.assertEqual(results["volume"]["value"], "100")
        self.assertEqual(results["pe_ttm"]["state"], "missing")
        self.assertIsNone(results["pe_ttm"]["value"])
        self.assertIsInstance(results["average_volume"]["value"], str)
        self.assertEqual(Decimal(results["average_volume"]["value"]), Decimal(100))
        self.assertEqual(results["average_volume"]["observed_count"], 1)
        self.assertEqual(results["average_volume"]["window_bar_count"], 2)
        self.assertEqual(results["average_volume"]["state"], "partial")
        self.assertEqual(results["average_volume"]["input_refs"], ["ohlcv[2026-10-02].volume"])

    def test_pe_annual_is_not_substituted_for_ttm(self):
        bundle = self.adapter.fetch.return_value
        bundle.quote.price = Decimal(40)
        bundle.eps["reported_annual"] = Fact(fact_id="eps_annual", value=Decimal(4),
            unit="PKR per unit", period="2025-07-01/2026-06-30", basis="consolidated",
            source_ref="stock_info", quality="source_reported")
        results = self.client.get("/api/v1/stocks/DCR/metrics").json()["metrics"]
        self.assertEqual(results["pe_annual"]["value"], "10")
        self.assertEqual(results["pe_annual"]["label"], "P/E annual")
        self.assertEqual(results["pe_annual"]["input_refs"], ["quote.price", "eps.reported_annual"])
        self.assertEqual(results["pe_ttm"]["state"], "missing")

    def test_pe_ttm_requires_positive_inputs_and_verified_metadata(self):
        for value, basis, period, unit, source, expected in (
            ("4", "consolidated", "2026-09-30", "PKR per unit", "stock_info", "available"),
            ("0", "consolidated", "2026-09-30", "PKR per unit", "stock_info", "not_meaningful"),
            ("-4", "consolidated", "2026-09-30", "PKR per unit", "stock_info", "not_meaningful"),
            ("4", "unknown", "2026-09-30", "PKR per unit", "stock_info", "missing"),
            ("4", "consolidated", None, "PKR per unit", "stock_info", "missing"),
            ("4", "consolidated", "2026-09-30", "USD per unit", "stock_info", "missing"),
            ("4", "consolidated", "2026-09-30", "PKR per unit", "unknown", "missing"),
        ):
            with self.subTest(value=value, basis=basis, period=period, unit=unit, source=source):
                bundle = sample_bundle()
                bundle.quote.price = Decimal(40)
                bundle.eps["reported_ttm"] = Fact(fact_id="eps_ttm", value=Decimal(value),
                    unit=unit, period=period, basis=basis, source_ref=source, quality="source_reported")
                with TestClient(create_app(ResearchService(Mock(fetch=Mock(return_value=bundle))))) as client:
                    metric = client.get("/api/v1/stocks/DCR/metrics").json()["metrics"]["pe_ttm"]
                self.assertEqual(metric["state"], expected)
                self.assertEqual(metric["value"], "10" if expected == "available" else None)

    def test_metrics_use_28_digits_without_mutating_decimal_context(self):
        bundle = self.adapter.fetch.return_value
        bundle.quote.price = Decimal(40)
        bundle.eps["reported_ttm"] = Fact(fact_id="eps_ttm", value=Decimal(3),
            unit="PKR per unit", period="2026-09-30", basis="consolidated",
            source_ref="stock_info", quality="source_reported")
        with localcontext() as context:
            context.prec = 6
            results = self.client.get("/api/v1/stocks/DCR/metrics").json()["metrics"]
            self.assertEqual(getcontext().prec, 6)
        self.assertEqual(results["pe_ttm"]["value"], "13.33333333333333333333333333")

    def test_volume_window_excludes_future_bars_and_keeps_missing_as_missing(self):
        bundle = self.adapter.fetch.return_value
        bundle.ohlcv = [DailyBar(date=date(2026, 9, 1) + timedelta(days=i),
            close=Decimal(40), volume=Decimal(i)) for i in range(33)]
        results = self.client.get("/api/v1/stocks/DCR/metrics").json()["metrics"]
        self.assertEqual(results["average_volume"]["observed_count"], 20)
        self.assertEqual(results["average_volume"]["value"], "21.5")
        self.assertEqual(results["average_volume"]["state"], "partial")
        bundle = sample_bundle()
        bundle.quote.price = None
        bundle.quote.volume = None
        for bar in bundle.ohlcv:
            bar.volume = None
        with TestClient(create_app(ResearchService(Mock(fetch=Mock(return_value=bundle))))) as client:
            results = client.get("/api/v1/stocks/DCR/metrics").json()["metrics"]
        for key in ("price", "volume", "average_volume"):
            self.assertEqual(results[key]["state"], "missing")
            self.assertIsNone(results[key]["value"])

    def test_metrics_reuse_cache_and_propagate_404_and_503(self):
        self.client.get("/api/v1/stocks/DCR")
        self.assertEqual(self.client.get("/api/v1/stocks/DCR/metrics").status_code, 200)
        self.adapter.fetch.assert_called_once_with("DCR")
        self.assertEqual(self.client.get("/api/v1/stocks/UNKNOWN/metrics").status_code, 404)
        self.adapter.fetch.side_effect = DataUnavailable("offline")
        self.assertEqual(self.client.get("/api/v1/stocks/AHCL/metrics").status_code, 503)
