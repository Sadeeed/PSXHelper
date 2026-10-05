import json
import unittest
from datetime import date, timedelta
from decimal import Decimal, getcontext, localcontext
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient
from test_data import sample_bundle

from psxhelper.data import DataUnavailable, ResearchService, UnknownSymbol
from psxhelper.main import create_app
from psxhelper.schemas import DailyBar, Fact, FinancialLine, FinancialPeriod


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

    def test_invalid_symbol_is_404_without_provider_call(self):
        self.assertEqual(self.client.get("/api/v1/stocks/DCR%3Fx=1").status_code, 404)
        self.adapter.fetch.assert_not_called()

    def test_provider_unknown_symbol_is_404(self):
        self.adapter.fetch.side_effect = UnknownSymbol("Provider does not recognize ticker")
        self.assertEqual(self.client.get("/api/v1/stocks/DEMO_UNKNOWN").status_code, 404)
        self.adapter.fetch.assert_called_once_with("DEMO_UNKNOWN")

    def test_non_demo_ticker_works_across_api_views_and_preview(self):
        bundle = self.adapter.fetch.return_value
        bundle.symbol = "DEMO_ALT"
        bundle.company_name = "Synthetic alternate company"
        self.assertEqual(self.client.get("/api/v1/stocks/demo_alt").json()["symbol"], "DEMO_ALT")
        for view in ("ohlcv", "metrics"):
            response = self.client.get(f"/api/v1/stocks/DEMO_ALT/{view}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["symbol"], "DEMO_ALT")
        response = self.client.post("/api/v1/allocations/preview",
            json={"symbol": " demo_alt ", "total_cash": "1000"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["symbol"], "DEMO_ALT")
        self.adapter.fetch.assert_called_once_with("DEMO_ALT")

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

    def test_history_market_cap_serializes_decimal_strings_and_missing_values(self):
        bundle = self.adapter.fetch.return_value
        bundle.ohlcv[1].market_cap = Decimal("123456789012345.125")
        data = self.client.get("/api/v1/stocks/DCR/ohlcv").json()
        aggregate = self.client.get("/api/v1/stocks/DCR").json()
        self.assertEqual(data["ohlcv"], aggregate["ohlcv"])
        self.assertIsNone(data["ohlcv"][0]["market_cap"])
        self.assertEqual(data["ohlcv"][1]["market_cap"], "123456789012345.125")
        self.assertEqual(data["market_cap_coverage"], aggregate["coverage"]["market_cap_history"])

    def test_health_and_discovery_do_not_fetch(self):
        self.assertEqual(self.client.get("/health").json()["local_model"], "not connected")
        discovery = self.client.get("/api/v1/tickers").json()
        self.assertEqual(discovery["symbols"], [])
        self.assertEqual(discovery["examples"], ["DCR", "AHCL"])
        self.assertEqual(discovery["lookup_mode"], "provider_validated")
        self.assertEqual(self.client.get("/").status_code, 200)
        self.adapter.fetch.assert_not_called()

    def test_research_page_is_html_and_does_not_fetch_on_server(self):
        for path in ("/", "/stocks/DCR", "/stocks/AHCL", "/stocks/DEMO_ALT", "/stocks/UNKNOWN"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])
                self.assertIn('id="lookup"', response.text)
                self.assertIn('id="symbol" name="symbol" type="text"', response.text)
                self.assertNotIn("['DCR','AHCL'].includes", response.text)
                self.assertIn("any provider-supported ticker", response.text)
                self.assertIn('id="chart"', response.text)
                self.assertIn('id="history-view"', response.text)
                self.assertIn('id="history-position" type="range"', response.text)
                self.assertIn('id="chart-readout"', response.text)
                self.assertIn('class="section-nav"', response.text)
                self.assertIn('id="headline-price"', response.text)
                self.assertIn('id="overview-stats"', response.text)
                self.assertIn('id="snapshot-values"', response.text)
                self.assertIn('id="financial-tabs"', response.text)
                self.assertIn('id="ratio-matrix"', response.text)
                self.assertNotIn('Read the company.<br>Then count the cost.', response.text)
                self.assertIn("market-cap observations leave gaps", response.text)
                self.assertIn("getScreenCTM", response.text)
                self.assertIn('id="financials"', response.text)
                self.assertIn('id="ratios"', response.text)
                self.assertIn("balance sheets are point-in-time", response.text)
                self.assertIn("Provider unit / scale", response.text)
                self.assertIn("PKR (user-assumed scaling)", response.text)
                self.assertIn('id="cash-form"', response.text)
                self.assertIn("/api/v1/allocations/preview", response.text)
                self.assertIn('id="assistant-form"', response.text)
                self.assertIn('id="assistant-period"', response.text)
                self.assertIn('id="assistant-question"', response.text)
                self.assertIn('id="assistant-evidence"', response.text)
                self.assertIn("/api/v1/assistant", response.text)
                self.assertIn("requestVersion !== version || selectedContext", response.text)
                self.assertIn("Qualitative explanations may be incorrect", response.text)
                self.assertIn("zero fees and taxes", response.text)
                self.assertNotIn("innerHTML", response.text)
        self.adapter.fetch.assert_not_called()

    def test_aggregate_exposes_statement_values_and_ratio_facts_without_scaling_claims(self):
        bundle = self.adapter.fetch.return_value
        for category, label, metric, value in (
            ("income", "Revenue", "FF_SALES", "120"),
            ("balance", "Cash", "FF_CASH_GENERIC", "12.5"),
            ("cashflow", "Cash movement", "FF_NET_INC_CF", "-2.5"),
        ):
            fact_id = f"{category}:{metric}:2025-06-30"
            bundle.facts[fact_id] = Fact(fact_id=fact_id, value=Decimal(value),
                unit="million PKR (user assumption)" if category == "income" else "unverified provider unit/scale",
                period="2025-06-30", basis="unknown", source_ref="stock_info",
                quality="Numeric-coerced; original precision not retained")
            setattr(bundle.financials, category, [FinancialPeriod(period_end=date(2025, 6, 30),
                period_kind="annual", basis="unknown", currency="PKR", original_scale="unverified",
                source_ref="stock_info", lines=[FinancialLine(reported_label=label, fact_id=fact_id,
                    decimal_value=Decimal(value) * 1000000 if category == "income" else Decimal(value))])])
        ratio_id = "ratios:FF_EPS_BASIC:2025-06-30"
        bundle.facts[ratio_id] = Fact(fact_id=ratio_id, value=Decimal("3.5"), unit="unverified provider unit/scale",
            period="2025-06-30", basis="unknown", source_ref="stock_info", quality="Provider numeric-coerced value")
        response = self.client.get("/api/v1/stocks/DCR")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["financials"]["income"][0]["lines"][0]["decimal_value"], "120000000")
        self.assertEqual(data["financials"]["balance"][0]["lines"][0]["decimal_value"], "12.5")
        self.assertIsNone(data["financials"]["balance"][0]["period_start"])
        self.assertEqual(data["financials"]["cashflow"][0]["lines"][0]["decimal_value"], "-2.5")
        self.assertEqual(data["facts"][ratio_id]["value"], "3.5")
        self.assertIsNone(data["eps"]["reported_annual"])

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
        self.adapter.fetch.side_effect = UnknownSymbol("Provider does not recognize ticker")
        self.assertEqual(self.client.get("/api/v1/stocks/UNKNOWN/metrics").status_code, 404)
        self.adapter.fetch.side_effect = DataUnavailable("offline")
        self.assertEqual(self.client.get("/api/v1/stocks/AHCL/metrics").status_code, 503)

    def test_cash_preview_whole_units_residual_and_source(self):
        self.adapter.fetch.return_value.quote.price = Decimal("40")
        for cash, quantity, notional, remaining in (
            ("1000", 25, "1000.00", "0.00"), ("1010", 25, "1000.00", "10.00"),
            ("39.99", 0, "0.00", "39.99"), ("0", 0, "0.00", "0.00"),
        ):
            with self.subTest(cash=cash):
                response = self.client.post("/api/v1/allocations/preview",
                    json={"symbol": "dcr", "total_cash": cash})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["quantity"], quantity)
                self.assertEqual(data["notional"], notional)
                self.assertEqual(data["remaining_cash"], remaining)
                self.assertEqual(data["fees"], "0.00")
                self.assertEqual(data["symbol"], "DCR")
                self.assertEqual(data["price"], "40")
                self.assertEqual(data["market_date"], "2026-10-02")
                self.assertEqual(data["bundle_hash"], self.adapter.fetch.return_value.bundle_hash)
                self.assertEqual(data["source_ref"], "stock_info")
                self.assertEqual(data["assumed_lot_size"], 1)
                self.assertTrue(any("Zero-fee" in warning for warning in data["warnings"]))
                self.assertTrue(any("lot size unverified" in warning for warning in data["warnings"]))
        self.adapter.fetch.assert_called_once_with("DCR")

    def test_cash_preview_rounding_affordability_and_exact_conservation(self):
        for price, cash, quantity in (
            ("40.125", "80.25", 2), ("40.125", "40.12", 0), ("40.125", "40.13", 1),
            ("0.3333", "1.00", 3), ("0.004", "0.01", 3),
            ("0.005", "0.01", 2), ("0.004", "0.00", 0),
        ):
            with self.subTest(price=price, cash=cash):
                bundle = sample_bundle()
                bundle.quote.price = Decimal(price)
                with TestClient(create_app(ResearchService(Mock(fetch=Mock(return_value=bundle))))) as client:
                    response = client.post("/api/v1/allocations/preview",
                        json={"symbol": "DCR", "total_cash": cash})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["quantity"], quantity)
                self.assertEqual(Decimal(data["notional"]) + Decimal(data["fees"])
                    + Decimal(data["remaining_cash"]), Decimal(cash))
                self.assertGreaterEqual(Decimal(data["remaining_cash"]), 0)
                self.assertIsInstance(data["total_cash"], str)

    def test_cash_preview_rejects_invalid_cash_before_provider_fetch(self):
        for cash in ("-1", "NaN", "Infinity", "-Infinity", "0.001", "bad", None,
                     "10000000000000000"):
            with self.subTest(cash=cash):
                response = self.client.post("/api/v1/allocations/preview",
                    json={"symbol": "DCR", "total_cash": cash})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/api/v1/allocations/preview", json={"symbol": "DCR"}).status_code, 422)
        self.adapter.fetch.assert_not_called()

    def test_cash_preview_requires_usable_price_and_preserves_errors(self):
        for price, status in ((None, 503), ("0", 422), ("-1", 422), ("NaN", 422), ("Infinity", 422)):
            with self.subTest(price=price):
                bundle = sample_bundle()
                bundle.quote.price = Decimal(price) if price is not None else None
                with TestClient(create_app(ResearchService(Mock(fetch=Mock(return_value=bundle))))) as client:
                    response = client.post("/api/v1/allocations/preview",
                        json={"symbol": "DCR", "total_cash": "1000"})
                self.assertEqual(response.status_code, status)
        self.adapter.fetch.side_effect = UnknownSymbol("Provider does not recognize ticker")
        self.assertEqual(self.client.post("/api/v1/allocations/preview",
            json={"symbol": "UNKNOWN", "total_cash": "1000"}).status_code, 404)
        self.adapter.fetch.assert_called_once_with("UNKNOWN")
        self.adapter.fetch.side_effect = DataUnavailable("offline")
        self.assertEqual(self.client.post("/api/v1/allocations/preview",
            json={"symbol": "DCR", "total_cash": "1000"}).status_code, 503)

    def test_assistant_returns_grounded_evidence_and_local_request_settings(self):
        self.adapter.fetch.return_value.symbol = "DEMO_TEST"
        bundle = self.client.get("/api/v1/stocks/DEMO_TEST").json()
        answer = {"explanation": "Earnings evidence is missing; valuation cannot be established.",
            "fact_refs": ["pe_ttm", "price"], "caveats": ["Accounting basis remains unverified."],
            "next_question": "Would you like to review the dated price observation?"}
        local = Mock(json=Mock(return_value={"done": True, "done_reason": "stop",
            "message": {"content": json.dumps(answer)}}))
        with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock, return_value=local) as post:
            response = self.client.post("/api/v1/assistant", json={"symbol": " demo_test ",
                "bundle_hash": bundle["bundle_hash"], "question": "What does P/E tell us?"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["symbol"], "DEMO_TEST")
        self.assertEqual(result["bundle_hash"], bundle["bundle_hash"])
        self.assertEqual(result["market_date"], "2026-10-02")
        self.assertEqual(result["basis"], "unknown")
        self.assertIsNone(result["period"])
        self.assertEqual(result["answer"], answer)
        self.assertEqual(result["evidence"]["price"]["observation"]["value"], "40.125")
        self.assertIn("stock_info", result["evidence"]["price"]["sources"])
        self.assertIsNone(result["evidence"]["pe_ttm"]["observation"]["value"])
        self.assertIn("last inference succeeded", self.client.get("/health").json()["local_model"])
        post.assert_awaited_once()
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/chat")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gemma4:e4b")
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 400)
        self.assertFalse(payload["format"]["additionalProperties"])
        self.assertEqual(payload["format"]["properties"]["fact_refs"]["maxItems"], 12)
        for field in payload["format"]["properties"].values():
            for constraint in ("minLength", "maxLength"):
                self.assertNotIn(constraint, field)
                self.assertNotIn(constraint, field.get("items", {}))
        context = json.loads(payload["messages"][1]["content"])
        self.assertIsNone(context["evidence"]["eps.reported_ttm"]["value"])
        self.assertIn("Cash preview is a separate", context["limitations"][-1])
        self.adapter.fetch.assert_called_once_with("DEMO_TEST")

    def test_assistant_rejects_invalid_input_without_fetch_or_inference(self):
        valid = {"symbol": "DEMO_TEST", "bundle_hash": "a" * 64, "question": "Explain P/E."}
        with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock) as post:
            for changes in ({"question": " "}, {"question": "x" * 801}, {"symbol": " "},
                            {"bundle_hash": "bad"}, {"basis": "guessed"}, {"period": "bad"},
                            {"price": "100"}):
                with self.subTest(changes=changes):
                    response = self.client.post("/api/v1/assistant", json={**valid, **changes})
                    self.assertEqual(response.status_code, 422)
            post.assert_not_awaited()
        self.adapter.fetch.assert_not_called()

    def test_assistant_rejects_stale_hash_and_invalid_period_or_basis(self):
        self.adapter.fetch.return_value.symbol = "DEMO_TEST"
        bundle = self.client.get("/api/v1/stocks/DEMO_TEST").json()
        valid = {"symbol": "DEMO_TEST", "bundle_hash": bundle["bundle_hash"], "question": "Explain P/E."}
        with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock) as post:
            for changes in ({"bundle_hash": "a" * 64}, {"period": "2025-06-30"},
                            {"basis": "consolidated"}):
                with self.subTest(changes=changes):
                    self.assertEqual(self.client.post("/api/v1/assistant", json={**valid, **changes}).status_code, 409)
            post.assert_not_awaited()

    def test_assistant_uses_selected_period_and_preserves_financial_units(self):
        bundle = self.adapter.fetch.return_value
        bundle.symbol = "DEMO_TEST"
        fact_id = "income:DEMO_REVENUE:2025-06-30"
        bundle.facts[fact_id] = Fact(fact_id=fact_id, value=Decimal("120"),
            unit="million PKR (user assumption)", period="2025-06-30", basis="unknown",
            source_ref="stock_info", quality="Unverified synthetic observation")
        bundle.financials.income = [FinancialPeriod(period_end=date(2025, 6, 30),
            period_kind="annual", basis="unknown", currency="PKR", original_scale="unverified",
            source_ref="stock_info", lines=[FinancialLine(reported_label="Revenue", fact_id=fact_id)])]
        answer = {"explanation": "Reported revenue uses an assumed monetary scale.",
            "fact_refs": [fact_id], "caveats": ["Accounting basis is unknown."],
            "next_question": "Would you like to inspect the source?"}
        local = Mock(json=Mock(return_value={"done": True, "done_reason": "stop",
            "message": {"content": json.dumps(answer)}}))
        with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock, return_value=local) as post:
            response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                "bundle_hash": bundle.bundle_hash, "question": "Explain revenue.",
                "period": "2025-06-30", "basis": "unknown"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["period"], "2025-06-30")
        evidence = response.json()["evidence"][fact_id]["observation"]
        self.assertEqual(evidence["value"], "120")
        self.assertEqual(evidence["unit"], "million PKR (user assumption)")
        self.assertEqual(evidence["basis"], "unknown")
        context = json.loads(post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertIn(fact_id, context["evidence"])

    def test_assistant_repairs_invalid_output_once(self):
        self.adapter.fetch.return_value.symbol = "DEMO_TEST"
        bundle = self.adapter.fetch.return_value
        answer = {"explanation": "The dated price is a provider observation.", "fact_refs": ["price"],
            "caveats": ["Not an executable price."], "next_question": "Review coverage?"}
        invalid = Mock(json=Mock(return_value={"done": True, "done_reason": "stop",
            "message": {"content": "not JSON"}}))
        valid = Mock(json=Mock(return_value={"done": True, "done_reason": "stop",
            "message": {"content": json.dumps(answer)}}))
        with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock,
                   side_effect=[invalid, valid]) as post:
            response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                "bundle_hash": bundle.bundle_hash, "question": "Explain price."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.await_count, 2)
        self.assertEqual(response.json()["answer"], answer)
        self.assertIn("Repair the answer", post.call_args.kwargs["json"]["messages"][-1]["content"])

    def test_assistant_withholds_unknown_refs_numeric_prose_and_truncated_output(self):
        self.adapter.fetch.return_value.symbol = "DEMO_TEST"
        bundle = self.adapter.fetch.return_value
        valid = {"explanation": "The price is a dated observation.", "fact_refs": ["price"],
            "caveats": ["Trading costs are excluded."], "next_question": "Review coverage?"}
        for changes in ({"fact_refs": ["invented"]}, {"fact_refs": []},
                        {"fact_refs": ["price", "price"]}, {"explanation": "The price is 999 PKR."},
                        {"caveats": ["Yield is 10%."]}, {"next_question": "Buy 100 units?"},
                        {"extra": "unsupported"}, {"explanation": "x" * 2001},
                        {"explanation": ""}, {"caveats": ["x" * 401]},
                        {"explanation": "Revenue is six thousand sixty point four million PKR."},
                        {"caveats": ["Yield is TEN percent."]},
                        {"next_question": "Buy twenty-five units?"}):
            with self.subTest(changes=changes):
                local = Mock(json=Mock(return_value={"done": True, "done_reason": "stop",
                    "message": {"content": json.dumps({**valid, **changes})}}))
                with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock, return_value=local) as post:
                    response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                        "bundle_hash": bundle.bundle_hash, "question": "Explain price."})
                self.assertEqual(response.status_code, 502)
                self.assertNotIn("answer", response.json())
                self.assertEqual(post.await_count, 2)
                self.assertFalse(self.client.app.state.assistant_lock.locked())
        for data in ({"done": True, "done_reason": "length", "message": {"content": json.dumps(valid)}},
                     {"done": False}, [], {"done": True, "done_reason": "stop", "message": None}):
            with self.subTest(data=data):
                local = Mock(json=Mock(return_value=data))
                with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock, return_value=local) as post:
                    response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                        "bundle_hash": bundle.bundle_hash, "question": "Explain price."})
                self.assertEqual(response.status_code, 502)
                self.assertEqual(post.await_count, 2)

    def test_assistant_busy_rejects_without_fetch_or_queue(self):
        with patch.object(self.client.app.state, "assistant_lock", Mock(locked=Mock(return_value=True))), \
                patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock) as post:
            response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                "bundle_hash": "a" * 64, "question": "Explain price."})
            self.assertEqual(response.status_code, 429)
            post.assert_not_awaited()
        self.adapter.fetch.assert_not_called()

    def test_assistant_timeout_and_unavailable_preserve_research_and_release_guard(self):
        self.adapter.fetch.return_value.symbol = "DEMO_TEST"
        bundle = self.adapter.fetch.return_value
        for failure, expected in ((httpx.ReadTimeout("timeout"), 504),
                                  (httpx.ConnectError("offline"), 503),
                                  (httpx.HTTPStatusError("missing model", request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
                                      response=httpx.Response(404)), 503)):
            with self.subTest(failure=failure):
                with patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=failure) as post:
                    response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                        "bundle_hash": bundle.bundle_hash, "question": "Explain price."})
                self.assertEqual(response.status_code, expected)
                post.assert_awaited_once()
                self.assertFalse(self.client.app.state.assistant_lock.locked())
                self.assertEqual(self.client.get("/api/v1/stocks/DEMO_TEST").status_code, 200)
        with patch("psxhelper.main.asyncio.timeout", side_effect=TimeoutError), \
                patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock) as post:
            response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                "bundle_hash": bundle.bundle_hash, "question": "Explain price."})
            self.assertEqual(response.status_code, 504)
            post.assert_not_awaited()
        self.assertFalse(self.client.app.state.assistant_lock.locked())

    def test_assistant_rechecks_hash_after_inference(self):
        bundle = self.adapter.fetch.return_value
        bundle.symbol = "DEMO_TEST"
        changed = bundle.model_copy(deep=True)
        changed.bundle_hash = "b" * 64
        answer = {"explanation": "The price is dated.", "fact_refs": ["price"],
            "caveats": [], "next_question": "Review coverage?"}
        local = Mock(json=Mock(return_value={"done": True, "done_reason": "stop",
            "message": {"content": json.dumps(answer)}}))
        with patch.object(self.client.app.state.research, "get", side_effect=[bundle, bundle, changed]), \
                patch("psxhelper.main.httpx.AsyncClient.post", new_callable=AsyncMock, return_value=local):
            response = self.client.post("/api/v1/assistant", json={"symbol": "DEMO_TEST",
                "bundle_hash": bundle.bundle_hash, "question": "Explain price."})
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("answer", response.json())
        self.assertFalse(self.client.app.state.assistant_lock.locked())
