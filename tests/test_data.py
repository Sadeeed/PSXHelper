"""Original synthetic inputs only; no stored real-market fixtures."""

import unittest
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import requests

from psxhelper.data import (
    DataUnavailable,
    ResearchService,
    SarmaayaAdapter,
    UnknownSymbol,
    content_hash,
)


def sample_bundle():
    # DCR is used only to test routing/mapping, not published as demo market data.
    return SarmaayaAdapter(client_factory=Mock()).normalize("DCR", {
        "symbol": "DCR", "name": "Synthetic test company", "close": "40.125",
        "volume": 100, "date": "2026-10-02T19:00:00Z", "sectorName": "Test sector",
    }, pd.DataFrame([
        {"date": "2026-10-02", "price": "40.125", "volume": 100},
        {"date": "2026-10-01", "price": "39", "volume": None},
    ]))


class TestData(unittest.TestCase):
    def test_decimal_strings_missing_values_and_provenance(self):
        bundle = sample_bundle()
        data = bundle.model_dump(mode="json")
        self.assertEqual(data["quote"]["price"], "40.125")
        self.assertIsNone(data["shares"]["total_outstanding"])
        self.assertIsNone(data["eps"]["reported_ttm"])
        self.assertIsNone(data["ohlcv"][0]["open"])
        self.assertIsNone(data["ohlcv"][0]["volume"])
        self.assertEqual(data["ohlcv"][0]["date"], "2026-10-01")
        self.assertEqual(bundle.facts["price"].value, Decimal("40.125"))
        self.assertEqual(bundle.manifest.content_hash, bundle.bundle_hash)
        self.assertEqual(bundle.data_mode, "remote_unverified")
        self.assertEqual(bundle.coverage["dividends"].status, "missing")

    def test_hash_stable_across_acquisition_times(self):
        first, second = sample_bundle(), sample_bundle()
        self.assertNotEqual(first.fetched_at, second.fetched_at)
        self.assertEqual(first.bundle_hash, second.bundle_hash)
        second.quote.price = Decimal(41)
        self.assertNotEqual(content_hash(second), first.bundle_hash)

    def test_provider_market_cap_history_preserves_values_dates_and_source(self):
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DEMO_ALT",
            {"symbol": "DEMO_ALT", "name": "Synthetic company"}, pd.DataFrame([
                {"date": "2026-10-02", "price": "4", "marketCap": "123456789012345.125"},
                {"date": "2026-10-01", "price": "3", "marketCap": "0"},
                {"date": "2026-09-30", "price": "2", "marketCap": None},
            ]))
        bars = bundle.model_dump(mode="json")["ohlcv"]
        self.assertIsNone(bars[0]["market_cap"])
        self.assertEqual(bars[1]["market_cap"], "0")
        self.assertEqual(bars[2]["market_cap"], "123456789012345.125")
        self.assertEqual(bars[2]["source_ref"], "price_history")
        self.assertEqual(bundle.coverage["market_cap_history"].status, "partial")
        self.assertIn("2 of 3", bundle.coverage["market_cap_history"].explanation)
        self.assertIn("Unit/scale and share basis unverified", bundle.coverage["market_cap_history"].explanation)
        self.assertIsNone(bundle.shares.total_outstanding)
        bundle.ohlcv[2].market_cap = Decimal("123")
        self.assertNotEqual(content_hash(bundle), bundle.bundle_hash)

    def test_invalid_optional_market_cap_does_not_discard_price_history(self):
        for value in ("-1", "Infinity", "-Infinity", "invalid"):
            with self.subTest(value=value):
                bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DEMO_ALT",
                    {"symbol": "DEMO_ALT", "name": "Synthetic company"},
                    pd.DataFrame([{"date": "2026-10-02", "price": "4", "marketCap": value}]))
                self.assertIsNone(bundle.ohlcv[0].market_cap)
                self.assertEqual(bundle.ohlcv[0].close, Decimal(4))
                self.assertEqual(bundle.coverage["market_cap_history"].status, "missing")
                self.assertIn("1 invalid values withheld", bundle.coverage["market_cap_history"].explanation)
                self.assertTrue(any("valid price/volume observations retained" in warning for warning in bundle.warnings))

    def test_absent_market_cap_stays_missing_without_share_reconstruction(self):
        bundle = sample_bundle()
        self.assertTrue(all(bar.market_cap is None for bar in bundle.ohlcv))
        self.assertEqual(bundle.coverage["market_cap_history"].status, "missing")
        self.assertIn("not reconstructed", bundle.coverage["market_cap_history"].explanation)

    def test_bad_history_and_symbol_mismatch_rejected(self):
        adapter = SarmaayaAdapter(client_factory=Mock())
        for history in [
            [{"date": "2026-10-01", "price": 1}, {"date": "2026-10-01", "price": 2}],
            [{"date": "invalid", "price": 1}],
            [{"date": "2026-10-01", "price": -1}],
            [{"date": "2026-10-01", "price": "Infinity"}],
            [{"date": "2026-10-01", "price": None}],
            [{"date": "2026-10-01", "price": 1, "symbol": "AHCL"}],
        ]:
            with self.subTest(history=history), self.assertRaises(ValueError):
                adapter.normalize("DCR", {"symbol": "DCR", "name": "Test"}, pd.DataFrame(history))
        with self.assertRaises(ValueError):
            adapter.normalize("DCR", {"symbol": "AHCL", "name": "Test"}, pd.DataFrame())

    def test_missing_quote_stays_missing(self):
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize(
            "DCR", {"symbol": "DCR", "name": "Test"}, pd.DataFrame())
        self.assertIsNone(bundle.quote.price)
        self.assertEqual(bundle.coverage["quote"].status, "missing")
        self.assertEqual(bundle.coverage["history"].status, "missing")
        self.assertNotIn("price", bundle.facts)

    def test_existing_client_reused_and_session_closed(self):
        client = Mock()
        client.get_stock_info.return_value = {"symbol": "DCR", "name": "Test"}
        client.get_price_history.return_value = pd.DataFrame()
        client.get_company_snapshot.return_value = []
        client.get_income_statement_df.return_value = pd.DataFrame()
        client.get_balance_sheet_df.return_value = pd.DataFrame()
        client.get_cash_flow_df.return_value = pd.DataFrame()
        client.get_financial_ratios_df.return_value = pd.DataFrame()
        client.get_dividends.return_value = pd.DataFrame()
        adapter = SarmaayaAdapter(client_factory=lambda: client)
        self.assertEqual(adapter.fetch(" dcr ").symbol, "DCR")
        client.get_stock_info.assert_called_once_with("DCR")
        client.get_price_history.assert_called_once_with("DCR", days=365)
        client.get_company_snapshot.assert_called_once_with("DCR")
        client.get_income_statement_df.assert_called_once_with("DCR")
        client.get_balance_sheet_df.assert_called_once_with("DCR")
        client.get_cash_flow_df.assert_called_once_with("DCR")
        client.get_financial_ratios_df.assert_called_once_with("DCR")
        client.get_income_statement.assert_not_called()
        client.get_dividends.assert_called_once_with("DCR")
        client.session.close.assert_called_once()

    def test_snapshot_share_counts_do_not_invent_dates_or_eps(self):
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), snapshot=[
                {"isin": "PKTEST", "metricMetric": "FF_COM_SHS_OUT", "afValue": "1000000", "afFiscalEndDate": "null"},
                {"isin": "PKTEST", "metricMetric": "FF_SHS_FLOAT", "afValue": "250000"},
                {"isin": "PKTEST", "metricMetric": "FF_EPS", "afValue": "4"},
                {"isin": "PKTEST", "metricMetric": "FF_DIV_YLD", "afValue": "5"},
            ])
        self.assertEqual(bundle.shares.total_outstanding, Decimal(1000000))
        self.assertEqual(bundle.shares.free_float, Decimal(250000))
        self.assertIsNone(bundle.shares.effective_date)
        self.assertIsNone(bundle.eps["reported_annual"])
        self.assertIsNone(bundle.eps["reported_ttm"])
        self.assertEqual(bundle.dividends, [])
        self.assertEqual(bundle.coverage["shares"].status, "partial")
        self.assertEqual(bundle.facts["total_outstanding"].source_ref, "company_snapshot")

    def test_income_uses_user_scale_preserving_coerced_values_and_unknown_basis(self):
        income = pd.DataFrame([
            {"statement_type": "income_statement", "name": "Revenue", "metric": "FF_SALES",
             "display_type": "wholeNumber", "period_end": "2025-06-30", "periodicity": "ANN", "value": 120},
            {"statement_type": "income_statement", "name": "Revenue", "metric": "FF_SALES",
             "display_type": "wholeNumber", "period_end": "2024-06-30", "periodicity": "ANN", "value": None},
            {"statement_type": "income_statement", "name": "Basic EPS", "metric": "FF_EPS_BASIC",
             "display_type": "float", "period_end": "2025-06-30", "periodicity": "ANN", "value": 4},
        ])
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), income=income)
        self.assertEqual(len(bundle.financials.income), 2)
        period = bundle.financials.income[0]
        self.assertEqual(str(period.period_end), "2025-06-30")
        self.assertEqual(str(period.period_start), "2024-07-01")
        self.assertIsNone(bundle.financials.income[1].period_start)
        self.assertEqual(period.currency, "PKR")
        self.assertIn("user assumption", period.original_scale)
        self.assertEqual(period.basis, "unknown")
        self.assertEqual(period.lines[0].decimal_value, Decimal(120000000))
        self.assertIsNone(period.lines[1].decimal_value)
        self.assertIsNone(bundle.financials.income[1].lines[0].decimal_value)
        self.assertEqual(period.lines[0].canonical_key, "revenue")
        self.assertEqual(bundle.facts["income:FF_SALES:2025-06-30"].value, Decimal(120))
        self.assertEqual(bundle.facts["income:FF_SALES:2025-06-30"].unit, "million PKR (user assumption)")
        self.assertNotIn("income:FF_SALES:2024-06-30", bundle.facts)
        self.assertIsNone(bundle.eps["reported_annual"])
        self.assertEqual(bundle.financials.balance, [])
        self.assertEqual(bundle.financials.cashflow, [])
        self.assertEqual(bundle.coverage["income"].status, "partial")
        self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))
        self.assertEqual(Decimal(bundle.model_dump(mode="json")["facts"]["income:FF_SALES:2025-06-30"]["value"]), Decimal(120))
        self.assertIn("original strings/precision not retained", bundle.facts["income:FF_SALES:2025-06-30"].quality)

    def test_income_start_requires_adjacent_comparable_annual_end(self):
        for dates, expected in (
            (["2025-06-30", "2024-06-30"], "2024-07-01"),
            (["2025-06-30", "2023-06-30"], None),
            (["2025-06-30", "2024-12-31"], None),
            (["2025-06-30"], None),
        ):
            with self.subTest(dates=dates):
                income = pd.DataFrame([
                    {"statement_type": "income_statement", "name": "Revenue", "metric": "FF_SALES",
                     "display_type": "wholeNumber", "period_end": end, "periodicity": "ANN", "value": 120}
                    for end in reversed(dates)])
                bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
                    {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), income=income)
                start = bundle.financials.income[0].period_start
                self.assertEqual(str(start) if start else None, expected)
                self.assertTrue(all(period.basis == "unknown" for period in bundle.financials.income))
                if expected:
                    self.assertTrue(any("derived from previous annual end" in warning for warning in bundle.warnings))
                self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))

    def test_unloaded_statement_categories_remain_missing_not_zero(self):
        bundle = sample_bundle()
        for category in ("balance", "cashflow"):
            self.assertEqual(getattr(bundle.financials, category), [])
            self.assertEqual(bundle.coverage[category].status, "missing")
            self.assertIn("not loaded", bundle.coverage[category].explanation)
            self.assertIn("not", bundle.coverage[category].explanation)

    def test_balance_cashflow_and_ratios_use_tidy_values_without_guessed_scale(self):
        info = {"symbol": "DEMO_ALT", "name": "Synthetic company", "isin": "PKTEST"}
        balance = pd.DataFrame([{"statement_type": "balance_sheet", "name": "Cash",
            "metric": "FF_CASH_GENERIC", "display_type": "wholeNumber", "period_end": end,
            "periodicity": "ANN", "value": 12.5, "year": 2025, "fiscal_period": 0}
            for end in ("2025-06-30", "2024-06-30")])
        cashflow = pd.DataFrame([{"statement_type": "cash_flow", "name": "Cash movement",
            "metric": "FF_NET_INC_CF", "display_type": "wholeNumber", "period_end": end,
            "periodicity": "ANN", "value": -2.5}
            for end in ("2025-06-30", "2024-06-30")])
        ratios = pd.DataFrame([{"statement_type": "ratios", "name": "Basic EPS",
            "metric": "FF_EPS_BASIC", "display_type": "float", "period_end": "2025-06-30",
            "periodicity": "ANN", "value": 3.5}])
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DEMO_ALT", info, pd.DataFrame(),
            balance=balance, cashflow=cashflow, ratios=ratios)
        self.assertEqual(len(bundle.financials.balance), 2)
        self.assertTrue(all(period.period_start is None for period in bundle.financials.balance))
        self.assertEqual(bundle.financials.balance[0].lines[0].decimal_value, Decimal("12.5"))
        self.assertEqual(str(bundle.financials.cashflow[0].period_start), "2024-07-01")
        self.assertEqual(bundle.financials.cashflow[0].lines[0].decimal_value, Decimal("-2.5"))
        self.assertIn("unverified provider units", bundle.financials.balance[0].original_scale)
        self.assertEqual(bundle.financials.balance[0].basis, "unknown")
        self.assertIn("point-in-time", bundle.coverage["balance"].explanation)
        for category in ("balance", "cashflow", "ratios"):
            self.assertEqual(bundle.coverage[category].status, "partial")
            self.assertIn(f"{category}_annual", bundle.source_refs)
        fact = bundle.facts["ratios:FF_EPS_BASIC:2025-06-30"]
        self.assertEqual(fact.value, Decimal("3.5"))
        self.assertIn("unverified provider unit/scale", fact.unit)
        self.assertIsNone(bundle.eps["reported_annual"])
        self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))
        self.assertEqual(bundle.model_dump(mode="json")["financials"]["balance"][0]["lines"][0]["decimal_value"], "12.5")

    def test_fundamentals_categories_fail_independently_without_accepting_partial_invalid_data(self):
        valid = {"statement_type": "balance_sheet", "name": "Cash", "metric": "FF_CASH_GENERIC",
            "display_type": "wholeNumber", "period_end": "2025-06-30", "periodicity": "ANN", "value": 5}
        for invalid in ({**valid, "value": float("inf")}, {**valid, "period_end": pd.NaT},
                        {**valid, "statement_type": "income_statement"}, {**valid, "periodicity": "LTM"}, valid):
            with self.subTest(invalid=invalid):
                bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DEMO_ALT",
                    {"symbol": "DEMO_ALT", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(),
                    balance=pd.DataFrame([valid, invalid]),
                    cashflow=pd.DataFrame([{**valid, "statement_type": "cash_flow"}]))
                self.assertEqual(bundle.financials.balance, [])
                self.assertEqual(bundle.coverage["balance"].status, "missing")
                self.assertNotIn("balance_annual", bundle.source_refs)
                self.assertFalse(any(key.startswith("balance:") for key in bundle.facts))
                self.assertEqual(len(bundle.financials.cashflow), 1)

    def test_fundamentals_missing_numeric_values_remain_missing(self):
        balance = pd.DataFrame([{"statement_type": "balance_sheet", "name": "Cash", "metric": "FF_CASH_GENERIC",
            "display_type": "wholeNumber", "period_end": "2025-06-30", "periodicity": "ANN", "value": float("nan")}])
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DEMO_ALT",
            {"symbol": "DEMO_ALT", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), balance=balance)
        self.assertIsNone(bundle.financials.balance[0].lines[0].decimal_value)
        self.assertNotIn("balance:FF_CASH_GENERIC:2025-06-30", bundle.facts)

    def test_dividend_announcements_are_deduplicated_references_not_events(self):
        row = {"symbol": "DCR", "sysAnnouncementId": "TEST-1",
               "announcementTitle": "Declaration of interim dividend", "postingDate": "2025-10-10"}
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test"}, pd.DataFrame(), announcements=pd.DataFrame([row, row]))
        self.assertEqual(bundle.dividends, [])
        self.assertEqual(bundle.coverage["dividends"].status, "missing")
        self.assertIn("1 distinct", bundle.coverage["dividends"].explanation)
        self.assertIn("not ex/declaration dates", bundle.coverage["dividends"].explanation)
        self.assertIn("dividend_announcement:TEST-1", bundle.source_refs)
        self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))

    def test_dividend_history_maps_reported_cash_without_inventing_exdates_or_status(self):
        row = {"symbol": "DCR", "announcementDate": "2026-07-01", "payoutType": "dividend",
               "dividendPerShare": "2", "faceValue": "10", "percentage": "20"}
        placeholder = {**row, "payoutType": "type", "announcementDate": "0000-00-00"}
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Synthetic test company"}, pd.DataFrame(),
            dividends=pd.DataFrame([row, row, placeholder]))
        self.assertEqual(len(bundle.dividends), 1)
        event = bundle.dividends[0]
        self.assertEqual(event["cash_per_unit"], "2")
        self.assertEqual(event["announcement_date"], "2026-07-01")
        self.assertIsNone(event["ex_date"])
        self.assertIsNone(event["declaration_date"])
        self.assertEqual(event["status"], "unknown")
        self.assertFalse(event["eligible_for_trailing_dps"])
        self.assertEqual(bundle.coverage["dividends"].status, "partial")
        self.assertEqual(len(event["reported_observations"]), 1)
        self.assertEqual(bundle.facts["dividend:DCR:2026-07-01:cash_per_unit"].value, Decimal("2"))
        self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))

    def test_dividend_summaries_serialize_years_decimal_values_and_provenance(self):
        history = pd.DataFrame()
        history.attrs["dividendsPerShare"] = [{"2025": "0.00", "2026": "2.50"}]
        history.attrs["dividendsYield"] = [{"2026": "5.25%"}]
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Synthetic test company"}, pd.DataFrame(), dividends=history)
        data = bundle.model_dump(mode="json")
        summaries = data["dividend_summaries"]
        self.assertEqual(summaries["dps"]["2025"]["value"], "0.00")
        self.assertEqual(summaries["dps"]["2026"]["value"], "2.50")
        self.assertEqual(summaries["yield"]["2026"]["value"], "5.25")
        self.assertEqual(summaries["yield"]["2026"]["unit"], "percent")
        self.assertEqual(summaries["dps"]["2026"]["unit"], "PKR per unit")
        self.assertEqual(summaries["dps"]["2026"]["basis"], "provider_reported_by_year")
        self.assertEqual(summaries["dps"]["2026"], data["facts"]["dividend_summary:dps:2026"])
        self.assertIn("5.25%", summaries["yield"]["2026"]["quality"])
        self.assertIn("dividend_summaries", bundle.source_refs)
        self.assertEqual(bundle.coverage["dividend_summaries"].status, "partial")
        self.assertEqual(bundle.dividends, [])
        self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))
        history.attrs["dividendsPerShare"] = [{"2026": "3"}]
        changed = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Synthetic test company"}, pd.DataFrame(), dividends=history)
        self.assertNotEqual(changed.bundle_hash, bundle.bundle_hash)

    def test_dividend_summary_failure_does_not_discard_other_summary_or_payouts(self):
        history = pd.DataFrame([{"symbol": "DCR", "announcementDate": "2026-07-01",
            "payoutType": "dividend", "dividendPerShare": "2"}])
        history.attrs["dividendsPerShare"] = [{"2026": "2"}]
        for invalid in ([{"2026": "Infinity"}], [{"2026": "-1"}], [{"bad": "2"}],
                        [{"2026": "5%"}, {"2026": "6%"}], "5%"):
            with self.subTest(invalid=invalid):
                history.attrs["dividendsYield"] = invalid
                bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
                    {"symbol": "DCR", "name": "Test"}, pd.DataFrame(), dividends=history)
                self.assertEqual(bundle.dividend_summaries["yield"], {})
                self.assertEqual(bundle.dividend_summaries["dps"]["2026"].value, Decimal("2"))
                self.assertEqual(len(bundle.dividends), 1)
                self.assertTrue(any("yearly yield summary failed" in warning for warning in bundle.warnings))

    def test_missing_dividend_summaries_remain_missing_not_zero(self):
        bundle = sample_bundle()
        self.assertEqual(bundle.dividend_summaries, {"dps": {}, "yield": {}})
        self.assertEqual(bundle.coverage["dividend_summaries"].status, "missing")

    def test_conflicting_dividend_values_are_withheld_not_corrected(self):
        row = {"symbol": "DCR", "announcementDate": "2026-07-01", "payoutType": "dividend",
               "dividendPerShare": "100", "faceValue": "1", "percentage": "100"}
        adapter = SarmaayaAdapter(client_factory=Mock())
        bundle = adapter.normalize("DCR", {"symbol": "DCR", "name": "Test"}, pd.DataFrame(), dividends=pd.DataFrame([row]))
        self.assertIsNone(bundle.dividends[0]["cash_per_unit"])
        self.assertEqual(bundle.dividends[0]["reported_observations"][0]["dividend_per_share"], "100")
        self.assertTrue(any("conflicts" in warning for warning in bundle.dividends[0]["warnings"]))
        row = {**row, "dividendPerShare": "1"}
        bundle = adapter.normalize("DCR", {"symbol": "DCR", "name": "Test"}, pd.DataFrame(),
            dividends=pd.DataFrame([row, {**row, "dividendPerShare": "2", "percentage": "200"}]))
        self.assertEqual(len(bundle.dividends), 1)
        self.assertIsNone(bundle.dividends[0]["cash_per_unit"])
        self.assertEqual(len(bundle.dividends[0]["reported_observations"]), 2)
        self.assertNotIn("dividend:DCR:2026-07-01:cash_per_unit", bundle.facts)

    def test_dividend_history_invalid_and_future_records_are_not_trailing_inputs(self):
        row = {"symbol": "DCR", "announcementDate": "2026-10-20", "payoutType": "dividend",
               "dividendPerShare": "1.25", "faceValue": "1", "percentage": "125"}
        adapter = SarmaayaAdapter(client_factory=Mock())
        info = {"symbol": "DCR", "name": "Test", "date": "2026-10-02T19:00:00Z"}
        bundle = adapter.normalize("DCR", info, pd.DataFrame(), dividends=pd.DataFrame([row]))
        self.assertFalse(bundle.dividends[0]["eligible_for_trailing_dps"])
        self.assertTrue(any("after selected quote" in warning for warning in bundle.dividends[0]["warnings"]))
        for bad in ({**row, "symbol": "AHCL"}, {**row, "dividendPerShare": "-1"}, {**row, "dividendPerShare": "Infinity"}):
            with self.subTest(bad=bad):
                bundle = adapter.normalize("DCR", info, pd.DataFrame(), dividends=pd.DataFrame([bad]))
                self.assertEqual(bundle.dividends, [])
                self.assertIn("validation failed", bundle.coverage["dividends"].explanation)
        bundle = adapter.normalize("DCR", info, pd.DataFrame(), dividends=pd.DataFrame([{**row, "announcementDate": "0000-00-00"}]))
        self.assertEqual(bundle.dividends, [])
        self.assertEqual(bundle.coverage["dividends"].status, "missing")

    def test_invalid_dividend_announcements_do_not_create_events(self):
        row = {"symbol": "DCR", "sysAnnouncementId": "TEST-1", "announcementTitle": "Dividend"}
        for rows in ([{**row, "symbol": "AHCL"}], [row, {**row, "announcementTitle": "Dividend cancelled"}]):
            with self.subTest(rows=rows):
                bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
                    {"symbol": "DCR", "name": "Test"}, pd.DataFrame(), announcements=pd.DataFrame(rows))
                self.assertEqual(bundle.dividends, [])
                self.assertIn("failed validation", bundle.coverage["dividends"].explanation)
                self.assertFalse(any(key.startswith("dividend_announcement:") for key in bundle.source_refs))

    def test_announcement_fetch_uses_quote_date_and_preserves_optional_failures(self):
        client = Mock()
        client.get_stock_info.return_value = {"symbol": "DCR", "name": "Test", "date": "2026-10-02T19:00:00Z"}
        client.get_price_history.return_value = pd.DataFrame()
        client.get_company_snapshot.return_value = []
        client.get_income_statement_df.return_value = pd.DataFrame()
        client.get_announcements.side_effect = requests.Timeout()
        bundle = SarmaayaAdapter(client_factory=lambda: client).fetch("DCR")
        client.get_announcements.assert_called_once_with("DCR", "2025-10-02", "2026-10-02")
        self.assertEqual(bundle.dividends, [])
        self.assertIn("Timeout", bundle.coverage["dividends"].explanation)
        client.session.close.assert_called_once()

    def test_optional_financial_failure_preserves_quote_history(self):
        client = Mock()
        client.get_stock_info.return_value = {"symbol": "DCR", "name": "Test", "close": "40"}
        client.get_price_history.return_value = pd.DataFrame()
        client.get_company_snapshot.side_effect = requests.HTTPError("503")
        client.get_income_statement_df.side_effect = requests.Timeout()
        bundle = SarmaayaAdapter(client_factory=lambda: client).fetch("DCR")
        self.assertEqual(bundle.quote.price, Decimal(40))
        self.assertEqual(bundle.coverage["income"].status, "missing")
        self.assertIn("Timeout", bundle.coverage["income"].explanation)
        self.assertIn("HTTPError", bundle.coverage["shares"].explanation)
        self.assertNotIn("401", str(bundle.coverage))
        client.session.close.assert_called_once()

    def test_repeated_nonzero_income_values_are_flagged_not_corrected(self):
        income = pd.DataFrame([{"statement_type": "income_statement", "name": label,
            "metric": metric, "display_type": display, "period_end": "2025-06-30",
            "periodicity": "ANN", "value": 7} for label, metric, display in (
            ("Revenue", "FF_SALES", "wholeNumber"),
            ("Net Income", "FF_NET_INC", "wholeNumber"),
            ("Basic EPS", "FF_EPS_BASIC", "float"),
        )])
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), income=income)
        self.assertTrue(any("at least three different metrics" in warning for warning in bundle.warnings))
        for line in bundle.financials.income[0].lines:
            self.assertEqual(line.decimal_value, None if line.reported_label == "Basic EPS" else Decimal(7000000))
            fact = bundle.facts[line.fact_id]
            self.assertEqual(fact.value, Decimal(7))
            self.assertIn("repeated cross-metric", fact.quality)
        self.assertEqual(bundle.manifest.content_hash, content_hash(bundle))
        income["value"] = 0
        zero_bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), income=income)
        self.assertFalse(any("at least three different metrics" in warning for warning in zero_bundle.warnings))

    def test_income_does_not_scale_percentages_eps_or_unknown_metrics(self):
        income = pd.DataFrame([{"statement_type": "income_statement", "name": label,
            "metric": metric, "display_type": display, "period_end": "2025-06-30",
            "periodicity": "ANN", "value": value} for label, metric, display, value in (
            ("Net Income", "FF_NET_INC", "wholeNumber", "-1.25"),
            ("Operating Margin", "FF_OPER_MGN", "floatPercent", "20"),
            ("Basic EPS", "FF_EPS_BASIC", "float", "4"),
            ("Unknown amount", "UNVERIFIED", "wholeNumber", "30"),
        )])
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(), income=income)
        lines = bundle.financials.income[0].lines
        self.assertEqual(lines[0].decimal_value, Decimal(-1250000))
        for line in lines[1:]:
            self.assertIsNone(line.decimal_value)
            self.assertIn("unverified provider unit", bundle.facts[line.fact_id].unit)
        self.assertIsNone(bundle.eps["reported_annual"])
        self.assertTrue(any("user assumption" in warning for warning in bundle.warnings))
        self.assertEqual(bundle.model_dump(mode="json")["financials"]["income"][0]["lines"][0]["decimal_value"], "-1250000.00")

    def test_income_requires_source_identity(self):
        income = pd.DataFrame([{"statement_type": "income_statement", "name": "Revenue", "metric": "FF_SALES",
            "display_type": "wholeNumber", "period_end": "2025-06-30", "periodicity": "ANN", "value": 120}])
        bundle = SarmaayaAdapter(client_factory=Mock()).normalize("DCR",
            {"symbol": "DCR", "name": "Test", "close": "40"}, pd.DataFrame(), income=income)
        self.assertEqual(bundle.quote.price, Decimal(40))
        self.assertEqual(bundle.financials.income, [])
        self.assertNotIn("income_annual", bundle.source_refs)
        self.assertEqual(bundle.coverage["income"].status, "missing")

    def test_optional_invalid_metadata_is_not_accepted(self):
        adapter = SarmaayaAdapter(client_factory=Mock())
        for snapshot in ([{"isin": "OTHER", "metricMetric": "FF_COM_SHS_OUT", "afValue": "100"}],
                         [{"isin": "PKTEST", "metricMetric": "FF_COM_SHS_OUT", "afValue": "-1"}]):
            with self.subTest(snapshot=snapshot):
                bundle = adapter.normalize("DCR", {"symbol": "DCR", "name": "Test", "isin": "PKTEST"},
                    pd.DataFrame(), snapshot=snapshot)
                self.assertIsNone(bundle.shares.total_outstanding)
                self.assertEqual(bundle.coverage["shares"].status, "missing")
        for data in (
            [{"date": "2025-06-30", "periodicity": "LTM", "value": "1"}],
            [{"date": "2025-06-30", "periodicity": "ANN", "value": "Infinity"}],
            [{"date": "2025-06-30", "periodicity": "ANN", "value": "1"}] * 2,
        ):
            with self.subTest(data=data):
                bundle = adapter.normalize("DCR", {"symbol": "DCR", "name": "Test", "isin": "PKTEST"}, pd.DataFrame(),
                    income=pd.DataFrame([{"statement_type": "income_statement", "name": "Revenue",
                        "metric": "FF_SALES", "display_type": "wholeNumber", "period_end": row["date"],
                        "periodicity": row["periodicity"], "value": row["value"]} for row in data]))
                self.assertEqual(bundle.financials.income, [])
                self.assertEqual(bundle.coverage["income"].status, "missing")
                self.assertFalse(any(key.startswith("income:") for key in bundle.facts))

    def test_errors_are_unavailable_and_close_session(self):
        for error in [requests.Timeout(), requests.HTTPError(), RuntimeError(), ValueError()]:
            client = Mock()
            client.get_stock_info.side_effect = error
            with self.subTest(error=type(error)), self.assertRaises(DataUnavailable):
                SarmaayaAdapter(client_factory=lambda: client).fetch("DCR")
            client.session.close.assert_called_once()

    def test_invalid_symbol_never_calls_provider(self):
        factory = Mock()
        for symbol in ["../DCR", "", "DCR?x=1", "DCR/OTHER", "A" * 17]:
            with self.subTest(symbol=symbol), self.assertRaises(UnknownSymbol):
                SarmaayaAdapter(client_factory=factory).fetch(symbol)
        factory.assert_not_called()

    def test_provider_backed_symbols_are_not_limited_to_demo_allowlist(self):
        for symbol in (" demo_alt ", "123", "DEMO.R1", "DEMO-X"):
            with self.subTest(symbol=symbol):
                normalized = symbol.strip().upper()
                client = Mock()
                client.get_stock_info.return_value = {"symbol": normalized, "name": "Synthetic test company"}
                client.get_price_history.return_value = pd.DataFrame()
                client.get_company_snapshot.return_value = []
                client.get_income_statement_df.return_value = pd.DataFrame()
                client.get_dividends.return_value = pd.DataFrame()
                service = ResearchService(SarmaayaAdapter(client_factory=lambda: client))
                bundle = service.get(symbol)
                self.assertEqual(bundle.symbol, normalized)
                self.assertEqual(service.get(normalized).bundle_hash, bundle.bundle_hash)
                client.get_stock_info.assert_called_once_with(normalized)
                client.session.close.assert_called_once()

    def test_provider_unknown_symbol_preserves_unknown_error_and_closes_session(self):
        response = requests.Response()
        response.status_code = 404
        missing = requests.HTTPError("not found", response=response)
        for payload in (None, {}, []):
            with self.subTest(payload=payload):
                client = Mock()
                client.get_stock_info.return_value = payload
                with self.assertRaises(UnknownSymbol):
                    SarmaayaAdapter(client_factory=lambda: client).fetch("DEMO_UNKNOWN")
                client.get_price_history.assert_not_called()
                client.session.close.assert_called_once()
        client = Mock()
        client.get_stock_info.side_effect = missing
        with self.assertRaises(UnknownSymbol):
            SarmaayaAdapter(client_factory=lambda: client).fetch("DEMO_UNKNOWN")
        client.get_price_history.assert_not_called()
        client.session.close.assert_called_once()

    def test_provider_lookup_errors_and_mismatches_are_not_unknown_tickers(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                client = Mock()
                response = requests.Response()
                response.status_code = status
                client.get_stock_info.side_effect = requests.HTTPError("unavailable", response=response)
                with self.assertRaises(DataUnavailable):
                    SarmaayaAdapter(client_factory=lambda: client).fetch("DEMO_ALT")
                client.session.close.assert_called_once()
        for payload in ({"symbol": "OTHER", "name": "Test"}, {"symbol": "DEMO_ALT"}, [1]):
            with self.subTest(payload=payload):
                client = Mock()
                client.get_stock_info.return_value = payload
                with self.assertRaises(DataUnavailable):
                    SarmaayaAdapter(client_factory=lambda: client).fetch("DEMO_ALT")
                client.get_price_history.assert_not_called()
                client.session.close.assert_called_once()

    def test_cache_and_failed_reload_preserve_original_data(self):
        adapter = Mock()
        adapter.fetch.return_value = sample_bundle()
        service = ResearchService(adapter)
        first = service.get("dcr")
        second = service.get("DCR")
        adapter.fetch.assert_called_once_with("DCR")
        second.quote.price = Decimal(999)
        self.assertEqual(service.get("DCR").quote.price, first.quote.price)
        adapter.fetch.side_effect = DataUnavailable("offline")
        fallback = service.get("DCR", reload=True)
        self.assertEqual(fallback.bundle_hash, first.bundle_hash)
        self.assertEqual(fallback.fetched_at, first.fetched_at)
        self.assertIn("Provider unavailable", fallback.warnings[-1])
        self.assertNotIn("Provider unavailable", service.get("DCR").warnings[-1])
        with self.assertRaises(DataUnavailable):
            service.get("AHCL")
