import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import requests
from urllib3.util.retry import Retry

from utils.sarmaaya_util import SarmaayaClient


class TestSarmaayaClient(unittest.TestCase):
    def setUp(self):
        session_patch = patch("utils.sarmaaya_util.requests.Session")
        self.addCleanup(session_patch.stop)
        self.session = session_patch.start().return_value
        self.client = SarmaayaClient()
        self.request_session = self.session.__enter__.return_value

    def test_session_headers_and_retry_policy(self):
        self.session.headers.update.assert_called_once_with(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64; rv:157.0) "
                    "Gecko/20100101 Firefox/157.0"
                ),
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Origin": "https://sarmaaya.pk",
                "Referer": "https://sarmaaya.pk/",
            }
        )
        self.session.mount.assert_called_once()
        scheme, adapter = self.session.mount.call_args.args
        self.assertEqual(scheme, "https://")
        retry = adapter.max_retries
        self.assertEqual(retry.total, 5)
        self.assertEqual(retry.backoff_factor, 1)
        self.assertEqual(list(retry.status_forcelist), [429, 500, 502, 503, 504])
        self.assertEqual(set(retry.allowed_methods), {"GET"})

    def test_request_arguments(self):
        for method in ("get", "post"):
            with self.subTest(method=method):
                request = getattr(self.request_session, method)
                request.return_value.json.return_value = {"response": [1]}
                kwargs = {"params": {"symbol": "FFL"}}
                if method == "post":
                    kwargs["json"] = {"key": "value"}
                result = getattr(self.client, f"_{method}")("/stocks", **kwargs)
                self.assertEqual(result, [1])
                request.assert_called_once_with(
                    f"{SarmaayaClient.BASE_URL}/stocks", timeout=30, **kwargs
                )
                request.return_value.raise_for_status.assert_called_once_with()

    def test_request_default_arguments(self):
        for method in ("get", "post"):
            with self.subTest(method=method):
                getattr(self.client, f"_{method}")("stocks")
                kwargs = {"params": None, "timeout": 30}
                if method == "post":
                    kwargs["json"] = None
                getattr(self.request_session, method).assert_called_once_with(
                    f"{SarmaayaClient.BASE_URL}/stocks", **kwargs
                )

    def test_response_unwrapping_and_fallback(self):
        cases = [
            ({"success": True, "response": {"isin": "PK123"}}, {"isin": "PK123"}),
            ({"response": None}, None),
            ({"response": []}, []),
            ({"response": 0}, 0),
            ({"success": True, "value": 2}, {"success": True, "value": 2}),
            ({}, {}),
            ([{"symbol": "FFL"}], [{"symbol": "FFL"}]),
            (None, None),
        ]
        for method in ("get", "post"):
            for payload, expected in cases:
                with self.subTest(method=method, payload=payload):
                    getattr(
                        self.request_session, method
                    ).return_value.json.return_value = payload
                    self.assertEqual(
                        getattr(self.client, f"_{method}")("stocks"), expected
                    )

    def test_api_failure_precedes_response_unwrapping(self):
        for method in ("get", "post"):
            with self.subTest(method=method):
                response = getattr(self.request_session, method).return_value
                response.json.return_value = {"success": False, "response": []}
                with self.assertRaisesRegex(RuntimeError, "Sarmaaya API error"):
                    getattr(self.client, f"_{method}")("stocks")

    def test_transport_http_and_json_errors_propagate(self):
        for method in ("get", "post"):
            for stage, error in (
                ("request", requests.Timeout("timeout")),
                ("status", requests.HTTPError("bad status")),
                ("json", ValueError("invalid json")),
            ):
                with self.subTest(method=method, stage=stage):
                    request = Mock()
                    setattr(self.request_session, method, request)
                    if stage == "request":
                        request.side_effect = error
                    elif stage == "status":
                        request.return_value.raise_for_status.side_effect = error
                    else:
                        request.return_value.json.side_effect = error
                    with self.assertRaises(type(error)) as caught:
                        getattr(self.client, f"_{method}")("stocks")
                    self.assertIs(caught.exception, error)
                    if stage == "status":
                        request.return_value.json.assert_not_called()

    def test_get_endpoints_normalize_symbols(self):
        cases = [
            ("get_stock_info", ("ffl",), "stocks/FFL", {}),
            (
                "get_snowflake_score",
                ("ffl",),
                "stock-score/snowflake",
                {"params": {"symbol": "FFL"}},
            ),
            (
                "get_sarmaaya_values",
                ("ffl",),
                "stock-score/sarmaaya-values",
                {"params": {"symbol": "FFL"}},
            ),
            ("get_market_view", (), "dashboard/market-view", {}),
            (
                "get_company_reports",
                ("ffl", 2025),
                "stocks/reports/FFL",
                {"params": {"year": 2025}},
            ),
        ]
        for method, args, endpoint, kwargs in cases:
            with (
                self.subTest(method=method),
                patch.object(self.client, "_get", return_value={}) as get,
            ):
                self.assertEqual(getattr(self.client, method)(*args), {})
                get.assert_called_once_with(endpoint, **kwargs)

    def test_performance_normalizes_symbols_and_numeric_horizons(self):
        rows = [
            {"symbol": "DEMO_ALT", "1w": "1.25", "1m": "-2.5", "3m": "0", "6m": None, "1y": "invalid"},
            {"symbol": "KSE100", "1w": "2", "1m": "3", "3m": "4", "6m": "5", "1y": "6"},
        ]
        with patch.object(self.client, "_get", return_value=rows) as get:
            result = self.client.get_performance("demo_alt")
        get.assert_called_once_with("stocks/returns/DEMO_ALT")
        expected = pd.DataFrame({
            "symbol": ["DEMO_ALT", "KSE100"], "1w": [1.25, 2.0], "1m": [-2.5, 3.0],
            "3m": [0, 4], "6m": [float("nan"), 5.0], "1y": [float("nan"), 6.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_performance_empty_and_partial_horizons(self):
        for rows in ([], [{"symbol": "DEMO_ALT", "1w": "1.5"}]):
            with self.subTest(rows=rows), patch.object(self.client, "_get", return_value=rows) as get:
                result = self.client.get_performance("demo_alt")
                get.assert_called_once_with("stocks/returns/DEMO_ALT")
                expected = pd.DataFrame(rows)
                if rows:
                    expected["1w"] = pd.Series([1.5])
                pd.testing.assert_frame_equal(result, expected)

    def test_stock_returns_delegates_to_performance(self):
        frame = pd.DataFrame({"symbol": ["DEMO_ALT"], "1w": [1.25]})
        with patch.object(self.client, "get_performance", return_value=frame) as performance:
            self.assertIs(self.client.get_stock_returns("demo_alt"), frame)
        performance.assert_called_once_with("demo_alt")

    def test_stock_returns_propagates_performance_failure(self):
        error = requests.HTTPError("unavailable")
        with patch.object(self.client, "_get", side_effect=error):
            with self.assertRaises(requests.HTTPError) as caught:
                self.client.get_stock_returns("demo_alt")
        self.assertIs(caught.exception, error)

    def test_dividends_preserve_payout_history_and_yearly_summaries(self):
        rows = [{"symbol": "DEMO_REIT", "announcementDate": "2026-07-01",
                 "payoutType": "dividend", "dividendPerShare": "2"}]
        payload = {"data": [], "dividendsYield": [{"2026": "5%"}],
                   "dividendsPerShare": [{"2026": "2"}], "payoutHistory": rows}
        with patch.object(self.client, "_get", return_value=payload) as get:
            result = self.client.get_dividends("demo_reit")
        get.assert_called_once_with("stocks/dividends/DEMO_REIT")
        pd.testing.assert_frame_equal(result, pd.DataFrame(rows))
        self.assertEqual(result.attrs["dividendsPerShare"], [{"2026": "2"}])
        self.assertEqual(result.attrs["dividendsYield"], [{"2026": "5%"}])
        self.assertEqual(result.attrs["source_url"], f"{SarmaayaClient.BASE_URL}/stocks/dividends/DEMO_REIT")
        self.assertIn("not app-calculated trailing", result.attrs["summary_basis"])

    def test_dividends_empty_history_and_invalid_schema(self):
        with patch.object(self.client, "_get", return_value={"payoutHistory": []}):
            result = self.client.get_dividends("DEMO_REIT")
            self.assertTrue(result.empty)
            self.assertIsNone(result.attrs["dividendsPerShare"])
            self.assertIsNone(result.attrs["dividendsYield"])
        for payload in ({}, None, [], {"payoutHistory": None}, {"payoutHistory": [1]}):
            with self.subTest(payload=payload), patch.object(self.client, "_get", return_value=payload):
                with self.assertRaises(ValueError):
                    self.client.get_dividends("DEMO_REIT")

    def test_dividend_summaries_survive_empty_payout_history(self):
        payload = {"payoutHistory": [], "dividendsPerShare": [{"2025": "0.00", "2026": "2.50"}],
                   "dividendsYield": [{"2026": "5.25%"}]}
        with patch.object(self.client, "_get", return_value=payload):
            result = self.client.get_dividends("DEMO_REIT")
        self.assertTrue(result.empty)
        self.assertEqual(result.attrs["dividendsPerShare"], payload["dividendsPerShare"])
        self.assertEqual(result.attrs["dividendsYield"], payload["dividendsYield"])

    def test_invalid_dividend_summary_schema_rejected(self):
        for key in ("dividendsPerShare", "dividendsYield"):
            for value in (None, "5%", {"2026": "2"}, [1]):
                with self.subTest(key=key, value=value), patch.object(self.client, "_get", return_value={"payoutHistory": [], key: value}):
                    with self.assertRaises(ValueError):
                        self.client.get_dividends("DEMO_REIT")

    def test_company_snapshot_uses_post(self):
        with patch.object(self.client, "_post", return_value={"name": "FFL"}) as post:
            self.assertEqual(self.client.get_company_snapshot("ffl"), {"name": "FFL"})
            post.assert_called_once_with("stocks/details/FFL")

    def test_isin_resolution_and_missing_values(self):
        with patch.object(self.client, "get_stock_info") as info:
            info.return_value = {"isin": "PK123"}
            self.assertEqual(self.client.get_isin("ffl"), "PK123")
            info.assert_called_once_with("ffl")
            for payload in ({}, {"isin": None}, {"isin": ""}):
                with self.subTest(payload=payload):
                    info.return_value = payload
                    with self.assertRaisesRegex(ValueError, "No ISIN found for FFL"):
                        self.client.get_isin("ffl")

    def test_price_history_sorts_utc_dates_and_resets_index(self):
        data = [
            {"date": "2025-01-02T00:00:00Z", "close": 20},
            {"date": "2025-01-01T00:00:00Z", "close": 10},
        ]
        with patch.object(self.client, "_get", return_value=data) as get:
            result = self.client.get_price_history("ffl", days=30)
            get.assert_called_once_with("stocks/price-history/FFL", params={"days": 30})
        expected = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
                "close": [10, 20],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_price_history_empty_and_without_dates(self):
        for data in ([], [{"close": 20}, {"close": 10}]):
            with (
                self.subTest(data=data),
                patch.object(self.client, "_get", return_value=data) as get,
            ):
                result = self.client.get_price_history("ffl")
                get.assert_called_once_with(
                    "stocks/price-history/FFL", params={"days": 365}
                )
                pd.testing.assert_frame_equal(result, pd.DataFrame(data))

    def test_index_constituents(self):
        for args, index in (((), "KSE100"), (("kse30",), "KSE30")):
            with self.subTest(index=index):
                data = [{"symbol": "FFL"}]
                with patch.object(self.client, "_get", return_value=data) as get:
                    result = self.client.get_index_constituents(*args)
                    get.assert_called_once_with(
                        "stocks/ticker", params={"index": index}
                    )
                    pd.testing.assert_frame_equal(result, pd.DataFrame(data))

    def test_index_symbols_columns_normalization_and_deduplication(self):
        for column in ("symbol", "ticker", "code"):
            with self.subTest(column=column):
                data = pd.DataFrame({column: ["ffl", None, "FFL", "ogdc", 123]})
                with patch.object(
                    self.client, "get_index_constituents", return_value=data
                ):
                    self.assertEqual(
                        self.client.get_index_symbols(), ["FFL", "OGDC", "123"]
                    )

    def test_index_symbols_column_precedence(self):
        data = pd.DataFrame({"symbol": ["ffl"], "ticker": ["ogdc"], "code": ["hubc"]})
        with patch.object(
            self.client, "get_index_constituents", return_value=data
        ) as get:
            self.assertEqual(self.client.get_index_symbols("kse30"), ["FFL"])
            get.assert_called_once_with("kse30")

    def test_index_symbols_empty_and_missing_columns(self):
        with patch.object(self.client, "get_index_constituents") as get:
            for data in (pd.DataFrame(), pd.DataFrame({"symbol": [None]})):
                with self.subTest(data=data):
                    get.return_value = data
                    self.assertEqual(self.client.get_index_symbols(), [])
            get.return_value = pd.DataFrame({"name": ["Company"]})
            with self.assertRaisesRegex(KeyError, "Available columns:.*name"):
                self.client.get_index_symbols()

    def test_fundamentals_resolve_isin_and_periodicity(self):
        for method, endpoint in (
            ("get_income_statement", "income-statement"),
            ("get_balance_sheet", "balance-sheet"),
            ("get_cash_flow", "cashflow"),
            ("get_financial_ratios", "ratios"),
        ):
            for kwargs, periodicity in (
                ({}, "ANN"),
                ({"periodicity": "ann"}, "ANN"),
                ({"periodicity": "ltm"}, "LTM"),
            ):
                with self.subTest(method=method, periodicity=periodicity):
                    data = {"Synthetic metric": {"metric": "FF_TEST", "displayType": "wholeNumber",
                        "data": [{"value": "10.00", "date": "2025-06-30T00:00:00.000Z",
                                  "year": 2025, "periodicity": periodicity, "fiscalPeriod": 0}]}}
                    with (
                        patch.object(
                            self.client, "get_isin", return_value="PK123"
                        ) as isin,
                        patch.object(self.client, "_get", return_value=data) as get,
                    ):
                        result = getattr(self.client, method)("ffl", **kwargs)
                        isin.assert_called_once_with("ffl")
                        get.assert_called_once_with(
                            f"stocks/fundamentals/{endpoint}",
                            params={"isin": "PK123", "periodicity": periodicity},
                        )
                        self.assertIs(result, data)

    def test_fundamentals_isolate_cookies_and_preserve_request_policy(self):
        for method, endpoint in (
            ("get_income_statement", "income-statement"),
            ("get_balance_sheet", "balance-sheet"),
            ("get_cash_flow", "cashflow"),
            ("get_financial_ratios", "ratios"),
        ):
            with self.subTest(method=method):
                self.session.reset_mock()
                self.session.headers = {"Accept": "application/json", "Referer": "https://sarmaaya.pk/"}
                self.session.cookies = requests.cookies.RequestsCookieJar()
                self.session.cookies.set("AWSALB", "sticky-backend")
                self.session.cookies.set("AWSALBCORS", "sticky-backend")
                stock = MagicMock()
                stock.__enter__.return_value = stock
                stock.get.return_value.json.return_value = {"response": {"isin": "PK123"}}
                retry = Retry(total=5, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
                self.session.get_adapter.return_value.max_retries = retry
                fresh = MagicMock()
                fresh.__enter__.return_value = fresh
                fresh.headers = {}
                fresh.cookies = requests.cookies.RequestsCookieJar()
                data = {"Synthetic metric": {"metric": "FF_TEST", "displayType": "wholeNumber",
                    "data": [{"value": "10.00", "date": "2025-06-30", "year": 2025,
                              "periodicity": "ANN", "fiscalPeriod": 0}]}}
                fresh.get.return_value.json.return_value = {"response": data}
                with patch("utils.sarmaaya_util.requests.Session", side_effect=[stock, fresh]) as factory:
                    result = getattr(self.client, method)("ffl")
                self.assertEqual(factory.call_count, 2)
                self.assertIs(result, data)
                self.session.get.assert_not_called()
                stock.get.assert_called_once_with(
                    f"{SarmaayaClient.BASE_URL}/stocks/FFL", params=None, timeout=30)
                fresh.get.assert_called_once_with(
                    f"{SarmaayaClient.BASE_URL}/stocks/fundamentals/{endpoint}",
                    params={"isin": "PK123", "periodicity": "ANN"}, timeout=30)
                self.assertEqual(fresh.headers, self.session.headers)
                self.assertEqual(len(fresh.cookies), 0)
                self.assertEqual(self.session.cookies.get("AWSALB"), "sticky-backend")
                self.assertEqual(self.session.cookies.get("AWSALBCORS"), "sticky-backend")
                scheme, adapter = fresh.mount.call_args.args
                self.assertEqual(scheme, "https://")
                self.assertIs(adapter.max_retries, retry)
                fresh.__exit__.assert_called_once()
                stock.__exit__.assert_called_once()

    def test_fundamentals_df_counterparts_delegate_with_statement_type(self):
        for raw_method, statement_type in (
            ("get_income_statement", "income_statement"),
            ("get_balance_sheet", "balance_sheet"),
            ("get_cash_flow", "cash_flow"),
            ("get_financial_ratios", "ratios"),
        ):
            for periodicity in ("ANN", "ltm"):
                with self.subTest(method=raw_method, periodicity=periodicity):
                    data = {"Test": {"metric": "FF_TEST", "displayType": "wholeNumber", "data": []}}
                    frame = pd.DataFrame()
                    with (patch.object(self.client, raw_method, return_value=data) as raw,
                          patch.object(self.client, "normalize_fundamentals", return_value=frame) as normalize):
                        result = getattr(self.client, raw_method + "_df")("demo_alt", periodicity)
                    self.assertIs(result, frame)
                    raw.assert_called_once_with("demo_alt", periodicity)
                    normalize.assert_called_once_with(data, statement_type=statement_type)

    def test_fundamentals_df_counterparts_normalize_actual_response_shape(self):
        for raw_method, statement_type in (
            ("get_income_statement", "income_statement"),
            ("get_balance_sheet", "balance_sheet"),
            ("get_cash_flow", "cash_flow"),
            ("get_financial_ratios", "ratios"),
        ):
            with self.subTest(method=raw_method):
                data = {
                    "Synthetic amount": {"metric": "FF_TEST", "displayType": "wholeNumber", "data": [
                        {"value": 12.5, "date": "2025-06-30T00:00:00.000Z", "year": 2025,
                         "periodicity": "ANN", "fiscalPeriod": 0},
                        {"value": "0.000", "date": "2024-06-30T00:00:00.000Z", "year": "2024",
                         "periodicity": "ANN", "fiscalPeriod": "0"}]},
                    "Synthetic margin": {"metric": "FF_TEST_MARGIN", "displayType": "floatPercent", "data": [
                        {"value": "-2.5", "date": "2025-06-30T00:00:00.000Z", "year": 2025,
                         "periodicity": "ANN", "fiscalPeriod": 0}]},
                }
                with patch.object(self.client, raw_method, return_value=data):
                    result = getattr(self.client, raw_method + "_df")("demo_alt")
                expected = pd.DataFrame({
                    "statement_type": [statement_type] * 3,
                    "name": ["Synthetic amount", "Synthetic amount", "Synthetic margin"],
                    "metric": ["FF_TEST", "FF_TEST", "FF_TEST_MARGIN"],
                    "display_type": ["wholeNumber", "wholeNumber", "floatPercent"],
                    "value": [12.5, 0.0, -2.5],
                    "period_end": pd.to_datetime(["2025-06-30", "2024-06-30", "2025-06-30"], utc=True),
                    "year": pd.Series([2025, 2024, 2025], dtype="Int64"),
                    "periodicity": ["ANN"] * 3,
                    "fiscal_period": pd.Series([0, 0, 0], dtype="Int64"),
                })
                pd.testing.assert_frame_equal(result, expected)

    def test_normalize_fundamentals_missing_values_and_ltm_are_not_invented(self):
        data = {"Synthetic metric": {"metric": "FF_TEST", "displayType": "float", "data": [
            {"value": "bad", "date": "bad", "year": None, "periodicity": "LTM", "fiscalPeriod": None},
            {"value": None, "date": None, "periodicity": "LTM"},
            {"value": "2", "date": "2026-03-31T05:00:00.000Z", "year": "2027",
             "periodicity": "LTM", "fiscalPeriod": "3"}]}}
        result = self.client.normalize_fundamentals(data, statement_type="ratios")
        self.assertEqual(len(result), 3)
        self.assertTrue(result.loc[:1, "value"].isna().all())
        self.assertTrue(result.loc[:1, "period_end"].isna().all())
        self.assertTrue(result.loc[:1, "year"].isna().all())
        self.assertTrue(result.loc[:1, "fiscal_period"].isna().all())
        self.assertEqual(result["periodicity"].tolist(), ["LTM"] * 3)
        self.assertEqual(result.loc[2, "period_end"], pd.Timestamp("2026-03-31T05:00:00Z"))
        self.assertEqual(result.loc[2, "year"], 2027)
        self.assertEqual(result.loc[2, "fiscal_period"], 3)
        self.assertEqual(result.loc[2, "value"], 2)

    def test_normalize_fundamentals_empty_and_non_object_entries(self):
        for data in (None, [], {}, {"bad": 1}, {"empty": {"data": []}},
                     {"bad": {"data": [None, 1, "bad"]}}):
            with self.subTest(data=data):
                self.assertTrue(self.client.normalize_fundamentals(data).empty)

    def test_dividends_isolate_cookies_and_preserve_request_policy(self):
        self.session.reset_mock()
        self.session.headers = {"Accept": "application/json", "Referer": "https://sarmaaya.pk/"}
        self.session.cookies = requests.cookies.RequestsCookieJar()
        self.session.cookies.set("AWSALB", "sticky-backend")
        self.session.cookies.set("AWSALBCORS", "sticky-backend")
        self.session.get.side_effect = requests.HTTPError("401 sticky backend")
        retry = Retry(total=5, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        self.session.get_adapter.return_value.max_retries = retry
        rows = [{"symbol": "DEMO_REIT", "dividendPerShare": "2"}]
        payload = {"payoutHistory": rows, "dividendsPerShare": [{"2026": "2"}],
                   "dividendsYield": [{"2026": "5%"}]}
        fresh = MagicMock()
        fresh.__enter__.return_value = fresh
        fresh.headers = {}
        fresh.cookies = requests.cookies.RequestsCookieJar()
        fresh.get.return_value.json.return_value = {"success": True, "response": payload}
        with patch("utils.sarmaaya_util.requests.Session", return_value=fresh) as factory:
            result = self.client.get_dividends("demo_reit")
        factory.assert_called_once_with()
        self.session.get.assert_not_called()
        fresh.get.assert_called_once_with(
            f"{SarmaayaClient.BASE_URL}/stocks/dividends/DEMO_REIT", params=None, timeout=30)
        fresh.get.return_value.raise_for_status.assert_called_once_with()
        pd.testing.assert_frame_equal(result, pd.DataFrame(rows))
        self.assertEqual(result.attrs["dividendsPerShare"], payload["dividendsPerShare"])
        self.assertEqual(result.attrs["dividendsYield"], payload["dividendsYield"])
        self.assertEqual(fresh.headers, self.session.headers)
        self.assertEqual(len(fresh.cookies), 0)
        self.assertEqual(self.session.cookies.get("AWSALB"), "sticky-backend")
        self.assertEqual(self.session.cookies.get("AWSALBCORS"), "sticky-backend")
        scheme, adapter = fresh.mount.call_args.args
        self.assertEqual(scheme, "https://")
        self.assertIs(adapter.max_retries, retry)
        fresh.__exit__.assert_called_once()

    def test_fundamentals_isolated_session_closes_on_transport_failure(self):
        fresh = MagicMock()
        fresh.__enter__.return_value = fresh
        error = requests.Timeout("timeout")
        fresh.get.side_effect = error
        self.session.get_adapter.return_value.max_retries = Retry(total=5)
        with patch("utils.sarmaaya_util.requests.Session", return_value=fresh):
            with self.assertRaises(requests.Timeout) as caught:
                self.client._get("/stocks/fundamentals/income-statement")
        self.assertIs(caught.exception, error)
        fresh.__exit__.assert_called_once()
        self.session.get.assert_not_called()

    def test_dividends_isolated_session_closes_on_transport_failure(self):
        fresh = MagicMock()
        fresh.__enter__.return_value = fresh
        error = requests.Timeout("timeout")
        fresh.get.side_effect = error
        self.session.get_adapter.return_value.max_retries = Retry(total=5)
        with patch("utils.sarmaaya_util.requests.Session", return_value=fresh):
            with self.assertRaises(requests.Timeout) as caught:
                self.client._get("/stocks/dividends/DEMO_REIT")
        self.assertIs(caught.exception, error)
        fresh.__exit__.assert_called_once()
        self.session.get.assert_not_called()

    def test_all_requests_use_distinct_cookie_free_sessions(self):
        self.session.headers = {"Accept": "application/json"}
        self.session.cookies = requests.cookies.RequestsCookieJar()
        self.session.cookies.set("AWSALBCORS", "sticky-backend")
        retry = Retry(total=5, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        self.session.get_adapter.return_value.max_retries = retry
        sessions = [MagicMock() for _ in range(3)]
        for session in sessions:
            session.__enter__.return_value = session
            session.headers = {}
            session.cookies = requests.cookies.RequestsCookieJar()
            session.get.return_value.json.return_value = {"response": {"symbol": "DEMO_REIT"}}
            session.post.return_value.json.return_value = {"response": []}
        with patch("utils.sarmaaya_util.requests.Session", side_effect=sessions) as factory:
            self.client.get_stock_info("demo_reit")
            sessions[0].cookies.set("AWSALBCORS", "stock-backend")
            self.assertEqual(self.client.get_company_snapshot("demo_reit"), [])
            self.client.get_stock_info("demo_reit")
        self.assertEqual(factory.call_count, 3)
        self.session.get.assert_not_called()
        self.session.post.assert_not_called()
        sessions[1].post.assert_called_once_with(
            f"{SarmaayaClient.BASE_URL}/stocks/details/DEMO_REIT",
            params=None, json=None, timeout=30)
        for session in sessions:
            self.assertEqual(session.headers, self.session.headers)
            scheme, adapter = session.mount.call_args.args
            self.assertEqual(scheme, "https://")
            self.assertIs(adapter.max_retries, retry)
            self.assertEqual(set(adapter.max_retries.allowed_methods), {"GET"})
            session.__exit__.assert_called_once()
        self.assertEqual(len(sessions[1].cookies), 0)
        self.assertEqual(len(sessions[2].cookies), 0)
        self.assertEqual(self.session.cookies.get("AWSALBCORS"), "sticky-backend")

    def test_post_isolated_session_closes_on_transport_failure(self):
        fresh = MagicMock()
        fresh.__enter__.return_value = fresh
        error = requests.Timeout("timeout")
        fresh.post.side_effect = error
        self.session.get_adapter.return_value.max_retries = Retry(total=5, allowed_methods=["GET"])
        with patch("utils.sarmaaya_util.requests.Session", return_value=fresh):
            with self.assertRaises(requests.Timeout) as caught:
                self.client._post("/stocks/details/DEMO_REIT")
        self.assertIs(caught.exception, error)
        fresh.__exit__.assert_called_once()
        self.session.post.assert_not_called()

    def test_invalid_periodicity_fails_before_requests(self):
        for method in ("get_income_statement", "get_balance_sheet", "get_cash_flow", "get_financial_ratios",
                       "get_income_statement_df", "get_balance_sheet_df", "get_cash_flow_df", "get_financial_ratios_df"):
            for periodicity in ("QTR", "", "annual"):
                with (
                    self.subTest(method=method, periodicity=periodicity),
                    patch.object(self.client, "get_isin") as isin,
                    patch.object(self.client, "_get") as get,
                ):
                    with self.assertRaisesRegex(ValueError, "periodicity must be"):
                        getattr(self.client, method)("ffl", periodicity)
                    isin.assert_not_called()
                    get.assert_not_called()

    def test_announcements_date_parameters_and_dataframe(self):
        data = [{"title": "Report"}]
        with patch.object(self.client, "_get", return_value=data) as get:
            result = self.client.get_announcements("ffl", "2025-01-01", "2025-12-31")
            get.assert_called_once_with(
                "stocks/announcements/FFL",
                params={"startDate": "2025-01-01", "endDate": "2025-12-31"},
            )
            pd.testing.assert_frame_equal(result, pd.DataFrame(data))

    def test_stock_snapshot_combines_normalized_symbol_and_scores(self):
        with (
            patch.object(
                self.client, "get_stock_info", return_value={"isin": "PK123"}
            ) as info,
            patch.object(
                self.client, "get_snowflake_score", return_value={"score": 1}
            ) as snow,
            patch.object(
                self.client, "get_sarmaaya_values", return_value={"value": 2}
            ) as values,
        ):
            self.assertEqual(
                self.client.get_stock_snapshot("ffl"),
                {
                    "symbol": "FFL",
                    "stock_info": {"isin": "PK123"},
                    "snowflake_score": {"score": 1},
                    "sarmaaya_values": {"value": 2},
                },
            )
            for method in (info, snow, values):
                method.assert_called_once_with("FFL")
