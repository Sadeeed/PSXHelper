"""One adapter over the existing utility; no guessed financial mappings."""

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from threading import Lock

import pandas as pd
import requests

from .schemas import (
    Coverage,
    DailyBar,
    DatasetManifest,
    Fact,
    FinancialLine,
    FinancialPeriod,
    Quote,
    Shares,
    SourceRef,
    TickerBundle,
)

SUPPORTED_SYMBOLS = ("DCR", "AHCL")


class UnknownSymbol(ValueError):
    pass


class DataUnavailable(RuntimeError):
    pass


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,15}", symbol):
        raise UnknownSymbol("Invalid ticker format")
    return symbol


def decimal_or_none(value):
    if value is None or pd.isna(value):
        return None
    return Decimal(str(value))


def content_hash(bundle: TickerBundle) -> str:
    content = bundle.model_dump(mode="json", exclude={"fetched_at", "bundle_hash", "manifest"})
    for source in content["source_refs"].values():
        source.pop("acquired_at", None)
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SarmaayaAdapter:
    def __init__(self, client_factory=None):
        # Source-checkout bridge: preserve the user's existing utility unchanged.
        if client_factory is None:
            from utils.sarmaaya_util import SarmaayaClient
            client_factory = SarmaayaClient
        self.client_factory = client_factory

    def fetch(self, symbol: str) -> TickerBundle:
        symbol = normalize_symbol(symbol)
        client = self.client_factory()
        try:
            try:
                info = client.get_stock_info(symbol)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    raise UnknownSymbol(f"Provider does not recognize ticker {symbol}") from exc
                raise
            if info is None or info == {} or info == []:
                raise UnknownSymbol(f"Provider returned no stock metadata for {symbol}")
            if not isinstance(info, dict) or info.get("symbol") != symbol or not info.get("name"):
                raise ValueError("Stock metadata does not match selected symbol")
            history = client.get_price_history(symbol, days=365)
            optional = {}
            failures = {}
            for category, method in (
                ("snapshot", client.get_company_snapshot),
                ("income", client.get_income_statement_df),
                ("balance", client.get_balance_sheet_df),
                ("cashflow", client.get_cash_flow_df),
                ("ratios", client.get_financial_ratios_df),
                ("dividends", client.get_dividends),
            ):
                try:
                    optional[category] = method(symbol)
                except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    failures[category] = f"{category} acquisition failed ({type(exc).__name__}); quote/history remain usable."
            if info.get("date"):
                try:
                    end = pd.to_datetime(info["date"], utc=True).date()
                    optional["announcements"] = client.get_announcements(
                        symbol, str(end - timedelta(days=365)), str(end))
                except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    failures["announcements"] = f"Dividend announcement acquisition failed ({type(exc).__name__}); no event coverage established."
            return self.normalize(symbol, info, history, **optional, failures=failures)
        except UnknownSymbol:
            raise
        except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise DataUnavailable("Sarmaaya data unavailable or failed validation") from exc
        finally:
            client.session.close()

    def normalize(self, symbol, info, history, snapshot=None, income=None, announcements=None, dividends=None, failures=None,
                  balance=None, cashflow=None, ratios=None) -> TickerBundle:
        symbol = normalize_symbol(symbol)
        if not isinstance(info, dict) or info.get("symbol") != symbol or not info.get("name"):
            raise ValueError("Stock metadata does not match selected symbol")
        now = datetime.now(UTC)
        source_time = pd.to_datetime(info.get("date"), utc=True) if info.get("date") else None
        quote = Quote(
            price=decimal_or_none(info.get("close")),
            market_date=source_time.date() if source_time is not None else None,
            source_time=source_time.to_pydatetime() if source_time is not None else None,
            volume=decimal_or_none(info.get("volume")),
        )
        bars = []
        invalid_market_caps = 0
        for row in history.to_dict(orient="records"):
            if row.get("symbol", symbol) != symbol:
                raise ValueError("History contains a different symbol")
            stamp = pd.to_datetime(row["date"], utc=True)
            if pd.isna(stamp):
                raise ValueError("History date missing")
            try:
                market_cap = decimal_or_none(row.get("marketCap"))
                if market_cap is not None and (not market_cap.is_finite() or market_cap < 0):
                    raise ValueError("Invalid market capitalization")
            except (InvalidOperation, ValueError, TypeError):
                market_cap = None
                invalid_market_caps += 1
            bars.append(DailyBar(
                date=stamp.date(), close=decimal_or_none(row.get("price")),
                volume=decimal_or_none(row.get("volume")),
                market_cap=market_cap,
            ))
        bars.sort(key=lambda bar: bar.date)
        refs = {
            "stock_info": SourceRef(provider=f"Sarmaaya; upstream {info.get('provider', 'unknown')}",
                url=f"https://beta-restapi.sarmaaya.pk/api/stocks/{symbol}",
                acquisition_method="GET via SarmaayaClient.get_stock_info", acquired_at=now),
            "price_history": SourceRef(provider="Sarmaaya",
                url=f"https://beta-restapi.sarmaaya.pk/api/stocks/price-history/{symbol}?days=365",
                acquisition_method="GET via SarmaayaClient.get_price_history", acquired_at=now),
        }
        coverage = {
            "profile": Coverage(status="partial", explanation="Name and sector available; business description not supplied by verified provider mapping."),
            "quote": Coverage(status="available" if quote.price is not None else "missing", explanation="Provider close; source timestamp is not a verified trading timestamp. Provider delay unknown."),
            "history": Coverage(status="partial" if bars else "missing", explanation=f"{len(bars)} observed close/volume bars; OHLC unavailable, adjustment basis and market-session completeness unverified."),
            "market_cap_history": Coverage(status="partial" if any(bar.market_cap is not None for bar in bars) else "missing",
                explanation=f"{sum(bar.market_cap is not None for bar in bars)} of {len(bars)} dated provider marketCap observations. Unit/scale and share basis unverified; values are not reconstructed from price or current shares. {invalid_market_caps} invalid values withheld. Missing records do not mean zero capitalization. Source: price_history."),
        }
        for category, reason in {
            "income": "Annual provider financial rows not loaded.",
            "balance": "Balance-sheet rows not loaded. Missing records do not mean zero balances.",
            "cashflow": "Cash-flow rows not loaded. Missing records do not mean zero cash flows.",
            "ratios": "Provider ratio observations not loaded; no ratios reconstructed.",
            "eps": "No verified EPS period and accounting basis.",
            "dividends": "No verified distribution-event mapping; empty records do not mean no payouts.",
            "index_memberships": "Constituents not ingested; absence does not mean exclusion.",
        }.items():
            coverage[category] = Coverage(status="unknown" if category == "index_memberships" else "missing", explanation=reason)
        facts = {}
        for key, value, unit in [("price", quote.price, "PKR per unit"), ("volume", quote.volume, "provider trading units")]:
            if value is not None:
                facts[key] = Fact(fact_id=key, value=value, unit=unit,
                    period=str(quote.market_date) if quote.market_date else None,
                    basis="provider_close", source_ref="stock_info", quality="provider observation")
        warnings = ["Access/presentation rights unverified; not a redistribution-permitted dataset.",
                    "Provider close is not a live intraday quote. Adjustment basis unverified.",
                    "Missing statement categories and dividend records do not mean zero activity."]
        if invalid_market_caps:
            warnings.append(f"Market-cap history: {invalid_market_caps} invalid provider values withheld; valid price/volume observations retained.")
        shares = Shares()
        statements = {"income": [], "balance": [], "cashflow": []}
        for category, message in (failures or {}).items():
            coverage[{"snapshot": "shares", "announcements": "dividends"}.get(category, category)] = Coverage(status="missing", explanation=message)
            warnings.append(message)
        if "shares" not in coverage:
            coverage["shares"] = Coverage(status="missing", explanation="Company snapshot not loaded.")
        if snapshot is not None:
            try:
                if not isinstance(snapshot, list):
                    raise ValueError("Company snapshot must be a list")
                values = {}
                for row in snapshot:
                    if not isinstance(row, dict) or not info.get("isin") or row.get("isin") != info["isin"]:
                        raise ValueError("Snapshot ISIN mismatch or missing identity")
                    key = row.get("metricMetric")
                    if key in {"FF_COM_SHS_OUT", "FF_SHS_FLOAT"}:
                        if key in values:
                            raise ValueError("Duplicate share observation")
                        values[key] = decimal_or_none(row.get("afValue"))
                shares = Shares(total_outstanding=values.get("FF_COM_SHS_OUT"),
                                free_float=values.get("FF_SHS_FLOAT"))
                refs["company_snapshot"] = SourceRef(provider="Sarmaaya standardized provider values",
                    url=f"https://beta-restapi.sarmaaya.pk/api/stocks/details/{symbol}",
                    acquisition_method="POST via SarmaayaClient.get_company_snapshot", acquired_at=now)
                for key, value in (("total_outstanding", shares.total_outstanding), ("free_float", shares.free_float)):
                    if value is not None:
                        facts[key] = Fact(fact_id=key, value=value, unit="shares/units",
                            basis="provider shares; effective date unknown", source_ref="company_snapshot",
                            quality="provider-standardized; source effective date unverified")
                coverage["shares"] = Coverage(status="partial" if values else "missing",
                    explanation="Provider share counts only; effective dates and trading-unit compatibility unverified. Snapshot ratios are not annual/TTM EPS or dividend events.")
                if values:
                    warnings.append("Share effective dates absent; do not infer them from retrieval time.")
            except (ValueError, KeyError, TypeError, InvalidOperation) as exc:
                shares = Shares()
                coverage["shares"] = Coverage(status="missing", explanation=f"Snapshot validation failed ({type(exc).__name__}); no share values accepted.")
                warnings.append(coverage["shares"].explanation)
        for category, frame, statement_type, endpoint, method in (
            ("income", income, "income_statement", "income-statement", "get_income_statement_df"),
            ("balance", balance, "balance_sheet", "balance-sheet", "get_balance_sheet_df"),
            ("cashflow", cashflow, "cash_flow", "cashflow", "get_cash_flow_df"),
            ("ratios", ratios, "ratios", "ratios", "get_financial_ratios_df"),
        ):
            if frame is None:
                continue
            try:
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError("Fundamentals response must be a DataFrame")
                if not frame.empty and not info.get("isin"):
                    raise ValueError("Fundamentals source requires an identified ISIN")
                source_ref = f"{category}_annual"
                grouped = {}
                income_facts = {}
                monetary_metrics = {
                    "FF_SALES", "FF_COGS", "FF_GROSS_INC", "FF_SGA", "FF_RD_EXP",
                    "FF_DEP_AMORT_EXP", "FF_OPER_EXP_OTH", "FF_OPER_EXP", "FF_OPER_INC",
                    "FF_EBITDA_OPER", "FF_INT_INC", "FF_INT_EXP_NET", "FF_INT_INC_NET",
                    "FF_PTX_INC", "FF_NET_INC", "FF_DIV_PFD",
                }
                canonical_keys = {
                    "FF_SALES": "revenue", "FF_COGS": "cost_of_sales", "FF_GROSS_INC": "gross_profit",
                    "FF_SGA": "selling_and_admin_expenses", "FF_RD_EXP": "research_and_development_expenses",
                    "FF_DEP_AMORT_EXP": "depreciation_and_amortization", "FF_OPER_EXP_OTH": "other_operating_expenses",
                    "FF_OPER_EXP": "operating_expenses", "FF_OPER_INC": "operating_profit",
                    "FF_EBITDA_OPER": "ebitda", "FF_INT_INC": "interest_income",
                    "FF_INT_EXP_NET": "net_interest_expense", "FF_INT_INC_NET": "net_interest_income",
                    "FF_PTX_INC": "profit_before_tax", "FF_NET_INC": "net_income", "FF_DIV_PFD": "preferred_dividends",
                }
                for observation in frame.to_dict(orient="records"):
                    metric, display, label = observation.get("metric"), observation.get("display_type"), observation.get("name")
                    if (not all(isinstance(item, str) and item for item in (metric, display, label))
                            or observation.get("statement_type") != statement_type
                            or observation.get("periodicity") != "ANN"):
                        raise ValueError("Unexpected normalized fundamentals schema or period")
                    end = pd.to_datetime(observation.get("period_end"), utc=True)
                    if pd.isna(end) or end is None:
                        raise ValueError("Missing financial period end")
                    end = end.date()
                    fact_id = f"{category}:{metric}:{end}"
                    if any(line.fact_id == fact_id for line in grouped.get(end, [])):
                        raise ValueError("Duplicate provider financial observation")
                    value = decimal_or_none(observation.get("value"))
                    if value is not None and not value.is_finite():
                        raise ValueError("Nonfinite financial observation")
                    scaled = category == "income" and metric in monetary_metrics
                    normalized_value = value if category != "income" else None
                    if value is not None and scaled:
                        with localcontext() as context:
                            context.prec = 28
                            normalized_value = value * Decimal(1000000)
                    grouped.setdefault(end, []).append(FinancialLine(
                        reported_label=label, canonical_key=canonical_keys.get(metric) if category == "income" else None,
                        decimal_value=normalized_value, fact_id=fact_id))
                    if value is not None:
                        income_facts[fact_id] = Fact(fact_id=fact_id, reported_label=label, value=value,
                            unit="million PKR (user assumption)" if scaled else f"unverified provider unit/scale; displayType={display}",
                            period=str(end), basis="unknown", source_ref=source_ref,
                            quality=f"provider-standardized; metric={metric}; displayType={display}; numeric-coerced by client DataFrame, original strings/precision not retained; provider year={observation.get('year')}; fiscal period={observation.get('fiscal_period')}; accounting basis unverified")
                periods = [FinancialPeriod(period_end=end, period_kind="annual", basis="unknown",
                    currency="PKR", original_scale="million PKR (user assumption; monetary rows only)" if category == "income" else "unverified provider units/scale; PKR currency assumption",
                    source_ref=source_ref,
                    lines=grouped[end]) for end in sorted(grouped, reverse=True)]
                income_warnings = []
                for index, period in enumerate(periods):
                    if category != "balance" and index + 1 < len(periods):
                        previous_end = periods[index + 1].period_end
                        if (period.period_end.year == previous_end.year + 1
                                and (period.period_end.month, period.period_end.day) == (previous_end.month, previous_end.day)):
                            period.period_start = previous_end + timedelta(days=1)
                            income_warnings.append(f"{category} period ending {period.period_end}: start {period.period_start} derived from previous annual end + 1 day, not supplied by provider.")
                    if period.period_start is None:
                        income_warnings.append(f"{category} period ending {period.period_end}: point-in-time balance" if category == "balance" else f"{category} period ending {period.period_end}: start unknown; no adjacent comparable annual end available.")
                for end, lines in grouped.items():
                    counts = {}
                    for line in lines:
                        fact = income_facts.get(line.fact_id)
                        if fact is not None and fact.value != 0:
                            counts[fact.value] = counts.get(fact.value, 0) + 1
                    repeated = {value for value, count in counts.items() if count >= 3}
                    if repeated:
                        income_warnings.append(f"{category} period {end}: identical nonzero values appear in at least three different metrics; verify against issuer statements before use.")
                        for line in lines:
                            fact = income_facts.get(line.fact_id)
                            if fact is not None and fact.value in repeated:
                                fact.quality += "; repeated cross-metric value requires issuer verification"
                refs[source_ref] = SourceRef(provider="Sarmaaya standardized provider values",
                    url=f"https://beta-restapi.sarmaaya.pk/api/stocks/fundamentals/{endpoint}?isin={info.get('isin', '')}&periodicity=ANN",
                    acquisition_method=f"GET via SarmaayaClient.{method} (isolated session; tidy numeric-coerced DataFrame)", acquired_at=now)
                facts.update(income_facts)
                warnings.extend(income_warnings)
                if category in statements:
                    statements[category] = periods
                coverage[category] = Coverage(status="partial" if periods else "missing",
                    explanation=f"{len(periods)} provider annual-date groups of standardized {category} observations; not verified issuer statements. Numeric-coerced provider values retained in facts, not original strings or precision. Accounting basis unknown. "
                    + ("Known income monetary rows use user-assumed million PKR; EPS/percentages not scaled. " if category == "income" else "Units/scale unverified; no million-PKR scaling applied. ")
                    + ("Balance sheet is point-in-time; no period start derived." if category == "balance" else "Starts derived only from adjacent comparable annual ends."))
                if periods:
                    warnings.append(f"{category} fundamentals are numeric-coerced, not raw provider observations; units and accounting basis require verification. No annual/TTM EPS inferred.")
                    if category == "income":
                        warnings.append("Income currency PKR is user-supplied; million PKR monetary scale is a user assumption, not provider-verified.")
            except (ValueError, KeyError, TypeError, InvalidOperation) as exc:
                coverage[category] = Coverage(status="missing", explanation=f"Annual {category} validation failed ({type(exc).__name__}); quote/history remain usable.")
                warnings.append(coverage[category].explanation)
        if announcements is not None:
            try:
                if not isinstance(announcements, pd.DataFrame):
                    raise ValueError("Announcements must be a DataFrame")
                announcement_refs = {}
                seen = {}
                for row in announcements.to_dict(orient="records"):
                    if row.get("symbol") != symbol or not isinstance(row.get("sysAnnouncementId"), str):
                        raise ValueError("Announcement identity mismatch")
                    identity = row["sysAnnouncementId"]
                    if identity in seen and seen[identity] != row:
                        raise ValueError("Conflicting duplicate announcement")
                    seen[identity] = row
                    title = row.get("announcementTitle")
                    if not isinstance(title, str):
                        raise ValueError("Announcement title missing")
                    if re.search(r"\b(dividend|distribution)\b", title, re.IGNORECASE):
                        announcement_refs[f"dividend_announcement:{identity}"] = SourceRef(
                            provider=f"Sarmaaya announcement: {title}",
                            url=next((attachment for attachment in row.get("attachments", [])
                                if isinstance(attachment, str) and attachment.startswith("https://dps.psx.com.pk/download/")),
                                f"https://beta-restapi.sarmaaya.pk/api/stocks/announcements/{symbol}"),
                            acquisition_method=f"GET via SarmaayaClient.get_announcements; record {identity}; attachment figures not extracted",
                            acquired_at=now)
                refs.update(announcement_refs)
                coverage["dividends"] = Coverage(status="missing",
                    explanation=f"{len(announcement_refs)} distinct dividend/distribution announcement references available, not verified distribution events. Posting dates are not ex/declaration dates; amounts, per-unit basis and event status unverified. Yield summaries and preferred-dividend income rows are not dividend events. No records does not confirm no payouts.")
                if announcement_refs:
                    warnings.append("Dividend announcements need verified attachment/event data before cash-per-unit, ex-date, status or trailing yield can be populated.")
            except (ValueError, KeyError, TypeError) as exc:
                coverage["dividends"] = Coverage(status="missing", explanation=f"Dividend announcements failed validation ({type(exc).__name__}); no events accepted.")
                warnings.append(coverage["dividends"].explanation)
        distributions = []
        dividend_note = coverage["dividends"].explanation
        if dividends is not None:
            try:
                if not isinstance(dividends, pd.DataFrame):
                    raise ValueError("Dividend history must be a DataFrame")
                events = {}
                payout_warnings = []
                for row in dividends.to_dict(orient="records"):
                    if row.get("symbol") != symbol:
                        raise ValueError("Dividend symbol mismatch")
                    if row.get("payoutType") != "dividend":
                        payout_warnings.append("Non-dividend/placeholder payout row excluded; not evidence of zero distributions.")
                        continue
                    raw_date = row.get("announcementDate")
                    if not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date) or raw_date == "0000-00-00":
                        payout_warnings.append("Dividend row without a usable announcement date excluded; no date invented.")
                        continue
                    announced = pd.to_datetime(raw_date, utc=True).date()
                    amount = decimal_or_none(row.get("dividendPerShare"))
                    face = decimal_or_none(row.get("faceValue"))
                    percentage = decimal_or_none(row.get("percentage"))
                    for value in (amount, face, percentage):
                        if value is not None and (not value.is_finite() or value < 0):
                            raise ValueError("Dividend amounts must be finite and nonnegative")
                    observation = {
                        "dividend_per_share": str(amount) if amount is not None else None,
                        "face_value": str(face) if face is not None else None,
                        "percentage": str(percentage) if percentage is not None else None,
                    }
                    identity = f"dividend:{symbol}:{announced}"
                    if identity in events:
                        event = events[identity]
                        if observation not in event["reported_observations"]:
                            event["reported_observations"].append(observation)
                            event["cash_per_unit"] = None
                            event["warnings"].append("Conflicting same-date records cannot establish distinct events; cash amount withheld.")
                        continue
                    event_warnings = ["Ex-date, declaration date and cancellation/supersession status not supplied; excluded from trailing DPS/yield."]
                    cash = amount
                    if amount is not None and face is not None and percentage is not None:
                        with localcontext() as context:
                            context.prec = 28
                            expected = face * percentage / Decimal("100")
                            if abs(amount - expected) > Decimal("0.005"):
                                cash = None
                                event_warnings.append("Reported per-share amount conflicts with face value times percentage; not corrected or used.")
                    if quote.market_date is not None and announced > quote.market_date:
                        event_warnings.append("Announcement dated after selected quote; not historical as-of that quote.")
                    events[identity] = {
                        "event_id": identity, "cash_per_unit": str(cash) if cash is not None else None,
                        "currency": "PKR", "announcement_date": str(announced),
                        "declaration_date": None, "ex_date": None, "status": "unknown",
                        "source_ref": "dividend_history", "quality": "provider-reported; event lifecycle unverified",
                        "eligible_for_trailing_dps": False, "reported_observations": [observation],
                        "warnings": event_warnings,
                    }
                distributions = [events[key] for key in sorted(events, reverse=True)]
                refs["dividend_history"] = SourceRef(provider="Sarmaaya payoutHistory",
                    url=f"https://beta-restapi.sarmaaya.pk/api/stocks/dividends/{symbol}",
                    acquisition_method="GET via SarmaayaClient.get_dividends; payoutHistory only, not annual yield/DPS summaries",
                    acquired_at=now)
                for event in distributions:
                    if event["cash_per_unit"] is not None:
                        key = f"{event['event_id']}:cash_per_unit"
                        facts[key] = Fact(fact_id=key, value=Decimal(event["cash_per_unit"]), unit="PKR per unit",
                            period=event["announcement_date"], basis="provider per-share amount; announcement date, not ex-date",
                            source_ref="dividend_history", quality="provider-reported; excludes unknown lifecycle from trailing calculations")
                warnings.extend(payout_warnings)
                coverage["dividends"] = Coverage(status="partial" if distributions else "missing",
                    explanation=f"{len(distributions)} deduplicated provider cash-dividend announcement records. Per-share amounts retained unless conflicting; raw observations preserved. Ex-dates and lifecycle status unknown; no complete trailing coverage or confirmed-zero claim. {dividend_note}")
            except (ValueError, KeyError, TypeError, InvalidOperation) as exc:
                distributions = []
                coverage["dividends"] = Coverage(status="missing",
                    explanation=f"Dividend history validation failed ({type(exc).__name__}); no payout records accepted. {dividend_note}")
                warnings.append(coverage["dividends"].explanation)
        if failures and "dividends" in failures:
            coverage["dividends"] = Coverage(status="missing", explanation=f"{failures['dividends']} {dividend_note}")
        dividend_summaries = {"dps": {}, "yield": {}}
        summary_warnings = []
        if isinstance(dividends, pd.DataFrame):
            for metric, attribute in (("dps", "dividendsPerShare"), ("yield", "dividendsYield")):
                reported = dividends.attrs.get(attribute)
                if reported is None:
                    continue
                try:
                    if not isinstance(reported, list) or not all(isinstance(row, dict) for row in reported):
                        raise ValueError("Summary must be a list of year-value objects")
                    accepted = {}
                    for row in reported:
                        for year, raw_value in row.items():
                            if not isinstance(year, str) or not re.fullmatch(r"\d{4}", year):
                                raise ValueError("Summary year must contain four digits")
                            raw = raw_value.strip() if isinstance(raw_value, str) else raw_value
                            if metric == "yield" and isinstance(raw, str):
                                raw = raw.removesuffix("%")
                            value = decimal_or_none(raw)
                            if value is None:
                                continue
                            if not value.is_finite() or value < 0:
                                raise ValueError("Summary values must be finite and nonnegative")
                            if year in accepted and accepted[year].value != value:
                                raise ValueError("Conflicting summary values for the same year")
                            accepted[year] = Fact(fact_id=f"dividend_summary:{metric}:{year}", value=value,
                                unit="PKR per unit" if metric == "dps" else "percent", period=year,
                                basis="provider_reported_by_year", source_ref="dividend_summaries",
                                quality=f"provider-reported; original value={raw_value}; calculation period/basis unverified; not app-calculated trailing DPS/yield")
                    dividend_summaries[metric] = accepted
                except (ValueError, TypeError, InvalidOperation) as exc:
                    summary_warnings.append(f"Provider yearly {metric} summary failed validation ({type(exc).__name__}); payout history remains independent.")
        count = sum(len(values) for values in dividend_summaries.values())
        if count:
            refs["dividend_summaries"] = SourceRef(provider="Sarmaaya provider-reported yearly DPS/yield",
                url=f"https://beta-restapi.sarmaaya.pk/api/stocks/dividends/{symbol}",
                acquisition_method="GET via SarmaayaClient.get_dividends; dividendsPerShare/dividendsYield DataFrame attrs",
                acquired_at=now)
            for values in dividend_summaries.values():
                facts.update({fact.fact_id: fact for fact in values.values()})
        coverage["dividend_summaries"] = Coverage(status="partial" if count else "missing",
            explanation=f"{count} provider-reported year-keyed DPS/yield observations. DPS in PKR per unit; yield in percent. Original values preserved in fact quality; calculation period/basis unverified. Not reconstructed trailing-365-day metrics; no summary records does not mean zero dividends.")
        warnings.extend(summary_warnings)
        dates = sorted({bar.date for bar in bars} | ({quote.market_date} if quote.market_date else set()))
        dataset_id = f"sarmaaya-{symbol.lower()}"
        bundle = TickerBundle(symbol=symbol, company_name=info["name"], sector=info.get("sectorName"),
            quote=quote, shares=shares, ohlcv=bars, financials=statements, dividends=distributions,
            dividend_summaries=dividend_summaries,
            source_refs=refs, facts=facts, coverage=coverage, warnings=warnings,
            dataset_id=dataset_id, fetched_at=now,
            manifest=DatasetManifest(dataset_id=dataset_id, acquisition_time=now,
                market_dates=dates, supported_symbols=[symbol]))
        bundle.bundle_hash = content_hash(bundle)
        bundle.manifest.content_hash = bundle.bundle_hash
        return bundle


class ResearchService:
    """Memory-only cache. A failed reload never replaces validated content."""

    def __init__(self, adapter=None):
        self.adapter = adapter or SarmaayaAdapter()
        self._cache = {}
        self._lock = Lock()

    def get(self, symbol: str, *, reload=False) -> TickerBundle:
        symbol = normalize_symbol(symbol)
        with self._lock:
            cached = self._cache.get(symbol)
            if cached is not None and not reload:
                return cached.model_copy(deep=True)
            try:
                bundle = self.adapter.fetch(symbol)
            except DataUnavailable:
                if cached is None:
                    raise
                fallback = cached.model_copy(deep=True)
                fallback.warnings.append("Provider unavailable; using previously validated in-memory data with original dates.")
                return fallback
            self._cache[symbol] = bundle.model_copy(deep=True)
            return bundle
