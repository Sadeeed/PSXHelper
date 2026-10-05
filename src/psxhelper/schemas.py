"""Normalized research contracts. Decimal values serialize as JSON strings."""

from datetime import date, datetime
from decimal import Decimal
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]
NonnegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssistantRequest(Record):
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16)]
    bundle_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)]
    period: date | None = None
    basis: Literal["consolidated", "unconsolidated", "unknown"] = "unknown"


class AssistantAnswer(Record):
    explanation: Annotated[str, Field(min_length=1, max_length=2000)]
    fact_refs: Annotated[list[str], Field(max_length=12)]
    caveats: Annotated[list[Annotated[str, Field(min_length=1, max_length=400)]], Field(max_length=8)]
    next_question: Annotated[str, Field(min_length=1, max_length=400)]

    @model_validator(mode="after")
    def reject_numeric_prose(self):
        if re.search(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|billion|trillion)\b",
                     " ".join([self.explanation, *self.caveats, self.next_question]), re.IGNORECASE):
            raise ValueError("Use evidence references, not generated numeric claims")
        if len(set(self.fact_refs)) != len(self.fact_refs):
            raise ValueError("Evidence references must be unique")
        return self


class Quote(Record):
    price: NonnegativeDecimal | None = None
    price_kind: str = "provider_close"
    market_date: date | None = None
    source_time: datetime | None = None
    volume: NonnegativeDecimal | None = None
    source_ref: str = "stock_info"


class Shares(Record):
    total_outstanding: NonnegativeDecimal | None = None
    free_float: NonnegativeDecimal | None = None
    effective_date: date | None = None


class DailyBar(Record):
    date: date
    open: NonnegativeDecimal | None = None
    high: NonnegativeDecimal | None = None
    low: NonnegativeDecimal | None = None
    close: NonnegativeDecimal
    volume: NonnegativeDecimal | None = None
    source_ref: str = "price_history"
    market_cap: NonnegativeDecimal | None = None


class FinancialLine(Record):
    reported_label: str
    canonical_key: str | None = None
    decimal_value: FiniteDecimal | None = None
    fact_id: str


class FinancialPeriod(Record):
    period_start: date | None = None
    period_end: date
    period_kind: Literal["annual", "ttm_reported"]
    basis: Literal["consolidated", "unconsolidated", "unknown"]
    currency: str
    original_scale: str
    source_ref: str
    lines: list[FinancialLine]


class Financials(Record):
    income: list[FinancialPeriod] = Field(default_factory=list)
    balance: list[FinancialPeriod] = Field(default_factory=list)
    cashflow: list[FinancialPeriod] = Field(default_factory=list)


class Fact(Record):
    fact_id: str
    reported_label: str | None = None
    value: FiniteDecimal
    unit: str
    period: str | None = None
    basis: str
    source_ref: str
    quality: str


class SourceRef(Record):
    provider: str
    url: str
    acquisition_method: str
    acquired_at: datetime


class Coverage(Record):
    status: Literal["available", "partial", "missing", "unknown"]
    explanation: str


class DatasetManifest(Record):
    dataset_id: str
    content_hash: str = ""
    data_mode: Literal["remote_unverified"] = "remote_unverified"
    provider: str = "Sarmaaya"
    acquisition_method: str = "User-selected SarmaayaClient API"
    acquisition_time: datetime
    market_dates: list[date]
    permission_evidence: str = "Not supplied; API availability is not permission evidence."
    retention_conditions: str = "Process memory only; no persistent provider fixtures."
    demonstration_conditions: str = "Public presentation and redistribution rights unverified."
    supported_symbols: list[str]
    currency: str = "PKR"
    scale: str = "PKR per trading unit; volume in provider units"
    statement_basis: str = "unknown"
    adjustment_basis: str = "unverified"


class TickerBundle(Record):
    symbol: str
    company_name: str
    security_type: str | None = None
    currency: str = "PKR"
    business_description: str | None = None
    sector: str | None = None
    quote: Quote
    shares: Shares = Field(default_factory=Shares)
    ohlcv: list[DailyBar] = Field(default_factory=list)
    financials: Financials = Field(default_factory=Financials)
    eps: dict[str, Fact | None] = Field(default_factory=lambda: {
        "reported_annual": None, "reported_ttm": None,
    })
    dividends: list = Field(default_factory=list)
    dividend_summaries: dict[
        Literal["dps", "yield"],
        dict[Annotated[str, Field(pattern=r"^\d{4}$")], Fact],
    ] = Field(default_factory=lambda: {"dps": {}, "yield": {}})
    index_memberships: list = Field(default_factory=list)
    facts: dict[str, Fact] = Field(default_factory=dict)
    source_refs: dict[str, SourceRef]
    coverage: dict[str, Coverage]
    warnings: list[str] = Field(default_factory=list)
    dataset_id: str
    bundle_hash: str = ""
    data_mode: Literal["remote_unverified"] = "remote_unverified"
    fetched_at: datetime
    manifest: DatasetManifest

    @model_validator(mode="after")
    def validate_references_and_dates(self):
        dates = [bar.date for bar in self.ohlcv]
        if dates != sorted(set(dates)):
            raise ValueError("Daily bars must have unique ascending dates")
        for key, fact in self.facts.items():
            if key != fact.fact_id or fact.source_ref not in self.source_refs:
                raise ValueError("Invalid fact identity or source reference")
        for item in [self.quote, *self.ohlcv]:
            if item.source_ref not in self.source_refs:
                raise ValueError("Unknown observation source")
        for metric, summaries in self.dividend_summaries.items():
            for year, fact in summaries.items():
                unit = f"{self.currency} per unit" if metric == "dps" else "percent"
                if (fact.period != year or fact.source_ref not in self.source_refs
                        or fact.unit != unit or fact.value < 0
                        or fact.basis != "provider_reported_by_year"):
                    raise ValueError("Dividend summary requires matching year, source, unit and provider-reported basis")
        return self
