"""Modeles Pydantic de l'extraction metier PDF."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from math import isclose
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceMethod(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class DocumentType(str, Enum):
    MONTHLY_FACT_SHEET = "MONTHLY_FACT_SHEET"
    MEETING_MINUTES = "MEETING_MINUTES"
    BOARD_MINUTES = "BOARD_MINUTES"
    COMMITTEE_MINUTES = "COMMITTEE_MINUTES"
    GENERAL_ASSEMBLY_MINUTES = "GENERAL_ASSEMBLY_MINUTES"
    FINANCIAL_REPORT = "FINANCIAL_REPORT"
    AUDITOR_REPORT = "AUDITOR_REPORT"
    LEGAL_DOCUMENT = "LEGAL_DOCUMENT"
    PRESENTATION = "PRESENTATION"
    NOTICE = "NOTICE"
    GOVERNANCE_REPORT = "GOVERNANCE_REPORT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class FinancialMetricType(str, Enum):
    NET_ASSET = "NET_ASSET"
    NAV = "NAV"
    PERFORMANCE = "PERFORMANCE"
    ALLOCATION = "ALLOCATION"
    BOND_ALLOCATION = "BOND_ALLOCATION"
    EQUITY_ALLOCATION = "EQUITY_ALLOCATION"
    CASH_ALLOCATION = "CASH_ALLOCATION"
    ASSET_VALUE = "ASSET_VALUE"
    LIABILITY_VALUE = "LIABILITY_VALUE"
    OTHER = "OTHER"


class Provenance(StrictModel):
    page_number: int = Field(ge=1)
    source_text: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0, le=1)
    source_method: SourceMethod = SourceMethod.EXPLICIT


class DocumentTypeInfo(Provenance):
    value: DocumentType
    source_method: SourceMethod


class DocumentFieldInfo(Provenance):
    value: str = Field(min_length=1)
    source_method: Literal["explicit"] = "explicit"


def _validate_iso_date(value: str | None) -> str | None:
    if value is not None:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("La date doit etre une date calendaire ISO YYYY-MM-DD.") from exc
    return value


class DocumentDateInfo(DocumentFieldInfo):
    value: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "DocumentDateInfo":
        _validate_iso_date(self.value)
        return self


class OPCVMInfo(Provenance):
    name: str = Field(min_length=1)


class FundFieldInfo(Provenance):
    value: str = Field(min_length=1)


class FundInformation(StrictModel):
    manager: FundFieldInfo | None = None
    depositary: FundFieldInfo | None = None
    distributor: FundFieldInfo | None = None
    constitution_date: FundFieldInfo | None = None
    valuation_frequency: FundFieldInfo | None = None
    fund_category: FundFieldInfo | None = None
    capitalization_type: FundFieldInfo | None = None
    valuation_date: FundFieldInfo | None = None


class MeetingInfo(Provenance):
    meeting_type: str | None = None
    meeting_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    meeting_time: str | None = None
    location: str | None = None

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "MeetingInfo":
        _validate_iso_date(self.meeting_date)
        return self


class ParticipantInfo(Provenance):
    name: str = Field(min_length=1)
    role: str | None = None
    organization: str | None = None


class DecisionInfo(Provenance):
    decision_type: str | None = None
    text: str = Field(min_length=1)
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    target_entity: str | None = None

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "DecisionInfo":
        _validate_iso_date(self.date)
        return self


class ResolutionInfo(Provenance):
    resolution_number: str | None = None
    text: str = Field(min_length=1)
    status: str | None = None


class TaskInfo(Provenance):
    title: str = Field(min_length=1)
    description: str | None = None
    assigned_to: str | None = None
    due_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: str | None = None
    priority: str | None = None

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "TaskInfo":
        _validate_iso_date(self.due_date)
        return self


class InvestmentPolicyInfo(Provenance):
    category: str | None = None
    raw_text: str = Field(min_length=1)
    percentage: float | None = Field(default=None, ge=0, le=100)
    minimum: float | None = Field(default=None, ge=0, le=100)
    maximum: float | None = Field(default=None, ge=0, le=100)


class HistoricalPerformanceInfo(Provenance):
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    raw_value: str = Field(min_length=1)
    numeric_value: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_values(self) -> "HistoricalPerformanceInfo":
        _validate_iso_date(self.date)
        if self.numeric_value is not None:
            parsed = conservative_numeric_value(self.raw_value)
            if parsed is None or not isclose(parsed, self.numeric_value, rel_tol=0, abs_tol=1e-9):
                self.numeric_value = None
        return self


class MandateInfo(Provenance):
    person: str = Field(min_length=1)
    role: str | None = None
    mandate_type: str | None = None
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    appointment_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    renewal_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def validate_calendar_dates(self) -> "MandateInfo":
        for value in (self.start_date, self.end_date, self.appointment_date, self.renewal_date):
            _validate_iso_date(value)
        return self


def conservative_numeric_value(raw_value: str) -> float | None:
    """Lit seulement un nombre simple non ambigu, sans separateur de milliers."""
    matches = re.findall(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?![\w])", raw_value)
    if len(matches) != 1:
        return None
    token = matches[0]
    if token.count(".") + token.count(",") > 1:
        return None
    separator = "." if "." in token else "," if "," in token else None
    if separator is not None:
        fraction = token.rsplit(separator, 1)[1]
        if len(fraction) == 3:
            return None
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


class FinancialMetric(Provenance):
    metric_type: FinancialMetricType
    label: str | None = None
    raw_value: str = Field(min_length=1)
    numeric_value: float | None = None
    unit: str | None = None
    currency: str | None = None
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    maturity: str | None = None
    percentage: float | None = None
    source_method: Literal["explicit"] = "explicit"

    @model_validator(mode="after")
    def validate_numeric_value(self) -> "FinancialMetric":
        _validate_iso_date(self.date)
        if self.numeric_value is not None:
            parsed = conservative_numeric_value(self.raw_value)
            if parsed is None or not isclose(parsed, self.numeric_value, rel_tol=0, abs_tol=1e-9):
                self.numeric_value = None
        return self


class DocumentExtraction(StrictModel):
    document_type: DocumentTypeInfo | None = None
    document_title: DocumentFieldInfo | None = None
    document_date: DocumentDateInfo | None = None
    opcvm: list[OPCVMInfo] = Field(default_factory=list)
    fund_information: FundInformation | None = None
    investment_policy: list[InvestmentPolicyInfo] = Field(default_factory=list)
    historical_performance: list[HistoricalPerformanceInfo] = Field(default_factory=list)
    meeting: MeetingInfo | None = None
    participants: list[ParticipantInfo] = Field(default_factory=list)
    decisions: list[DecisionInfo] = Field(default_factory=list)
    resolutions: list[ResolutionInfo] = Field(default_factory=list)
    tasks: list[TaskInfo] = Field(default_factory=list)
    mandates: list[MandateInfo] = Field(default_factory=list)
    financial_information: list[FinancialMetric] = Field(default_factory=list)
