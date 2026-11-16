from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.legacy_extraction.schemas import (
    DocumentExtraction,
    DocumentType,
    DocumentTypeInfo,
    FinancialMetric,
    FinancialMetricType,
    FundFieldInfo,
    FundInformation,
    HistoricalPerformanceInfo,
    InvestmentPolicyInfo,
)


class SchemaTests(unittest.TestCase):
    def test_document_extraction_defaults_match_expected_shape(self) -> None:
        result = DocumentExtraction.model_validate({})
        self.assertIsNone(result.document_type)
        self.assertIsNone(result.document_title)
        self.assertEqual(result.opcvm, [])
        self.assertIsNone(result.fund_information)
        self.assertEqual(result.investment_policy, [])
        self.assertEqual(result.historical_performance, [])
        self.assertEqual(result.financial_information, [])

    def test_monthly_fact_sheet_is_supported_as_explicit_type(self) -> None:
        value = DocumentTypeInfo(
            value=DocumentType.MONTHLY_FACT_SHEET,
            source_method="explicit",
            page_number=1,
            source_text="FICHE MENSUELLE",
            confidence=0.99,
        )
        self.assertEqual(value.value, DocumentType.MONTHLY_FACT_SHEET)

    def test_fund_information_supports_provenance_and_absent_fields(self) -> None:
        manager = FundFieldInfo(
            value="Gestionnaire test",
            page_number=1,
            source_text="Gestionnaire : Gestionnaire test",
            confidence=0.9,
            source_method="explicit",
        )
        information = FundInformation(manager=manager)
        self.assertEqual(information.manager.value, "Gestionnaire test")
        self.assertIsNone(information.depositary)

    def test_investment_policy_preserves_raw_text_without_forcing_numbers(self) -> None:
        policy = InvestmentPolicyInfo(
            category="Obligations",
            raw_text="Une part significative en obligations",
            page_number=1,
            source_text="Une part significative en obligations",
            confidence=0.8,
        )
        self.assertIsNone(policy.percentage)
        self.assertIsNone(policy.minimum)
        self.assertIsNone(policy.maximum)

    def test_historical_performance_rejects_ambiguous_numeric_conversion(self) -> None:
        performance = HistoricalPerformanceInfo(
            date="2025-12-31",
            raw_value="61.205%",
            numeric_value=61205,
            unit="%",
            page_number=1,
            source_text="31/12/2025 61.205%",
            confidence=0.9,
        )
        self.assertIsNone(performance.numeric_value)

    def test_existing_financial_metric_types_remain_supported(self) -> None:
        expected = {
            "NAV",
            "NET_ASSET",
            "PERFORMANCE",
            "ALLOCATION",
            "BOND_ALLOCATION",
            "CASH_ALLOCATION",
            "EQUITY_ALLOCATION",
        }
        self.assertTrue(expected.issubset({item.value for item in FinancialMetricType}))

    def test_document_type_accepts_only_controlled_values(self) -> None:
        value = DocumentTypeInfo(
            value=DocumentType.FINANCIAL_REPORT,
            source_method="inferred",
            page_number=1,
            source_text="Rapport annuel",
            confidence=0.8,
        )
        self.assertEqual(value.source_method, "inferred")
        with self.assertRaises(ValidationError):
            DocumentTypeInfo(
                value="UNKNOWN_KIND",
                source_method="inferred",
                page_number=1,
                source_text="Rapport annuel",
                confidence=0.8,
            )

    def test_confidence_range_is_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            DocumentTypeInfo(
                value="OTHER",
                source_method="explicit",
                page_number=1,
                source_text="Autre document",
                confidence=1.1,
            )

    def test_source_text_stays_short_with_complete_provenance(self) -> None:
        value = DocumentTypeInfo(
            value="MONTHLY_FACT_SHEET",
            source_method="explicit",
            page_number=1,
            source_text="FICHE MENSUELLE",
            confidence=0.99,
        )
        self.assertLessEqual(len(value.source_text), 120)
        self.assertEqual(value.page_number, 1)
        self.assertEqual(value.source_method, "explicit")
        with self.assertRaises(ValidationError):
            DocumentTypeInfo(
                value="OTHER",
                source_method="explicit",
                page_number=1,
                source_text="x" * 121,
                confidence=0.9,
            )

    def test_financial_value_preserves_raw_and_rejects_ambiguous_conversion(self) -> None:
        metric = FinancialMetric(
            metric_type=FinancialMetricType.NAV,
            raw_value="61.205",
            numeric_value=61205,
            page_number=1,
            source_text="Valeur liquidative 61.205",
            confidence=0.9,
        )
        self.assertEqual(metric.raw_value, "61.205")
        self.assertIsNone(metric.numeric_value)

    def test_currency_and_unit_are_independent(self) -> None:
        metric = FinancialMetric(
            metric_type="NET_ASSET",
            raw_value="67,722 MD",
            numeric_value=67722,
            unit="MD",
            currency=None,
            page_number=1,
            source_text="Actif net 67,722 MD",
            confidence=0.9,
        )
        self.assertIsNone(metric.numeric_value)
        self.assertEqual(metric.unit, "MD")
        self.assertIsNone(metric.currency)


if __name__ == "__main__":
    unittest.main()
