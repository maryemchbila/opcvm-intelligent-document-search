from __future__ import annotations

import unittest

from app.legacy_extraction.schemas import DocumentExtraction
from app.legacy_extraction.validators import validate_allocation_business_rules


def _metric(
    label: str,
    percentage: float,
    *,
    metric_type: str = "ALLOCATION",
    maturity: str | None = None,
) -> dict[str, object]:
    raw_value = f"{percentage:g}%"
    return {
        "metric_type": metric_type,
        "label": label,
        "raw_value": raw_value,
        "percentage": percentage,
        "maturity": maturity,
        "page_number": 1,
        "source_text": f"{label} {raw_value}",
        "confidence": 0.95,
    }


def _extraction(*metrics: dict[str, object]) -> DocumentExtraction:
    return DocumentExtraction.model_validate({"financial_information": list(metrics)})


class AllocationBusinessValidationTests(unittest.TestCase):
    def _correct_metrics(self) -> list[dict[str, object]]:
        return [
            _metric("Obligation", 58.30, metric_type="BOND_ALLOCATION"),
            _metric("Placement monétaires", 29.52),
            _metric("Liquidités", 12.18, metric_type="CASH_ALLOCATION"),
            _metric("OPCVM", 4.25),
            _metric("Actions", 0.02, metric_type="EQUITY_ALLOCATION"),
        ]

    def test_correct_global_allocation_is_success(self) -> None:
        extraction = _extraction(*self._correct_metrics())
        before = extraction.model_dump()
        result = validate_allocation_business_rules(extraction)
        self.assertEqual(result.validation_status, "SUCCESS")
        self.assertEqual(result.validation_warnings, ())
        self.assertEqual(extraction.model_dump(), before)

    def test_missing_liquidity_requires_review(self) -> None:
        metrics = [
            metric for metric in self._correct_metrics() if metric["label"] != "Liquidités"
        ]
        result = validate_allocation_business_rules(_extraction(*metrics))
        codes = {warning.code for warning in result.validation_warnings}
        self.assertEqual(result.validation_status, "REVIEW_REQUIRED")
        self.assertIn("ALLOCATION_CATEGORY_MISSING", codes)
        self.assertIn("ALLOCATION_GROUP_INCOMPLETE", codes)

    def test_invalid_global_sum_requires_review(self) -> None:
        metrics = self._correct_metrics()
        metrics[1] = _metric("Placement monétaires", 20.0)
        result = validate_allocation_business_rules(_extraction(*metrics))
        self.assertIn(
            "ALLOCATION_SUM_INVALID",
            {warning.code for warning in result.validation_warnings},
        )

    def test_duplicate_label_is_reported(self) -> None:
        metrics = self._correct_metrics()
        metrics.append(_metric("OPCVM", 4.25))
        result = validate_allocation_business_rules(_extraction(*metrics))
        self.assertIn(
            "ALLOCATION_DUPLICATE_LABEL",
            {warning.code for warning in result.validation_warnings},
        )

    def test_bond_maturities_are_excluded_from_global_sum(self) -> None:
        metrics = self._correct_metrics()
        metrics.extend(
            [
                _metric("< 1 an", 28.05, metric_type="BOND_ALLOCATION", maturity="< 1 an"),
                _metric(
                    "entre 1 à 3 ans",
                    15.12,
                    metric_type="BOND_ALLOCATION",
                    maturity="entre 1 à 3 ans",
                ),
                _metric(
                    "entre 3 à 5 ans",
                    12.46,
                    metric_type="BOND_ALLOCATION",
                    maturity="entre 3 à 5 ans",
                ),
                _metric("> 5 ans", 3.11, metric_type="BOND_ALLOCATION", maturity="> 5 ans"),
            ]
        )
        result = validate_allocation_business_rules(_extraction(*metrics))
        self.assertEqual(result.validation_status, "SUCCESS")

    def test_conflicting_values_for_same_label_are_reported(self) -> None:
        metrics = self._correct_metrics()
        metrics.append(_metric("Actions", 2.0, metric_type="EQUITY_ALLOCATION"))
        result = validate_allocation_business_rules(_extraction(*metrics))
        self.assertIn(
            "ALLOCATION_CONFLICT",
            {warning.code for warning in result.validation_warnings},
        )

    def test_same_percentage_for_different_categories_is_suspicious(self) -> None:
        metrics = self._correct_metrics()
        metrics[4] = _metric("Actions", 4.25, metric_type="EQUITY_ALLOCATION")
        result = validate_allocation_business_rules(_extraction(*metrics))
        self.assertIn(
            "ALLOCATION_CONFLICT",
            {warning.code for warning in result.validation_warnings},
        )


if __name__ == "__main__":
    unittest.main()
