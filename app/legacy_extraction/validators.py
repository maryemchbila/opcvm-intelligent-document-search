"""Validations locales complementaires sans modifier les donnees Gemini."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .schemas import (
    DocumentExtraction,
    FinancialMetric,
    FinancialMetricType,
    conservative_numeric_value,
)

ValidationStatus = Literal["SUCCESS", "REVIEW_REQUIRED"]

_ALLOCATION_TYPES = {
    FinancialMetricType.ALLOCATION,
    FinancialMetricType.BOND_ALLOCATION,
    FinancialMetricType.CASH_ALLOCATION,
    FinancialMetricType.EQUITY_ALLOCATION,
}
_EXPECTED_GLOBAL_CATEGORIES = (
    "Obligation",
    "Placement monétaires",
    "Liquidités",
    "OPCVM",
    "Actions",
)
_SUMMED_GLOBAL_CATEGORIES = (
    "Obligation",
    "Placement monétaires",
    "Liquidités",
)
_MATURITY_PATTERN = re.compile(
    r"\b(?:maturite|moins de|plus de|entre\s+\d+\s+(?:a|et)\s+\d+\s+ans?)\b"
    r"|[<>]\s*\d+\s*ans?"
)
_ISSUER_PATTERN = re.compile(r"\bemprunts?\s+(?:nationaux?|societes?)\b")


@dataclass(frozen=True, slots=True)
class ValidationWarning:
    code: str
    severity: Literal["WARNING"]
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BusinessValidationResult:
    validation_status: ValidationStatus
    validation_warnings: tuple[ValidationWarning, ...]


def _warning(code: str, message: str) -> ValidationWarning:
    return ValidationWarning(code=code, severity="WARNING", message=message)


def _normalized_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    return "".join(
        character for character in text if not unicodedata.combining(character)
    ).lower()


def _allocation_context(metric: FinancialMetric) -> str:
    return _normalized_text(" ".join(filter(None, (metric.label, metric.source_text))))


def _allocation_group_and_category(
    metric: FinancialMetric,
) -> tuple[str | None, str | None]:
    if metric.metric_type not in _ALLOCATION_TYPES:
        return None, None

    context = _allocation_context(metric)
    if metric.maturity is not None or _MATURITY_PATTERN.search(context):
        return "BOND_MATURITY_ALLOCATION", None
    if _ISSUER_PATTERN.search(context):
        return "BOND_ISSUER_ALLOCATION", None
    if "placement monetaire" in context:
        return "GLOBAL_ALLOCATION", "Placement monétaires"
    if re.search(r"\b(?:liquidites?|cash|quasi[- ]liquidites?)\b", context):
        return "GLOBAL_ALLOCATION", "Liquidités"
    if re.search(r"\bopcvm\b", context):
        return "GLOBAL_ALLOCATION", "OPCVM"
    if re.search(r"\b(?:actions?|equity)\b", context):
        return "GLOBAL_ALLOCATION", "Actions"
    if "obligat" in context:
        return "GLOBAL_ALLOCATION", "Obligation"
    return None, None


def validate_allocation_business_rules(
    extraction: DocumentExtraction,
    *,
    sum_tolerance: float = 1.0,
) -> BusinessValidationResult:
    """Controle les allocations sans corriger ni reconstruire aucune valeur."""
    if sum_tolerance < 0:
        raise ValueError("La tolerance de somme doit etre positive ou nulle.")

    warnings: list[ValidationWarning] = []
    global_categories: dict[str, list[FinancialMetric]] = {
        category: [] for category in _EXPECTED_GLOBAL_CATEGORIES
    }
    for metric in extraction.financial_information:
        group, category = _allocation_group_and_category(metric)
        if group == "GLOBAL_ALLOCATION" and category is not None:
            global_categories[category].append(metric)

    # Sans indice d'une repartition globale, aucune categorie n'est supposee visible.
    if not any(global_categories.values()):
        return BusinessValidationResult("SUCCESS", ())

    missing_categories = [
        category for category, metrics in global_categories.items() if not metrics
    ]
    for category in missing_categories:
        warnings.append(
            _warning(
                "ALLOCATION_CATEGORY_MISSING",
                f"{category} absente de la repartition globale.",
            )
        )

    if missing_categories:
        warnings.append(
            _warning(
                "ALLOCATION_GROUP_INCOMPLETE",
                "La repartition globale ne contient pas toutes les categories attendues.",
            )
        )

    for category, metrics in global_categories.items():
        if len(metrics) < 2:
            continue
        percentages = {
            metric.percentage for metric in metrics if metric.percentage is not None
        }
        if len(percentages) > 1:
            warnings.append(
                _warning(
                    "ALLOCATION_CONFLICT",
                    f"Plusieurs valeurs contradictoires existent pour {category}.",
                )
            )
        else:
            warnings.append(
                _warning(
                    "ALLOCATION_DUPLICATE_LABEL",
                    f"Plusieurs entrees existent pour la categorie {category}.",
                )
            )

    percentages_by_value: dict[float, list[str]] = {}
    for category, metrics in global_categories.items():
        if len(metrics) != 1 or metrics[0].percentage is None:
            continue
        percentage = metrics[0].percentage
        if percentage <= 0:
            continue
        percentages_by_value.setdefault(round(percentage, 9), []).append(category)
    for percentage, categories in percentages_by_value.items():
        if len(categories) > 1:
            warnings.append(
                _warning(
                    "ALLOCATION_CONFLICT",
                    "Des categories differentes partagent le meme pourcentage "
                    f"inhabituel ({percentage:g} %).",
                )
            )

    summed_metrics: list[FinancialMetric] = []
    sum_is_available = True
    for category in _SUMMED_GLOBAL_CATEGORIES:
        metrics = global_categories[category]
        if len(metrics) != 1 or metrics[0].percentage is None:
            sum_is_available = False
            break
        summed_metrics.append(metrics[0])

    if sum_is_available:
        total = sum(metric.percentage or 0 for metric in summed_metrics)
        if abs(total - 100.0) > sum_tolerance:
            warnings.append(
                _warning(
                    "ALLOCATION_SUM_INVALID",
                    f"La somme de la repartition globale est {total:g} %, hors tolerance.",
                )
            )
    elif not missing_categories:
        warnings.append(
            _warning(
                "ALLOCATION_GROUP_INCOMPLETE",
                "La somme globale ne peut pas etre verifiee faute de pourcentages complets.",
            )
        )

    status: ValidationStatus = "REVIEW_REQUIRED" if warnings else "SUCCESS"
    return BusinessValidationResult(status, tuple(warnings))


def validate_pdf_path(value: str | Path) -> Path:
    """Valide le fichier PDF explicitement demande."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable: {path.name}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Le fichier fourni doit avoir l'extension .pdf.")
    with path.open("rb") as pdf_file:
        if pdf_file.read(5) != b"%PDF-":
            raise ValueError("Le fichier ne semble pas etre un PDF valide.")
    return path


def validate_extraction(extraction: DocumentExtraction) -> DocumentExtraction:
    """Verifie les valeurs sans modifier l'extraction Gemini."""
    for metric in extraction.financial_information:
        if metric.numeric_value is not None:
            parsed = conservative_numeric_value(metric.raw_value)
            if parsed is None:
                raise ValueError("numeric_value doit rester null si raw_value est ambigue.")
    return extraction
