from __future__ import annotations

from typing import Annotated, Literal

from pydantic import PlainSerializer, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, StrictBool


IssueTuple = Annotated[
    tuple["ValidationIssue", ...],
    PlainSerializer(lambda value: list(value), return_type=list, when_used="always"),
]


class ValidationIssue(ImmutableContractModel):
    code: str
    message: str
    path: str
    severity: Literal["error", "warning"]


class ValidationReport(ImmutableContractModel):
    valid: StrictBool
    issues: IssueTuple

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        expected = not any(issue.severity == "error" for issue in self.issues)
        if self.valid != expected:
            raise ValueError("valid must be true if and only if no error-severity issues exist")
        return self


def build_report(issues) -> ValidationReport:
    copied = tuple(issues)
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in copied),
        issues=copied,
    )


def merge_reports(*reports: ValidationReport) -> ValidationReport:
    return build_report(issue for report in reports for issue in report.issues)


__all__ = ["ValidationIssue", "ValidationReport", "build_report", "merge_reports"]
