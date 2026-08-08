"""Strict-v3 Core human review, adjudication, and release gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Literal
import hashlib
import json
import unicodedata
from pathlib import Path

from pydantic import ConfigDict, computed_field, field_validator, model_validator

from mub.vnext.audit.core_candidate import (
    CoreCandidateValidationReceipt,
    verify_core_candidate_validation_receipt,
)
from mub.vnext.audit.core import (
    CORE_AUDIT_FAMILIES,
    CORE_AUDIT_SCHEMA_VERSION,
    CoreAuditSelection,
    CoreAuditSelectionPackage,
    _task_conditions,
    core_audit_review_context_hash,
    core_audit_selection_hash,
)
from mub.vnext.contracts import TaskFamily
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.contracts.common import ImmutableContractModel


ReviewOutcome = Literal["pass", "fail", "not_applicable"]
ReviewVerdict = Literal["pass", "block", "needs_revision"]
ReviewerRole = Literal["primary", "secondary", "adjudicator"]
_CHECK_FIELDS = (
    "answer_unique",
    "gold_actions_scope_correct",
    "event_roles_correct",
    "selector_history_evidence_correct",
    "four_surface_semantic_equivalence",
    "surface_natural",
)
_DUAL_REVIEW_FAMILIES = frozenset(
    {
        TaskFamily.DELETION_FORGETTING,
        TaskFamily.CURRENT_HISTORICAL_QUERY,
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
    }
)
_SELECTOR_REVIEW_FAMILIES = frozenset(
    {
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
        TaskFamily.DELETION_FORGETTING,
        TaskFamily.CURRENT_HISTORICAL_QUERY,
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
    }
)
_MAX_REVIEW_RECORDS = 1024
_FULL_CORE_TASK_COUNT = 12_000
_VERIFIED_GATE_TOKEN = object()


class _StrictFrozenCoreReviewModel(ImmutableContractModel):
    model_config = ConfigDict(strict=True)


def _hash_value(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be lowercase sha256")
    return value


def _is_default_ignorable(character: str) -> bool:
    """Explicit Unicode Default_Ignorable_Code_Point policy used for audit text."""
    codepoint = ord(character)
    return (
        unicodedata.category(character) == "Cf"
        or codepoint in {0x00AD, 0x034F, 0x061C, 0xFEFF}
        or 0x115F <= codepoint <= 0x1160
        or 0x17B4 <= codepoint <= 0x17B5
        or 0x180B <= codepoint <= 0x180F
        or 0x200B <= codepoint <= 0x200F
        or 0x202A <= codepoint <= 0x202E
        or 0x2060 <= codepoint <= 0x206F
        or codepoint in {0x3164, 0xFFA0}
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0x1BCA0 <= codepoint <= 0x1BCA3
        or 0x1D173 <= codepoint <= 0x1D17A
        or 0xE0000 <= codepoint <= 0xE007F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _normalized_observation(value: str) -> str:
    """NFKC/casefold, drop default ignorables, and normalize punctuation/space."""
    characters = []
    for character in unicodedata.normalize("NFKC", value).casefold():
        category = unicodedata.category(character)
        if _is_default_ignorable(character):
            continue
        if character.isspace() or category.startswith(("P", "S")):
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def applicable_core_audit_checks(family: TaskFamily) -> frozenset[str]:
    """Return the exact reviewed applicability policy for one Core family."""
    if type(family) is not TaskFamily or family not in CORE_AUDIT_FAMILIES:
        raise ValueError("family must be one of Core A-G")
    applicable = set(_CHECK_FIELDS)
    if family not in _SELECTOR_REVIEW_FAMILIES:
        applicable.remove("selector_history_evidence_correct")
    return frozenset(applicable)


class CoreAuditChecks(_StrictFrozenCoreReviewModel):
    answer_unique: ReviewOutcome
    gold_actions_scope_correct: ReviewOutcome
    event_roles_correct: ReviewOutcome
    selector_history_evidence_correct: ReviewOutcome
    four_surface_semantic_equivalence: ReviewOutcome
    surface_natural: ReviewOutcome

    def values(self) -> tuple[ReviewOutcome, ...]:
        return tuple(getattr(self, field) for field in _CHECK_FIELDS)


class CoreAuditDecision(_StrictFrozenCoreReviewModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    audit_id: str
    task_id: str
    task_hash: str
    source_task_manifest_hash: str
    selection_hash: str
    review_context_hash: str
    reviewer_id: str
    reviewer_role: ReviewerRole
    review_record_id: str
    independent_review_attestation: bool
    verdict: ReviewVerdict
    checks: CoreAuditChecks
    task_specific_observation: str
    notes: str

    @field_validator(
        "audit_id",
        "task_id",
        "review_record_id",
        "task_specific_observation",
    )
    @classmethod
    def _nonblank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("reviewer_id")
    @classmethod
    def _canonical_reviewer_id(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        if value != normalized:
            raise ValueError(
                "reviewer_id must already be canonical NFKC lowercase without surrounding whitespace"
            )
        if not value or any(
            not (character.isascii() and (character.isalnum() or character in "._-"))
            for character in value
        ):
            raise ValueError("reviewer_id must be a nonblank lowercase ASCII identifier")
        return value

    @field_validator(
        "task_hash",
        "source_task_manifest_hash",
        "selection_hash",
        "review_context_hash",
    )
    @classmethod
    def _hash(cls, value: str, info) -> str:
        return _hash_value(value, info.field_name)

    @model_validator(mode="after")
    def _attested(self) -> CoreAuditDecision:
        if self.independent_review_attestation is not True:
            raise ValueError("human review requires an explicit independence attestation")
        normalized_observation = _normalized_observation(
            self.task_specific_observation
        )
        if not normalized_observation or not any(
            character.isalnum() for character in normalized_observation
        ):
            raise ValueError(
                "task_specific_observation must contain substantive letters or digits"
            )
        return self


class CoreAuditDecisionTemplate(_StrictFrozenCoreReviewModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    audit_id: str
    task_id: str
    task_hash: str
    source_task_manifest_hash: str
    selection_hash: str
    review_context_hash: str
    reviewer_role: ReviewerRole
    reviewer_id: None = None
    review_record_id: None = None
    independent_review_attestation: None = None
    verdict: None = None
    checks: None = None
    task_specific_observation: None = None
    notes: None = None

    @field_validator(
        "task_hash",
        "source_task_manifest_hash",
        "selection_hash",
        "review_context_hash",
    )
    @classmethod
    def _hash(cls, value: str, info) -> str:
        return _hash_value(value, info.field_name)

    @computed_field(return_type=bool)
    @property
    def release_ready(self) -> bool:
        return False


class CoreAuditInputObservation(_StrictFrozenCoreReviewModel):
    source: Literal["decision", "adjudication"]
    index: int
    record_digest: str
    audit_id: str | None = None
    accepted: bool
    rejection_reason: str | None = None

    @field_validator("record_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _hash_value(value, "record_digest")

    @model_validator(mode="after")
    def _rejection_coherence(self) -> CoreAuditInputObservation:
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted input observations cannot have rejection_reason")
        if not self.accepted and (
            self.rejection_reason is None or not self.rejection_reason.strip()
        ):
            raise ValueError("rejected input observations require rejection_reason")
        return self


class CoreAuditRemediation(_StrictFrozenCoreReviewModel):
    audit_id: str
    generator_stratum: str
    template_stratum: str
    required_action: Literal["regenerate_reselect"] = "regenerate_reselect"
    instruction: Literal[
        "fix the generator/template stratum, regenerate the candidate, and reselect; never rewrite or mechanically rebind human decisions"
    ] = "fix the generator/template stratum, regenerate the candidate, and reselect; never rewrite or mechanically rebind human decisions"


class CoreAuditGateReport(_StrictFrozenCoreReviewModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    candidate_scope: Literal["full", "bounded_test", "unattested"]
    full_candidate: bool
    candidate_validation_receipt: CoreCandidateValidationReceipt | None = None
    selection_package: CoreAuditSelectionPackage
    source_task_manifest: TaskManifestV3
    source_task_manifest_hash: str
    selection_hash: str
    review_context_hash: str
    surface_context_evidence: tuple[MemUpdateTaskV3, ...]
    input_observations: tuple[CoreAuditInputObservation, ...]
    decision_evidence: tuple[CoreAuditDecision, ...] = ()
    adjudication_evidence: tuple[CoreAuditDecision, ...] = ()
    missing_review_roles: tuple[str, ...] = ()
    duplicate_review_roles: tuple[str, ...] = ()
    unknown_audit_ids: tuple[str, ...] = ()
    binding_mismatch_audit_ids: tuple[str, ...] = ()
    malformed_review_records: tuple[str, ...] = ()
    duplicate_review_record_ids: tuple[str, ...] = ()
    copied_observation_audit_ids: tuple[str, ...] = ()
    non_independent_audit_ids: tuple[str, ...] = ()
    invalid_applicability_audit_ids: tuple[str, ...] = ()
    required_adjudication_ids: tuple[str, ...] = ()
    unresolved_adjudication_ids: tuple[str, ...] = ()
    adjudicated_audit_ids: tuple[str, ...] = ()
    non_pass_terminal_audit_ids: tuple[str, ...] = ()
    terminal_pass_audit_ids: tuple[str, ...] = ()
    remediations: tuple[CoreAuditRemediation, ...] = ()
    agreement_item_count: int = 0
    raw_agreement: float | None = None
    cohens_kappa: float | None = None
    issues: tuple[str, ...] = ()

    @field_validator(
        "source_task_manifest_hash", "selection_hash", "review_context_hash"
    )
    @classmethod
    def _hash(cls, value: str, info) -> str:
        return _hash_value(value, info.field_name)

    @field_validator("input_observations", mode="before")
    @classmethod
    def _input_observations(cls, value: Any) -> tuple[CoreAuditInputObservation, ...]:
        if type(value) not in (list, tuple) or len(value) > 2048:
            raise ValueError("input_observations must be a bounded list or tuple")
        observations = tuple(
            item
            if type(item) is CoreAuditInputObservation
            else CoreAuditInputObservation.model_validate(item)
            for item in value
        )
        expected = tuple(
            (source, index)
            for source in ("decision", "adjudication")
            for index in range(
                sum(item.source == source for item in observations)
            )
        )
        observed = tuple((item.source, item.index) for item in observations)
        if observed != expected:
            raise ValueError("input_observations must be contiguous in source/index order")
        return observations

    @field_validator("decision_evidence", "adjudication_evidence", mode="before")
    @classmethod
    def _evidence(cls, value: Any) -> tuple[CoreAuditDecision, ...]:
        if type(value) not in (list, tuple) or len(value) > _MAX_REVIEW_RECORDS:
            raise ValueError("review evidence must be a bounded list or tuple")
        result = []
        for item in value:
            if type(item) is CoreAuditDecision:
                raw = object.__getattribute__(item, "__dict__")
                if type(raw) is not dict or set(raw) != set(CoreAuditDecision.model_fields):
                    raise ValueError("gate evidence requires intact CoreAuditDecision records")
                payload = dict(raw)
            elif type(item) is dict:
                payload = item
            else:
                raise ValueError(
                    "gate evidence requires CoreAuditDecision records or serialized dictionaries"
                )
            result.append(CoreAuditDecision.model_validate(payload))
        return tuple(sorted(result, key=_decision_key))

    @field_validator("surface_context_evidence", mode="before")
    @classmethod
    def _surface_context(cls, value: Any) -> tuple[MemUpdateTaskV3, ...]:
        if type(value) not in (list, tuple) or len(value) != 896:
            raise ValueError("surface context evidence requires exactly 896 task records")
        return tuple(
            item
            if type(item) is MemUpdateTaskV3
            else MemUpdateTaskV3.model_validate(item)
            for item in value
        )

    @field_validator("remediations", mode="before")
    @classmethod
    def _remediations(cls, value: Any) -> tuple[CoreAuditRemediation, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("remediations must be a list or tuple")
        return tuple(
            item
            if type(item) is CoreAuditRemediation
            else CoreAuditRemediation.model_validate(item)
            for item in value
        )

    @field_validator(
        "missing_review_roles",
        "duplicate_review_roles",
        "unknown_audit_ids",
        "binding_mismatch_audit_ids",
        "malformed_review_records",
        "duplicate_review_record_ids",
        "copied_observation_audit_ids",
        "non_independent_audit_ids",
        "invalid_applicability_audit_ids",
        "required_adjudication_ids",
        "unresolved_adjudication_ids",
        "adjudicated_audit_ids",
        "non_pass_terminal_audit_ids",
        "terminal_pass_audit_ids",
        "issues",
        mode="before",
    )
    @classmethod
    def _sorted_text(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("gate issue fields must be lists or tuples")
        result = tuple(sorted(value))
        if any(type(item) is not str or not item.strip() for item in result):
            raise ValueError("gate issue fields require nonblank strings")
        if len(result) != len(set(result)):
            raise ValueError("gate issue fields must be unique")
        return result

    @model_validator(mode="after")
    def _candidate_scope_is_unattested(self) -> CoreAuditGateReport:
        if self.full_candidate or self.candidate_scope != "unattested":
            raise ValueError(
                "structural gate reports cannot claim authoritative candidate scope"
            )
        receipt = self.candidate_validation_receipt
        if receipt is not None:
            manifest_count = sum(self.source_task_manifest.split_counts.values())
            if (
                receipt.source_task_manifest_hash
                != self.source_task_manifest_hash
                or receipt.task_count != manifest_count
            ):
                raise ValueError(
                    "candidate validation receipt metadata does not match the report"
                )
        return self._validate_input_observation_bindings()

    def _validate_input_observation_bindings(self) -> CoreAuditGateReport:
        evidence_by_source = {
            "decision": self.decision_evidence,
            "adjudication": self.adjudication_evidence,
        }
        for source, evidence in evidence_by_source.items():
            accepted = tuple(
                observation
                for observation in self.input_observations
                if observation.source == source and observation.accepted
            )
            if len(accepted) != len(evidence):
                raise ValueError("accepted input observations must bind all retained evidence")
            unmatched = list(evidence)
            for observation in accepted:
                match = next(
                    (
                        item
                        for item in unmatched
                        if item.audit_id == observation.audit_id
                        and _input_digest(item, source, observation.index)
                        == observation.record_digest
                    ),
                    None,
                )
                if match is None:
                    raise ValueError("input observation digest does not bind retained evidence")
                unmatched.remove(match)
        return self

    def validated_replace(self, **changes) -> CoreAuditGateReport:
        raw = object.__getattribute__(self, "__dict__")
        if type(raw) is not dict or set(raw) != set(type(self).model_fields):
            raise ValueError("CoreAuditGateReport requires intact fields")
        payload = dict(raw)
        payload.update(changes)
        return type(self).model_validate(payload)

    @computed_field(return_type=bool)
    @property
    def release_ready(self) -> bool:
        """Structural reports are never authoritative without explicit verification."""
        return False

    def _structural_terminal_ready(self) -> bool:
        try:
            raw = object.__getattribute__(self, "__dict__")
            snapshot = type(self).model_validate(dict(raw))
            if any(
                not observation.accepted
                for observation in snapshot.input_observations
            ):
                return False
            package = snapshot.selection_package
            selected_ids = {item.audit_id for item in package.selections}
            selected_tasks_by_id = {
                task.task_id: task for task in snapshot.surface_context_evidence
            }
            selected_tasks = tuple(
                selected_tasks_by_id[item.task_id] for item in package.selections
            )
            recomputed = evaluate_core_audit_gate(
                package,
                snapshot.decision_evidence,
                snapshot.adjudication_evidence,
                source_task_manifest=snapshot.source_task_manifest,
                candidate_validation_receipt=snapshot.candidate_validation_receipt,
                selected_tasks=selected_tasks,
                surface_context_tasks=snapshot.surface_context_evidence,
            )
            recomputed_raw = object.__getattribute__(recomputed, "__dict__")
            snapshot_raw = object.__getattribute__(snapshot, "__dict__")
            compared_fields = set(type(self).model_fields) - {
                "selection_package",
                "surface_context_evidence",
                "input_observations",
                "decision_evidence",
                "adjudication_evidence",
            }
            if any(
                recomputed_raw[field] != snapshot_raw[field]
                for field in compared_fields
            ):
                return False
            return bool(
                set(recomputed.terminal_pass_audit_ids) == selected_ids
                and len(selected_ids) == 224
                and not recomputed.issues
                and not recomputed.unresolved_adjudication_ids
                and not recomputed.non_pass_terminal_audit_ids
            )
        except Exception:
            return False


class VerifiedCoreAuditGateReport:
    """Immutable result of verifying one exact structural report and root snapshot."""

    __slots__ = (
        "_report",
        "_report_hash",
        "_receipt_hash",
        "_trusted_candidate_root",
        "_candidate_root_digest",
        "_candidate_scope",
        "_full_candidate",
        "_release_ready",
    )

    def __init__(
        self,
        *,
        report: CoreAuditGateReport,
        report_hash: str,
        receipt_hash: str,
        trusted_candidate_root: str,
        candidate_root_digest: str,
        candidate_scope: Literal["full", "bounded_test"],
        full_candidate: bool,
        release_ready: bool,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_GATE_TOKEN:
            raise TypeError(
                "VerifiedCoreAuditGateReport can only be created by explicit verification"
            )
        values = {
            "_report": report,
            "_report_hash": report_hash,
            "_receipt_hash": receipt_hash,
            "_trusted_candidate_root": trusted_candidate_root,
            "_candidate_root_digest": candidate_root_digest,
            "_candidate_scope": candidate_scope,
            "_full_candidate": full_candidate,
            "_release_ready": release_ready,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("VerifiedCoreAuditGateReport is immutable")

    @property
    def report(self) -> CoreAuditGateReport:
        return self._report

    @property
    def report_hash(self) -> str:
        return self._report_hash

    @property
    def receipt_hash(self) -> str:
        return self._receipt_hash

    @property
    def trusted_candidate_root(self) -> str:
        return self._trusted_candidate_root

    @property
    def candidate_root_digest(self) -> str:
        return self._candidate_root_digest

    @property
    def candidate_scope(self) -> Literal["full", "bounded_test"]:
        return self._candidate_scope

    @property
    def full_candidate(self) -> bool:
        return self._full_candidate

    @property
    def release_ready(self) -> bool:
        return self._release_ready

    def __getattr__(self, name: str) -> Any:
        return getattr(self._report, name)


def _observed_audit_id(value: Any) -> str | None:
    if type(value) is dict:
        candidate = value.get("audit_id")
    else:
        try:
            raw = object.__getattribute__(value, "__dict__")
        except Exception:
            raw = None
        candidate = raw.get("audit_id") if type(raw) is dict else None
    return candidate if type(candidate) is str and candidate.strip() else None


def _input_digest(value: Any, source: str, index: int) -> str:
    try:
        if hasattr(value, "model_dump"):
            payload = canonical_json_bytes(value)
        elif type(value) is dict:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        else:
            raise TypeError
    except Exception:
        payload = (
            f"unserializable:{type(value).__module__}.{type(value).__qualname__}:"
            f"{_observed_audit_id(value) or ''}"
        ).encode("utf-8")
    envelope = (
        source.encode("ascii")
        + b":"
        + str(index).encode("ascii")
        + b":"
        + hashlib.sha256(payload).hexdigest().encode("ascii")
    )
    return hashlib.sha256(envelope).hexdigest()


def _decision_key(item: CoreAuditDecision) -> tuple[str, str, str, str]:
    return (
        item.audit_id,
        item.reviewer_role,
        item.reviewer_id,
        item.review_record_id,
    )


def _expected_roles(selection: CoreAuditSelection) -> tuple[str, ...]:
    if selection.family in _DUAL_REVIEW_FAMILIES:
        return ("primary", "secondary")
    return ("primary",)


def core_audit_decision_templates(
    package: CoreAuditSelectionPackage,
) -> tuple[CoreAuditDecisionTemplate, ...]:
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    if core_audit_selection_hash(package) != package.selection_hash:
        raise ValueError("selection package hash mismatch")
    context_hash = core_audit_review_context_hash(package)
    templates = []
    for selected in package.selections:
        for role in _expected_roles(selected):
            templates.append(
                CoreAuditDecisionTemplate(
                    audit_id=selected.audit_id,
                    task_id=selected.task_id,
                    task_hash=selected.task_hash,
                    source_task_manifest_hash=package.source_task_manifest_hash,
                    selection_hash=package.selection_hash,
                    review_context_hash=context_hash,
                    reviewer_role=role,
                )
            )
    return tuple(sorted(templates, key=lambda item: (item.audit_id, item.reviewer_role)))


def core_audit_adjudication_templates(
    package: CoreAuditSelectionPackage,
    audit_ids: Sequence[str] | Any,
) -> tuple[CoreAuditDecisionTemplate, ...]:
    """Build blank adjudicator rows only for caller-declared required audit IDs."""
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    requested = tuple(audit_ids)
    if len(requested) > len(package.selections):
        raise ValueError("adjudication template request exceeds the selection")
    if any(type(audit_id) is not str or not audit_id.strip() for audit_id in requested):
        raise ValueError("adjudication audit IDs must be nonblank strings")
    if len(requested) != len(set(requested)):
        raise ValueError("adjudication audit IDs must be unique")
    selected_by_id = {item.audit_id: item for item in package.selections}
    unknown = set(requested) - selected_by_id.keys()
    if unknown:
        raise ValueError(f"unknown adjudication audit IDs: {sorted(unknown)}")
    context_hash = core_audit_review_context_hash(package)
    return tuple(
        CoreAuditDecisionTemplate(
            audit_id=selected_by_id[audit_id].audit_id,
            task_id=selected_by_id[audit_id].task_id,
            task_hash=selected_by_id[audit_id].task_hash,
            source_task_manifest_hash=package.source_task_manifest_hash,
            selection_hash=package.selection_hash,
            review_context_hash=context_hash,
            reviewer_role="adjudicator",
        )
        for audit_id in sorted(requested)
    )


def _snapshot_sequence(value: Sequence[Any], name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be an exact list or tuple")
    if len(value) > _MAX_REVIEW_RECORDS:
        raise ValueError(f"{name} exceeds the bounded review limit")
    return tuple(value)


def validate_core_audit_review_context(
    package: CoreAuditSelectionPackage,
    selected_tasks: Sequence[Any],
    surface_context_tasks: Sequence[Any],
) -> tuple[MemUpdateTaskV3, ...]:
    """Authenticate exact selected rows and all four surfaces against selection hashes."""
    selected_input = _snapshot_sequence(selected_tasks, "selected_tasks")
    context_input = _snapshot_sequence(surface_context_tasks, "surface_context_tasks")
    if len(selected_input) != 224 or len(context_input) != 896:
        raise ValueError("review context requires exactly 224 selected and 896 surface tasks")
    expected_selected = tuple(item.task_id for item in package.selections)
    expected_context = tuple(
        variant.task_id
        for item in package.selections
        for variant in item.surface_variants
    )
    if tuple(getattr(task, "task_id", None) for task in selected_input) != expected_selected:
        raise ValueError("selected task context IDs/order do not match the selection")
    if tuple(getattr(task, "task_id", None) for task in context_input) != expected_context:
        raise ValueError("four-surface context IDs/order do not match the selection")
    expected_hashes = {
        variant.task_id: variant.task_hash
        for item in package.selections
        for variant in item.surface_variants
    }
    expected_core_surface = {
        variant.task_id: (item.semantic_core_id, variant.surface_id)
        for item in package.selections
        for variant in item.surface_variants
    }
    snapshots = []
    for task in context_input:
        if type(task) is not MemUpdateTaskV3:
            raise TypeError("review context requires exact MemUpdateTaskV3 records")
        if sha256_model(task) != expected_hashes[task.task_id]:
            raise ValueError(f"review context task hash mismatch for {task.task_id}")
        expected_core, expected_surface = expected_core_surface[task.task_id]
        if (
            task.metadata.split_key.semantic_core_id != expected_core
            or task.metadata.extra.get("surface_template") != expected_surface
        ):
            raise ValueError("review context semantic-core/surface binding mismatch")
        snapshots.append(task)
    selected_by_id = {task.task_id: task for task in snapshots}
    if any(
        sha256_model(task) != expected_hashes[task.task_id]
        or sha256_model(task) != sha256_model(selected_by_id[task.task_id])
        for task in selected_input
    ):
        raise ValueError("selected task bytes differ from the authenticated surface context")
    for selection in package.selections:
        task = selected_by_id[selection.task_id]
        if (
            TaskFamily(task.task_family) is not selection.family
            or task.difficulty is not selection.difficulty
            or task.metadata.split is not selection.split
            or task.metadata.split_key.semantic_core_id != selection.semantic_core_id
            or task.metadata.extra.get("surface_template") != selection.surface_id
            or _task_conditions(task) != selection.covered_conditions
        ):
            raise ValueError(
                f"selection fields do not match authenticated task projection for {selection.task_id}"
            )
    return tuple(snapshots)


def _is_bound(
    decision: CoreAuditDecision,
    selected: CoreAuditSelection,
    package: CoreAuditSelectionPackage,
    review_context_hash: str,
) -> bool:
    return (
        decision.task_id == selected.task_id
        and decision.task_hash == selected.task_hash
        and decision.source_task_manifest_hash == package.source_task_manifest_hash
        and decision.selection_hash == package.selection_hash
        and decision.review_context_hash == review_context_hash
    )


def _valid_applicability(
    decision: CoreAuditDecision, family: TaskFamily
) -> bool:
    applicable = applicable_core_audit_checks(family)
    return all(
        (getattr(decision.checks, field) != "not_applicable")
        if field in applicable
        else (getattr(decision.checks, field) == "not_applicable")
        for field in _CHECK_FIELDS
    )


def _human_pass(decision: CoreAuditDecision, family: TaskFamily) -> bool:
    if not _valid_applicability(decision, family):
        return False
    applicable = applicable_core_audit_checks(family)
    return decision.verdict == "pass" and all(
        getattr(decision.checks, field) == "pass" for field in applicable
    )


def _review_signature(decision: CoreAuditDecision, family: TaskFamily) -> tuple[str, ...]:
    applicable = applicable_core_audit_checks(family)
    return (
        decision.verdict,
        *(getattr(decision.checks, field) for field in _CHECK_FIELDS if field in applicable),
    )


def _agreement(
    package: CoreAuditSelectionPackage,
    by_role: dict[tuple[str, str], list[CoreAuditDecision]],
    excluded_audit_ids: set[str],
) -> tuple[int, float | None, float | None]:
    first: list[str] = []
    second: list[str] = []
    for selected in package.selections:
        if (
            selected.family not in _DUAL_REVIEW_FAMILIES
            or selected.audit_id in excluded_audit_ids
        ):
            continue
        primary = by_role.get((selected.audit_id, "primary"), [])
        secondary = by_role.get((selected.audit_id, "secondary"), [])
        if len(primary) != 1 or len(secondary) != 1:
            continue
        p_signature = _review_signature(primary[0], selected.family)
        s_signature = _review_signature(secondary[0], selected.family)
        first.extend(p_signature)
        second.extend(s_signature)
    count = len(first)
    if count == 0:
        return 0, None, None
    observed = sum(a == b for a, b in zip(first, second)) / count
    first_counts = Counter(first)
    second_counts = Counter(second)
    categories = set(first_counts) | set(second_counts)
    expected = sum(
        (first_counts[value] / count) * (second_counts[value] / count)
        for value in categories
    )
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return count, observed, kappa


def evaluate_core_audit_gate(
    package: CoreAuditSelectionPackage,
    decisions: Sequence[Any],
    adjudications: Sequence[Any],
    *,
    source_task_manifest: TaskManifestV3,
    candidate_validation_receipt: CoreCandidateValidationReceipt | None = None,
    selected_tasks: Sequence[Any],
    surface_context_tasks: Sequence[Any],
) -> CoreAuditGateReport:
    """Fail-closed gate over bound primary/secondary and adjudication records."""
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    if core_audit_selection_hash(package) != package.selection_hash:
        raise ValueError("selection package hash mismatch")
    if type(source_task_manifest) is not TaskManifestV3:
        raise TypeError("gate requires an exact strict-v3 source TaskManifestV3")
    if sha256_model(source_task_manifest) != package.source_task_manifest_hash:
        raise ValueError("source task manifest hash does not match the selection")
    manifest_task_count = sum(source_task_manifest.split_counts.values())
    full_candidate = False
    candidate_scope = "unattested"
    if candidate_validation_receipt is not None:
        if (
            candidate_validation_receipt.source_task_manifest_hash
            != package.source_task_manifest_hash
            or candidate_validation_receipt.task_count != manifest_task_count
        ):
            raise ValueError(
                "candidate validation receipt metadata does not match the evaluation"
            )
    context_evidence = validate_core_audit_review_context(
        package, selected_tasks, surface_context_tasks
    )
    if any(
        source_task_manifest.task_record_hashes.get(task.task_id)
        != sha256_model(task)
        for task in context_evidence
    ):
        raise ValueError("review context is not authenticated by the source task manifest")
    context_hash = core_audit_review_context_hash(package)
    decision_input = _snapshot_sequence(decisions, "decisions")
    adjudication_input = _snapshot_sequence(adjudications, "adjudications")
    selected_by_id = {item.audit_id: item for item in package.selections}
    malformed: set[str] = set()
    unknown: set[str] = set()
    binding: set[str] = set()
    invalid_applicability: set[str] = set()
    valid_decisions: list[CoreAuditDecision] = []
    valid_adjudications: list[CoreAuditDecision] = []
    input_observations: list[CoreAuditInputObservation] = []

    def observe(
        item: Any,
        source: str,
        index: int,
        *,
        accepted: bool,
        rejection_reason: str | None = None,
    ) -> None:
        input_observations.append(
            CoreAuditInputObservation(
                source=source,
                index=index,
                record_digest=_input_digest(item, source, index),
                audit_id=_observed_audit_id(item),
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
        )

    def collect(
        values: tuple[Any, ...], expected_adjudication: bool
    ) -> list[CoreAuditDecision]:
        result = []
        source = "adjudication" if expected_adjudication else "decision"
        for index, item in enumerate(values):
            if type(item) is not CoreAuditDecision:
                malformed.add(f"<index:{index}:{source}>")
                observe(item, source, index, accepted=False, rejection_reason="malformed_record_type")
                continue
            try:
                raw = object.__getattribute__(item, "__dict__")
                snapshot = CoreAuditDecision.model_validate(dict(raw))
            except Exception:
                malformed.add(getattr(item, "audit_id", f"<index:{index}>") or f"<index:{index}>")
                observe(item, source, index, accepted=False, rejection_reason="malformed_record")
                continue
            selected = selected_by_id.get(snapshot.audit_id)
            if selected is None:
                unknown.add(snapshot.audit_id)
                observe(item, source, index, accepted=False, rejection_reason="unknown_audit_id")
                continue
            if not _is_bound(snapshot, selected, package, context_hash):
                binding.add(snapshot.audit_id)
                observe(item, source, index, accepted=False, rejection_reason="binding_mismatch")
                continue
            if (snapshot.reviewer_role == "adjudicator") != expected_adjudication:
                malformed.add(snapshot.audit_id)
                observe(item, source, index, accepted=False, rejection_reason="wrong_reviewer_role")
                continue
            if not _valid_applicability(snapshot, selected.family):
                invalid_applicability.add(snapshot.audit_id)
            result.append(snapshot)
            observe(item, source, index, accepted=True)
        return result

    valid_decisions = collect(decision_input, False)
    valid_adjudications = collect(adjudication_input, True)
    by_role: dict[tuple[str, str], list[CoreAuditDecision]] = defaultdict(list)
    for item in valid_decisions:
        by_role[(item.audit_id, item.reviewer_role)].append(item)
    adjudication_by_id: dict[str, list[CoreAuditDecision]] = defaultdict(list)
    for item in valid_adjudications:
        adjudication_by_id[item.audit_id].append(item)

    missing = set()
    duplicates = set()
    for selected in package.selections:
        for role in _expected_roles(selected):
            rows = by_role.get((selected.audit_id, role), [])
            label = f"{selected.audit_id}:{role}"
            if not rows:
                missing.add(label)
            elif len(rows) > 1:
                duplicates.add(label)
        unexpected_roles = {
            role
            for (audit_id, role), rows in by_role.items()
            if audit_id == selected.audit_id and role not in _expected_roles(selected) and rows
        }
        duplicates.update(f"{selected.audit_id}:{role}" for role in unexpected_roles)

    record_counts = Counter(
        item.review_record_id for item in (*valid_decisions, *valid_adjudications)
    )
    duplicate_records = {
        record_id for record_id, count in record_counts.items() if count > 1
    }
    copied_observations: set[str] = set()
    per_audit_groups: dict[tuple[str, str], list[CoreAuditDecision]] = defaultdict(list)
    global_groups: dict[str, list[CoreAuditDecision]] = defaultdict(list)
    for item in (*valid_decisions, *valid_adjudications):
        fingerprint = _normalized_observation(item.task_specific_observation)
        per_audit_groups[(item.audit_id, fingerprint)].append(item)
        global_groups[fingerprint].append(item)
    for groups in (per_audit_groups, global_groups):
        for rows in groups.values():
            if len(rows) > 1:
                copied_observations.update(item.audit_id for item in rows)

    non_independent: set[str] = set()
    required_adjudication: set[str] = set()
    terminal_pass: set[str] = set()
    non_pass_terminal: set[str] = set()
    adjudicated: set[str] = set()
    for selected in package.selections:
        expected = _expected_roles(selected)
        rows = [
            by_role[(selected.audit_id, role)][0]
            for role in expected
            if len(by_role.get((selected.audit_id, role), [])) == 1
        ]
        if len(rows) == 2 and rows[0].reviewer_id == rows[1].reviewer_id:
            non_independent.add(selected.audit_id)
        complete = len(rows) == len(expected)
        clean_rows = complete and selected.audit_id not in invalid_applicability
        disagreement = (
            len(rows) == 2
            and _review_signature(rows[0], selected.family)
            != _review_signature(rows[1], selected.family)
        )
        any_nonpass = bool(rows) and any(
            not _human_pass(item, selected.family) for item in rows
        )
        needs_adjudication = complete and clean_rows and (disagreement or any_nonpass)
        if needs_adjudication:
            required_adjudication.add(selected.audit_id)
        adjudication_rows = adjudication_by_id.get(selected.audit_id, [])
        if needs_adjudication:
            if len(adjudication_rows) == 1:
                adjudicator = adjudication_rows[0]
                if adjudicator.reviewer_id in {item.reviewer_id for item in rows}:
                    non_independent.add(selected.audit_id)
                else:
                    adjudicated.add(selected.audit_id)
                    if _human_pass(adjudicator, selected.family):
                        terminal_pass.add(selected.audit_id)
                    else:
                        non_pass_terminal.add(selected.audit_id)
            elif len(adjudication_rows) > 1:
                duplicates.add(f"{selected.audit_id}:adjudicator")
        elif adjudication_rows:
            duplicates.add(f"{selected.audit_id}:unexpected_adjudicator")
        elif complete and clean_rows and all(
            _human_pass(item, selected.family) for item in rows
        ):
            terminal_pass.add(selected.audit_id)
        elif complete and clean_rows:
            non_pass_terminal.add(selected.audit_id)

    record_issue_audits = {
        item.audit_id
        for item in (*valid_decisions, *valid_adjudications)
        if item.review_record_id in duplicate_records
    }
    duplicate_role_audits = {
        label.split(":", 1)[0] for label in duplicates
    }
    dirty_audits = (
        record_issue_audits
        | copied_observations
        | non_independent
        | invalid_applicability
        | duplicate_role_audits
        | binding
        | {value for value in malformed if value in selected_by_id}
    )
    terminal_pass.difference_update(dirty_audits)
    non_pass_terminal.difference_update(dirty_audits)
    adjudicated.difference_update(dirty_audits)
    unresolved = required_adjudication - adjudicated
    agreement_count, raw_agreement, kappa = _agreement(
        package, by_role, dirty_audits
    )
    issue_sets = {
        "missing_review_roles": missing,
        "duplicate_review_roles": duplicates,
        "unknown_audit_ids": unknown,
        "binding_mismatch_audit_ids": binding,
        "malformed_review_records": malformed,
        "duplicate_review_record_ids": duplicate_records,
        "copied_observation_audit_ids": copied_observations,
        "non_independent_audit_ids": non_independent,
        "invalid_applicability_audit_ids": invalid_applicability,
        "unresolved_adjudication_ids": unresolved,
        "non_pass_terminal_audit_ids": non_pass_terminal,
    }
    issues = tuple(
        f"{name}:{value}"
        for name, values in issue_sets.items()
        for value in sorted(values)
    )
    if candidate_validation_receipt is None:
        issues = tuple(
            sorted(
                (
                    *issues,
                    "candidate_unattested:no_current_root_validation_receipt",
                )
            )
        )
    remediations = tuple(
        CoreAuditRemediation(
            audit_id=audit_id,
            generator_stratum=(
                f"{selected_by_id[audit_id].family.value}:"
                + ",".join(selected_by_id[audit_id].covered_conditions)
            ),
            template_stratum=selected_by_id[audit_id].surface_id,
        )
        for audit_id in sorted(non_pass_terminal)
    )
    return CoreAuditGateReport(
        candidate_scope=candidate_scope,
        full_candidate=full_candidate,
        candidate_validation_receipt=candidate_validation_receipt,
        selection_package=package,
        source_task_manifest=source_task_manifest,
        source_task_manifest_hash=package.source_task_manifest_hash,
        selection_hash=package.selection_hash,
        review_context_hash=context_hash,
        surface_context_evidence=context_evidence,
        input_observations=tuple(input_observations),
        decision_evidence=tuple(valid_decisions),
        adjudication_evidence=tuple(valid_adjudications),
        missing_review_roles=tuple(missing),
        duplicate_review_roles=tuple(duplicates),
        unknown_audit_ids=tuple(unknown),
        binding_mismatch_audit_ids=tuple(binding),
        malformed_review_records=tuple(malformed),
        duplicate_review_record_ids=tuple(duplicate_records),
        copied_observation_audit_ids=tuple(copied_observations),
        non_independent_audit_ids=tuple(non_independent),
        invalid_applicability_audit_ids=tuple(invalid_applicability),
        required_adjudication_ids=tuple(required_adjudication),
        unresolved_adjudication_ids=tuple(unresolved),
        adjudicated_audit_ids=tuple(adjudicated),
        non_pass_terminal_audit_ids=tuple(non_pass_terminal),
        terminal_pass_audit_ids=tuple(terminal_pass),
        remediations=remediations,
        agreement_item_count=agreement_count,
        raw_agreement=raw_agreement,
        cohens_kappa=kappa,
        issues=issues,
    )


def verify_core_audit_gate_report(
    report: CoreAuditGateReport,
    *,
    trusted_candidate_root: Path | str,
) -> VerifiedCoreAuditGateReport | None:
    """Return an immutable capability for one verified report/root snapshot."""
    if type(report) is not CoreAuditGateReport:
        return None
    receipt = report.candidate_validation_receipt
    if receipt is None:
        return None
    try:
        raw = object.__getattribute__(report, "__dict__")
        if type(raw) is not dict or set(raw) != set(CoreAuditGateReport.model_fields):
            return None
        snapshot = CoreAuditGateReport.model_validate(dict(raw))
        trusted_root = Path(trusted_candidate_root).resolve(strict=True)
        if str(trusted_root) != receipt.candidate_dir:
            return None
        if not verify_core_candidate_validation_receipt(
            receipt, trusted_candidate_root=trusted_root
        ):
            return None
        manifest_task_count = sum(snapshot.source_task_manifest.split_counts.values())
        full_candidate = bool(
            receipt.expected_full
            and receipt.task_count == _FULL_CORE_TASK_COUNT
            and manifest_task_count == _FULL_CORE_TASK_COUNT
        )
        candidate_scope: Literal["full", "bounded_test"] = (
            "full" if full_candidate else "bounded_test"
        )
        return VerifiedCoreAuditGateReport(
            report=snapshot,
            report_hash=sha256_model(snapshot),
            receipt_hash=receipt.receipt_hash,
            trusted_candidate_root=str(trusted_root),
            candidate_root_digest=receipt.candidate_root_digest,
            candidate_scope=candidate_scope,
            full_candidate=full_candidate,
            release_ready=bool(
                full_candidate and snapshot._structural_terminal_ready()
            ),
            _token=_VERIFIED_GATE_TOKEN,
        )
    except Exception:
        return None


__all__ = [
    "CoreAuditChecks",
    "CoreAuditDecision",
    "CoreAuditDecisionTemplate",
    "CoreAuditGateReport",
    "CoreAuditInputObservation",
    "CoreAuditRemediation",
    "VerifiedCoreAuditGateReport",
    "ReviewOutcome",
    "ReviewVerdict",
    "ReviewerRole",
    "applicable_core_audit_checks",
    "core_audit_adjudication_templates",
    "core_audit_decision_templates",
    "evaluate_core_audit_gate",
    "validate_core_audit_review_context",
    "verify_core_audit_gate_report",
]
