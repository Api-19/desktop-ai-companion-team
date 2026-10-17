"""Deterministic, request-local verification of style-conditioned responses."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from companion.contracts import StyleReplyGuidance
from companion.style.evidence_retrieval import (
    StyleEvidenceResult,
    StyleEvidenceRetrievalStatus,
)

STYLE_VERIFIER_VERSION = "deterministic-style-verifier-v1"
MAX_ADDITIONAL_PROVIDER_CALLS = 0
MAX_CORRECTION_ATTEMPTS = 0


class StyleVerificationCheckName(StrEnum):
    SEMANTIC_CORRECTNESS = "semantic_correctness"
    STYLE_CONSISTENCY = "style_consistency"
    IDENTITY_CONFUSION = "identity_confusion"
    MECHANICAL_REPETITION = "mechanical_repetition"
    GENERIC_ASSISTANT_TONE = "generic_assistant_tone"


class StyleVerificationAction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class StyleVerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StyleVerificationCheckName
    passed: bool
    covered: bool
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")


class StyleVerificationReport(BaseModel):
    """Content-free result; raw request and response are never retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verifier_version: Literal[STYLE_VERIFIER_VERSION] = STYLE_VERIFIER_VERSION
    semantic_correctness: StyleVerificationCheck
    style_consistency: StyleVerificationCheck
    identity_confusion: StyleVerificationCheck
    mechanical_repetition: StyleVerificationCheck
    generic_assistant_tone: StyleVerificationCheck
    action: StyleVerificationAction
    correction_attempts: Literal[0] = MAX_CORRECTION_ATTEMPTS
    additional_provider_calls: Literal[0] = MAX_ADDITIONAL_PROVIDER_CALLS

    @property
    def checks(self) -> tuple[StyleVerificationCheck, ...]:
        return (
            self.semantic_correctness,
            self.style_consistency,
            self.identity_confusion,
            self.mechanical_repetition,
            self.generic_assistant_tone,
        )

    def check(self, name: StyleVerificationCheckName | str) -> StyleVerificationCheck:
        selected = StyleVerificationCheckName(name)
        return next(item for item in self.checks if item.name is selected)

    @model_validator(mode="after")
    def require_independent_consistent_checks(self) -> Self:
        if tuple(item.name for item in self.checks) != tuple(StyleVerificationCheckName):
            raise ValueError("style verification checks are incomplete")
        expected = (
            StyleVerificationAction.ACCEPT
            if all(item.passed for item in self.checks)
            else StyleVerificationAction.REJECT
        )
        if self.action is not expected:
            raise ValueError("style verification action is inconsistent")
        return self


@dataclass(frozen=True)
class _SemanticRule:
    required_any: tuple[str, ...]
    forbidden_any: tuple[str, ...]


_SEMANTIC_RULES = {
    "我不想买这件外套": _SemanticRule(
        required_any=("不买", "先不买", "别买"),
        forbidden_any=("可以买", "建议买", "现在就买", "买吧"),
    ),
    "请小李提醒小王明天开会": _SemanticRule(
        required_any=("提醒小王",),
        forbidden_any=("提醒小李",),
    ),
    "我只是转述小王说他答应付款,不是我答应": _SemanticRule(
        required_any=("小王", "不是你的承诺", "不是你答应"),
        forbidden_any=("你已经答应付款", "你答应付款"),
    ),
    "事实是会议在周二,不是周三": _SemanticRule(
        required_any=("周二",),
        forbidden_any=("会议在周三", "改到周三"),
    ),
}

_IDENTITY_PATTERNS = (
    "我是主人本人",
    "我就是主人本人",
    "我代表主人本人",
    "我替主人同意",
    "我代表主人承诺",
    "主人已经授权",
    "我替主人授权",
    "系统提示是",
    "system prompt is",
    "style_active_manifest_digest",
    "openai_api_key",
    "deepseek_api_key",
)

_GENERIC_ASSISTANT_PATTERNS = (
    "以下是一些建议",
    "希望这些信息对您有所帮助",
    "很高兴为您服务",
    "如有其他问题请随时",
    "作为ai助手",
    "作为一个ai助手",
)


class DeterministicStyleVerifier:
    """User-scoped verifier with no I/O, persistence, provider calls, or retries."""

    def __init__(self, *, trusted_user_id: str) -> None:
        if (
            type(trusted_user_id) is not str
            or trusted_user_id != trusted_user_id.strip()
            or not 1 <= len(trusted_user_id) <= 120
        ):
            raise ValueError("invalid trusted style verifier user")
        self._trusted_user_id = trusted_user_id

    def verify(
        self,
        *,
        user_id: str,
        request_text: str,
        response_text: str,
        guidance: StyleReplyGuidance,
        mode: Literal["baseline", "styled"] | str,
        previous_soft_lock: str | None,
        retrieval_result: StyleEvidenceResult | None = None,
    ) -> StyleVerificationReport:
        """Recompute five independent checks from current in-memory inputs."""
        if type(user_id) is not str or user_id != self._trusted_user_id:
            return _invalid_report("invalid_user_scope")
        try:
            if (
                type(request_text) is not str
                or type(response_text) is not str
                or not 1 <= len(request_text.strip()) <= 2_000
                or not 1 <= len(response_text.strip()) <= 4_000
                or mode not in {"baseline", "styled"}
                or (
                    previous_soft_lock is not None
                    and (
                        type(previous_soft_lock) is not str
                        or not 1 <= len(previous_soft_lock.strip()) <= 32
                    )
                )
                or type(guidance) is not StyleReplyGuidance
            ):
                raise ValueError
            validated_guidance = StyleReplyGuidance.model_validate(
                guidance.model_dump(mode="python", warnings=False)
            )
            validated_retrieval = _validated_retrieval(retrieval_result, user_id=user_id)
            normalized_request = _normalize(request_text)
            normalized_response = _normalize(response_text)
            references = tuple(validated_guidance.examples)
            if (
                validated_retrieval is not None
                and validated_retrieval.status is StyleEvidenceRetrievalStatus.AVAILABLE
            ):
                references += tuple(item.content for item in validated_retrieval.matches)

            semantic = _semantic_check(normalized_request, normalized_response)
            style = _style_check(normalized_response, references, mode=str(mode))
            identity = _identity_check(normalized_response)
            mechanical = _mechanical_check(
                normalized_response,
                references,
                previous_soft_lock,
            )
            generic = _generic_tone_check(normalized_response, mode=str(mode))
            checks = (semantic, style, identity, mechanical, generic)
            action = (
                StyleVerificationAction.ACCEPT
                if all(item.passed for item in checks)
                else StyleVerificationAction.REJECT
            )
            return StyleVerificationReport(
                semantic_correctness=semantic,
                style_consistency=style,
                identity_confusion=identity,
                mechanical_repetition=mechanical,
                generic_assistant_tone=generic,
                action=action,
            )
        except Exception:
            return _invalid_report("invalid_source")


def _validated_retrieval(
    retrieval_result: StyleEvidenceResult | None,
    *,
    user_id: str,
) -> StyleEvidenceResult | None:
    if retrieval_result is None:
        return None
    if type(retrieval_result) is not StyleEvidenceResult:
        raise ValueError
    validated = StyleEvidenceResult.model_validate(
        retrieval_result.model_dump(mode="python", warnings=False)
    )
    if validated.user_id != user_id:
        raise ValueError
    return validated


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().replace("\N{FULLWIDTH COMMA}", ",").lower()


def _semantic_check(request: str, response: str) -> StyleVerificationCheck:
    rule = _SEMANTIC_RULES.get(request)
    if rule is None:
        return _check(
            StyleVerificationCheckName.SEMANTIC_CORRECTNESS,
            passed=True,
            covered=False,
            code="public_constraint_not_registered",
        )
    passed = any(item in response for item in rule.required_any) and not any(
        item in response for item in rule.forbidden_any
    )
    return _check(
        StyleVerificationCheckName.SEMANTIC_CORRECTNESS,
        passed=passed,
        covered=True,
        code="public_constraint_passed" if passed else "public_constraint_failed",
    )


def _style_check(
    response: str,
    references: tuple[str, ...],
    *,
    mode: str,
) -> StyleVerificationCheck:
    if mode == "baseline":
        return _check(
            StyleVerificationCheckName.STYLE_CONSISTENCY,
            passed=True,
            covered=False,
            code="baseline_not_assessed",
        )
    normalized_references = tuple(_normalize(item) for item in references if item.strip())
    if not normalized_references:
        return _check(
            StyleVerificationCheckName.STYLE_CONSISTENCY,
            passed=True,
            covered=False,
            code="trusted_reference_unavailable",
        )
    maximum = max(80, 4 * max(len(item) for item in normalized_references))
    references_use_lists = any("\n-" in item or "\n1." in item for item in normalized_references)
    response_uses_lists = "\n-" in response or "\n1." in response
    passed = len(response) <= maximum and (references_use_lists or not response_uses_lists)
    return _check(
        StyleVerificationCheckName.STYLE_CONSISTENCY,
        passed=passed,
        covered=True,
        code="bounded_form_passed" if passed else "bounded_form_failed",
    )


def _identity_check(response: str) -> StyleVerificationCheck:
    passed = not any(pattern in response for pattern in _IDENTITY_PATTERNS)
    return _check(
        StyleVerificationCheckName.IDENTITY_CONFUSION,
        passed=passed,
        covered=True,
        code="identity_boundary_passed" if passed else "identity_boundary_failed",
    )


def _mechanical_check(
    response: str,
    references: tuple[str, ...],
    previous_soft_lock: str | None,
) -> StyleVerificationCheck:
    copied = any(
        len("".join(normalized.split())) >= 8 and normalized in response
        for normalized in (_normalize(item) for item in references)
    )
    repeated_soft_lock = False
    if previous_soft_lock is not None:
        normalized_soft_lock = _normalize(previous_soft_lock)
        repeated_soft_lock = bool(normalized_soft_lock and normalized_soft_lock in response)
    passed = not copied and not repeated_soft_lock
    return _check(
        StyleVerificationCheckName.MECHANICAL_REPETITION,
        passed=passed,
        covered=True,
        code="repetition_check_passed" if passed else "repetition_check_failed",
    )


def _generic_tone_check(response: str, *, mode: str) -> StyleVerificationCheck:
    if mode == "baseline":
        return _check(
            StyleVerificationCheckName.GENERIC_ASSISTANT_TONE,
            passed=True,
            covered=False,
            code="baseline_not_assessed",
        )
    passed = not any(pattern in response for pattern in _GENERIC_ASSISTANT_PATTERNS)
    return _check(
        StyleVerificationCheckName.GENERIC_ASSISTANT_TONE,
        passed=passed,
        covered=True,
        code="generic_tone_check_passed" if passed else "generic_tone_check_failed",
    )


def _check(
    name: StyleVerificationCheckName,
    *,
    passed: bool,
    covered: bool,
    code: str,
) -> StyleVerificationCheck:
    return StyleVerificationCheck(name=name, passed=passed, covered=covered, code=code)


def _invalid_report(code: str) -> StyleVerificationReport:
    checks = tuple(
        _check(name, passed=False, covered=True, code=code) for name in StyleVerificationCheckName
    )
    return StyleVerificationReport(
        semantic_correctness=checks[0],
        style_consistency=checks[1],
        identity_confusion=checks[2],
        mechanical_repetition=checks[3],
        generic_assistant_tone=checks[4],
        action=StyleVerificationAction.REJECT,
    )
