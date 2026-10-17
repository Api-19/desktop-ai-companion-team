from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from companion.dialogue.models import (
    BudgetConfig,
    DialogueConfig,
    DialogueSecrets,
    ProductionStyleSources,
    SnippetStyleSources,
    StyleRuntimeSources,
)
from companion.runtime.settings import Settings


class DialogueConfigError(RuntimeError):
    """Raised when dialogue configuration cannot be loaded safely."""


PRIVATE_STYLE_CONFIG_FIELDS = frozenset(
    {
        "active_manifest_source",
        "active_style_source",
        "approved_nicknames",
        "calibration_source",
        "candidate_source",
        "expected_calibration_fingerprint",
        "expected_count",
        "true_name_terms",
    }
)

FRAMEWORK_SOURCE_FIELDS = frozenset(ProductionStyleSources.model_fields)


def _reject_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise ValueError("private root contains symlink")
        if current == current.parent:
            return
        current = current.parent


def validate_framework_sources(sources: ProductionStyleSources, private_root: Path) -> None:
    """Recheck fixed source paths, including symlink drift, before constructing stores."""
    _reject_symlink_ancestors(private_root)
    root = private_root.resolve(strict=False)
    paths = tuple(getattr(sources, field) for field in sorted(FRAMEWORK_SOURCE_FIELDS))
    if any(not isinstance(path, Path) for path in paths):
        raise ValueError("invalid framework source")
    for path in paths:
        if not path.is_absolute() or not path.is_relative_to(root) or path == root:
            raise ValueError("framework source escapes root")
        if path.resolve(strict=False) != path:
            raise ValueError("framework source contains traversal or symlink")
        current = path
        while current != root:
            if current.is_symlink():
                raise ValueError("framework source contains symlink")
            current = current.parent
    for index, first in enumerate(paths):
        if any(
            first == second or first in second.parents or second in first.parents
            for second in paths[index + 1 :]
        ):
            raise ValueError("framework sources overlap")


def _framework_sources(payload: object, private_root: Path) -> ProductionStyleSources:
    if not isinstance(payload, dict) or set(payload) != FRAMEWORK_SOURCE_FIELDS:
        raise ValueError("invalid framework sources")
    resolved = {}
    for field in FRAMEWORK_SOURCE_FIELDS:
        value = payload[field]
        if type(value) is not str or not value or ".." in Path(value).parts:
            raise ValueError("invalid framework source")
        raw = Path(value)
        resolved[field] = private_root / raw if not raw.is_absolute() else raw
    sources = ProductionStyleSources.model_validate(resolved)
    validate_framework_sources(sources, private_root)
    return sources


def _resolve_private_source(value: object, private_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid private source")
    source = Path(value)
    resolved = (private_root / source).resolve() if not source.is_absolute() else source.resolve()
    if not resolved.is_relative_to(private_root):
        raise ValueError("private source escapes root")
    return resolved


def _snippet_sources(payload: object, private_root: Path) -> SnippetStyleSources:
    if not isinstance(payload, dict):
        raise ValueError("invalid snippet style configuration")
    card = payload.get("habit_card_source")
    terms = payload.get("shared_terms_source")
    return SnippetStyleSources.model_validate(
        {
            **payload,
            "library_source": _resolve_private_source(payload.get("library_source"), private_root),
            "habit_card_source": (
                None if card is None else _resolve_private_source(card, private_root)
            ),
            "shared_terms_source": (
                None if terms is None else _resolve_private_source(terms, private_root)
            ),
        }
    )


def _validated_config(payload: object, settings: Settings, digest: str | None) -> DialogueConfig:
    if not isinstance(payload, dict):
        raise ValueError("invalid dialogue configuration")
    _reject_symlink_ancestors(settings.private_data_dir)
    private_root = settings.private_data_dir.resolve()
    if "snippet_style" in payload:
        return DialogueConfig.model_validate(
            {
                **payload,
                "style": None,
                "style_framework": None,
                "snippet_style": _snippet_sources(payload["snippet_style"], private_root),
            }
        )
    if digest is None:
        return DialogueConfig.model_validate({**payload, "style": None, "style_framework": None})
    style_payload = payload.get("style")
    if not isinstance(style_payload, dict):
        raise ValueError("invalid style configuration")
    if set(style_payload) - PRIVATE_STYLE_CONFIG_FIELDS:
        raise ValueError("unexpected private style configuration")
    style = StyleRuntimeSources.model_validate(
        {
            **style_payload,
            "candidate_source": _resolve_private_source(
                style_payload.get("candidate_source"), private_root
            ),
            "active_style_source": _resolve_private_source(
                style_payload.get("active_style_source"), private_root
            ),
            "active_manifest_source": _resolve_private_source(
                style_payload.get("active_manifest_source"), private_root
            ),
            "calibration_source": _resolve_private_source(
                style_payload.get("calibration_source"), private_root
            ),
            "expected_active_manifest_digest": digest,
            "private_data_dir": private_root,
        }
    )
    framework = (
        _framework_sources(payload["style_framework"], private_root)
        if "style_framework" in payload
        else None
    )
    return DialogueConfig.model_validate({**payload, "style": style, "style_framework": framework})


def load_dialogue_config(
    settings: Settings,
    environ: Mapping[str, str] | None = None,
    *,
    require_style_manifest: bool = True,
) -> tuple[DialogueConfig, DialogueSecrets]:
    """Load runtime settings, optionally skipping legacy style and framework sources."""
    environment = os.environ if environ is None else environ
    try:
        _reject_symlink_ancestors(settings.private_data_dir)
        payload: Any = json.loads(
            (settings.private_data_dir / "dialogue" / "runtime-config.json").read_text(
                encoding="utf-8"
            )
        )
        digest = None
        if require_style_manifest and "snippet_style" not in payload:
            digest = environment["STYLE_ACTIVE_MANIFEST_DIGEST"]
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("invalid manifest digest")
        config = _validated_config(payload, settings, digest)
        api_key = environment[config.provider.api_key_env]
        if not api_key.strip():
            raise ValueError("missing api key")
        return config, DialogueSecrets(api_key=api_key)
    except Exception:
        settings = None  # type: ignore[assignment]
        environ = None
        environment = None  # type: ignore[assignment]
        payload = None
        config = None  # type: ignore[assignment]
        api_key = None  # type: ignore[assignment]
        digest = None  # type: ignore[assignment]
        raise DialogueConfigError("dialogue configuration unavailable") from None


def load_budget_config(settings: Settings) -> BudgetConfig:
    """Read only the budget segment at the fixed private configuration path."""
    path: Path | None = None
    payload: object = None
    budget: BudgetConfig | None = None
    descriptor: int | None = None
    before = current = stream = None
    failure = False
    try:
        path = settings.private_data_dir / "dialogue" / "runtime-config.json"
        _reject_symlink_ancestors(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("invalid budget configuration file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            current = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("budget configuration changed")
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
                payload = json.load(stream)
        finally:
            os.close(descriptor)
        if type(payload) is not dict:
            raise ValueError("invalid budget configuration")
        budget = BudgetConfig.model_validate(payload["budget"])
    except Exception:
        failure = True
    del settings, path, payload, descriptor, before, current, stream
    if failure:
        del budget
        raise DialogueConfigError("dialogue configuration unavailable")
    return budget
