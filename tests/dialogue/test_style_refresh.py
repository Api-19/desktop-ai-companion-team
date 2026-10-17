from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest
from dialogue_support import synthetic_dialogue_config

from companion.contracts import StyleIntent, StyleReplyGuidance
from companion.dialogue import preflight_style_refresh
from companion.runtime.settings import Settings
from companion.style import SingleUserStyleAdapter

EXPECTED_INTENTS = {
    "我还没吃饭": StyleIntent.NOT_EATEN_YET_CARE,
    "我准备休息了": StyleIntent.SLEEP_FAREWELL,
    "今晚又熬夜了": StyleIntent.SLEEP_REMINDER,
    "外面好像要下雨": StyleIntent.RAIN_REMINDER,
    "家务刚做完": StyleIntent.HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT,
    "我又想买东西了": StyleIntent.SHOPPING_DESIRE_ACKNOWLEDGEMENT,
    "我准备出门了": StyleIntent.GOING_OUT_CARE,
}


class SyntheticPreciseStyleProvider:
    def __init__(
        self,
        *,
        trusted_user_id: str = "synthetic-preflight-owner",
        status_override: str | None = None,
        wrong_target_intent: bool = False,
    ) -> None:
        self.trusted_user_id = trusted_user_id
        self.status_override = status_override
        self.wrong_target_intent = wrong_target_intent
        self.call_count = 0
        self.user_ids: list[str] = []

    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str and user_id == self.trusted_user_id

    def guidance_for(
        self,
        user_id: str,
        text: str,
        recent_context: list[str],
        idle_seconds: int | None,
    ) -> StyleReplyGuidance:
        self.call_count += 1
        self.user_ids.append(user_id)
        if self.call_count == 1 and self.status_override == "no_match":
            return StyleReplyGuidance(status="no_match", style_version="synthetic-v1")
        if self.call_count == 1 and self.status_override == "unavailable":
            return StyleReplyGuidance(status="unavailable")
        intent = EXPECTED_INTENTS.get(text, StyleIntent.CONVERSATION_REOPEN)
        if self.wrong_target_intent and text == "我还没吃饭":
            intent = StyleIntent.CATEGORY_DEFAULT
        return StyleReplyGuidance(
            status="matched",
            style_version="synthetic-v1",
            category="synthetic",
            intent=intent,
            examples=["synthetic safe example"],
        )


def synthetic_settings(root: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=root / "private",
        runtime_data_dir=root / "runtime",
        test_results_dir=root / "results",
    )


def test_preflight_without_style_is_not_ready(tmp_path: Path) -> None:
    config = synthetic_dialogue_config(tmp_path).model_copy(update={"style": None})
    summary = preflight_style_refresh(synthetic_settings(tmp_path), config)
    assert summary.status == "not_ready"
    assert summary.provider_requests == 0


def test_preflight_style_refresh_validates_twenty_and_seven_without_provider(
    tmp_path: Path,
) -> None:
    style = SyntheticPreciseStyleProvider()

    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        synthetic_dialogue_config(tmp_path),
        style_adapter=style,
    )

    assert summary.status == "ready"
    assert summary.fixed_scenario_count == 20
    assert summary.matched_count == 20
    assert summary.no_match_count == 0
    assert summary.unavailable_count == 0
    assert summary.refresh_scenario_count == 7
    assert summary.refresh_precise_count == 7
    assert summary.request_capacity == 15
    assert summary.budget_capacity is True
    assert summary.prompt_pair_count == 7
    assert summary.provider_requests == 0
    assert style.call_count == 20


def test_preflight_style_refresh_uses_adapter_owner_for_all_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from companion.dialogue import style_refresh

    style = SyntheticPreciseStyleProvider(trusted_user_id="bound-owner")
    request_ids: list[str] = []
    prepare = style_refresh.prepare_dialogue_input

    def capture_request(request):
        request_ids.append(request.user_id)
        return prepare(request)

    monkeypatch.setattr(style_refresh, "prepare_dialogue_input", capture_request)

    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        synthetic_dialogue_config(tmp_path),
        style_adapter=style,
    )

    assert summary.status == "ready"
    assert style.user_ids == ["bound-owner"] * 20
    assert request_ids == ["bound-owner"] * 7


@pytest.mark.parametrize("owner", (None, "bound-owner"))
def test_preflight_style_refresh_concrete_adapter_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner: str | None,
) -> None:
    from companion.style import single_user_adapter

    expected_owner = owner or "synthetic-preflight-owner"
    style = SyntheticPreciseStyleProvider(trusted_user_id=expected_owner)

    def guidance(text, recent_context, idle_seconds, *, user_id, **kwargs):
        return style.guidance_for(user_id, text, recent_context, idle_seconds)

    monkeypatch.setattr(single_user_adapter, "build_style_reply_guidance", guidance)
    config = synthetic_dialogue_config(tmp_path)
    adapter = (
        SingleUserStyleAdapter(trusted_user_id=owner, **config.style.as_call_kwargs())
        if owner is not None
        else None
    )
    if adapter is not None:
        assert adapter.trusted_user_id == owner
        with pytest.raises(AttributeError):
            adapter.trusted_user_id = "different-owner"

    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        config,
        style_adapter=adapter,
    )

    assert summary.status == "ready"
    assert style.user_ids == [expected_owner] * 20


def test_preflight_style_refresh_rejects_missing_or_unaccepted_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style = SyntheticPreciseStyleProvider(trusted_user_id="bound-owner")
    monkeypatch.setattr(style, "accepts_user", lambda user_id: False)
    config = synthetic_dialogue_config(tmp_path)
    settings = synthetic_settings(tmp_path)

    assert preflight_style_refresh(settings, config, style_adapter=style).status == "not_ready"
    del style.trusted_user_id
    assert preflight_style_refresh(settings, config, style_adapter=style).status == "not_ready"
    assert style.call_count == 0
    assert not settings.runtime_data_dir.exists()


@pytest.mark.parametrize("trusted_user_id", ("", " ", "x" * 121))
def test_preflight_style_refresh_rejects_invalid_adapter_owner(
    tmp_path: Path,
    trusted_user_id: str,
) -> None:
    style = SyntheticPreciseStyleProvider(trusted_user_id=trusted_user_id)

    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        synthetic_dialogue_config(tmp_path),
        style_adapter=style,
    )

    assert summary.status == "not_ready"
    assert summary.provider_requests == 0
    assert style.call_count == 0


@pytest.mark.parametrize("status", ("no_match", "unavailable"))
def test_preflight_style_refresh_fails_closed_for_nonmatched_guidance(
    tmp_path: Path,
    status: str,
) -> None:
    style = SyntheticPreciseStyleProvider(status_override=status)

    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        synthetic_dialogue_config(tmp_path),
        style_adapter=style,
    )

    assert summary.status == "not_ready"
    assert summary.provider_requests == 0


def test_preflight_style_refresh_rejects_wrong_precise_intent(tmp_path: Path) -> None:
    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        synthetic_dialogue_config(tmp_path),
        style_adapter=SyntheticPreciseStyleProvider(wrong_target_intent=True),
    )

    assert summary.status == "not_ready"
    assert summary.refresh_precise_count == 6
    assert summary.provider_requests == 0


def test_preflight_style_refresh_fails_closed_when_capacity_is_insufficient(
    tmp_path: Path,
) -> None:
    config = synthetic_dialogue_config(tmp_path)
    config = config.model_copy(
        update={
            "budget": config.budget.model_copy(update={"worst_case_request_cny": Decimal("7.00")})
        }
    )

    summary = preflight_style_refresh(
        synthetic_settings(tmp_path),
        config,
        style_adapter=SyntheticPreciseStyleProvider(),
    )

    assert summary.status == "not_ready"
    assert summary.budget_capacity is False
    assert summary.provider_requests == 0


def load_preflight_script() -> ModuleType:
    path = Path("scripts/preflight_dialogue_style_refresh.py").resolve()
    spec = importlib.util.spec_from_file_location("preflight_dialogue_style_refresh", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_script_prints_only_aggregate_summary(monkeypatch, capsys) -> None:
    module = load_preflight_script()
    expected = preflight_style_refresh(
        synthetic_settings(Path("synthetic-root")),
        synthetic_dialogue_config(Path("synthetic-root")),
        style_adapter=SyntheticPreciseStyleProvider(),
    )
    monkeypatch.setattr(module, "load_settings", lambda root: object())
    monkeypatch.setattr(module, "load_dialogue_config", lambda settings: (object(), object()))
    monkeypatch.setattr(module, "preflight_style_refresh", lambda settings, config: expected)

    assert module.main() == 0
    assert json.loads(capsys.readouterr().out) == expected.model_dump(mode="json")
