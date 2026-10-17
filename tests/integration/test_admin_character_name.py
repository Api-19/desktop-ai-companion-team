# ruff: noqa: RUF001

"""Synthetic Admin save and Dialogue restart, without real secrets or providers."""

import html
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from io import BytesIO
from urllib.parse import urlencode

from companion.admin.app import build_admin_app, load_admin_config
from companion.contracts import AddresseeRegister, DialogueRequest, StyleEvidence
from companion.dialogue.config import load_dialogue_config
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import ProviderResult, ProviderUsage
from companion.dialogue.orchestrator import DialogueOrchestrator
from companion.dialogue.usage import MonthlyUsageLedger
from companion.runtime.settings import Settings

NOW = datetime(2026, 10, 17, tzinfo=UTC)
ORIGIN = "http://127.0.0.1:12345"


class SyntheticStyle:
    def accepts_user(self, user_id: str) -> bool:
        return type(user_id) is str

    def evidence_for(self, *args: object, **kwargs: object) -> StyleEvidence:
        return StyleEvidence(
            style_version="synthetic-snippet-v1", addressee_register=AddresseeRegister.NEUTRAL
        )


class CaptureProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[PromptMessage, ...]] = []

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        self.requests.append(tuple(messages))
        return ProviderResult(
            text="合成安全回答",
            usage=ProviderUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        )


def call(app, path, *, cookie="", form=None):
    body = urlencode(form).encode() if form is not None else b""
    captured = []
    page = b"".join(
        app(
            {
                "REQUEST_METHOD": "POST" if form is not None else "GET",
                "PATH_INFO": path,
                "HTTP_HOST": "127.0.0.1:12345",
                "HTTP_ORIGIN": ORIGIN if form is not None else "",
                "HTTP_COOKIE": cookie,
                "CONTENT_LENGTH": str(len(body)),
                "wsgi.input": BytesIO(body),
            },
            lambda status, headers: captured.append((status, dict(headers))),
        )
    ).decode()
    status, headers = captured[0]
    return int(status[:3]), headers, page


def test_saved_character_name_is_used_after_rebuild(tmp_path):
    settings = Settings(
        environment="test",
        log_level="ERROR",
        private_data_dir=tmp_path / "synthetic-private",
        runtime_data_dir=tmp_path / "synthetic-runtime",
        test_results_dir=tmp_path / "synthetic-results",
    )
    settings.private_data_dir.mkdir(mode=0o700)
    admin_dir = settings.private_data_dir / "admin"
    admin_dir.mkdir(mode=0o700)
    admin_config = admin_dir / "config.toml"
    admin_config.write_text('enabled = true\nowner_id = "synthetic-owner"\n')
    admin_config.chmod(0o600)
    path = settings.private_data_dir / "dialogue" / "runtime-config.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "character_name": "旧合成角色",
                "provider": {
                    "base_url": "https://provider.invalid/v1",
                    "model": "synthetic-model",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 120,
                },
                "budget": {
                    "monthly_cap_cny": "100.00",
                    "worst_case_request_cny": "1.00",
                    "input_cny_per_million": "1.00",
                    "output_cny_per_million": "1.00",
                    "max_output_tokens": 256,
                },
                "snippet_style": {
                    "library_source": "synthetic-snippets.jsonl",
                    "partner_user_id": "synthetic-partner",
                },
            }
        )
    )
    environment = {"OPENAI_API_KEY": "synthetic-key-only"}
    original_config, _ = load_dialogue_config(
        settings, environ=environment, require_style_manifest=False
    )
    old_provider = CaptureProvider()
    old = DialogueOrchestrator(
        config=original_config,
        snippet_style=SyntheticStyle(),
        provider=old_provider,
        ledger=MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", original_config.budget),
        clock=lambda: NOW,
    )
    request = DialogueRequest(user_id="synthetic-owner", text="合成问题", recent_context=[])
    first = old.reply(request, trusted_user_id="synthetic-owner")
    assert first.response.blocked_reason is None
    assert first.response.text.startswith("我是 旧合成角色，一个 AI")
    assert "名字叫“旧合成角色”" in old_provider.requests[-1][0].content

    app = build_admin_app(settings, load_admin_config(settings), port=12345)
    code = app.issue_start_code()
    _, headers, page = call(app, "/admin")
    pre_cookie = headers["Set-Cookie"].split(";", 1)[0]
    csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    status, headers, _ = call(
        app, "/admin/unlock", cookie=pre_cookie, form={"csrf": csrf, "code": code}
    )
    assert status == 303
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    status, _, page = call(app, "/admin/character", cookie=cookie)
    assert status == 200
    values = {
        key: html.unescape(value)
        for key, value in re.findall(r'name="([^"]+)" value="([^"]*)"', page)
    }
    values["name"] = "新合成角色"
    status, _, page = call(app, "/admin/character", cookie=cookie, form=values)
    assert status == 200 and "角色名称已保存，重启对话后生效。" in page
    assert json.loads(path.read_bytes())["character_name"] == "新合成角色"

    still_old = old.reply(
        request.model_copy(update={"text": "另一个合成问题"}), trusted_user_id="synthetic-owner"
    )
    assert still_old.response.blocked_reason is None
    assert "名字叫“旧合成角色”" in old_provider.requests[-1][0].content
    assert "新合成角色" not in old_provider.requests[-1][0].content
    old_first = old.reply(
        request.model_copy(update={"user_id": "synthetic-second-user"}),
        trusted_user_id="synthetic-second-user",
    )
    assert old_first.response.text.startswith("我是 旧合成角色，一个 AI")

    rebuilt_config, _ = load_dialogue_config(
        settings, environ=environment, require_style_manifest=False
    )
    new_provider = CaptureProvider()
    rebuilt = DialogueOrchestrator(
        config=rebuilt_config,
        snippet_style=SyntheticStyle(),
        provider=new_provider,
        ledger=MonthlyUsageLedger(settings.runtime_data_dir / "dialogue", rebuilt_config.budget),
        clock=lambda: NOW,
    )
    result = rebuilt.reply(request, trusted_user_id="synthetic-owner")
    assert result.response.blocked_reason is None
    assert result.response.disclosed_ai_identity
    assert result.response.text.startswith("我是 新合成角色，一个 AI")
    assert "名字叫“新合成角色”" in new_provider.requests[0][0].content
    assert "旧合成角色" not in new_provider.requests[0][0].content
