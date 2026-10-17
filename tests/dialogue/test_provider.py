from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
from pydantic import SecretStr

import companion.dialogue as dialogue_package
import companion.dialogue.provider as provider_module
from companion.dialogue.context import PromptMessage
from companion.dialogue.models import ProviderConfig, ProviderUsage
from companion.dialogue.provider import (
    InvalidProviderResponse,
    OpenAICompatibleProvider,
    OpenAIProviderUnavailable,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailableKind,
    ProviderUnavailableStage,
    Transport,
)

_REQUEST_TYPE = urllib.request.Request


class RecordingTransport:
    def __init__(self, response: object) -> None:
        self._response = response
        self.call_count = 0
        self.request_payload: dict[str, object] = {}
        self.method = ""
        self.url = ""
        self.headers: dict[str, str] = {}

    def __call__(self, request: urllib.request.Request, timeout: int) -> bytes:
        self.call_count += 1
        self.request_payload = json.loads(request.data or b"{}")
        self.method = request.get_method()
        self.url = request.full_url
        self.headers = dict(request.header_items())
        assert timeout == 120
        return json.dumps(self._response).encode()


class RaisingTransport:
    def __init__(self, exception_factory: Callable[[], Exception]) -> None:
        self._exception_factory = exception_factory
        self.call_count = 0

    def __call__(self, request: urllib.request.Request, timeout: int) -> bytes:
        self.call_count += 1
        raise self._exception_factory()


def make_provider(transport: Transport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        config=ProviderConfig(
            base_url="https://jojocode.com/v1",
            model="gpt-5.6-terra",
            timeout_seconds=120,
        ),
        api_key=SecretStr("synthetic-api-key"),
        max_output_tokens=256,
        transport=transport,
    )


def make_default_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        config=ProviderConfig(
            base_url="https://jojocode.com/v1",
            model="gpt-5.6-terra",
            timeout_seconds=120,
        ),
        api_key=SecretStr("synthetic-api-key"),
        max_output_tokens=256,
    )


def sample_messages() -> tuple[PromptMessage, PromptMessage]:
    return (
        PromptMessage(role="system", content="synthetic system"),
        PromptMessage(role="user", content="synthetic request"),
    )


@pytest.mark.parametrize(
    ("base_url", "disable_thinking"),
    [
        ("https://api.deepseek.com", True),
        ("https://api.deepseek.com/v1", True),
        ("https://api.deepseek.com:443/v1", True),
        ("https://api.deepseek.com/beta", True),
        ("https://api.deepseek.com:8443/v1", False),
        ("https://api.deepseek.com.example.com/v1", False),
        ("https://example.com/api.deepseek.com", False),
        ("https://api.openai.com/v1", False),
        ("https://jojocode.com/v1", False),
    ],
)
def test_only_official_deepseek_requests_disable_thinking(
    base_url: str, disable_thinking: bool
) -> None:
    transport = RecordingTransport({"choices": [{"message": {"content": "synthetic answer"}}]})
    provider = OpenAICompatibleProvider(
        config=ProviderConfig(base_url=base_url, model="deepseek-flash", timeout_seconds=120),
        api_key=SecretStr("synthetic-api-key"),
        max_output_tokens=256,
        transport=transport,
    )
    assert provider.complete(sample_messages()).text == "synthetic answer"
    expected: dict[str, object] = {
        "model": "deepseek-flash",
        "messages": [
            {"role": "system", "content": "synthetic system"},
            {"role": "user", "content": "synthetic request"},
        ],
        "max_tokens": 256,
    }
    if disable_thinking:
        expected["thinking"] = {"type": "disabled"}
    assert transport.request_payload == expected
    assert transport.call_count == 1
    assert transport.method == "POST"
    expected_endpoint = (
        "https://api.deepseek.com/v1/chat/completions"
        if base_url == "https://api.deepseek.com:443/v1"
        else base_url.rstrip("/") + "/chat/completions"
    )
    assert transport.url == expected_endpoint
    assert transport.headers == {
        "Authorization": "Bearer synthetic-api-key",
        "Content-type": "application/json",
    }


@pytest.mark.parametrize("content", ["", None])
def test_reasoning_content_is_never_used_as_a_reply(content: object) -> None:
    transport = RecordingTransport(
        {"choices": [{"message": {"content": content, "reasoning_content": "PRIVATE_REASONING"}}]}
    )
    provider = make_provider(transport)
    with pytest.raises(InvalidProviderResponse, match=r"^invalid provider response$"):
        provider.complete(sample_messages())
    assert transport.call_count == 1


def test_provider_unavailable_diagnostic_contract_is_fixed_and_enum_only() -> None:
    assert {item.value for item in ProviderUnavailableKind} == {
        "dns_unavailable",
        "connect_unavailable",
        "tls_unavailable",
        "redirect_refused",
        "http_400_bad_request",
        "http_401_unauthorized",
        "http_403_forbidden",
        "http_404_not_found",
        "http_408_timeout",
        "http_429_rate_limited",
        "http_4xx_other",
        "http_5xx",
        "transport_unknown",
    }
    assert {item.value for item in ProviderUnavailableStage} == {
        "request_serialization",
        "request_construction",
        "opener_construction",
        "opener_open",
        "response_read",
        "response_close",
        "transport_call",
    }
    error = OpenAIProviderUnavailable(
        ProviderUnavailableKind.DNS_UNAVAILABLE,
        ProviderUnavailableStage.OPENER_OPEN,
    )
    assert str(error) == "provider unavailable"
    assert type(error.kind) is ProviderUnavailableKind
    assert type(error.stage) is ProviderUnavailableStage
    assert error.stage is ProviderUnavailableStage.OPENER_OPEN
    assert error.__cause__ is None
    assert error.__context__ is None
    with pytest.raises(TypeError):
        OpenAIProviderUnavailable("dns_unavailable", ProviderUnavailableStage.OPENER_OPEN)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpenAIProviderUnavailable(True, ProviderUnavailableStage.OPENER_OPEN)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpenAIProviderUnavailable()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OpenAIProviderUnavailable(ProviderUnavailableKind.DNS_UNAVAILABLE)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OpenAIProviderUnavailable(ProviderUnavailableKind.DNS_UNAVAILABLE, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpenAIProviderUnavailable(ProviderUnavailableKind.DNS_UNAVAILABLE, "opener_open")  # type: ignore[arg-type]

    class OtherStage(StrEnum):
        OPENER_OPEN = "opener_open"

    with pytest.raises(TypeError):
        OpenAIProviderUnavailable(ProviderUnavailableKind.DNS_UNAVAILABLE, OtherStage.OPENER_OPEN)  # type: ignore[arg-type]

    interactions: list[str] = []

    class HostileEnum(StrEnum):
        VALUE = "opener_open"

        def __str__(self) -> str:
            interactions.append("str")
            raise RuntimeError("PRIVATE_ENUM_STRING_SENTINEL")

        def __repr__(self) -> str:
            interactions.append("repr")
            raise RuntimeError("PRIVATE_ENUM_REPR_SENTINEL")

    with pytest.raises(TypeError, match=r"^invalid provider unavailable kind$"):
        OpenAIProviderUnavailable(  # type: ignore[arg-type]
            HostileEnum.VALUE,
            ProviderUnavailableStage.OPENER_OPEN,
        )
    with pytest.raises(TypeError, match=r"^invalid provider unavailable stage$"):
        OpenAIProviderUnavailable(  # type: ignore[arg-type]
            ProviderUnavailableKind.DNS_UNAVAILABLE,
            HostileEnum.VALUE,
        )
    assert interactions == []
    assert "ProviderUnavailableKind" not in dialogue_package.__all__
    assert "ProviderUnavailableStage" not in dialogue_package.__all__
    assert "OpenAIProviderUnavailable" not in dialogue_package.__all__


def test_provider_unavailable_constructor_rejects_hostile_exception_subclass_without_hooks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactions: list[str] = []

    class HostileUnavailable(OpenAIProviderUnavailable):
        def __setattr__(self, name: str, value: object) -> None:
            interactions.append(name)
            print("PRIVATE_UNAVAILABLE_SETATTR_SENTINEL")
            raise RuntimeError("PRIVATE_UNAVAILABLE_SETATTR_SENTINEL")

    with pytest.raises(TypeError, match=r"^invalid provider unavailable type$"):
        HostileUnavailable(
            ProviderUnavailableKind.DNS_UNAVAILABLE,
            ProviderUnavailableStage.OPENER_OPEN,
        )

    assert interactions == []
    captured = capsys.readouterr()
    assert "PRIVATE_UNAVAILABLE_" not in captured.out + captured.err


class _StageResponse:
    def __init__(
        self,
        *,
        read_error: Exception | None = None,
        close_error: Exception | None = None,
        response: bytes = b'{"choices":[{"message":{"content":"synthetic"}}]}',
    ) -> None:
        self._read_error = read_error
        self._close_error = close_error
        self._response = response
        self.close_count = 0

    def __enter__(self) -> _StageResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        self.close()
        return False

    def read(self, size: int = -1) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._response if size < 0 else self._response[:size]

    def close(self) -> None:
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error


class _StageOpener:
    def __init__(
        self,
        *,
        response: _StageResponse | None = None,
        open_error: Exception | None = None,
    ) -> None:
        self._response = response
        self._open_error = open_error
        self.open_count = 0

    def open(self, request: urllib.request.Request, timeout: int) -> _StageResponse:
        self.open_count += 1
        if self._open_error is not None:
            raise self._open_error
        assert self._response is not None
        return self._response


def test_default_transport_caps_read_and_closes_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes: list[int] = []

    class LargeResponse(_StageResponse):
        def read(self, size: int = -1) -> bytes:
            sizes.append(size)
            return b" " * (1024 * 1024 + 1)

    response = LargeResponse()
    opener = _StageOpener(response=response)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(InvalidProviderResponse, match=r"^invalid provider response$"):
        make_default_provider().complete(sample_messages())
    assert sizes == [1024 * 1024 + 1]
    assert response.close_count == 1
    assert opener.open_count == 1


@pytest.mark.parametrize("extra", [0, 1])
def test_response_size_boundary_is_checked_before_json_parsing(extra: int) -> None:
    raw = b'{"choices":[{"message":{"content":"synthetic"}}]}'
    raw += b" " * (1024 * 1024 + extra - len(raw))
    calls = 0

    def transport(request: urllib.request.Request, timeout: int) -> bytes:
        nonlocal calls
        calls += 1
        return raw

    provider = make_provider(transport)
    if extra:
        with pytest.raises(InvalidProviderResponse, match=r"^invalid provider response$"):
            provider.complete(sample_messages())
    else:
        assert provider.complete(sample_messages()).text == "synthetic"
    assert calls == 1


def _assert_unavailable_stage(
    error: OpenAIProviderUnavailable,
    stage: ProviderUnavailableStage,
    *,
    kind: ProviderUnavailableKind = ProviderUnavailableKind.TRANSPORT_UNKNOWN,
) -> None:
    assert error.kind is kind
    assert error.stage is stage
    assert str(error) == "provider unavailable"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_request_serialization_failure_has_exact_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport({})

    def fail_dump(self: PromptMessage) -> dict[str, object]:
        raise OSError("PRIVATE_SERIALIZATION_SENTINEL")

    monkeypatch.setattr(PromptMessage, "model_dump", fail_dump)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.REQUEST_SERIALIZATION)
    assert transport.call_count == 0


def test_request_construction_failure_has_exact_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport({})

    def fail_request(*args: object, **kwargs: object) -> None:
        raise OSError("PRIVATE_REQUEST_SENTINEL")

    monkeypatch.setattr(provider_module.urllib.request, "Request", fail_request)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.REQUEST_CONSTRUCTION)
    assert transport.call_count == 0


def test_default_opener_construction_failure_has_exact_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build(*args: object, **kwargs: object) -> None:
        raise OSError("PRIVATE_OPENER_SENTINEL")

    monkeypatch.setattr(provider_module.urllib.request, "build_opener", fail_build)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.OPENER_CONSTRUCTION)


def test_default_open_failure_has_exact_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = _StageOpener(open_error=OSError("PRIVATE_OPEN_SENTINEL"))
    monkeypatch.setattr(provider_module.urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.OPENER_OPEN)
    assert opener.open_count == 1


def test_default_read_failure_has_exact_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _StageResponse(read_error=http.client.IncompleteRead(b"", 1))
    opener = _StageOpener(response=response)
    monkeypatch.setattr(provider_module.urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.RESPONSE_READ)
    assert response.close_count == 1


def test_default_close_failure_has_exact_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _StageResponse(close_error=OSError("PRIVATE_CLOSE_SENTINEL"))
    opener = _StageOpener(response=response)
    monkeypatch.setattr(provider_module.urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.RESPONSE_CLOSE)
    assert response.close_count == 1


def test_default_close_failure_takes_precedence_over_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StageResponse(
        read_error=http.client.IncompleteRead(b"", 1),
        close_error=OSError("PRIVATE_CLOSE_SENTINEL"),
    )
    opener = _StageOpener(response=response)
    monkeypatch.setattr(provider_module.urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.RESPONSE_CLOSE)
    assert response.close_count == 1


def test_injected_transport_failure_has_transport_call_stage() -> None:
    transport = RaisingTransport(lambda: OSError("PRIVATE_TRANSPORT_SENTINEL"))
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.TRANSPORT_CALL)
    assert transport.call_count == 1


@pytest.mark.parametrize(
    ("failure", "kind"),
    (
        (
            urllib.error.HTTPError("https://public.invalid", 408, "x", {}, None),
            ProviderUnavailableKind.HTTP_408_TIMEOUT,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 503, "x", {}, None),
            ProviderUnavailableKind.HTTP_5XX,
        ),
        (urllib.error.URLError(socket.gaierror()), ProviderUnavailableKind.DNS_UNAVAILABLE),
        (
            urllib.error.URLError(ConnectionRefusedError()),
            ProviderUnavailableKind.CONNECT_UNAVAILABLE,
        ),
        (urllib.error.URLError(ssl.SSLError()), ProviderUnavailableKind.TLS_UNAVAILABLE),
        (provider_module._RedirectRefused(), ProviderUnavailableKind.REDIRECT_REFUSED),
    ),
)
def test_default_open_known_unavailable_keeps_kind_and_has_open_stage(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    kind: ProviderUnavailableKind,
) -> None:
    opener = _StageOpener(open_error=failure)
    monkeypatch.setattr(provider_module.urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(
        caught.value,
        ProviderUnavailableStage.OPENER_OPEN,
        kind=kind,
    )


def test_default_open_exact_timeout_keeps_provider_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _StageOpener(open_error=TimeoutError("PRIVATE_TIMEOUT_SENTINEL"))
    monkeypatch.setattr(provider_module.urllib.request, "build_opener", lambda *args: opener)

    with pytest.raises(ProviderTimeout, match=r"^provider timeout$"):
        make_default_provider().complete(sample_messages())


def test_injected_transport_cannot_claim_default_internal_stage() -> None:
    marker = provider_module._DefaultTransportFailure(
        provider_module._AttemptStatus.UNAVAILABLE,
        ProviderUnavailableKind.HTTP_5XX,
        ProviderUnavailableStage.OPENER_OPEN,
    )
    transport = RaisingTransport(lambda: marker)

    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.TRANSPORT_CALL)
    assert transport.call_count == 1


def test_rebound_default_name_cannot_let_injected_transport_claim_internal_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = provider_module._DefaultTransportFailure(
        provider_module._AttemptStatus.UNAVAILABLE,
        ProviderUnavailableKind.HTTP_5XX,
        ProviderUnavailableStage.OPENER_OPEN,
    )
    transport = RaisingTransport(lambda: marker)
    monkeypatch.setattr(provider_module, "urllib_transport", transport)

    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.TRANSPORT_CALL)
    assert transport.call_count == 1


def test_original_default_transport_keeps_internal_identity_after_name_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_default = provider_module.urllib_transport
    provider = make_provider(original_default)
    monkeypatch.setattr(
        provider_module.urllib.request,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(OSError("PRIVATE_OPENER_SENTINEL")),
    )
    monkeypatch.setattr(provider_module, "urllib_transport", RecordingTransport({}))

    with pytest.raises(OpenAIProviderUnavailable) as caught:
        provider.complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.OPENER_CONSTRUCTION)


@pytest.mark.parametrize("boundary", tuple(ProviderUnavailableStage))
def test_each_stage_releases_hostile_exception_without_interaction_or_exposure(
    boundary: ProviderUnavailableStage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactions: list[str] = []

    class HostileFailure(OSError):
        def __getattribute__(self, name: str) -> object:
            if name == "args":
                interactions.append("args")
                print("PRIVATE_STAGE_ARGS_SENTINEL")
                raise RuntimeError("PRIVATE_STAGE_ARGS_SENTINEL")
            return super().__getattribute__(name)

        def __setattr__(self, name: str, value: object) -> None:
            interactions.append(f"set:{name}")
            print("PRIVATE_STAGE_SETATTR_SENTINEL")
            raise RuntimeError("PRIVATE_STAGE_SETATTR_SENTINEL")

        def __str__(self) -> str:
            interactions.append("str")
            print("PRIVATE_STAGE_STRING_SENTINEL")
            raise RuntimeError("PRIVATE_STAGE_STRING_SENTINEL")

        def __repr__(self) -> str:
            interactions.append("repr")
            print("PRIVATE_STAGE_REPR_SENTINEL")
            raise RuntimeError("PRIVATE_STAGE_REPR_SENTINEL")

    failure = OSError.__new__(HostileFailure)
    BaseException.__init__(failure)
    transport: Transport = RecordingTransport({})

    if boundary is ProviderUnavailableStage.REQUEST_SERIALIZATION:
        monkeypatch.setattr(
            PromptMessage,
            "model_dump",
            lambda self: (_ for _ in ()).throw(failure),
        )
    elif boundary is ProviderUnavailableStage.REQUEST_CONSTRUCTION:
        monkeypatch.setattr(
            provider_module.urllib.request,
            "Request",
            lambda *args, **kwargs: (_ for _ in ()).throw(failure),
        )
    elif boundary is ProviderUnavailableStage.OPENER_CONSTRUCTION:
        monkeypatch.setattr(
            provider_module.urllib.request,
            "build_opener",
            lambda *args: (_ for _ in ()).throw(failure),
        )
        transport = provider_module.urllib_transport
    elif boundary is ProviderUnavailableStage.OPENER_OPEN:
        monkeypatch.setattr(
            provider_module.urllib.request,
            "build_opener",
            lambda *args: _StageOpener(open_error=failure),
        )
        transport = provider_module.urllib_transport
    elif boundary is ProviderUnavailableStage.RESPONSE_READ:
        monkeypatch.setattr(
            provider_module.urllib.request,
            "build_opener",
            lambda *args: _StageOpener(response=_StageResponse(read_error=failure)),
        )
        transport = provider_module.urllib_transport
    elif boundary is ProviderUnavailableStage.RESPONSE_CLOSE:
        monkeypatch.setattr(
            provider_module.urllib.request,
            "build_opener",
            lambda *args: _StageOpener(
                response=_StageResponse(
                    close_error=failure,
                    response=b"PRIVATE_RESPONSE_RAW_SENTINEL",
                )
            ),
        )
        transport = provider_module.urllib_transport
    else:
        transport = RaisingTransport(lambda: failure)

    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    _assert_unavailable_stage(caught.value, boundary)
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=(
            "PRIVATE_STAGE_",
            "PRIVATE_RESPONSE_RAW_SENTINEL",
            "synthetic-api-key",
            "synthetic system",
            "synthetic request",
            "https://jojocode.com/v1",
            "Authorization",
        ),
    )
    assert interactions == []
    captured = capsys.readouterr()
    assert "PRIVATE_STAGE_" not in captured.out + captured.err + caplog.text


def test_default_marker_subclass_cannot_expose_internal_stage_or_hooks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactions: list[str] = []

    class HostileMarker(provider_module._DefaultTransportFailure):
        def __getattribute__(self, name: str) -> object:
            if name in {"status", "kind", "stage"}:
                interactions.append(name)
                print("PRIVATE_MARKER_DESCRIPTOR_SENTINEL")
                raise RuntimeError("PRIVATE_MARKER_DESCRIPTOR_SENTINEL")
            return super().__getattribute__(name)

    marker = RuntimeError.__new__(HostileMarker)
    RuntimeError.__init__(marker)
    monkeypatch.setattr(
        provider_module,
        "_raise_default_transport_failure",
        lambda *args: (_ for _ in ()).throw(marker),
    )
    monkeypatch.setattr(
        provider_module.urllib.request,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(OSError()),
    )

    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_default_provider().complete(sample_messages())

    _assert_unavailable_stage(caught.value, ProviderUnavailableStage.TRANSPORT_CALL)
    assert interactions == []
    captured = capsys.readouterr()
    assert "PRIVATE_MARKER_" not in captured.out + captured.err


@pytest.mark.parametrize("code", (True, type("IntSubclass", (int,), {})(401)))
def test_untrusted_http_status_types_fail_closed(code: object) -> None:
    failure = urllib.error.HTTPError("https://public.invalid", 500, "x", {}, None)
    failure.code = code  # type: ignore[assignment]
    transport = RaisingTransport(lambda: failure)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())
    assert caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL


def test_url_error_reason_subclass_fails_closed() -> None:
    class GaiSubclass(socket.gaierror):
        pass

    transport = RaisingTransport(lambda: urllib.error.URLError(GaiSubclass()))
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())
    assert caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL


def test_provider_sends_one_minimal_non_streaming_request() -> None:
    transport = RecordingTransport(
        {
            "choices": [{"message": {"content": "synthetic safe answer"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_cache_hit_tokens": 2,
                "prompt_cache_miss_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    )

    result = make_provider(transport).complete(sample_messages())

    assert result.text == "synthetic safe answer"
    assert result.usage == ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    assert transport.call_count == 1
    assert transport.method == "POST"
    assert transport.url == "https://jojocode.com/v1/chat/completions"
    assert transport.headers["Content-type"] == "application/json"
    assert transport.headers["Authorization"] == "Bearer synthetic-api-key"
    assert transport.request_payload == {
        "model": "gpt-5.6-terra",
        "messages": [
            {"role": "system", "content": "synthetic system"},
            {"role": "user", "content": "synthetic request"},
        ],
        "max_tokens": 256,
    }


@pytest.mark.parametrize(
    ("response", "response_bytes"),
    [
        (None, b"synthetic-invalid-response-sentinel"),
        ({}, None),
        ({"choices": []}, None),
        ({"choices": [{"message": {"content": "   "}}]}, None),
        ({"choices": [{"message": {"content": 123}}]}, None),
        (
            {
                "choices": [{"message": {"content": "synthetic"}}],
                "usage": {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
            },
            None,
        ),
        (
            {
                "choices": [{"message": {"content": "synthetic"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
            },
            None,
        ),
        ({"choices": [{"message": {"content": "x" * 4_001}}]}, None),
    ],
)
def test_invalid_provider_response_is_fixed_safe_error_without_traceback_material(
    response: object, response_bytes: bytes | None
) -> None:
    if response_bytes is None:
        transport: Transport = RecordingTransport(response)
    else:
        transport = _BytesTransport(response_bytes)

    with pytest.raises(InvalidProviderResponse, match=r"^invalid provider response$") as caught:
        make_provider(transport).complete(sample_messages())

    assert _call_count(transport) == 1
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=(
            "synthetic-api-key",
            "synthetic system",
            "synthetic request",
            "synthetic-invalid-response-sentinel",
            "Authorization",
            "https://jojocode.com/v1",
        ),
    )


def test_timeout_is_fixed_safe_error_without_retry_or_secret() -> None:
    transport = RaisingTransport(lambda: TimeoutError("synthetic-timeout-sentinel"))

    with pytest.raises(ProviderTimeout, match=r"^provider timeout$") as caught:
        make_provider(transport).complete(sample_messages())

    assert transport.call_count == 1
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=(
            "synthetic-api-key",
            "synthetic system",
            "synthetic request",
            "synthetic-timeout-sentinel",
            "Authorization",
            "https://jojocode.com/v1",
        ),
    )


def test_transport_error_is_fixed_safe_error_without_traceback_material() -> None:
    transport = RaisingTransport(lambda: OSError("synthetic-transport-sentinel"))

    with pytest.raises(ProviderError, match=r"^provider unavailable$") as caught:
        make_provider(transport).complete(sample_messages())

    assert caught.value.__cause__ is None
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL
    assert transport.call_count == 1
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=(
            "synthetic-api-key",
            "synthetic system",
            "synthetic request",
            "synthetic-transport-sentinel",
            "Authorization",
            "https://jojocode.com/v1",
        ),
    )


@pytest.mark.parametrize(
    ("failure", "kind"),
    (
        (
            urllib.error.HTTPError("https://public.invalid", 400, "x", {}, None),
            ProviderUnavailableKind.HTTP_400_BAD_REQUEST,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 401, "x", {}, None),
            ProviderUnavailableKind.HTTP_401_UNAUTHORIZED,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 403, "x", {}, None),
            ProviderUnavailableKind.HTTP_403_FORBIDDEN,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 404, "x", {}, None),
            ProviderUnavailableKind.HTTP_404_NOT_FOUND,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 408, "x", {}, None),
            ProviderUnavailableKind.HTTP_408_TIMEOUT,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 429, "x", {}, None),
            ProviderUnavailableKind.HTTP_429_RATE_LIMITED,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 422, "x", {}, None),
            ProviderUnavailableKind.HTTP_4XX_OTHER,
        ),
        (
            urllib.error.HTTPError("https://public.invalid", 503, "x", {}, None),
            ProviderUnavailableKind.HTTP_5XX,
        ),
        (urllib.error.URLError(socket.gaierror()), ProviderUnavailableKind.DNS_UNAVAILABLE),
        (
            urllib.error.URLError(ConnectionRefusedError()),
            ProviderUnavailableKind.CONNECT_UNAVAILABLE,
        ),
        (urllib.error.URLError(ssl.SSLError()), ProviderUnavailableKind.TLS_UNAVAILABLE),
        (OSError(), ProviderUnavailableKind.TRANSPORT_UNKNOWN),
    ),
)
def test_transport_failures_have_fixed_safe_kinds(
    failure: Exception, kind: ProviderUnavailableKind
) -> None:
    transport = RaisingTransport(lambda: failure)
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())
    assert caught.value.kind is kind
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL
    assert str(caught.value) == "provider unavailable"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert transport.call_count == 1
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=("synthetic-api-key", "https://public.invalid", "Authorization"),
    )


def test_missing_http_code_and_unknown_url_reason_fail_closed() -> None:
    http_failure = urllib.error.HTTPError("https://public.invalid", 500, "x", {}, None)
    del http_failure.code
    unknown_reason = urllib.error.URLError(object())
    for failure in (http_failure, unknown_reason):
        transport = RaisingTransport(lambda failure=failure: failure)
        with pytest.raises(OpenAIProviderUnavailable) as caught:
            make_provider(transport).complete(sample_messages())
        assert caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
        assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL


def test_hostile_transport_exceptions_are_not_interacted_with_or_exposed(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    interactions: list[str] = []

    class HostileUrlError(urllib.error.URLError):
        @property
        def reason(self) -> object:
            interactions.append("reason")
            print("PRIVATE_DESCRIPTOR_SENTINEL")
            raise RuntimeError("PRIVATE_DESCRIPTOR_SENTINEL")

        def __str__(self) -> str:
            interactions.append("str")
            raise RuntimeError("PRIVATE_STRING_SENTINEL")

        def __repr__(self) -> str:
            interactions.append("repr")
            raise RuntimeError("PRIVATE_REPR_SENTINEL")

    transport = RaisingTransport(lambda: HostileUrlError("PRIVATE_BODY_SENTINEL"))
    with pytest.raises(OpenAIProviderUnavailable) as caught:
        make_provider(transport).complete(sample_messages())

    assert caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL
    assert interactions == []
    captured = capsys.readouterr()
    assert "PRIVATE_" not in captured.out + captured.err + caplog.text
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=("PRIVATE_", "synthetic-api-key", "https://jojocode.com/v1"),
    )

    class HostileHttpError(urllib.error.HTTPError):
        @property
        def code(self) -> object:
            interactions.append("code")
            print("PRIVATE_HTTP_CODE_SENTINEL")
            raise RuntimeError("PRIVATE_HTTP_CODE_SENTINEL")

    hostile_http = urllib.error.HTTPError.__new__(HostileHttpError)
    BaseException.__init__(hostile_http)
    http_transport = RaisingTransport(lambda: hostile_http)
    with pytest.raises(OpenAIProviderUnavailable) as http_caught:
        make_provider(http_transport).complete(sample_messages())
    assert http_caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
    assert http_caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL
    assert interactions == []


def test_timeout_subclass_fails_closed_without_interaction_or_exposure(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    interactions: list[str] = []

    class HostileTimeout(TimeoutError):
        def __str__(self) -> str:
            interactions.append("str")
            print("PRIVATE_TIMEOUT_STRING_SENTINEL")
            raise RuntimeError("PRIVATE_TIMEOUT_STRING_SENTINEL")

        def __repr__(self) -> str:
            interactions.append("repr")
            print("PRIVATE_TIMEOUT_REPR_SENTINEL")
            raise RuntimeError("PRIVATE_TIMEOUT_REPR_SENTINEL")

    transport = RaisingTransport(lambda: HostileTimeout("PRIVATE_TIMEOUT_BODY_SENTINEL"))
    with pytest.raises(OpenAIProviderUnavailable, match=r"^provider unavailable$") as caught:
        make_provider(transport).complete(sample_messages())

    assert caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=("PRIVATE_TIMEOUT_", "synthetic-api-key", "https://jojocode.com/v1"),
    )
    assert interactions == []
    captured = capsys.readouterr()
    assert "PRIVATE_TIMEOUT_" not in captured.out + captured.err + caplog.text


def test_redirect_subclass_fails_closed_without_interaction_or_exposure(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    interactions: list[str] = []

    class HostileRedirect(provider_module._RedirectRefused):
        def __str__(self) -> str:
            interactions.append("str")
            print("PRIVATE_REDIRECT_STRING_SENTINEL")
            raise RuntimeError("PRIVATE_REDIRECT_STRING_SENTINEL")

        def __repr__(self) -> str:
            interactions.append("repr")
            print("PRIVATE_REDIRECT_REPR_SENTINEL")
            raise RuntimeError("PRIVATE_REDIRECT_REPR_SENTINEL")

    transport = RaisingTransport(lambda: HostileRedirect("PRIVATE_REDIRECT_BODY_SENTINEL"))
    with pytest.raises(OpenAIProviderUnavailable, match=r"^provider unavailable$") as caught:
        make_provider(transport).complete(sample_messages())

    assert caught.value.kind is ProviderUnavailableKind.TRANSPORT_UNKNOWN
    assert caught.value.stage is ProviderUnavailableStage.TRANSPORT_CALL
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=("PRIVATE_REDIRECT_", "synthetic-api-key", "https://jojocode.com/v1"),
    )
    assert interactions == []
    captured = capsys.readouterr()
    assert "PRIVATE_REDIRECT_" not in captured.out + captured.err + caplog.text


def test_redirect_refusal_does_not_interact_with_untrusted_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactions: list[str] = []

    class HostileResponse:
        def __getattribute__(self, name: str) -> object:
            interactions.append(name)
            print("PRIVATE_REDIRECT_SENTINEL")
            raise RuntimeError("PRIVATE_REDIRECT_SENTINEL")

    handler = provider_module._NoRedirectHandler()
    with pytest.raises(provider_module._RedirectRefused):
        handler._reject_redirect(None, HostileResponse(), 302, "x", {})  # type: ignore[arg-type]

    captured = capsys.readouterr()
    assert interactions == []
    assert "PRIVATE_REDIRECT_SENTINEL" not in captured.out + captured.err


def test_redirect_refusal_does_not_close_hostile_exact_http_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    interactions: list[str] = []

    class HostileFile:
        def flush(self) -> None:
            interactions.append("flush")
            print("PRIVATE_HTTP_RESPONSE_SENTINEL")

        def close(self) -> None:
            interactions.append("close")
            print("PRIVATE_HTTP_RESPONSE_SENTINEL")

    response = object.__new__(http.client.HTTPResponse)
    object.__setattr__(response, "fp", HostileFile())
    handler = provider_module._NoRedirectHandler()
    with pytest.raises(provider_module._RedirectRefused):
        handler._reject_redirect(None, response, 302, "x", {})  # type: ignore[arg-type]

    captured = capsys.readouterr()
    assert interactions == []
    assert "PRIVATE_HTTP_RESPONSE_SENTINEL" not in captured.out + captured.err


def test_redirect_refusal_best_effort_closes_trusted_standard_library_socket() -> None:
    local, peer = socket.socketpair()
    response = http.client.HTTPResponse(local)
    try:
        handler = provider_module._NoRedirectHandler()
        with pytest.raises(provider_module._RedirectRefused):
            handler._reject_redirect(None, response, 302, "x", {})  # type: ignore[arg-type]
        assert local.fileno() == -1
    finally:
        local.close()
        peer.close()


def test_redirect_refusal_best_effort_closes_trusted_tls_socket() -> None:
    local, peer = socket.socketpair()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    tls = context.wrap_socket(
        local,
        server_hostname="public.invalid",
        do_handshake_on_connect=False,
    )
    response = http.client.HTTPResponse(tls)
    try:
        handler = provider_module._NoRedirectHandler()
        with pytest.raises(provider_module._RedirectRefused):
            handler._reject_redirect(None, response, 302, "x", {})  # type: ignore[arg-type]
        assert tls.fileno() == -1
    finally:
        tls.close()
        peer.close()


class _RedirectTargetHandler(BaseHTTPRequestHandler):
    request_count = 0
    authorization_values: ClassVar[list[str | None]] = []

    def do_GET(self) -> None:
        self._record_request()

    def do_POST(self) -> None:
        self._record_request()

    def _record_request(self) -> None:
        type(self).request_count += 1
        type(self).authorization_values.append(self.headers.get("Authorization"))
        payload = json.dumps(
            {"choices": [{"message": {"content": "synthetic redirected answer"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _redirect_handler(location: str) -> type[BaseHTTPRequestHandler]:
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    return RedirectHandler


def _same_origin_handler() -> type[BaseHTTPRequestHandler]:
    class SameOriginHandler(_RedirectTargetHandler):
        def do_POST(self) -> None:
            if self.path == "/v1/chat/completions":
                self.send_response(302)
                self.send_header("Location", "/redirect-target")
                self.end_headers()
                return
            self._record_request()

    return SameOriginHandler


@contextmanager
def _serve(
    handler: type[BaseHTTPRequestHandler],
) -> Iterator[tuple[str, type[BaseHTTPRequestHandler]]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", handler
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _real_transport_provider(base_url: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        config=ProviderConfig(
            base_url=f"{base_url}/v1",
            model="gpt-5.6-terra",
            timeout_seconds=5,
        ),
        api_key=SecretStr("synthetic-redirect-api-key"),
        max_output_tokens=256,
    )


def test_default_transport_rejects_cross_origin_redirect_without_contacting_target() -> None:
    _RedirectTargetHandler.request_count = 0
    _RedirectTargetHandler.authorization_values = []
    with (
        _serve(_RedirectTargetHandler) as (target_root, _),
        _serve(_redirect_handler(f"{target_root}/redirect-target")) as (source_root, _),
        pytest.raises(OpenAIProviderUnavailable, match=r"^provider unavailable$") as caught,
    ):
        _real_transport_provider(source_root).complete(sample_messages())

    assert _RedirectTargetHandler.request_count == 0
    assert _RedirectTargetHandler.authorization_values == []
    assert caught.value.kind is ProviderUnavailableKind.REDIRECT_REFUSED
    assert caught.value.stage is ProviderUnavailableStage.OPENER_OPEN
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=(
            "synthetic-redirect-api-key",
            target_root,
            source_root,
            "Authorization",
        ),
    )


def test_default_transport_rejects_same_origin_redirect_without_contacting_target() -> None:
    handler = _same_origin_handler()
    handler.request_count = 0
    handler.authorization_values = []
    with (
        _serve(handler) as (server_root, _),
        pytest.raises(OpenAIProviderUnavailable, match=r"^provider unavailable$") as caught,
    ):
        _real_transport_provider(server_root).complete(sample_messages())

    assert handler.request_count == 0
    assert handler.authorization_values == []
    assert caught.value.kind is ProviderUnavailableKind.REDIRECT_REFUSED
    assert caught.value.stage is ProviderUnavailableStage.OPENER_OPEN
    _assert_sanitized_provider_traceback(
        caught.value,
        forbidden_values=(
            "synthetic-redirect-api-key",
            server_root,
            "Authorization",
        ),
    )


class _BytesTransport:
    def __init__(self, response: bytes) -> None:
        self._response = response
        self.call_count = 0

    def __call__(self, request: urllib.request.Request, timeout: int) -> bytes:
        self.call_count += 1
        return self._response


def _call_count(transport: Transport) -> int:
    return transport.call_count  # type: ignore[attr-defined]


def _assert_sanitized_provider_traceback(
    error: Exception, forbidden_values: tuple[str, ...]
) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(value not in str(error) for value in forbidden_values)
    for frame, _ in traceback.walk_tb(error.__traceback__):
        filename = frame.f_code.co_filename.replace("/", "\\")
        if filename.endswith("src\\companion\\dialogue\\provider.py"):
            assert not any(
                isinstance(value, (ProviderConfig, PromptMessage, SecretStr, _REQUEST_TYPE))
                for value in frame.f_locals.values()
            )
            assert all(
                forbidden not in repr(value)
                for value in frame.f_locals.values()
                for forbidden in forbidden_values
            )
