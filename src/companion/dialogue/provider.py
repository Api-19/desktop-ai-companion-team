from __future__ import annotations

import http.client
import io
import json
import socket
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import suppress
from enum import StrEnum
from typing import Protocol

from pydantic import SecretStr

from companion.dialogue.context import PromptMessage
from companion.dialogue.models import ProviderConfig, ProviderResult, ProviderUsage

_MAX_RESPONSE_BYTES = 1024 * 1024


class ProviderError(RuntimeError):
    """Raised when the configured provider cannot safely complete a request."""


class ProviderTimeout(ProviderError):
    """Raised when the provider does not answer before the configured timeout."""


class InvalidProviderResponse(ProviderError):
    """Raised when a provider response cannot be safely consumed."""


class ProviderUnavailableKind(StrEnum):
    DNS_UNAVAILABLE = "dns_unavailable"
    CONNECT_UNAVAILABLE = "connect_unavailable"
    TLS_UNAVAILABLE = "tls_unavailable"
    REDIRECT_REFUSED = "redirect_refused"
    HTTP_400_BAD_REQUEST = "http_400_bad_request"
    HTTP_401_UNAUTHORIZED = "http_401_unauthorized"
    HTTP_403_FORBIDDEN = "http_403_forbidden"
    HTTP_404_NOT_FOUND = "http_404_not_found"
    HTTP_408_TIMEOUT = "http_408_timeout"
    HTTP_429_RATE_LIMITED = "http_429_rate_limited"
    HTTP_4XX_OTHER = "http_4xx_other"
    HTTP_5XX = "http_5xx"
    TRANSPORT_UNKNOWN = "transport_unknown"


class ProviderUnavailableStage(StrEnum):
    REQUEST_SERIALIZATION = "request_serialization"
    REQUEST_CONSTRUCTION = "request_construction"
    OPENER_CONSTRUCTION = "opener_construction"
    OPENER_OPEN = "opener_open"
    RESPONSE_READ = "response_read"
    RESPONSE_CLOSE = "response_close"
    TRANSPORT_CALL = "transport_call"


class OpenAIProviderUnavailable(ProviderError):
    def __init__(
        self,
        kind: ProviderUnavailableKind,
        stage: ProviderUnavailableStage,
    ) -> None:
        if type(self) is not OpenAIProviderUnavailable:
            raise TypeError("invalid provider unavailable type")
        if type(kind) is not ProviderUnavailableKind:
            raise TypeError("invalid provider unavailable kind")
        if type(stage) is not ProviderUnavailableStage:
            raise TypeError("invalid provider unavailable stage")
        super().__init__("provider unavailable")
        self.kind = kind
        self.stage = stage


def _raise_openai_provider_unavailable(
    kind: ProviderUnavailableKind,
    stage: ProviderUnavailableStage,
) -> None:
    raise OpenAIProviderUnavailable(kind, stage) from None


class Transport(Protocol):
    def __call__(self, request: urllib.request.Request, timeout: int) -> bytes: ...


class DialogueProvider(Protocol):
    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult: ...


class _AttemptStatus(StrEnum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"


class _DefaultTransportFailure(RuntimeError):
    def __init__(
        self,
        status: _AttemptStatus,
        kind: ProviderUnavailableKind | None,
        stage: ProviderUnavailableStage,
    ) -> None:
        if type(status) is not _AttemptStatus:
            raise TypeError("invalid default transport status")
        if status is _AttemptStatus.TIMEOUT:
            if kind is not None:
                raise TypeError("invalid default transport kind")
        elif status is _AttemptStatus.UNAVAILABLE:
            if type(kind) is not ProviderUnavailableKind:
                raise TypeError("invalid default transport kind")
        else:
            raise TypeError("invalid default transport status")
        if type(stage) is not ProviderUnavailableStage:
            raise TypeError("invalid default transport stage")
        super().__init__("default transport failed")
        self.status = status
        self.kind = kind
        self.stage = stage


def _raise_default_transport_failure(
    status: _AttemptStatus,
    kind: ProviderUnavailableKind | None,
    stage: ProviderUnavailableStage,
) -> None:
    raise _DefaultTransportFailure(status, kind, stage) from None


class _RedirectRefused(RuntimeError):
    pass


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def _reject_redirect(
        self,
        request: urllib.request.Request,
        response: object,
        code: int,
        message: str,
        headers: object,
    ) -> None:
        del request, code, message, headers
        _best_effort_close_redirect_response(response)
        del response
        raise _RedirectRefused from None

    http_error_301 = _reject_redirect
    http_error_302 = _reject_redirect
    http_error_303 = _reject_redirect
    http_error_307 = _reject_redirect
    http_error_308 = _reject_redirect


def _best_effort_close_redirect_response(response: object) -> None:
    if type(response) is not http.client.HTTPResponse:
        return
    try:
        fp = object.__getattribute__(response, "fp")
        if type(fp) is io.BufferedReader:
            raw = object.__getattribute__(fp, "raw")
        elif type(fp) is socket.SocketIO:
            raw = fp
        else:
            return
        if type(raw) is not socket.SocketIO:
            return
        sock = object.__getattribute__(raw, "_sock")
        sock_type = type(sock)
        if sock_type not in (socket.socket, ssl.SSLSocket):
            return
    except BaseException:
        return
    with suppress(BaseException):
        if sock_type is socket.socket:
            socket.socket.close(sock)
        else:
            ssl.SSLSocket.close(sock)
    with suppress(BaseException):
        object.__setattr__(response, "fp", None)


def urllib_transport(request: urllib.request.Request, timeout: int) -> bytes:
    opener: urllib.request.OpenerDirector | None = None
    response: object | None = None
    raw = b""
    failure_status = _AttemptStatus.UNAVAILABLE
    failure_kind: ProviderUnavailableKind | None = ProviderUnavailableKind.TRANSPORT_UNKNOWN
    failed = False

    try:
        opener = urllib.request.build_opener(_NoRedirectHandler())
    except Exception as error:
        failure_status, failure_kind = _classify_transport_exception(error)
        error = None  # type: ignore[assignment]
        failed = True
    if failed:
        del request, opener, response, raw
        _raise_default_transport_failure(
            failure_status,
            failure_kind,
            ProviderUnavailableStage.OPENER_CONSTRUCTION,
        )

    try:
        response = opener.open(request, timeout=timeout)
    except Exception as error:
        failure_status, failure_kind = _classify_transport_exception(error)
        error = None  # type: ignore[assignment]
        failed = True
    if failed:
        del request, opener, response, raw
        _raise_default_transport_failure(
            failure_status,
            failure_kind,
            ProviderUnavailableStage.OPENER_OPEN,
        )

    read_failed = False
    try:
        raw = response.read(_MAX_RESPONSE_BYTES + 1)  # type: ignore[union-attr]
    except Exception as error:
        failure_status, failure_kind = _classify_transport_exception(error)
        error = None  # type: ignore[assignment]
        read_failed = True

    close_failed = False
    close_status = _AttemptStatus.UNAVAILABLE
    close_kind: ProviderUnavailableKind | None = ProviderUnavailableKind.TRANSPORT_UNKNOWN
    try:
        response.close()  # type: ignore[union-attr]
    except Exception as error:
        close_status, close_kind = _classify_transport_exception(error)
        error = None  # type: ignore[assignment]
        close_failed = True

    del request, opener, response
    if close_failed:
        del raw
        _raise_default_transport_failure(
            close_status,
            close_kind,
            ProviderUnavailableStage.RESPONSE_CLOSE,
        )
    if read_failed:
        del raw
        _raise_default_transport_failure(
            failure_status,
            failure_kind,
            ProviderUnavailableStage.RESPONSE_READ,
        )
    return raw


def _capture_default_transport_identity(
    default_transport: Transport,
) -> Callable[[Transport], bool]:
    def is_default_transport(candidate: Transport) -> bool:
        return candidate is default_transport

    return is_default_transport


_is_default_transport = _capture_default_transport_identity(urllib_transport)


class OpenAICompatibleProvider:
    """Minimal one-shot adapter for a non-streaming Chat Completions endpoint."""

    def __init__(
        self,
        *,
        config: ProviderConfig,
        api_key: SecretStr,
        max_output_tokens: int,
        transport: Transport = urllib_transport,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens
        self._transport = transport
        self._transport_is_default = _is_default_transport(transport)

    def complete(self, messages: Sequence[PromptMessage]) -> ProviderResult:
        """Make exactly one request and expose only fixed public failures."""
        result: ProviderResult | None = None
        status = _AttemptStatus.UNAVAILABLE
        kind = ProviderUnavailableKind.TRANSPORT_UNKNOWN
        stage = ProviderUnavailableStage.TRANSPORT_CALL
        try:
            status, result, returned_kind, returned_stage = _complete_once(
                config=self._config,
                api_key=self._api_key.get_secret_value(),
                max_output_tokens=self._max_output_tokens,
                transport=self._transport,
                transport_is_default=self._transport_is_default,
                messages=messages,
            )
            if type(returned_kind) is ProviderUnavailableKind:
                kind = returned_kind
            if type(returned_stage) is ProviderUnavailableStage:
                stage = returned_stage
        except Exception:
            status = _AttemptStatus.UNAVAILABLE
            kind = ProviderUnavailableKind.TRANSPORT_UNKNOWN
            stage = ProviderUnavailableStage.TRANSPORT_CALL

        if status is _AttemptStatus.SUCCESS:
            if result is None:
                status = _AttemptStatus.INVALID_RESPONSE
            else:
                return result

        del self, messages, result
        if status is _AttemptStatus.TIMEOUT:
            raise ProviderTimeout("provider timeout")
        if status is _AttemptStatus.INVALID_RESPONSE:
            raise InvalidProviderResponse("invalid provider response")
        _raise_openai_provider_unavailable(kind, stage)


def _complete_once(
    *,
    config: ProviderConfig,
    api_key: str,
    max_output_tokens: int,
    transport: Transport,
    transport_is_default: bool,
    messages: Sequence[PromptMessage],
) -> tuple[
    _AttemptStatus,
    ProviderResult | None,
    ProviderUnavailableKind | None,
    ProviderUnavailableStage | None,
]:
    body = b""
    headers: dict[str, str] = {}
    request: urllib.request.Request | None = None
    raw = b""
    transport_status = _AttemptStatus.SUCCESS
    unavailable_kind: ProviderUnavailableKind | None = None
    unavailable_stage: ProviderUnavailableStage | None = None

    serialization_failed = False
    try:
        body = json.dumps(
            {
                "model": config.model,
                "messages": [message.model_dump() for message in messages],
                "max_tokens": max_output_tokens,
                **(
                    {"thinking": {"type": "disabled"}}
                    if config.base_url.scheme == "https"
                    and config.base_url.host == "api.deepseek.com"
                    and config.base_url.port == 443
                    and config.base_url.username is None
                    and config.base_url.password is None
                    else {}
                ),
            }
        ).encode("utf-8")
    except Exception:
        serialization_failed = True
    if serialization_failed:
        del config, api_key, max_output_tokens, transport, messages, body, headers, request, raw
        return (
            _AttemptStatus.UNAVAILABLE,
            None,
            ProviderUnavailableKind.TRANSPORT_UNKNOWN,
            ProviderUnavailableStage.REQUEST_SERIALIZATION,
        )

    construction_failed = False
    endpoint = ""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{str(config.base_url).rstrip('/')}/chat/completions"
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    except Exception:
        construction_failed = True
    if construction_failed:
        del config, api_key, max_output_tokens, transport, messages, body, headers, request, raw
        del endpoint
        return (
            _AttemptStatus.UNAVAILABLE,
            None,
            ProviderUnavailableKind.TRANSPORT_UNKNOWN,
            ProviderUnavailableStage.REQUEST_CONSTRUCTION,
        )

    try:
        raw = transport(request, config.timeout_seconds)
    except _DefaultTransportFailure as error:
        if transport_is_default is True and type(error) is _DefaultTransportFailure:
            returned_status = object.__getattribute__(error, "status")
            returned_kind = object.__getattribute__(error, "kind")
            returned_stage = object.__getattribute__(error, "stage")
            if (
                type(returned_status) is _AttemptStatus
                and returned_status in (_AttemptStatus.TIMEOUT, _AttemptStatus.UNAVAILABLE)
                and type(returned_stage) is ProviderUnavailableStage
                and (
                    (returned_status is _AttemptStatus.TIMEOUT and returned_kind is None)
                    or (
                        returned_status is _AttemptStatus.UNAVAILABLE
                        and type(returned_kind) is ProviderUnavailableKind
                    )
                )
            ):
                transport_status = returned_status
                unavailable_kind = returned_kind
                unavailable_stage = returned_stage
            else:
                transport_status = _AttemptStatus.UNAVAILABLE
                unavailable_kind = ProviderUnavailableKind.TRANSPORT_UNKNOWN
                unavailable_stage = ProviderUnavailableStage.TRANSPORT_CALL
        else:
            transport_status = _AttemptStatus.UNAVAILABLE
            unavailable_kind = ProviderUnavailableKind.TRANSPORT_UNKNOWN
            unavailable_stage = ProviderUnavailableStage.TRANSPORT_CALL
        error = None  # type: ignore[assignment]
    except Exception as error:
        transport_status, unavailable_kind = _classify_transport_exception(error)
        unavailable_stage = ProviderUnavailableStage.TRANSPORT_CALL
        error = None  # type: ignore[assignment]
    finally:
        del body, headers, request
        del endpoint

    if transport_status is not _AttemptStatus.SUCCESS:
        del config, api_key, max_output_tokens, transport, messages, raw
        return transport_status, None, unavailable_kind, unavailable_stage

    payload: object | None = None
    try:
        result = _parse_result(raw)
    except Exception:
        status = _AttemptStatus.INVALID_RESPONSE
        result = None
    else:
        status = _AttemptStatus.SUCCESS
    finally:
        del raw, payload
    del config, api_key, max_output_tokens, transport, messages
    return status, result, None, None


def _classify_transport_exception(
    error: Exception,
) -> tuple[_AttemptStatus, ProviderUnavailableKind | None]:
    error_type = type(error)
    if error_type is TimeoutError:
        return _AttemptStatus.TIMEOUT, None
    if error_type is urllib.error.HTTPError:
        return _AttemptStatus.UNAVAILABLE, _classify_http_error(error)
    if error_type is urllib.error.URLError:
        return _classify_url_error(error)
    if error_type is _RedirectRefused:
        return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.REDIRECT_REFUSED
    return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.TRANSPORT_UNKNOWN


def _classify_http_error(error: urllib.error.HTTPError) -> ProviderUnavailableKind:
    if type(error) is not urllib.error.HTTPError:
        return ProviderUnavailableKind.TRANSPORT_UNKNOWN
    try:
        code = object.__getattribute__(error, "code")
    except BaseException:
        return ProviderUnavailableKind.TRANSPORT_UNKNOWN
    if type(code) is not int:
        return ProviderUnavailableKind.TRANSPORT_UNKNOWN
    exact = {
        400: ProviderUnavailableKind.HTTP_400_BAD_REQUEST,
        401: ProviderUnavailableKind.HTTP_401_UNAUTHORIZED,
        403: ProviderUnavailableKind.HTTP_403_FORBIDDEN,
        404: ProviderUnavailableKind.HTTP_404_NOT_FOUND,
        408: ProviderUnavailableKind.HTTP_408_TIMEOUT,
        429: ProviderUnavailableKind.HTTP_429_RATE_LIMITED,
    }
    if code in exact:
        return exact[code]
    if 400 <= code < 500:
        return ProviderUnavailableKind.HTTP_4XX_OTHER
    if 500 <= code < 600:
        return ProviderUnavailableKind.HTTP_5XX
    return ProviderUnavailableKind.TRANSPORT_UNKNOWN


def _classify_url_error(
    error: urllib.error.URLError,
) -> tuple[_AttemptStatus, ProviderUnavailableKind | None]:
    if type(error) is not urllib.error.URLError:
        return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.TRANSPORT_UNKNOWN
    try:
        reason = object.__getattribute__(error, "reason")
    except BaseException:
        return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.TRANSPORT_UNKNOWN
    reason_type = type(reason)
    if reason_type in (TimeoutError, socket.timeout):
        return _AttemptStatus.TIMEOUT, None
    if reason_type is socket.gaierror:
        return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.DNS_UNAVAILABLE
    if reason_type in (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError):
        return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.CONNECT_UNAVAILABLE
    if reason_type in (ssl.SSLError, ssl.SSLCertVerificationError):
        return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.TLS_UNAVAILABLE
    return _AttemptStatus.UNAVAILABLE, ProviderUnavailableKind.TRANSPORT_UNKNOWN


def _parse_result(raw: bytes) -> ProviderResult:
    if type(raw) is not bytes or len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("provider payload exceeds size limit")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("provider payload is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("provider payload has no usable choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("provider payload has no text content")
    text = message["content"].strip()
    usage_payload = payload.get("usage")
    usage = _parse_usage(usage_payload) if usage_payload is not None else None
    return ProviderResult(text=text, usage=usage)


def _parse_usage(usage_payload: object) -> ProviderUsage:
    if not isinstance(usage_payload, dict):
        raise ValueError("provider usage is invalid")
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(
        not isinstance(usage_payload.get(field), int) or isinstance(usage_payload.get(field), bool)
        for field in token_fields
    ):
        raise ValueError("provider usage is invalid")
    return ProviderUsage.model_validate({field: usage_payload[field] for field in token_fields})
