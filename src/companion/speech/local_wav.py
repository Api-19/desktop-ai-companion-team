from __future__ import annotations

import os
import stat
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode

_PATH_TYPE = type(Path())
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_RESOURCE_TOKEN = object()
_OWNED_FILE_IDENTITIES: dict[Path, tuple[int, int]] = {}
_RIFF_HEADER_SIZE = 12
_CHUNK_HEADER_SIZE = 8
_PCM_FORMAT_SIZE = 16


class _CreationCleanupFailed(Exception):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class PcmWavData:
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    samples_int16: tuple[int, ...] = field(repr=False)
    samples_float: tuple[float, ...] = field(repr=False)


class OwnedWavFile:
    __slots__ = ("__path", "__run_root", "__closed")  # noqa: RUF023

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("OwnedWavFile cannot be subclassed")

    def __init__(self, path: Path, run_root: Path, token: object) -> None:
        if token is not _RESOURCE_TOKEN:
            raise TypeError("OwnedWavFile must be created by this module")
        _record_created_file(path)
        self.__path = path
        self.__run_root = run_root
        self.__closed = False

    def _capture_bytes(self) -> bytes:
        if self.__closed or self.__path is None:
            _raise_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
        try:
            payload = _capture_verified_bytes(self.__path, self.__run_root)
        except Exception:
            payload = None
        if payload is None:
            _raise_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
        return payload

    def close(self) -> None:
        if self.__closed:
            return
        path = self.__path
        self.__path = None
        self.__closed = True
        failed = False
        try:
            if path is None:
                raise ValueError("missing owned file")
            _verify_owned_file(path, self.__run_root).unlink()
        except Exception:
            failed = True
        if path is not None:
            _OWNED_FILE_IDENTITIES.pop(path, None)
        if failed:
            path = None
            _raise_error(LocalSpeechAdapterErrorCode.CLEANUP_FAILED)


def _raise_error(code: LocalSpeechAdapterErrorCode) -> NoReturn:
    raise LocalSpeechAdapterError(code)


def _is_link_or_reparse(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _absolute_no_follow(path: Path) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if _is_link_or_reparse(current):
            raise ValueError("link or reparse point")
    return absolute.resolve(strict=True)


def _inside_worktree(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(_WORKTREE_ROOT)
    except ValueError:
        return False
    except OSError:
        return True
    return True


def _verified_run_root(run_root: Path) -> Path:
    if type(run_root) is not _PATH_TYPE or not run_root.is_absolute() or _inside_worktree(run_root):
        raise ValueError("untrusted run root")
    resolved = _absolute_no_follow(run_root)
    if not resolved.is_dir():
        raise ValueError("run root is not a directory")
    return resolved


def _verify_owned_file(path: Path, run_root: Path) -> Path:
    if type(path) is not _PATH_TYPE or type(run_root) is not _PATH_TYPE:
        raise ValueError("untrusted owned path")
    resolved = _absolute_no_follow(path)
    if resolved.parent != run_root or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("owned file escaped its root")
    expected_identity = _OWNED_FILE_IDENTITIES.get(path)
    if expected_identity is not None and _file_identity(resolved) != expected_identity:
        raise ValueError("owned file was replaced")
    return resolved


def _file_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _record_created_file(path: Path, identity: tuple[int, int] | None = None) -> None:
    _OWNED_FILE_IDENTITIES[path] = _file_identity(path) if identity is None else identity


def _capture_verified_bytes(path: Path, run_root: Path) -> bytes:
    verified = _verify_owned_file(path, run_root)
    expected_identity = _OWNED_FILE_IDENTITIES.get(path)
    if expected_identity is None:
        raise ValueError("missing owned identity")
    with verified.open("rb") as owned_file:
        if _stat_identity(os.fstat(owned_file.fileno())) != expected_identity:
            raise ValueError("opened file identity changed")
        payload = owned_file.read()
        if _stat_identity(os.fstat(owned_file.fileno())) != expected_identity:
            raise ValueError("opened file identity changed")
    _verify_owned_file(path, run_root)
    return payload


def _stat_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _new_path(run_root: Path, suffix: str) -> Path:
    return run_root / f"speech-{uuid.uuid4().hex}{suffix}"


def _create_exclusive(path: Path, payload: bytes = b"") -> None:
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as file:
        identity: tuple[int, int] | None = None
        try:
            identity = _file_identity(path)
            _record_created_file(path, identity)
        except Exception:
            try:
                file.close()
            except Exception:
                raise _CreationCleanupFailed from None
            if identity is None or not _unlink_matching_identity(path, path.parent, identity):
                raise _CreationCleanupFailed from None
            raise
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _unlink_matching_identity(path: Path, run_root: Path, identity: tuple[int, int]) -> bool:
    try:
        resolved = _absolute_no_follow(path)
        if (
            resolved.parent != run_root
            or not stat.S_ISREG(resolved.stat().st_mode)
            or _file_identity(resolved) != identity
        ):
            return False
        resolved.unlink()
    except Exception:
        return False
    return True


def create_owned_wav(run_root: Path) -> OwnedWavFile:
    created: Path | None = None
    root: Path | None = None
    path: Path | None = None
    verified: Path | None = None
    try:
        root = _verified_run_root(run_root)
        for _ in range(16):
            path = _new_path(root, ".wav")
            created = path
            try:
                _create_exclusive(path)
            except FileExistsError:
                created = None
                continue
            verified = _verify_owned_file(path, root)
            return OwnedWavFile(verified, root, _RESOURCE_TOKEN)
    except Exception as error:
        clean = (
            False
            if type(error) is _CreationCleanupFailed
            else _remove_created(root, (created,))
            if created is not None
            else True
        )
    else:
        clean = True
    run_root = None  # type: ignore[assignment]
    created = None
    root = None
    path = None
    verified = None
    if not clean:
        _raise_error(LocalSpeechAdapterErrorCode.CLEANUP_FAILED)
    _raise_error(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)


def _parse_pcm_wav(payload: bytes, expected_sample_rate_hz: int | None) -> PcmWavData:
    if type(payload) is not bytes:
        raise ValueError("payload must be bytes")
    if expected_sample_rate_hz is not None and type(expected_sample_rate_hz) is not int:
        raise ValueError("invalid expected sample rate")
    if len(payload) < _RIFF_HEADER_SIZE or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("not a WAVE file")
    if struct.unpack_from("<I", payload, 4)[0] != len(payload) - 8:
        raise ValueError("bad RIFF size")

    offset = _RIFF_HEADER_SIZE
    fmt_payload: bytes | None = None
    data_payload: bytes | None = None
    while offset < len(payload):
        if len(payload) - offset < _CHUNK_HEADER_SIZE:
            raise ValueError("truncated chunk header")
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        offset += _CHUNK_HEADER_SIZE
        end = offset + chunk_size
        if end > len(payload):
            raise ValueError("truncated chunk payload")
        chunk_payload = payload[offset:end]
        if chunk_id == b"fmt ":
            if fmt_payload is not None:
                raise ValueError("duplicate format chunk")
            fmt_payload = chunk_payload
        elif chunk_id == b"data":
            if data_payload is not None:
                raise ValueError("duplicate data chunk")
            data_payload = chunk_payload
        offset = end
        if chunk_size & 1:
            if offset >= len(payload) or payload[offset] != 0:
                raise ValueError("invalid chunk padding")
            offset += 1
    if offset != len(payload) or fmt_payload is None or data_payload is None:
        raise ValueError("missing or trailing WAVE data")
    if len(fmt_payload) != _PCM_FORMAT_SIZE:
        raise ValueError("invalid PCM format chunk")

    audio_format, channels, sample_rate_hz, byte_rate, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", fmt_payload
    )
    if (
        audio_format != 1
        or channels != 1
        or sample_rate_hz <= 0
        or bits_per_sample != 16
        or block_align != 2
        or byte_rate != sample_rate_hz * block_align
        or len(data_payload) % block_align
    ):
        raise ValueError("unsupported PCM")
    if expected_sample_rate_hz is not None and sample_rate_hz != expected_sample_rate_hz:
        raise ValueError("unexpected sample rate")

    frame_count = len(data_payload) // block_align
    samples_int16 = struct.unpack(f"<{frame_count}h", data_payload) if frame_count else ()
    return PcmWavData(
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=2,
        frame_count=frame_count,
        samples_int16=samples_int16,
        samples_float=tuple(sample / 32768.0 for sample in samples_int16),
    )


def read_owned_pcm_wav(
    audio: OwnedWavFile, expected_sample_rate_hz: int | None = None
) -> PcmWavData:
    if type(audio) is not OwnedWavFile:
        del audio
        _raise_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
    try:
        payload = audio._capture_bytes()
        pcm = _parse_pcm_wav(payload, expected_sample_rate_hz)
    except LocalSpeechAdapterError:
        raise
    except Exception:
        payload = None
        pcm = None
    if pcm is None:
        _raise_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
    return pcm


def _pcm_wav_payload(samples: tuple[int, ...], sample_rate_hz: int) -> bytes:
    data = struct.pack(f"<{len(samples)}h", *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate_hz, sample_rate_hz * 2, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _valid_write_inputs(samples: object, sample_rate_hz: object) -> tuple[int, ...] | None:
    if type(samples) is not tuple or type(sample_rate_hz) is not int or sample_rate_hz <= 0:
        return None
    if any(type(sample) is not int or not -32768 <= sample <= 32767 for sample in samples):
        return None
    return samples


def _remove_created(run_root: Path | None, paths: tuple[Path | None, ...]) -> bool:
    clean = True
    for path in paths:
        if path is None or path not in _OWNED_FILE_IDENTITIES:
            continue
        try:
            if run_root is None:
                raise ValueError("missing run root")
            _verify_owned_file(path, run_root).unlink()
        except Exception:
            clean = False
        finally:
            _OWNED_FILE_IDENTITIES.pop(path, None)
    return clean


def write_validated_pcm_wav(
    run_root: Path, samples: tuple[int, ...], sample_rate_hz: int
) -> OwnedWavFile:
    valid_samples = _valid_write_inputs(samples, sample_rate_hz)
    if valid_samples is None:
        run_root = None  # type: ignore[assignment]
        samples = ()
        _raise_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
    temporary: Path | None = None
    final: Path | None = None
    root: Path | None = None
    payload: bytes | None = None
    captured: bytes | None = None
    candidate: Path | None = None
    expected_identity: tuple[int, int] | None = None
    verified: Path | None = None
    paths: tuple[Path, ...] = ()
    try:
        root = _verified_run_root(run_root)
        payload = _pcm_wav_payload(valid_samples, sample_rate_hz)
        for _ in range(16):
            candidate = _new_path(root, ".tmp")
            temporary = candidate
            try:
                _create_exclusive(candidate, payload)
            except FileExistsError:
                temporary = None
                continue
            temporary = _verify_owned_file(candidate, root)
            break
        if temporary is None:
            raise ValueError("unable to create temporary file")
        captured = _capture_verified_bytes(temporary, root)
        _parse_pcm_wav(captured, sample_rate_hz)
        final = _new_path(root, ".wav")
        expected_identity = _OWNED_FILE_IDENTITIES[temporary]
        os.link(temporary, final, follow_symlinks=False)
        _OWNED_FILE_IDENTITIES[final] = expected_identity
        if _file_identity(final) != expected_identity:
            raise ValueError("final file identity changed")
        _verify_owned_file(final, root)
        _verify_owned_file(temporary, root).unlink()
        _OWNED_FILE_IDENTITIES.pop(temporary, None)
        temporary = None
        verified = _verify_owned_file(final, root)
        return OwnedWavFile(verified, root, _RESOURCE_TOKEN)
    except Exception as error:
        paths = tuple(path for path in (temporary, final) if path is not None)
        clean = False if type(error) is _CreationCleanupFailed else _remove_created(root, paths)
    run_root = None  # type: ignore[assignment]
    samples = ()
    valid_samples = None
    temporary = None
    final = None
    root = None
    payload = None
    captured = None
    candidate = None
    expected_identity = None
    verified = None
    paths = ()
    if not clean:
        _raise_error(LocalSpeechAdapterErrorCode.CLEANUP_FAILED)
    _raise_error(LocalSpeechAdapterErrorCode.INVALID_AUDIO)
