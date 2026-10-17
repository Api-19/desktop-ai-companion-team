from __future__ import annotations

import os
import stat
import struct
import traceback
from pathlib import Path

import pytest

import companion.speech.local_wav as local_wav
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_wav import (
    OwnedWavFile,
    create_owned_wav,
    read_owned_pcm_wav,
    write_validated_pcm_wav,
)


def _pcm_wav_bytes(samples: tuple[int, ...] = (-32768, 0, 32767)) -> bytes:
    data = struct.pack(f"<{len(samples)}h", *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _owned_with_bytes(tmp_path: Path, payload: bytes):
    audio = create_owned_wav(tmp_path)
    audio._OwnedWavFile__path.write_bytes(payload)  # type: ignore[attr-defined]
    return audio


def _module_frame_locals(error: BaseException, module_name: str) -> str:
    rendered: list[str] = []
    current = error.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == module_name:
            rendered.extend(
                f"{name}={value!r}" for name, value in current.tb_frame.f_locals.items()
            )
        current = current.tb_next
    return "\n".join(rendered)


def test_owned_wav_close_unlinks_once_and_is_idempotent(tmp_path: Path) -> None:
    audio = _owned_with_bytes(tmp_path, _pcm_wav_bytes())

    assert len(list(tmp_path.iterdir())) == 1
    audio.close()
    audio.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode contract")
@pytest.mark.parametrize("with_samples", [False, True])
def test_wav_created_owner_only_under_permissive_umask(tmp_path: Path, with_samples: bool) -> None:
    previous = os.umask(0)
    try:
        audio = (
            write_validated_pcm_wav(tmp_path, (1, 2, 3), 16000)
            if with_samples
            else create_owned_wav(tmp_path)
        )
    finally:
        os.umask(previous)
    try:
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in tmp_path.iterdir())
    finally:
        audio.close()


def test_reader_uses_one_immutable_capture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = _owned_with_bytes(tmp_path, _pcm_wav_bytes())
    calls = 0
    original_open = Path.open

    def once(path: Path, *args: object, **kwargs: object):
        nonlocal calls
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "rb":
            calls += 1
            if calls > 1:
                raise AssertionError("reader reopened the path")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", once)

    assert read_owned_pcm_wav(audio).samples_int16 == (-32768, 0, 32767)
    assert calls == 1


def test_reader_rejects_trailing_bytes_and_never_exposes_pcm_or_path(tmp_path: Path) -> None:
    marker = "WAV_PATH_SENTINEL"
    root = tmp_path / marker
    root.mkdir()
    audio = _owned_with_bytes(root, _pcm_wav_bytes() + b"TAIL")

    with pytest.raises(LocalSpeechAdapterError) as caught:
        read_owned_pcm_wav(audio)

    rendered = "".join(traceback.format_exception(caught.value))
    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert marker not in str(caught.value)
    assert marker not in rendered
    assert "TAIL" not in rendered


def test_writer_round_trips_endpoint_scaling_and_returns_owned_file(tmp_path: Path) -> None:
    audio = write_validated_pcm_wav(
        tmp_path,
        samples=(-32768, 0, 32767),
        sample_rate_hz=16_000,
    )

    pcm = read_owned_pcm_wav(audio, expected_sample_rate_hz=16_000)

    assert type(audio) is OwnedWavFile
    assert pcm.samples_int16 == (-32768, 0, 32767)
    assert pcm.samples_float == (-1.0, 0.0, 32767 / 32768)
    audio.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "samples",
    [(32768,), (True,), (1.0,)],
)
def test_writer_rejects_invalid_pcm_and_leaves_no_artifacts(
    tmp_path: Path, samples: tuple[object, ...]
) -> None:
    with pytest.raises(LocalSpeechAdapterError) as caught:
        write_validated_pcm_wav(tmp_path, samples=samples, sample_rate_hz=16_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert list(tmp_path.iterdir()) == []


def test_resource_rejects_public_arbitrary_path_construction_and_hides_path(tmp_path: Path) -> None:
    audio = create_owned_wav(tmp_path)

    with pytest.raises(TypeError):
        OwnedWavFile(tmp_path / "caller.wav", tmp_path)  # type: ignore[call-arg]

    assert not hasattr(audio, "path")
    assert str(tmp_path) not in repr(audio)
    with pytest.raises(AttributeError):
        audio.extra = "not allowed"  # type: ignore[attr-defined]
    audio.close()


def test_reader_rejects_regular_path_replacement(tmp_path: Path) -> None:
    audio = _owned_with_bytes(tmp_path, _pcm_wav_bytes())
    path = audio._OwnedWavFile__path  # type: ignore[attr-defined]
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(_pcm_wav_bytes((1,)))
    replacement.replace(path)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        read_owned_pcm_wav(audio)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    with pytest.raises(LocalSpeechAdapterError) as cleanup:
        audio.close()
    assert cleanup.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    assert path.read_bytes() == _pcm_wav_bytes((1,))


def test_reader_rechecks_identity_after_its_single_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = _owned_with_bytes(tmp_path, _pcm_wav_bytes())
    path = audio._OwnedWavFile__path  # type: ignore[attr-defined]
    replacement = tmp_path / "replacement-during-read.wav"
    replacement.write_bytes(_pcm_wav_bytes((1,)))

    original_open = Path.open

    def replace_then_open(opened_path: Path, *args: object, **kwargs: object):
        if opened_path == path:
            replacement.replace(path)
        return original_open(opened_path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", replace_then_open)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        read_owned_pcm_wav(audio)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO


def test_reader_rejects_a_different_open_handle_even_when_path_identity_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = _owned_with_bytes(tmp_path, _pcm_wav_bytes((111,)))
    owned_path = audio._OwnedWavFile__path  # type: ignore[attr-defined]
    replacement = tmp_path / "replacement-handle.wav"
    replacement.write_bytes(_pcm_wav_bytes((222,)))
    original_open = Path.open

    def open_replacement(path: Path, *args: object, **kwargs: object):
        if path == owned_path and (args[0] if args else kwargs.get("mode", "r")) == "rb":
            return original_open(replacement, "rb")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", open_replacement)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        read_owned_pcm_wav(audio)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    audio.close()


def test_fixed_wav_errors_do_not_retain_path_or_pcm_in_module_frame_locals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker_root = tmp_path / "PRIVATE_WAV_PATH_SENTINEL"
    marker_root.mkdir()
    invalid = _owned_with_bytes(marker_root, b"PRIVATE_PCM_BYTES_SENTINEL")

    with pytest.raises(LocalSpeechAdapterError) as read_caught:
        read_owned_pcm_wav(invalid)

    read_locals = _module_frame_locals(read_caught.value, local_wav.__name__)
    assert str(marker_root) not in read_locals
    assert "PRIVATE_PCM_BYTES_SENTINEL" not in read_locals

    with pytest.raises(LocalSpeechAdapterError) as type_caught:
        read_owned_pcm_wav("PRIVATE_AUDIO_OBJECT_SENTINEL")  # type: ignore[arg-type]
    type_locals = _module_frame_locals(type_caught.value, local_wav.__name__)
    assert "PRIVATE_AUDIO_OBJECT_SENTINEL" not in type_locals

    with pytest.raises(LocalSpeechAdapterError) as input_caught:
        write_validated_pcm_wav(marker_root, ("PRIVATE_PCM_VALUE_SENTINEL",), 16_000)  # type: ignore[arg-type]
    input_locals = _module_frame_locals(input_caught.value, local_wav.__name__)
    assert str(marker_root) not in input_locals
    assert "PRIVATE_PCM_VALUE_SENTINEL" not in input_locals

    audio = _owned_with_bytes(marker_root, _pcm_wav_bytes((12_345, -23_456)))
    monkeypatch.setattr(
        local_wav,
        "_verify_owned_file",
        lambda *_args: (_ for _ in ()).throw(OSError("CLOSE_SENTINEL")),
    )
    with pytest.raises(LocalSpeechAdapterError) as close_caught:
        audio.close()
    close_locals = _module_frame_locals(close_caught.value, local_wav.__name__)
    assert str(marker_root) not in close_locals

    monkeypatch.undo()
    monkeypatch.setattr(
        local_wav.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("LINK_SENTINEL")),
    )
    with pytest.raises(LocalSpeechAdapterError) as write_caught:
        write_validated_pcm_wav(marker_root, (12_345, -23_456), 16_000)
    write_locals = _module_frame_locals(write_caught.value, local_wav.__name__)
    assert str(marker_root) not in write_locals
    assert "12_345" not in write_locals
    assert "12345" not in write_locals
    assert "-23456" not in write_locals


def test_writer_requires_strict_readback_before_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        local_wav,
        "_parse_pcm_wav",
        lambda *_args: (_ for _ in ()).throw(ValueError),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        write_validated_pcm_wav(tmp_path, samples=(1,), sample_rate_hz=16_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert list(tmp_path.iterdir()) == []


def test_factory_failure_removes_its_created_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_verify = local_wav._verify_owned_file
    calls = 0

    def fail_first_verify(*args: object) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic failure")
        return original_verify(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(
        local_wav,
        "_verify_owned_file",
        fail_first_verify,
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        create_owned_wav(tmp_path)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert list(tmp_path.iterdir()) == []


def test_factory_removes_a_partially_created_file_after_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(local_wav.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError()))

    with pytest.raises(LocalSpeechAdapterError) as caught:
        create_owned_wav(tmp_path)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert list(tmp_path.iterdir()) == []


def test_reader_rejects_invalid_format_and_expected_rate(tmp_path: Path) -> None:
    invalid = _owned_with_bytes(tmp_path, b"RIFF\x00\x00\x00\x00WAVE")
    with pytest.raises(LocalSpeechAdapterError) as invalid_caught:
        read_owned_pcm_wav(invalid)
    assert invalid_caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    invalid.close()

    valid = _owned_with_bytes(tmp_path, _pcm_wav_bytes())
    with pytest.raises(LocalSpeechAdapterError) as rate_caught:
        read_owned_pcm_wav(valid, expected_sample_rate_hz=44_100)
    assert rate_caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    valid.close()


def test_writer_failure_after_temporary_write_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        local_wav.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        write_validated_pcm_wav(tmp_path, samples=(1,), sample_rate_hz=16_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert list(tmp_path.iterdir()) == []


def test_writer_empty_pcm_round_trips_without_deciding_silence_policy(tmp_path: Path) -> None:
    audio = write_validated_pcm_wav(tmp_path, samples=(), sample_rate_hz=16_000)

    pcm = read_owned_pcm_wav(audio, expected_sample_rate_hz=16_000)

    assert pcm.frame_count == 0
    assert pcm.samples_int16 == ()
    assert pcm.samples_float == ()
    audio.close()
    assert list(tmp_path.iterdir()) == []


def test_writer_refuses_to_overwrite_a_preexisting_final_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temporary = tmp_path / "created.tmp"
    preexisting_final = tmp_path / "preexisting.wav"
    sentinel = b"PREEXISTING_DESTINATION"
    preexisting_final.write_bytes(sentinel)
    paths = iter((temporary, preexisting_final))
    monkeypatch.setattr(local_wav, "_new_path", lambda _root, _suffix: next(paths))

    with pytest.raises(LocalSpeechAdapterError) as caught:
        write_validated_pcm_wav(tmp_path, samples=(1,), sample_rate_hz=16_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.INVALID_AUDIO
    assert preexisting_final.read_bytes() == sentinel
    assert list(tmp_path.iterdir()) == [preexisting_final]


def test_writer_rejects_temp_replacement_during_readback_without_deleting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temporary = tmp_path / "created.tmp"
    replacement = tmp_path / "replacement.tmp"
    replacement_payload = b"TEMP_REPLACEMENT_SENTINEL"
    replacement.write_bytes(replacement_payload)
    paths = iter((temporary, tmp_path / "final.wav"))
    monkeypatch.setattr(local_wav, "_new_path", lambda _root, _suffix: next(paths))

    original_open = Path.open
    replaced = False

    def replace_then_open(path: Path, *args: object, **kwargs: object):
        nonlocal replaced
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == temporary and mode == "rb" and not replaced:
            replacement.replace(path)
            replaced = True
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", replace_then_open)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        write_validated_pcm_wav(tmp_path, samples=(1,), sample_rate_hz=16_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    with temporary.open("rb") as replacement_file:
        assert replacement_file.read() == replacement_payload
    assert temporary not in local_wav._OWNED_FILE_IDENTITIES


def test_writer_rejects_final_replacement_after_atomic_link_without_deleting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_link = local_wav.os.link
    replacement = tmp_path / "replacement-final.wav"
    replacement_payload = b"FINAL_REPLACEMENT_SENTINEL"
    replacement.write_bytes(replacement_payload)

    def link_then_replace(source: Path, destination: Path, **kwargs: object) -> None:
        original_link(source, destination, **kwargs)
        replacement.replace(destination)

    monkeypatch.setattr(local_wav.os, "link", link_then_replace)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        write_validated_pcm_wav(tmp_path, samples=(1,), sample_rate_hz=16_000)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    finals = list(tmp_path.glob("*.wav"))
    assert len(finals) == 1
    with finals[0].open("rb") as replacement_file:
        assert replacement_file.read() == replacement_payload
    assert finals[0] not in local_wav._OWNED_FILE_IDENTITIES


def test_cleanup_failure_forgets_the_replaced_path_identity(tmp_path: Path) -> None:
    audio = _owned_with_bytes(tmp_path, _pcm_wav_bytes())
    path = audio._OwnedWavFile__path  # type: ignore[attr-defined]
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(_pcm_wav_bytes((7,)))
    replacement.replace(path)

    with pytest.raises(LocalSpeechAdapterError) as caught:
        audio.close()

    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
    assert path not in local_wav._OWNED_FILE_IDENTITIES
    assert path.read_bytes() == _pcm_wav_bytes((7,))


def test_exclusive_creation_record_failure_removes_the_created_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        local_wav,
        "_record_created_file",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic record failure")),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        create_owned_wav(tmp_path)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED
    assert list(tmp_path.iterdir()) == []


def test_exclusive_creation_without_reliable_identity_fails_as_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        local_wav,
        "_file_identity",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic identity failure")),
    )

    with pytest.raises(LocalSpeechAdapterError) as caught:
        create_owned_wav(tmp_path)

    assert caught.value.code is LocalSpeechAdapterErrorCode.CLEANUP_FAILED
