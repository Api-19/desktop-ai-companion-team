from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode

SHERPA_ONNX_VERSION = "1.13.4"
SHERPA_ONNX_CORE_VERSION = "1.13.4"
_PATH_TYPE = type(Path())
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]

_SENSEVOICE_SOURCE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
)
_SILERO_SOURCE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models"
_VITS_SOURCE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"
)


@dataclass(frozen=True, slots=True)
class LocalSpeechModelFile:
    identifier: str
    relative_path: Path
    sha256: str
    license_id: str
    source_url: str
    byte_count: int | None = None


@dataclass(frozen=True, slots=True)
class LocalSpeechManifest:
    sensevoice_identifier: str
    silero_identifier: str
    vits_identifier: str
    sensevoice_model: LocalSpeechModelFile
    sensevoice_tokens: LocalSpeechModelFile
    silero_model: LocalSpeechModelFile
    vits_model: LocalSpeechModelFile
    vits_lexicon: LocalSpeechModelFile
    vits_tokens: LocalSpeechModelFile
    vits_date_fst: LocalSpeechModelFile
    vits_number_fst: LocalSpeechModelFile
    sensevoice_tree_sha256: str
    vits_tree_sha256: str
    vits_required_tree_sha256: str
    vits_archive_sha256: str
    vits_archive_license_id: str
    vits_archive_source_url: str

    @property
    def vits_required_files(self) -> tuple[LocalSpeechModelFile, ...]:
        return (
            self.vits_model,
            self.vits_lexicon,
            self.vits_tokens,
            self.vits_date_fst,
            self.vits_number_fst,
        )


LOCAL_SPEECH_MANIFEST = LocalSpeechManifest(
    sensevoice_identifier="sensevoice-2024-07-17-int8",
    silero_identifier="silero-vad-onnx",
    vits_identifier="vits-melo-tts-zh_en",
    sensevoice_model=LocalSpeechModelFile(
        "sensevoice-2024-07-17-int8-model",
        Path("sensevoice-2024-07-17-int8/model.int8.onnx"),
        "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
        "FunASR-Model-License-1.1",
        _SENSEVOICE_SOURCE_URL,
    ),
    sensevoice_tokens=LocalSpeechModelFile(
        "sensevoice-2024-07-17-int8-tokens",
        Path("sensevoice-2024-07-17-int8/tokens.txt"),
        "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
        "FunASR-Model-License-1.1",
        _SENSEVOICE_SOURCE_URL,
    ),
    silero_model=LocalSpeechModelFile(
        "silero-vad-onnx",
        Path("silero_vad.onnx"),
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        "MIT",
        _SILERO_SOURCE_URL,
    ),
    vits_model=LocalSpeechModelFile(
        "vits-melo-tts-zh-en-model",
        Path("vits-melo-tts-zh_en/model.onnx"),
        "bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d",
        "MIT",
        _VITS_SOURCE_URL,
    ),
    vits_lexicon=LocalSpeechModelFile(
        "vits-melo-tts-zh-en-lexicon",
        Path("vits-melo-tts-zh_en/lexicon.txt"),
        "7236884b02435ac5d10cf69b4be40a61b45aa676b5300f0e412f185748fee528",
        "MIT",
        _VITS_SOURCE_URL,
    ),
    vits_tokens=LocalSpeechModelFile(
        "vits-melo-tts-zh-en-tokens",
        Path("vits-melo-tts-zh_en/tokens.txt"),
        "d18664a7e12bd7ea1022ddaf951e534e136815016c5a809d6b64156bffb4369d",
        "MIT",
        _VITS_SOURCE_URL,
    ),
    vits_date_fst=LocalSpeechModelFile(
        "vits-melo-tts-zh-en-date-fst",
        Path("vits-melo-tts-zh_en/date.fst"),
        "eb8aa079ae3cb81d8f4404992f39d61a0cb990947512b5b8d1e54d1f6980e718",
        "MIT",
        _VITS_SOURCE_URL,
    ),
    vits_number_fst=LocalSpeechModelFile(
        "vits-melo-tts-zh-en-number-fst",
        Path("vits-melo-tts-zh_en/number.fst"),
        "743f402181fcfebf76cc2f0546b71fa26476e626fbe4e460fb7b4c3a7a8bd5bd",
        "MIT",
        _VITS_SOURCE_URL,
    ),
    sensevoice_tree_sha256="782cbe9421d975b92f8484df9a7c751cdd2f475866f595c16034f9a978881e19",
    vits_tree_sha256="7c4491c59e61659fca2a1e1555bef674a098ddde4be131300437e8fad0dfe1f4",
    vits_required_tree_sha256="e4683fd2a372235d02972eb6f85619597892ca10ac00bd4daf66f22390356ca5",
    vits_archive_sha256="e58351ed7149f290a54534538badd4077cdbe6fddc964b24d0bee870415d1514",
    vits_archive_license_id="MIT",
    vits_archive_source_url=_VITS_SOURCE_URL,
)

ALL_MODEL_FILES = (
    LOCAL_SPEECH_MANIFEST.sensevoice_model,
    LOCAL_SPEECH_MANIFEST.sensevoice_tokens,
    LOCAL_SPEECH_MANIFEST.silero_model,
    *LOCAL_SPEECH_MANIFEST.vits_required_files,
)


@dataclass(frozen=True, slots=True, repr=False, weakref_slot=True)
class LocalSpeechModelConfig:
    model_root: Path
    run_root: Path
    sensevoice_identifier: str
    silero_identifier: str
    vits_identifier: str
    sensevoice_model_relative_path: Path
    sensevoice_tokens_relative_path: Path
    silero_model_relative_path: Path
    vits_model_relative_path: Path
    vits_lexicon_relative_path: Path
    vits_tokens_relative_path: Path
    vits_date_fst_relative_path: Path
    vits_number_fst_relative_path: Path
    sherpa_onnx_version: str
    sherpa_onnx_core_version: str
    sensevoice_model_sha256: str
    sensevoice_tokens_sha256: str
    silero_model_sha256: str
    vits_model_sha256: str
    vits_lexicon_sha256: str
    vits_tokens_sha256: str
    vits_date_fst_sha256: str
    vits_number_fst_sha256: str
    sensevoice_tree_sha256: str
    vits_tree_sha256: str
    vits_required_tree_sha256: str

    def __post_init__(self) -> None:
        if type(self.model_root) is not _PATH_TYPE or type(self.run_root) is not _PATH_TYPE:
            _raise_configuration_failed()
        if not self.model_root.is_absolute() or not self.run_root.is_absolute():
            _raise_configuration_failed()
        if _path_is_inside_worktree(self.model_root) or _path_is_inside_worktree(self.run_root):
            _raise_configuration_failed()
        if _same_resolved_path(self.model_root, self.run_root):
            _raise_configuration_failed()
        for value in _relative_paths_from_config(self):
            _validate_relative_path(value)
        for value in _declarations_from_config(self):
            if type(value) is not str:
                _raise_configuration_failed()
        if not _config_matches_manifest(self):
            _raise_configuration_failed()


def _model_config_from_validated_runtime(
    settings: object,
    value: object,
) -> LocalSpeechModelConfig:
    from companion.speech.runtime_config import _require_current_validated_runtime

    validated: object = None
    model_root: Path | None = None
    run_root: Path | None = None
    values: dict[str, object] | None = None
    result: LocalSpeechModelConfig | None = None
    failed = False
    try:
        validated = _require_current_validated_runtime(settings, value)  # type: ignore[arg-type]
        model_root = validated.model_root  # type: ignore[attr-defined]
        run_root = validated.run_root  # type: ignore[attr-defined]
        values = {
            "model_root": model_root,
            "run_root": run_root,
            "sensevoice_identifier": LOCAL_SPEECH_MANIFEST.sensevoice_identifier,
            "silero_identifier": LOCAL_SPEECH_MANIFEST.silero_identifier,
            "vits_identifier": LOCAL_SPEECH_MANIFEST.vits_identifier,
            "sensevoice_model_relative_path": LOCAL_SPEECH_MANIFEST.sensevoice_model.relative_path,
            "sensevoice_tokens_relative_path": (
                LOCAL_SPEECH_MANIFEST.sensevoice_tokens.relative_path
            ),
            "silero_model_relative_path": LOCAL_SPEECH_MANIFEST.silero_model.relative_path,
            "vits_model_relative_path": LOCAL_SPEECH_MANIFEST.vits_model.relative_path,
            "vits_lexicon_relative_path": LOCAL_SPEECH_MANIFEST.vits_lexicon.relative_path,
            "vits_tokens_relative_path": LOCAL_SPEECH_MANIFEST.vits_tokens.relative_path,
            "vits_date_fst_relative_path": LOCAL_SPEECH_MANIFEST.vits_date_fst.relative_path,
            "vits_number_fst_relative_path": LOCAL_SPEECH_MANIFEST.vits_number_fst.relative_path,
            "sherpa_onnx_version": SHERPA_ONNX_VERSION,
            "sherpa_onnx_core_version": SHERPA_ONNX_CORE_VERSION,
            "sensevoice_model_sha256": LOCAL_SPEECH_MANIFEST.sensevoice_model.sha256,
            "sensevoice_tokens_sha256": LOCAL_SPEECH_MANIFEST.sensevoice_tokens.sha256,
            "silero_model_sha256": LOCAL_SPEECH_MANIFEST.silero_model.sha256,
            "vits_model_sha256": LOCAL_SPEECH_MANIFEST.vits_model.sha256,
            "vits_lexicon_sha256": LOCAL_SPEECH_MANIFEST.vits_lexicon.sha256,
            "vits_tokens_sha256": LOCAL_SPEECH_MANIFEST.vits_tokens.sha256,
            "vits_date_fst_sha256": LOCAL_SPEECH_MANIFEST.vits_date_fst.sha256,
            "vits_number_fst_sha256": LOCAL_SPEECH_MANIFEST.vits_number_fst.sha256,
            "sensevoice_tree_sha256": LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256,
            "vits_tree_sha256": LOCAL_SPEECH_MANIFEST.vits_tree_sha256,
            "vits_required_tree_sha256": LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256,
        }
        result = object.__new__(LocalSpeechModelConfig)
        for field_name, field_value in values.items():
            object.__setattr__(result, field_name, field_value)
        if not _config_matches_manifest(result):
            raise ValueError("manifest mismatch")
    except Exception:
        failed = True
    settings = None
    value = None
    validated = None
    model_root = None
    run_root = None
    values = None
    field_name = None  # type: ignore[assignment]
    field_value = None
    if failed or result is None:
        result = None
        _raise_configuration_failed()
    return result


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedLocalSpeechPaths:
    sensevoice_model: Path
    sensevoice_tokens: Path
    silero_model: Path
    vits_model: Path
    vits_lexicon: Path
    vits_tokens: Path
    vits_rule_fsts: tuple[Path, Path]
    run_root: Path


def _raise_configuration_failed() -> NoReturn:
    raise LocalSpeechAdapterError(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)


def _path_is_inside_worktree(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(_WORKTREE_ROOT)
    except ValueError:
        return False
    except OSError:
        return True
    return True


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return True


def _validate_relative_path(path: Path) -> None:
    if type(path) is not _PATH_TYPE:
        _raise_configuration_failed()
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        _raise_configuration_failed()


def _relative_paths_from_config(config: LocalSpeechModelConfig) -> tuple[Path, ...]:
    return (
        config.sensevoice_model_relative_path,
        config.sensevoice_tokens_relative_path,
        config.silero_model_relative_path,
        config.vits_model_relative_path,
        config.vits_lexicon_relative_path,
        config.vits_tokens_relative_path,
        config.vits_date_fst_relative_path,
        config.vits_number_fst_relative_path,
    )


def _declarations_from_config(config: LocalSpeechModelConfig) -> tuple[str, ...]:
    return (
        config.sensevoice_identifier,
        config.silero_identifier,
        config.vits_identifier,
        config.sherpa_onnx_version,
        config.sherpa_onnx_core_version,
        config.sensevoice_model_sha256,
        config.sensevoice_tokens_sha256,
        config.silero_model_sha256,
        config.vits_model_sha256,
        config.vits_lexicon_sha256,
        config.vits_tokens_sha256,
        config.vits_date_fst_sha256,
        config.vits_number_fst_sha256,
        config.sensevoice_tree_sha256,
        config.vits_tree_sha256,
        config.vits_required_tree_sha256,
    )


def _config_matches_manifest(config: LocalSpeechModelConfig) -> bool:
    return _relative_paths_from_config(config) == tuple(
        item.relative_path for item in ALL_MODEL_FILES
    ) and _declarations_from_config(config) == (
        LOCAL_SPEECH_MANIFEST.sensevoice_identifier,
        LOCAL_SPEECH_MANIFEST.silero_identifier,
        LOCAL_SPEECH_MANIFEST.vits_identifier,
        SHERPA_ONNX_VERSION,
        SHERPA_ONNX_CORE_VERSION,
        *(item.sha256 for item in ALL_MODEL_FILES),
        LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256,
        LOCAL_SPEECH_MANIFEST.vits_tree_sha256,
        LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256,
    )


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


def _verified_root(path: Path) -> Path:
    if _path_is_inside_worktree(path):
        raise ValueError("root is in worktree")
    resolved = _absolute_no_follow(path)
    if not resolved.is_dir():
        raise ValueError("root is not a directory")
    return resolved


def _verified_file(root: Path, relative_path: Path) -> Path:
    candidate = root / relative_path
    resolved = _absolute_no_follow(candidate)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("model path escaped its root") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("model path is not a regular file")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(65_536):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_tree_files(
    root: Path, *, casefold_order: bool = False
) -> tuple[tuple[Path, Path], ...]:
    files: list[tuple[Path, Path]] = []
    for candidate in root.rglob("*"):
        if _is_link_or_reparse(candidate):
            raise ValueError("link or reparse point")
        if candidate.is_dir():
            continue
        if not stat.S_ISREG(candidate.stat().st_mode):
            raise ValueError("tree contains non-regular path")
        files.append((candidate.relative_to(root), candidate))
    key = (
        (lambda item: _tree_sort_key(item[0]))
        if casefold_order
        else (lambda item: item[0].as_posix())
    )
    return tuple(sorted(files, key=key))


def _tree_sort_key(path: Path) -> tuple[str, str]:
    relative = path.as_posix()
    return relative.casefold(), relative


def _sensevoice_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path, path in _regular_tree_files(root, casefold_order=True):
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _vits_tree_sha256(root: Path, *, required_paths: tuple[Path, ...] | None = None) -> str:
    if required_paths is None:
        entries = _regular_tree_files(root)
    else:
        entries = tuple(
            (relative_path, _verified_file(root, relative_path))
            for relative_path in sorted(required_paths, key=Path.as_posix)
        )
    digest = hashlib.sha256()
    for relative_path, path in entries:
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).lower().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_local_speech_config(config: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
    if type(config) is not LocalSpeechModelConfig or not _config_matches_manifest(config):
        raise ValueError("untrusted manifest declaration")
    model_root = _verified_root(config.model_root)
    run_root = _verified_root(config.run_root)
    if model_root == run_root:
        raise ValueError("equal roots")
    paths = tuple(_verified_file(model_root, item.relative_path) for item in ALL_MODEL_FILES)
    for path, item in zip(paths, ALL_MODEL_FILES, strict=True):
        if _file_sha256(path) != item.sha256:
            raise ValueError("file digest mismatch")
    sensevoice_root = paths[0].parent
    vits_root = paths[3].parent
    if _sensevoice_tree_sha256(sensevoice_root) != LOCAL_SPEECH_MANIFEST.sensevoice_tree_sha256:
        raise ValueError("sensevoice tree digest mismatch")
    if _vits_tree_sha256(vits_root) != LOCAL_SPEECH_MANIFEST.vits_tree_sha256:
        raise ValueError("vits tree digest mismatch")
    required_paths = tuple(
        item.relative_path.relative_to(vits_root.relative_to(model_root))
        for item in ALL_MODEL_FILES[3:]
    )
    if (
        _vits_tree_sha256(vits_root, required_paths=required_paths)
        != LOCAL_SPEECH_MANIFEST.vits_required_tree_sha256
    ):
        raise ValueError("required vits tree digest mismatch")
    return VerifiedLocalSpeechPaths(
        sensevoice_model=paths[0],
        sensevoice_tokens=paths[1],
        silero_model=paths[2],
        vits_model=paths[3],
        vits_lexicon=paths[4],
        vits_tokens=paths[5],
        vits_rule_fsts=(paths[6], paths[7]),
        run_root=run_root,
    )


def verify_local_speech_config(config: LocalSpeechModelConfig) -> VerifiedLocalSpeechPaths:
    try:
        verified = _verify_local_speech_config(config)
    except Exception:
        verified = None
    if verified is None:
        del config
        _raise_configuration_failed()
    return verified


def load_sherpa_runtime(config: LocalSpeechModelConfig) -> object:
    try:
        valid = type(config) is LocalSpeechModelConfig and _config_matches_manifest(config)
        versions = (
            importlib.metadata.version("sherpa-onnx"),
            importlib.metadata.version("sherpa-onnx-core"),
        )
        if not valid or versions != (SHERPA_ONNX_VERSION, SHERPA_ONNX_CORE_VERSION):
            raise ValueError("unsupported sherpa runtime")
        runtime = importlib.import_module("sherpa_onnx")
    except Exception:
        runtime = None
    if runtime is None:
        del config
        _raise_configuration_failed()
    return runtime
