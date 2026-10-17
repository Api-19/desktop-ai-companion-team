from __future__ import annotations

from companion.contracts import TranscriptConfidenceStatus
from companion.speech.controller_models import RecognitionInput
from companion.speech.controller_ports import RecordedAudio
from companion.speech.local_errors import LocalSpeechAdapterError, LocalSpeechAdapterErrorCode
from companion.speech.local_model_config import (
    LocalSpeechModelConfig,
    VerifiedLocalSpeechPaths,
    load_sherpa_runtime,
    verify_local_speech_config,
)
from companion.speech.local_wav import OwnedWavFile, PcmWavData, read_owned_pcm_wav
from companion.speech.recognition_gate import preflight_recognition_audio
from companion.speech.silero_quality import analyze_silero_quality

_SAMPLE_RATE_HZ = 16_000
_VAD_WINDOW_SIZE = 512


def _load_runtime_and_paths(
    config: LocalSpeechModelConfig,
) -> tuple[VerifiedLocalSpeechPaths, object]:
    return verify_local_speech_config(config), load_sherpa_runtime(config)


def _raise(code: LocalSpeechAdapterErrorCode) -> None:
    raise LocalSpeechAdapterError(code)


class SenseVoiceSileroRecognizer:
    __slots__ = ("__busy", "__closed", "__recognizer", "__vad")

    def __init__(self, config: LocalSpeechModelConfig) -> None:
        self.__busy = False
        self.__closed = True
        self.__vad: object | None = None
        self.__recognizer: object | None = None
        if type(config) is not LocalSpeechModelConfig:
            _raise(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)

        runtime: object | None = None
        vad: object | None = None
        recognizer: object | None = None
        try:
            paths, runtime = _load_runtime_and_paths(config)
            vad_config = runtime.VadModelConfig()  # type: ignore[attr-defined]
            paths = verify_local_speech_config(config)
            vad_config.silero_vad.model = str(paths.silero_model)
            vad_config.silero_vad.threshold = 0.25
            vad_config.silero_vad.min_silence_duration = 0.5
            vad_config.silero_vad.min_speech_duration = 0.5
            vad_config.silero_vad.max_speech_duration = 10.0
            vad_config.silero_vad.window_size = _VAD_WINDOW_SIZE
            vad_config.sample_rate = _SAMPLE_RATE_HZ
            vad_config.num_threads = 1
            vad_config.provider = "cpu"
            vad_config.debug = False
            vad = runtime.VoiceActivityDetector(  # type: ignore[attr-defined]
                vad_config, buffer_size_in_seconds=30.0
            )
            paths = verify_local_speech_config(config)
            recognizer = runtime.OfflineRecognizer.from_sense_voice(  # type: ignore[attr-defined]
                model=str(paths.sensevoice_model),
                tokens=str(paths.sensevoice_tokens),
                num_threads=4,
                sample_rate=_SAMPLE_RATE_HZ,
                feature_dim=80,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
                language="zh",
                use_itn=False,
            )
        except Exception:
            runtime = None
            vad = None
            recognizer = None
        if vad is None or recognizer is None:
            runtime = None
            paths = None  # type: ignore[assignment]
            vad_config = None  # type: ignore[assignment]
            _raise(LocalSpeechAdapterErrorCode.CONFIGURATION_FAILED)
        self.__vad = vad
        self.__recognizer = recognizer
        self.__closed = False

    def close(self) -> None:
        self.__vad = None
        self.__recognizer = None
        self.__closed = True

    def recognize(self, audio: RecordedAudio) -> RecognitionInput:
        if self.__closed or self.__busy:
            del audio
            _raise(LocalSpeechAdapterErrorCode.RECOGNITION_FAILED)
        if type(audio) is not OwnedWavFile:
            del audio
            _raise(LocalSpeechAdapterErrorCode.INVALID_AUDIO)

        failure: LocalSpeechAdapterErrorCode | None = None
        output: RecognitionInput | None = None
        self.__busy = True
        try:
            try:
                pcm = read_owned_pcm_wav(audio, expected_sample_rate_hz=_SAMPLE_RATE_HZ)
            except Exception:
                pcm = None
                failure = LocalSpeechAdapterErrorCode.INVALID_AUDIO
            if pcm is not None:
                try:
                    intervals = self._vad_intervals(pcm)
                    evidence, vad_passed = analyze_silero_quality(pcm, intervals)
                    rejection = preflight_recognition_audio(
                        audio=evidence,
                        vad_passed=vad_passed,
                    )
                except Exception:
                    evidence = None
                    vad_passed = False
                    rejection = None
                    failure = LocalSpeechAdapterErrorCode.RECOGNITION_FAILED
                if failure is None and evidence is not None:
                    if rejection is not None:
                        output = RecognitionInput(
                            audio=evidence,
                            vad_passed=vad_passed,
                            text=None,
                            confidence=None,
                            confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
                            confidence_threshold=None,
                        )
                    else:
                        text = self._recognize_pcm(pcm)
                        if text is None:
                            failure = LocalSpeechAdapterErrorCode.RECOGNITION_FAILED
                        else:
                            output = RecognitionInput(
                                audio=evidence,
                                vad_passed=vad_passed,
                                text=text,
                                confidence=None,
                                confidence_status=TranscriptConfidenceStatus.UNAVAILABLE,
                                confidence_threshold=None,
                            )
        finally:
            self.__busy = False

        if failure is not None or output is None:
            pcm = None
            intervals = None  # type: ignore[assignment]
            evidence = None
            rejection = None
            text = None
            del audio
            _raise(failure or LocalSpeechAdapterErrorCode.RECOGNITION_FAILED)
        return output

    def _vad_intervals(self, pcm: PcmWavData) -> list[tuple[int, int]]:
        vad = self.__vad
        if vad is None:
            raise ValueError("closed")
        vad.reset()
        samples = pcm.samples_float
        for offset in range(0, len(samples), _VAD_WINDOW_SIZE):
            chunk = samples[offset : offset + _VAD_WINDOW_SIZE]
            if len(chunk) < _VAD_WINDOW_SIZE:
                chunk = chunk + (0.0,) * (_VAD_WINDOW_SIZE - len(chunk))
            vad.accept_waveform(chunk)
        vad.flush()

        intervals: list[tuple[int, int]] = []
        maximum_segments = (pcm.frame_count + _VAD_WINDOW_SIZE - 1) // _VAD_WINDOW_SIZE + 1
        for _ in range(maximum_segments):
            empty = vad.empty()
            if type(empty) is not bool:
                raise ValueError("malformed VAD state")
            if empty:
                return intervals
            front = vad.front
            start = front.start
            samples = front.samples
            length = len(samples)
            if type(start) is not int or type(length) is not int or length < 0:
                raise ValueError("malformed VAD segment")
            intervals.append((start, start + length))
            vad.pop()
        empty = vad.empty()
        if type(empty) is not bool or not empty:
            raise ValueError("unbounded VAD state")
        return intervals

    def _recognize_pcm(self, pcm: PcmWavData) -> str | None:
        recognizer = self.__recognizer
        if recognizer is None:
            return None
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(_SAMPLE_RATE_HZ, pcm.samples_float)
            recognizer.decode_stream(stream)
            text = stream.result.text
        except Exception:
            return None
        return text if type(text) is str else None
