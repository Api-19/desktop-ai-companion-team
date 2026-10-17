# ruff: noqa: RUF001
"""Synthetic-only acceptance evidence for the frozen M1 labeler contract."""

import hashlib
import json
import stat
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.dialogue.models import ProviderResult, ProviderUsage
from companion.dialogue.provider import ProviderError
from scripts import m1_labeler as labeler

MARKER = "SYNTHETIC_CHAT_CANARY_M1_7321"
USAGE = ProviderUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
LABELS = [dict(code=code, name=f"类别{code}", hint=f"提示{code}") for code in "ABCDE"]


def tasks(count=5):
    return {
        "version": 1, "labels": LABELS,
        "items": {f"r{i:03}": {"context": [f"他：{MARKER}", "她：前文"],
                                "partner": ["问题一", "问题二"], "reply": ["回答一", "回答二"]}
                  for i in range(count)},
        "rounds": [{"id": "test", "title": "合成", "opens_on": None,
                    "order": [f"r{i:03}" for i in reversed(range(count))]}],
    }


def gold(codes):
    return {f"r{i:03}": {"label": code, "at": "2026-10-09T00:00:00+00:00"}
            for i, code in enumerate(codes)}


class FakeProvider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return ProviderResult(text=reply, usage=USAGE)


class FakeLedger:
    def __init__(self, cost="0.10"):
        self.cost = Decimal(cost)
        self.total = Decimal("12.00")
        self.events = []

    def snapshot(self, now):
        self.events.append(("snapshot", now))
        return SimpleNamespace(accounted_cny=self.total)

    def reserve(self, count, now):
        self.events.append(("reserve", count, now))
        return ("synthetic-reservation",)

    def settle(self, reservation, usage, now):
        self.events.append(("settle", reservation, usage, now))
        self.total += self.cost


def test_frozen_prompt_and_user_string():
    expected = (
        '你是对话标注员。下面每条是一段聊天：可能有前文，然后是"她"说的话，最后是"他"的回复。'
        '只给"他"的回复标一个标签，标的是他回复中第一个有内容的动作。'
        '开头的纯附和词（可以可以、嗯嗯、哦哦、好呀、ok、嘿嘿、行吧、哎呦等）先跳过；'
        '整条回复只有附和词时标 A。只有表情时按表情作用标：安慰类（如[拥抱]）标 A，'
        '搞怪类（如[猪头]）标 D。只看这段对话本身，不要推测两人关系处在什么阶段。\n标签：\n'
        + '\n'.join(f'{entry["code"]} {entry["name"]}：{entry["hint"]}' for entry in LABELS)
        + '\n? 无法判断\n边界：目的是要她哄、要她关注、表达喜欢标 E；目的是逗她、损她、搞笑标 D；'
        '一边接话一边损她标 D；针对她刚说的话反问标 A，问她新的情况标 B。\n'
        '输出格式：每条一行，"编号:标签"，不要输出其他任何文字。'
    )
    messages = labeler.build_messages(tasks(), ["r001", "r000"])
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == expected
    assert hashlib.sha256(messages[0].content.encode()).hexdigest() == hashlib.sha256(
        expected.encode()).hexdigest()
    one = f"前文 他：{MARKER}\n前文 她：前文\n她：问题一\n她：问题二\n他：回答一\n他：回答二"
    assert messages[1].content == f"[r001]\n{one}\n\n[r000]\n{one}"


def test_parse_fullwidth_missing_conflict_and_other_batch():
    parsed = labeler.parse_labels(
        " [r000] ： A \n r001:？\nr002:B\nr002:B\nr003:C\nr003:D\nr003:C\n"
        "r999:E\nnoise\nr004:F\nr005:? trailing", [f"r{i:03}" for i in range(6)])
    assert parsed == {"r000": "A", "r001": "?", "r002": "B", "r003": "invalid",
                      "r004": "invalid", "r005": "invalid"}


@pytest.mark.parametrize(("predictions", "kappa", "f1", "reject", "passed"), [
    (list("ABCDE"), 1, 1, 0, True),
    (list("BCDEA"), -0.25, 0, 0, False),
    (["A", "A", "?", "invalid", "E"], 0.5, 1 / 3, 0.4, False),
])
def test_hand_calculated_metrics(predictions, kappa, f1, reject, passed):
    answers = dict(zip(gold("ABCDE"), predictions, strict=True))
    metrics = labeler.compute_metrics(gold("ABCDE"), answers)
    assert metrics["kappa"] == pytest.approx(kappa)
    assert metrics["macro_f1"] == pytest.approx(f1)
    assert metrics["reject_rate"] == pytest.approx(reject)
    assert labeler.is_passed(metrics) is passed
    assert set(metrics["confusion"]) == set("ABCDE")
    assert set(metrics["confusion"]["A"]) == {*"ABCDE", "?", "invalid", "not_run"}
    assert metrics["per_class"]["A"]["support"] == 1
    if "invalid" in predictions:
        assert metrics["per_class"]["A"]["precision"] == 0.5
        assert metrics["per_class"]["A"]["recall"] == 1
        assert metrics["confusion"]["C"]["?"] == 1
        assert metrics["confusion"]["D"]["invalid"] == 1


def test_unevaluated_gold_not_run_and_absent_support():
    metrics = labeler.compute_metrics(gold("AAA?"), {
        "r000": "A", "r001": "A", "r002": "not_run", "r003": "invalid"})
    assert metrics["kappa"] == 1
    assert metrics["macro_f1"] == pytest.approx(0.8)
    assert metrics["reject_rate"] == 0.5
    assert metrics["confusion"]["A"]["not_run"] == 1
    assert metrics["per_class"]["B"] == dict(support=0, precision=0, recall=0, f1=0)


@pytest.mark.parametrize(("field", "value"), [
    ("kappa", 0.5999), ("macro_f1", 0.7499), ("reject_rate", 0.1501), ("recall", 0),
])
def test_each_gate_independently_fails(field, value):
    metrics = labeler.compute_metrics(gold("ABCDE"), dict(zip(gold("ABCDE"), "ABCDE", strict=True)))
    metrics.update(kappa=0.6, macro_f1=0.75, reject_rate=0.15)
    assert labeler.is_passed(metrics)
    if field == "recall":
        metrics["per_class"]["E"]["recall"] = value
    else:
        metrics[field] = value
    assert not labeler.is_passed(metrics)


def test_empty_and_no_valid_intersection():
    for source, predictions in [(gold("AB"), {"r000": "?", "r001": "invalid"}), ({}, {})]:
        metrics = labeler.compute_metrics(source, predictions)
        assert metrics["kappa"] == metrics["macro_f1"] == 0
        assert not labeler.is_passed(metrics)


def test_batch_failure_settles_and_continues(tmp_path):
    provider = FakeProvider([ProviderError(MARKER), "r002:C\nr001:B", "r000:A"])
    ledger = FakeLedger()
    report = labeler.run_labeler(tasks(), "test", gold("ABCDE"), tmp_path,
                                 provider=provider, ledger=ledger, model="synthetic-model",
                                 max_cny=Decimal("2"), max_requests=10, batch_size=2,
                                 min_interval=0)
    assert report["labels"] == {"r004": "invalid", "r003": "invalid", "r002": "C",
                                "r001": "B", "r000": "A"}
    assert report["requests"] == 3 and report["spent_cny"] == "0.30"
    assert [event[2] for event in ledger.events if event[0] == "settle"] == [None, USAGE, USAGE]
    assert [event[0] for event in ledger.events if event[0] != "snapshot"] == [
        "reserve", "settle", "reserve", "settle", "reserve", "settle"]
    assert all(event[-1].tzinfo == UTC for event in ledger.events)
    assert "[r004]" in provider.calls[0][1].content
    assert "[r002]" in provider.calls[1][1].content


@pytest.mark.parametrize(("max_requests", "max_cny", "requests"), [
    (1, "2", 1), (10, "0.25", 1), (0, "2", 0), (10, "0.19", 0), (10, "0.20", 1),
])
def test_caps_leave_remaining_not_run(tmp_path, max_requests, max_cny, requests):
    ledger = FakeLedger()
    provider = FakeProvider(["r004:E"])
    report = labeler.run_labeler(tasks(), "test", gold("ABCDE"), tmp_path,
                                 provider=provider, ledger=ledger, batch_size=1,
                                 max_cny=Decimal(max_cny), max_requests=max_requests,
                                 min_interval=0)
    assert report["requests"] == len(provider.calls) == requests
    assert list(report["labels"].values()).count("not_run") == 5 - requests
    assert report["spent_cny"] == ("0.10" if requests else "0.00")
    assert sum(event[0] == "snapshot" for event in ledger.events) >= requests + 2


def test_request_spacing_injected_sleep(tmp_path):
    clock = [0.0]
    starts, sleeps = [], []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    provider = FakeProvider(["r002:C", "r001:B", "r000:A"])
    complete = provider.complete

    def timed(messages):
        starts.append(clock[0])
        clock[0] += 0.5
        return complete(messages)

    provider.complete = timed
    labeler.run_labeler(tasks(3), "test", gold("ABC"), tmp_path, provider=provider,
                        batch_size=1, min_interval=3, sleep=sleep, monotonic=lambda: clock[0])
    assert starts == [0, 3, 6]
    assert sleeps == [2.5, 2.5]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    task_path, gold_path = tmp_path / "label_tasks.json", tmp_path / "labels_test.json"
    task_path.write_text(json.dumps(tasks()))
    gold_path.write_text(json.dumps(gold("ABCDE")))
    out = tmp_path / "private" / "labeler"
    args = ["--tasks", str(task_path), "--round", "test", "--gold", str(gold_path),
            "--out-dir", str(out), "--min-interval", "0"]
    settings = SimpleNamespace(private_data_dir=tmp_path / "private",
                               runtime_data_dir=tmp_path / "runtime")
    monkeypatch.setattr(labeler, "load_settings", lambda root: settings)
    return args, out, settings


def test_cli_real_assembly_fake_transport_permissions_and_privacy(cli, monkeypatch, capsys):
    args, out, settings = cli
    provider = FakeProvider(["r000:A\nr001:B\nr002:C\nr003:D\nr004:E"])
    ledger = FakeLedger()
    config = SimpleNamespace(provider=SimpleNamespace(model="synthetic-model"),
                             budget=SimpleNamespace(max_output_tokens=123))
    secret = object()

    def load(received, *, require_style_manifest):
        assert received is settings and require_style_manifest is False
        return config, SimpleNamespace(api_key=secret)

    def build(**kwargs):
        assert kwargs == dict(config=config.provider, api_key=secret, max_output_tokens=123)
        return provider

    def accounting(root, budget):
        assert root == settings.runtime_data_dir / "dialogue" and budget is config.budget
        return ledger

    monkeypatch.setattr(labeler, "load_dialogue_config", load)
    monkeypatch.setattr(labeler, "OpenAICompatibleProvider", build)
    monkeypatch.setattr(labeler, "MonthlyUsageLedger", accounting)
    assert labeler.main([*args, "--real", "--max-cny", "2", "--max-requests", "10"]) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and len(captured.out.splitlines()) == 1
    summary = json.loads(captured.out)
    assert summary == dict(status="ok", requests=1, spent_cny="0.10", kappa=1,
                           macro_f1=1, reject_rate=0, passed=True,
                           per_class={code: {"recall": 1} for code in "ABCDE"})
    files = list(out.glob("labeler_test_*.json"))
    assert len(files) == 1 and stat.S_IMODE(files[0].stat().st_mode) == 0o600
    raw = files[0].read_text()
    report = json.loads(raw)
    assert MARKER not in raw + captured.out
    assert report["version"] == 1 and report["model"] == "synthetic-model"
    digest = hashlib.sha256(provider.calls[0][0].content.encode()).hexdigest()
    assert report["prompt_sha256"] == digest
    assert set(report) == {"version", "model", "prompt_sha256", "labels", "metrics",
                           "passed", "requests", "spent_cny"}
    timestamp = files[0].stem.removeprefix("labeler_test_")
    datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")


def test_fake_injection_output(tmp_path):
    report = labeler.run_labeler(tasks(1), "test", gold("A"), tmp_path,
                                 provider=FakeProvider(["r000:A"]), min_interval=0)
    assert report["model"] == "fake" and report["spent_cny"] == "0.00"


def test_cli_without_real_does_not_read_data(cli, monkeypatch, capsys):
    args, _, _ = cli
    monkeypatch.setattr(labeler, "load_settings", lambda *a: pytest.fail("settings read"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("input read"))
    assert labeler.main(args) == 2
    assert capsys.readouterr().out == '{"status":"real_required"}\n'


@pytest.mark.parametrize("location", ["outside", "root", "symlink"])
def test_cli_private_output_boundary(cli, tmp_path, monkeypatch, capsys, location):
    args, _, settings = cli
    target = tmp_path / "outside"
    if location == "root":
        target = settings.private_data_dir
    elif location == "symlink":
        settings.private_data_dir.mkdir()
        link = settings.private_data_dir / "escape"
        link.symlink_to(tmp_path, target_is_directory=True)
        target = link / "outside"
    args[args.index("--out-dir") + 1] = str(target)
    monkeypatch.setattr(labeler, "load_dialogue_config",
                        lambda *a, **kw: pytest.fail("secrets read"))
    assert labeler.main([*args, "--real", "--max-cny", "1", "--max-requests", "2"]) == 2
    assert capsys.readouterr().out == '{"status":"out_dir_not_private"}\n'


@pytest.mark.parametrize("extra", [[], ["--max-cny", "NaN", "--max-requests", "1"],
                                   ["--max-cny", "1", "--max-requests", "-1"],
                                   ["--max-cny", "1", "--max-requests", "1", "--batch-size", "0"],
                                   ["--max-cny", "1", "--max-requests", "1",
                                    "--min-interval", "nan"]])
def test_invalid_cli_is_sanitized(cli, capsys, extra):
    args, _, _ = cli
    assert labeler.main([*args, "--real", *extra]) == 1
    assert capsys.readouterr().out == '{"status":"failed"}\n'


def test_unexpected_exception_is_sanitized(cli, monkeypatch, capsys):
    args, _, _ = cli

    def fail(*a, **kw):
        raise RuntimeError(MARKER)

    monkeypatch.setattr(labeler, "load_dialogue_config", fail)
    assert labeler.main([*args, "--real", "--max-cny", "1", "--max-requests", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"status":"failed"}\n' and captured.err == ""


def test_parser_error_does_not_echo_input(capsys):
    assert labeler.main(["--bad", MARKER]) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"status":"failed"}\n' and captured.err == ""


def test_executable_cli_without_real(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [str(root / ".venv/bin/python"), str(root / "scripts/m1_labeler.py"),
         "--tasks", str(tmp_path / "missing.json"), "--round", "test",
         "--gold", str(tmp_path / "missing-gold.json"), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert result.returncode == 2
    assert result.stdout == '{"status":"real_required"}\n' and result.stderr == ""


def test_non_evaluable_gold_does_not_add_false_positives():
    metrics = labeler.compute_metrics(gold("AA?"), {
        "r000": "A", "r001": "A", "r002": "A"})
    assert metrics["per_class"]["A"] == dict(support=2, precision=1, recall=1, f1=1)
    assert metrics["macro_f1"] == metrics["kappa"] == 1
    assert metrics["reject_rate"] == 0


@pytest.mark.parametrize("corruption", ["round", "id", "code", "gold", "timestamp"])
def test_invalid_input_fails_before_secret_loading(cli, monkeypatch, capsys, corruption):
    args, _, _ = cli
    task_path = Path(args[args.index("--tasks") + 1])
    gold_path = Path(args[args.index("--gold") + 1])
    source, answers = tasks(), gold("ABCDE")
    if corruption == "round":
        args[args.index("--round") + 1] = "../escape"
    elif corruption == "id":
        source["items"][MARKER] = source["items"].pop("r000")
        source["rounds"][0]["order"][-1] = MARKER
    elif corruption == "code":
        source["labels"] = [*LABELS[:-1], dict(code="F", name="未知", hint="未知")]
    elif corruption == "gold":
        answers["r000"]["label"] = MARKER
    else:
        answers["r000"]["at"] = "2026-10-09T00:00:00"
    task_path.write_text(json.dumps(source))
    gold_path.write_text(json.dumps(answers))
    monkeypatch.setattr(labeler, "load_dialogue_config",
                        lambda *a, **kw: pytest.fail("secrets before validation"))
    assert labeler.main([*args, "--real", "--max-cny", "1", "--max-requests", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"status":"failed"}\n' and captured.err == ""


def test_report_mode_is_0600_even_with_restrictive_umask(tmp_path):
    import os

    original = os.umask(0o777)
    try:
        labeler.run_labeler(tasks(1), "test", gold("A"), tmp_path,
                            provider=FakeProvider(["r000:A"]), min_interval=0)
    finally:
        os.umask(original)
    path = next(tmp_path.glob("labeler_test_*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
