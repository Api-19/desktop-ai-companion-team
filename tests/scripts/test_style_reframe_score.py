from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts import style_reframe_eval as ev


def ballot(n=30, wins=24):
    key = {str(i): dict(zip("ABC", ev.ARMS, strict=True)) for i in range(n)}
    choices = {str(i): "C" if i < wins else "A" for i in range(n)}
    return choices, key


@pytest.mark.parametrize("n, wins, passed", [(30, 24, True), (30, 23, False), (29, 24, False)])
def test_score_pass_boundary(n, wins, passed):
    choices, key = ballot(n, wins)
    result = ev.score_choices(choices, key)
    assert result == {
        "total": n, "candidate_wins": wins,
        "counts": {"legacy": n - wins, "general": 0, "full": wins, "并列": 0, "都不像": 0},
        "passed": passed,
    }


def test_score_counts_and_per_item_labels():
    key = {"a": {"A": "full", "B": "legacy"}, "b": {"A": "legacy", "B": "full"},
           "c": {"A": "general"}, "d": {"A": "full"}, "e": {"A": "full"}}
    result = ev.score_choices({"a": "A", "b": "A", "c": "A", "d": "并列", "e": "都不像"}, key)
    assert result["candidate_wins"] == 1
    assert result["counts"] == {"full": 1, "legacy": 1, "general": 1, "并列": 1, "都不像": 1}
    assert ev.score_choices({"a": "B"}, {"a": key["a"]}, "legacy")["candidate_wins"] == 1


@pytest.mark.parametrize("choices", [{}, {"a": "A", "extra": "A"}, {"a": "C"},
                                    {"a": "未知"}, {"a": ["A"]}, []])
def test_score_invalid_choices(choices):
    with pytest.raises(ValueError):
        ev.score_choices(choices, {"a": {"A": "full", "B": "legacy"}})


@pytest.mark.parametrize("key", [[], {"a": []}, {"a": {"A": 1}}])
def test_score_invalid_key(key):
    with pytest.raises(ValueError):
        ev.score_choices({"a": "A"}, key)


@pytest.mark.parametrize("kind", ["valid", "choice", "json", "missing", "shape", "encoding"])
def test_score_cli(tmp_path, kind):
    choices, key = ballot()
    choice_path, key_path = tmp_path / "choices.json", tmp_path / "key.json"
    if kind == "choice":
        choices["0"] = "PRIVATE_SYNTHETIC_INVALID"
    choice_path.write_text(json.dumps(choices), encoding="utf-8")
    key_path.write_text(json.dumps(key), encoding="utf-8")
    if kind == "json":
        choice_path.write_text("PRIVATE_SYNTHETIC_INVALID", encoding="utf-8")
    elif kind == "missing":
        choice_path.unlink()
    elif kind == "shape":
        key_path.write_text("[]", encoding="utf-8")
    elif kind == "encoding":
        choice_path.write_bytes(b"\xff")
    result = subprocess.run(
        [sys.executable, str(ev.ROOT / "scripts/style_reframe_score.py"),
         "--choices", str(choice_path), "--answer-key", str(key_path)],
        env={**os.environ, "PYTHONPATH": str(ev.ROOT / "src")}, cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    if kind == "valid":
        assert result.returncode == 0
        assert json.loads(result.stdout) == {"status": "scored", **ev.score_choices(choices, key)}
    else:
        assert result.returncode == 2
        assert json.loads(result.stdout) == {"status": "invalid_input"}


def test_score_cli_card(tmp_path):
    key = {str(i): {"A": "full", "B": "card"} for i in range(30)}
    choices = {str(i): "B" if i < 24 else "A" for i in range(30)}
    choice_path, key_path = tmp_path / "choices.json", tmp_path / "key.json"
    choice_path.write_text(json.dumps(choices), encoding="utf-8")
    key_path.write_text(json.dumps(key), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ev.ROOT / "scripts/style_reframe_score.py"),
         "--choices", str(choice_path), "--answer-key", str(key_path),
         "--candidate-arm", "card"],
        env={**os.environ, "PYTHONPATH": str(ev.ROOT / "src")}, cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "status": "scored", **ev.score_choices(choices, key, candidate_arm="card")}
    assert json.loads(result.stdout)["passed"] is True
