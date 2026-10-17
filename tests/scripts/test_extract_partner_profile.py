"""Local extraction acceptance using synthetic identities and messages only."""

# ruff: noqa: RUF001

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from companion.contracts import ConsentRecord
from companion.memory import MemoryService
from companion.memory.profile import ProfileStatus

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/extract_partner_profile.py"
FIXTURE = ROOT / "fixtures/synthetic/partner_profile_chat_v1.json"
USER = "synthetic-partner"


def run_cli(chat, db, ledger, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--chat",
            str(chat),
            "--memory-db",
            str(db),
            "--user-id",
            USER,
            "--ledger",
            str(ledger),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def assert_safe(result, chat=FIXTURE):
    output = result.stdout + result.stderr
    for message in json.loads(chat.read_text()):
        text = message["text"]
        if text:
            assert text[:8] not in output
    assert result.stderr == ""


def grant(service, user=USER, fields=("profile",)):
    service.grant_consent(
        ConsentRecord(
            user_id=user,
            granted=True,
            granted_at=datetime.now(UTC),
            allowed_fields=list(fields),
        )
    )


@pytest.fixture
def storage(tmp_path):
    db = tmp_path / "memory.sqlite3"
    service = MemoryService(db, clock=lambda: datetime.now(UTC))
    grant(service)
    return service, db, tmp_path / "ledger.json"


def write_chat(tmp_path, texts, *, direction="inbound", content_type="text"):
    template = json.loads(FIXTURE.read_text())[0]
    chat = tmp_path / "chat.json"
    chat.write_text(
        json.dumps(
            [
                template
                | {
                    "message_id": f"synthetic-{i}",
                    "text": text,
                    "direction": direction,
                    "content_type": content_type,
                }
                for i, text in enumerate(texts)
            ],
            ensure_ascii=False,
        )
    )
    return chat


def test_synthetic_acceptance_and_counts(storage):
    service, db, ledger = storage
    result = run_cli(FIXTURE, db, ledger)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_safe(result)
    assert json.loads(result.stdout) == {
        "scanned": 18,
        "self_statements": 16,
        "candidates_added": 8,
        "skipped_duplicate": 1,
        "skipped_processed": 0,
        "excluded_owner": 1,
        "excluded_question": 1,
        "excluded_hypothetical": 1,
        "excluded_past": 1,
        "excluded_third_party": 1,
        "excluded_health": 1,
        "excluded_quote": 1,
        "excluded_other": 3,
    }
    items = service.list_profile(USER)
    assert {item.topic for item in items} == {
        "喝水",
        "作息",
        "奶茶",
        "咖啡",
        "运动",
        "吃辣",
        "早餐",
        "零食",
    }
    assert all(item.status == "candidate" and item.source == "self_stated" for item in items)
    assert all(item.evidence_date.isoformat() == "2026-10-10" for item in items)
    assert all(len(item.evidence_quote) <= 30 and len(item.text) <= 60 for item in items)
    water = next(item for item in items if item.topic == "喝水")
    assert water.text == "我每天喝水。" and water.category == "habit"
    assert water.confidence == "high"
    assert next(item for item in items if item.topic == "奶茶").confidence == "med"
    messages = json.loads(FIXTURE.read_text())
    saved = json.loads(ledger.read_text())
    assert sorted(saved) == sorted(
        hashlib.sha256(m["message_id"].encode()).hexdigest() for m in messages
    )
    assert len(saved) == len(set(saved))


def test_second_run_and_deleted_candidate_never_resurrect(storage):
    service, db, ledger = storage
    assert run_cli(FIXTURE, db, ledger).returncode == 0
    before = ledger.read_bytes()
    for delete in (False, True):
        if delete:
            item = service.list_profile(USER)[0]
            service.delete_profile(user_id=USER, profile_id=item.profile_id)
        result = run_cli(FIXTURE, db, ledger)
        assert result.returncode == 0
        counts = json.loads(result.stdout)
        assert counts["skipped_processed"] == counts["scanned"] == 18
        assert all(
            value == 0
            for key, value in counts.items()
            if key not in {"scanned", "skipped_processed"}
        )
        assert len(service.list_profile(USER)) == (7 if delete else 8)
        assert ledger.read_bytes() == before
        assert_safe(result)


@pytest.mark.parametrize("consent", ["missing", "ordinary", "other_user", "absent_db"])
def test_no_profile_consent_adds_no_candidates_or_ledger_entries(tmp_path, consent):
    db, ledger = tmp_path / "memory.sqlite3", tmp_path / "ledger.json"
    if consent != "absent_db":
        service = MemoryService(db, clock=lambda: datetime.now(UTC))
        if consent == "ordinary":
            grant(service, fields=("preference",))
        if consent == "other_user":
            grant(service, user="synthetic-other")
        original_count = len(service.list_profile(USER))
    else:
        original_count = 0
    ledger.write_text("[]")
    result = run_cli(FIXTURE, db, ledger)
    assert result.returncode == 2
    assert result.stdout == "profile_consent_required\n"
    assert ledger.read_text() == "[]"
    assert db.exists()
    service = MemoryService(db, clock=lambda: datetime.now(UTC))
    assert len(service.list_profile(USER)) == original_count
    assert_safe(result)


def test_unauthorized_preflight_allows_service_cleanup_but_adds_no_candidates(tmp_path):
    db, ledger = tmp_path / "memory.sqlite3", tmp_path / "ledger.json"
    old = datetime.now(UTC) - timedelta(days=40)
    service = MemoryService(db, clock=lambda: old)
    grant(service, user="synthetic-other")
    service.add_profile_candidate(
        user_id="synthetic-other",
        category="habit",
        topic="喝水",
        text="我每天喝水",
        source="self_stated",
        evidence_date=old.date(),
        confidence="high",
    )
    original_count = len(service.list_profile(USER))
    assert len(service.list_profile("synthetic-other")) == 1
    result = run_cli(FIXTURE, db, ledger)
    assert result.returncode == 2
    assert result.stdout == "profile_consent_required\n"
    assert len(service.list_profile(USER)) == original_count
    assert service.list_profile("synthetic-other") == ()
    assert not ledger.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_dry_run_adds_no_candidates_or_ledger_entries(tmp_path, existing):
    db, ledger = tmp_path / "memory.sqlite3", tmp_path / "ledger.json"
    if existing:
        service = MemoryService(db, clock=lambda: datetime.now(UTC))
        grant(service)
        service.add_profile_candidate(
            user_id=USER,
            category="habit",
            topic="喝水",
            text="我每天喝水。",
            source="self_stated",
            evidence_date=datetime.now(UTC).date(),
            confidence="high",
        )
        original_count = len(service.list_profile(USER))
        ledger.write_text("[]")
    else:
        original_count = 0
    result = run_cli(FIXTURE, db, ledger, "--dry-run")
    assert result.returncode == 0
    assert json.loads(result.stdout)["candidates_added"] == 0
    assert_safe(result)
    assert db.exists()
    service = MemoryService(db, clock=lambda: datetime.now(UTC))
    assert len(service.list_profile(USER)) == original_count
    if existing:
        assert ledger.read_text() == "[]"
    else:
        assert not ledger.exists()


@pytest.mark.parametrize("status", list(ProfileStatus))
def test_duplicate_in_every_state_and_user_isolation(storage, tmp_path, status):
    service, db, ledger = storage
    values = dict(
        category="preference",
        topic="咖啡",
        text="我喜欢咖啡。",
        source="self_stated",
        evidence_date=datetime.now(UTC).date(),
        confidence="med",
    )
    grant(service, user="synthetic-other")
    service.add_profile_candidate(user_id="synthetic-other", **values)
    item = service.add_profile_candidate(user_id=USER, **values)
    if status in {ProfileStatus.OWNER_APPROVED, ProfileStatus.CONFIRMED}:
        service.approve_profile(user_id=USER, profile_id=item.profile_id)
    if status == ProfileStatus.CONFIRMED:
        service.confirm_profile(
            user_id=USER,
            profile_id=item.profile_id,
            version=1,
            subject_id=USER,
            use_in_chat=False,
            simulated=True,
        )
    if status == ProfileStatus.DISABLED:
        service.disable_profile(user_id=USER, profile_id=item.profile_id, reason="owner")
    chat = write_chat(tmp_path, ["我喜欢咖啡。", "我喜欢咖啡。", "我喜欢奶茶。"])
    result = run_cli(chat, db, ledger)
    assert result.returncode == 0
    assert json.loads(result.stdout)["skipped_duplicate"] == 2
    assert json.loads(result.stdout)["candidates_added"] == 1
    assert len(service.list_profile(USER)) == 2


@pytest.mark.parametrize(
    ("text", "excluded"),
    [
        ("我每天喝水吗。", "question"),
        ("我喜欢咖啡？", "question"),
        ("我每天喝水?", "question"),
        ("要是我每天喝水。", "hypothetical"),
        ("假如我喜欢奶茶。", "hypothetical"),
        ("万一我喜欢咖啡。", "hypothetical"),
        ("之前我每天喝水。", "past"),
        ("小时候我喜欢零食。", "past"),
        ("那时候我喜欢奶茶。", "past"),
        ("我曾经喜欢咖啡。", "past"),
        ("我妈喜欢咖啡，我喜欢奶茶。", "third_party"),
        ("我爸每天喝水，我喜欢咖啡。", "third_party"),
        ("朋友喜欢运动，我喜欢跑步。", "third_party"),
        ("室友喜欢奶茶，我喜欢咖啡。", "third_party"),
        ("同学每天喝水，我喜欢奶茶。", "third_party"),
        ("我经常吃药喝水。", "health"),
        ("我喜欢咖啡但焦虑。", "health"),
        ("我每天运动但抑郁。", "health"),
        ("我喜欢咖啡但疼。", "health"),
        ("我喜欢零食但痛。", "health"),
        ("我每天到医院喝水。", "health"),
        ("我喜欢咖啡，确诊了。", "health"),
        ("我说：我喜欢咖啡。", "quote"),
        ("“我喜欢奶茶”。", "quote"),
        ("她说我喜欢咖啡。", "quote"),
        ("“我喜欢奶茶。”", "quote"),
        ("我说自己喜欢咖啡。", "quote"),
        ("我喜欢星星。", "other"),
        ("我刚喝水。", "other"),
    ],
)
def test_exclusion_rules(storage, tmp_path, text, excluded):
    service, db, ledger = storage
    chat = write_chat(tmp_path, [text])
    result = run_cli(chat, db, ledger)
    assert result.returncode == 0
    counts = json.loads(result.stdout)
    assert counts[f"excluded_{excluded}"] == 1
    assert sum(value for key, value in counts.items() if key.startswith("excluded_")) == 1
    assert service.list_profile(USER) == ()
    assert_safe(result, chat)


def test_long_sentence_limits_and_all_delimiters(storage, tmp_path):
    service, db, ledger = storage
    long = "我喜欢零食" + "合成脆片" * 30 + "啊。"
    chat = write_chat(tmp_path, [long, "我每天喝水？我喜欢奶茶！我喜欢咖啡?\n我习惯跑步。"])
    result = run_cli(chat, db, ledger)
    assert result.returncode == 0
    items = service.list_profile(USER)
    assert len(items) == 3
    snack = next(item for item in items if item.topic == "零食")
    assert snack.text == long[:-2][:60]
    assert snack.evidence_quote == long[:30]
    assert json.loads(result.stdout)["excluded_question"] == 2


def test_speech_report_is_excluded_but_novel_word_is_not(storage, tmp_path):
    service, db, ledger = storage
    chat = write_chat(
        tmp_path,
        [
            "我听朋友说她喜欢咖啡。",
            "我说过他喜欢奶茶。",
            "我喜欢咖啡小说。",
        ],
    )
    result = run_cli(chat, db, ledger)
    assert result.returncode == 0
    assert json.loads(result.stdout)["excluded_quote"] == 2
    (item,) = service.list_profile(USER)
    assert item.text == "我喜欢咖啡小说。"


@pytest.mark.parametrize(
    ("predicate", "category", "confidence"),
    [
        ("每天", "habit", "high"),
        ("经常", "habit", "med"),
        ("总是", "habit", "high"),
        ("老是", "habit", "med"),
        ("一般", "habit", "med"),
        ("从来不", "habit", "high"),
        ("很少", "habit", "med"),
        ("不爱", "preference", "med"),
        ("不喜欢", "preference", "med"),
        ("讨厌", "preference", "med"),
        ("爱", "preference", "med"),
        ("喜欢", "preference", "med"),
        ("习惯", "habit", "med"),
    ],
)
def test_predicate_categories_and_confidence(storage, tmp_path, predicate, category, confidence):
    service, db, ledger = storage
    chat = write_chat(tmp_path, [f"我{predicate}喝水。"])
    result = run_cli(chat, db, ledger)
    assert result.returncode == 0
    (item,) = service.list_profile(USER)
    assert item.category == category and item.confidence == confidence


@pytest.mark.parametrize(
    ("word", "topic"),
    [
        ("水", "喝水"),
        ("熬夜", "作息"),
        ("早睡", "作息"),
        ("起床", "作息"),
        ("睡觉", "作息"),
        ("运动", "运动"),
        ("健身", "运动"),
        ("早餐", "早餐"),
    ],
)
def test_topic_synonyms(storage, tmp_path, word, topic):
    service, db, ledger = storage
    result = run_cli(write_chat(tmp_path, [f"我喜欢{word}。"]), db, ledger)
    assert result.returncode == 0
    (item,) = service.list_profile(USER)
    assert item.topic == topic


@pytest.mark.parametrize(
    "content_type", ["image", "voice", "video", "emoji", "quote", "system", "other"]
)
def test_nontext_is_never_extracted_even_with_text(storage, tmp_path, content_type):
    service, db, ledger = storage
    chat = write_chat(tmp_path, ["我每天喝水。"], content_type=content_type)
    result = run_cli(chat, db, ledger)
    assert result.returncode == 0
    assert json.loads(result.stdout)["excluded_other"] == 1
    assert service.list_profile(USER) == ()


def test_different_user_same_text_does_not_block_candidate(storage, tmp_path):
    service, db, ledger = storage
    grant(service, user="synthetic-other")
    service.add_profile_candidate(
        user_id="synthetic-other",
        category="habit",
        topic="喝水",
        text="我每天喝水。",
        source="self_stated",
        evidence_date=datetime.now(UTC).date(),
        confidence="high",
    )
    result = run_cli(write_chat(tmp_path, ["我每天喝水。"]), db, ledger)
    assert result.returncode == 0
    assert json.loads(result.stdout)["candidates_added"] == 1
    assert len(service.list_profile(USER)) == 1


@pytest.mark.parametrize(
    "failure",
    ["invalid_chat", "invalid_ledger", "invalid_argument", "aliased_output", "unwritable_ledger"],
)
def test_errors_never_echo_source_or_arguments(storage, tmp_path, failure):
    _, db, ledger = storage
    chat = write_chat(tmp_path, ["我喜欢合成独特零食。"])
    extra = []
    if failure == "invalid_chat":
        records = json.loads(chat.read_text())
        records[0]["timestamp"] = records[0]["text"]
        chat.write_text(json.dumps(records))
    if failure == "invalid_ledger":
        ledger.write_text('["我喜欢合成独特零食"]')
    if failure == "invalid_argument":
        extra = ["--我喜欢合成独特零食"]
    if failure == "aliased_output":
        ledger = chat
    if failure == "unwritable_ledger":
        ledger.mkdir()
    before = db.read_bytes()
    result = run_cli(chat, db, ledger, *extra)
    assert result.returncode != 0
    assert result.stdout.strip() in {"failed", "invalid_arguments"}
    assert_safe(result, chat)
    assert db.read_bytes() == before


def test_script_has_no_network_imports():
    source = SCRIPT.read_text()
    assert all(term not in source for term in ("urllib", "requests", "http"))
