"""Frozen partner profile rules, using only synthetic identities and content."""

import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from companion.contracts import ConsentRecord
from companion.memory import (
    CreateMemoryRequest,
    MemoryAuthorizationError,
    MemoryService,
    MemoryValidationError,
)
from companion.memory.profile import (
    PROFILE_PERMISSION,
    ProfileCategory,
    ProfileItem,
    ProfileSource,
    ProfileStatus,
    ProfileView,
)

A = "synthetic-profile-a"
B = "synthetic-profile-b"
NOW = datetime(2026, 1, 31, 12, tzinfo=UTC)


@pytest.fixture
def profile_service(tmp_path):
    clock = [NOW]
    path = tmp_path / "profiles.sqlite3"
    service = MemoryService(path, clock=lambda: clock[0])
    return service, clock, path


def grant(service, user=A, fields=(PROFILE_PERMISSION,)):
    return service.grant_consent(
        ConsentRecord(user_id=user, granted=True, granted_at=NOW, allowed_fields=list(fields))
    )


def candidate(service, user=A, **changes):
    values = dict(
        user_id=user,
        category="habit",
        topic="喝水",
        text="习惯少喝水",
        source="self_stated",
        evidence_date=NOW.date(),
        evidence_quote="合成:我不爱喝水",
        confidence="high",
    )
    return service.add_profile_candidate(**(values | changes))


def confirm(service, item, **changes):
    service.approve_profile(user_id=item.user_id, profile_id=item.profile_id)
    return service.confirm_profile(
        **(
            dict(
                user_id=item.user_id,
                profile_id=item.profile_id,
                version=item.version,
                subject_id=item.user_id,
                use_in_chat=True,
                simulated=True,
            )
            | changes
        )
    )


def retrieve(service, user=A, **changes):
    return service.retrieve_profile(**(dict(user_id=user, query_texts=["记得喝水"]) | changes))


def test_permission_is_separate_and_no_unauthorized_row(profile_service):
    service, _, path = profile_service
    with pytest.raises(MemoryAuthorizationError):
        candidate(service)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    grant(service, fields=("preference",))
    with pytest.raises(MemoryAuthorizationError):
        candidate(service)
    with pytest.raises(ValidationError):
        CreateMemoryRequest(
            user_id=A,
            field="profile",
            value="合成",
            source_timestamp=NOW,
            explicitly_requested=True,
        )
    grant(service)
    assert candidate(service).status == ProfileStatus.CANDIDATE


@pytest.mark.parametrize("operation", ["approve", "edit", "disable", "delete", "confirm"])
def test_cross_user_ids_cannot_be_operated_on(profile_service, operation):
    service, _, _ = profile_service
    grant(service)
    grant(service, B)
    item = candidate(service)
    kwargs = dict(user_id=B, profile_id=item.profile_id)
    kwargs.update(
        {
            "approve": {},
            "edit": {"text": "合成新内容"},
            "disable": {"reason": "owner"},
            "delete": {},
            "confirm": dict(version=1, subject_id=B, use_in_chat=True, simulated=True),
        }[operation]
    )
    with pytest.raises(MemoryAuthorizationError):
        getattr(service, operation + "_profile")(**kwargs)
    assert service.list_profile(A) == (item,)
    assert service.list_profile(B) == ()
    confirm(service, item)
    assert retrieve(service, B) == ()
    assert len(retrieve(service)) == 1


def test_states_confirmation_version_and_projection(profile_service):
    service, clock, _ = profile_service
    grant(service)
    item = candidate(service)
    assert retrieve(service) == ()
    approved = service.approve_profile(user_id=A, profile_id=item.profile_id)
    assert approved.status == ProfileStatus.OWNER_APPROVED
    assert retrieve(service) == ()
    kwargs = dict(
        user_id=A,
        profile_id=item.profile_id,
        version=1,
        subject_id=A,
        use_in_chat=True,
        simulated=True,
    )
    with pytest.raises(MemoryAuthorizationError):
        service.confirm_profile(**(kwargs | {"subject_id": B}))
    with pytest.raises(MemoryValidationError):
        service.confirm_profile(**(kwargs | {"version": 2}))
    confirmed = service.confirm_profile(**(kwargs | {"use_in_chat": False}))
    assert confirmed.evidence_quote is None
    assert confirmed.confirmed_by == A and confirmed.confirmed_version == 1
    assert confirmed.simulated is True and confirmed.confirmed_at == NOW
    assert retrieve(service) == ()
    service.confirm_profile(**kwargs)
    views = retrieve(service)
    assert views == (ProfileView(topic="喝水", text="习惯少喝水", category="habit"),)
    assert type(views[0]) is ProfileView
    assert set(views[0].model_dump()) == {"topic", "text", "category"}
    clock[0] += timedelta(days=1)
    edited = service.edit_profile(user_id=A, profile_id=item.profile_id, text="习惯多喝水")
    assert edited.version == 2 and edited.status == ProfileStatus.OWNER_APPROVED
    assert edited.evidence_quote is None
    assert edited.confirmed_at is None and edited.confirmed_by is None
    assert edited.confirmed_version is None and edited.use_in_chat is False
    assert edited.simulated is False
    assert edited.evidence_date == NOW.date()
    assert retrieve(service) == ()
    with pytest.raises(MemoryValidationError):
        service.confirm_profile(**kwargs)
    service.confirm_profile(**(kwargs | {"version": 2}))
    assert len(retrieve(service)) == 1
    disabled = service.disable_profile(user_id=A, profile_id=item.profile_id, reason="self")
    assert disabled.status == ProfileStatus.DISABLED and disabled.disabled_reason == "self"
    assert retrieve(service) == ()
    assert service.list_profile(A) == (disabled,)


@pytest.mark.parametrize("stage", ["candidate", "disabled"])
def test_confirmation_requires_reviewed_active_state(profile_service, stage):
    service, _, _ = profile_service
    grant(service)
    item = candidate(service)
    if stage == "disabled":
        service.disable_profile(user_id=A, profile_id=item.profile_id, reason="owner")
    with pytest.raises(MemoryValidationError):
        service.confirm_profile(
            user_id=A,
            profile_id=item.profile_id,
            version=1,
            subject_id=A,
            use_in_chat=True,
            simulated=True,
        )


@pytest.mark.parametrize(
    "start,end",
    [
        (date(2026, 1, 31), date(2026, 5, 31)),
        (date(2025, 10, 31), date(2026, 2, 28)),
    ],
)
def test_calendar_month_expiry_boundary_and_cleanup(profile_service, start, end):
    service, clock, _ = profile_service
    clock[0] = datetime.combine(start, datetime.min.time(), UTC)
    grant(service)
    item = confirm(service, candidate(service, evidence_date=start))
    assert item.valid_until == end
    clock[0] = datetime.combine(end, datetime.min.time(), UTC) - timedelta(microseconds=1)
    assert len(retrieve(service)) == 1
    clock[0] += timedelta(microseconds=1)
    assert retrieve(service) == ()
    expired = service.list_profile(A)[0]
    assert expired.status == ProfileStatus.DISABLED and expired.disabled_reason == "expired"
    clock[0] += timedelta(days=30) - timedelta(microseconds=1)
    assert service.list_profile(A)
    clock[0] += timedelta(microseconds=1)
    assert service.list_profile(A) == ()


def test_reconfirmation_renews_without_changing_evidence_date(profile_service):
    service, clock, _ = profile_service
    grant(service)
    item = confirm(service, candidate(service, source="owner_added"))
    clock[0] = datetime(2026, 3, 31, tzinfo=UTC)
    renewed = service.confirm_profile(
        user_id=A,
        profile_id=item.profile_id,
        version=1,
        subject_id=A,
        use_in_chat=True,
        simulated=True,
    )
    assert renewed.valid_until == date(2026, 7, 31)
    assert renewed.evidence_date == item.evidence_date
    assert renewed.source == ProfileSource.OWNER_ADDED
    future = candidate(service, topic="作息", evidence_date=date(2026, 4, 30))
    assert confirm(service, future).valid_until == date(2026, 8, 30)


def test_conflict_disables_all_confirmed_topic_items_only_for_this_user(profile_service):
    service, _, _ = profile_service
    grant(service)
    grant(service, B)
    first = confirm(service, candidate(service))
    same = confirm(service, candidate(service))
    other = confirm(service, candidate(service, B, text="习惯多喝水"))
    assert len(retrieve(service)) == 2
    different = confirm(service, candidate(service, text="习惯多喝水"))
    assert different.disabled_reason == "conflict"
    assert retrieve(service) == ()
    assert {x.profile_id for x in service.list_profile(A)} == {
        first.profile_id,
        same.profile_id,
        different.profile_id,
    }
    assert all(
        x.status == ProfileStatus.DISABLED and x.disabled_reason == "conflict"
        for x in service.list_profile(A)
    )
    assert service.list_profile(B) == (other,)


@pytest.mark.parametrize("stage", ["candidate", "disabled"])
def test_cleanup_after_exactly_30_days_on_ordinary_service_call(profile_service, stage):
    service, clock, path = profile_service
    grant(service)
    item = candidate(service)
    if stage == "disabled":
        clock[0] += timedelta(days=5)
        service.disable_profile(user_id=A, profile_id=item.profile_id, reason="owner")
    clock[0] += timedelta(days=30) - timedelta(microseconds=1)
    assert len(service.list_profile(A)) == 1
    clock[0] += timedelta(microseconds=1)
    service.get_consent(A)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0


def test_edit_disabled_does_not_extend_retention(profile_service):
    service, clock, _ = profile_service
    grant(service)
    item = candidate(service)
    service.disable_profile(user_id=A, profile_id=item.profile_id, reason="owner")
    clock[0] += timedelta(days=29)
    service.edit_profile(user_id=A, profile_id=item.profile_id, topic="饮水")
    clock[0] += timedelta(days=1)
    assert service.list_profile(A) == ()


def test_consent_narrowing_revocation_and_restart(profile_service):
    service, clock, path = profile_service
    grant(service, fields=("profile", "preference", "explicit_note"))
    candidate(service)
    for field in ["preference", "explicit_note"]:
        service.create_memory(
            CreateMemoryRequest(
                user_id=A,
                field=field,
                value="合成普通记忆",
                source_timestamp=NOW,
                explicitly_requested=True,
            )
        )
    grant(service, fields=("preference",))
    assert service.list_profile(A) == ()
    assert [m.field for m in service.list_memories(A)] == ["preference"]
    restarted = MemoryService(path, clock=lambda: clock[0])
    assert restarted.list_profile(A) == ()
    grant(restarted, fields=("profile", "preference"))
    candidate(restarted)
    assert restarted.revoke_consent(A).deleted_count == 2
    restarted = MemoryService(path, clock=lambda: clock[0])
    assert restarted.list_profile(A) == () and restarted.list_memories(A) == ()
    assert retrieve(restarted) == ()


def test_single_delete_is_immediate_and_user_scoped(profile_service):
    service, _, path = profile_service
    grant(service)
    item = confirm(service, candidate(service))
    receipt = service.delete_profile(user_id=A, profile_id=item.profile_id)
    assert receipt.user_id == A and receipt.deleted_count == 1 and receipt.completed_at == NOW
    assert retrieve(service) == ()
    assert MemoryService(path, clock=lambda: NOW).list_profile(A) == ()


def test_overlap_ranking_limit_and_management_order(profile_service):
    service, clock, _ = profile_service
    grant(service)
    for topic, text in [
        ("饮水", "多喝白水"),
        ("喝水", "喝水时间"),
        ("水杯", "喝水时间水杯"),
        ("提醒", "请喝水时间水杯"),
    ]:
        confirm(service, candidate(service, topic=topic, text=text))
        clock[0] += timedelta(seconds=1)
    assert [x.topic for x in service.list_profile(A)] == ["饮水", "喝水", "水杯", "提醒"]
    assert retrieve(service, query_texts=["完全无交集"]) == ()
    assert [x.topic for x in retrieve(service, query_texts=["喝水时间水杯"], limit=10)] == [
        "水杯",
        "提醒",
        "喝水",
    ]
    assert len(retrieve(service, limit=1)) == 1
    assert retrieve(service, limit=0) == ()
    assert retrieve(service, query_texts=[]) == ()
    with pytest.raises(MemoryValidationError):
        retrieve(service, query_texts="喝水")
    with pytest.raises(MemoryValidationError):
        retrieve(service, limit=-1)


@pytest.mark.parametrize(
    "changes",
    [
        {"topic": ""},
        {"topic": "水" * 13},
        {"text": ""},
        {"text": "水" * 61},
        {"evidence_quote": "水" * 31},
        {"category": "disease"},
        {"source": "inferred"},
        {"confidence": "certain"},
    ],
)
def test_profile_input_bounds(profile_service, changes):
    service, _, _ = profile_service
    grant(service)
    with pytest.raises(MemoryValidationError):
        candidate(service, **changes)
    assert service.list_profile(A) == ()


def test_types_and_all_edit_fields(profile_service):
    service, _, _ = profile_service
    grant(service)
    item = candidate(service, topic="水" * 12, text="水" * 60, evidence_quote="水" * 30)
    assert type(item) is ProfileItem
    assert item.category is ProfileCategory.HABIT
    confirm(service, item)
    for field, value in [
        ("category", "preference"),
        ("topic", "饮水"),
        ("text", "合成爱喝水"),
        ("source", "owner_added"),
    ]:
        item = service.edit_profile(user_id=A, profile_id=item.profile_id, **{field: value})
        assert item.confirmed_at is None and item.status == ProfileStatus.OWNER_APPROVED
        service.confirm_profile(
            user_id=A,
            profile_id=item.profile_id,
            version=item.version,
            subject_id=A,
            use_in_chat=True,
            simulated=True,
        )
    assert item.version == 5


def test_owner_added_provenance_cannot_be_relabelled(profile_service):
    service, _, _ = profile_service
    grant(service)
    item = candidate(service, source="owner_added")
    with pytest.raises(MemoryValidationError):
        service.edit_profile(user_id=A, profile_id=item.profile_id, source="self_stated")
    assert service.list_profile(A) == (item,)


def test_simulated_confirmation_cannot_migrate_even_after_edit_and_restart(profile_service):
    service, clock, path = profile_service
    grant(service)
    item = confirm(service, candidate(service))
    kwargs = dict(
        user_id=A,
        profile_id=item.profile_id,
        version=1,
        subject_id=A,
        use_in_chat=True,
        simulated=False,
    )
    with pytest.raises(MemoryValidationError):
        service.confirm_profile(**kwargs)
    service.edit_profile(user_id=A, profile_id=item.profile_id, text="合成多喝水")
    restarted = MemoryService(path, clock=lambda: clock[0])
    with pytest.raises(MemoryValidationError):
        restarted.confirm_profile(**(kwargs | {"version": 2}))
    assert restarted.list_profile(A)[0].status == ProfileStatus.OWNER_APPROVED


def test_stale_confirmed_version_not_retrieved(profile_service):
    service, _, path = profile_service
    grant(service)
    item = confirm(service, candidate(service))
    # Synthetic stored record models a stale confirmation, independently of the edit operation.
    stale = item.model_copy(update={"version": 2})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE profiles SET payload = ? WHERE user_id = ? AND profile_id = ?",
            (stale.model_dump_json(), A, item.profile_id),
        )
    assert retrieve(service) == ()


def test_confirmed_quote_is_removed_from_persisted_database(profile_service):
    service, _, path = profile_service
    grant(service)
    marker = "synthetic-evidence-marker"
    item = candidate(service, evidence_quote=marker)
    assert marker.encode() in path.read_bytes()
    confirm(service, item)
    assert marker.encode() not in path.read_bytes()
    for suffix in ["-wal", "-journal", "-shm"]:
        artifact = path.with_name(path.name + suffix)
        if artifact.exists():
            assert marker.encode() not in artifact.read_bytes()


def test_narrowing_ordinary_fields_preserves_profile_and_other_user(profile_service):
    service, _, _ = profile_service
    grant(service, fields=("profile", "preference"))
    grant(service, B)
    item = candidate(service)
    other = candidate(service, B)
    service.create_memory(
        CreateMemoryRequest(
            user_id=A,
            field="preference",
            value="合成偏好",
            source_timestamp=NOW,
            explicitly_requested=True,
        )
    )
    grant(service)
    assert service.list_memories(A) == ()
    assert service.list_profile(A) == (item,)
    grant(service, fields=())
    assert service.list_profile(A) == ()
    assert service.list_profile(B) == (other,)
