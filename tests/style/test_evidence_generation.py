"""Public synthetic tests for trusted Phase 5 source binding and generation."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from companion import style
from companion.style import StyleEvidenceRetrievalStatus
from companion.style.errors import StyleValidationError
from tests.style.test_profile_generation import package, state_for
from tests.style.test_style_evidence_retrieval import NOW, _catalog, _record, _snapshot


class Gateway:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def current_state(self, user_id):
        self.calls.append(user_id)
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


def inputs(texts=None, indices=(0, 1)):
    current = package(texts or ["selected synthetic phrase", "UNSELECTED_BODY_SENTINEL"])
    state = state_for(current, indices).model_copy(update={"owner_id": "owner-a"})
    original = _snapshot()
    consent = original.consent.model_copy(
        update={
            "scope": "non_text_style_features_local_persistence",
            "source_binding": state,
        }
    )
    snapshot = original.model_copy(update={"consent": consent})
    return current, state, snapshot


def generate(current, state, snapshot, **overrides):
    kwargs = dict(
        user_id="owner-a",
        package=current,
        source_state=state,
        snapshot=snapshot,
        selections=((0, "gratitude"),),
        observed_at=NOW,
    )
    kwargs.update(overrides)
    return style.generate_style_evidence_catalog(**kwargs)


def store(tmp_path, gateway, strict=True):
    return style.SingleTrustedUserStyleEvidenceStore(
        trusted_user_id="owner-a",
        private_data_dir=tmp_path,
        evidence_source=tmp_path / "evidence.json",
        index_source=tmp_path / "index.json",
        cache_source=tmp_path / "cache.json",
        version_source=tmp_path / "version.json",
        strict_source_binding=strict,
        source_gateway=gateway,
    )


def route(snapshot, text="谢谢你啦"):
    catalog = style.build_formal_style_scenario_catalog(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=snapshot,
        expected_catalog_version=style.FORMAL_STYLE_SCENARIO_CATALOG_VERSION,
    )
    delta = style.build_pragmatic_context_delta(
        user_id="owner-a",
        trusted_user_id="owner-a",
        text=text,
        recent_context=(),
        idle_seconds=None,
        observed_at=NOW,
    )
    return style.route_open_set_style(
        user_id="owner-a",
        trusted_user_id="owner-a",
        snapshot=snapshot,
        context_delta=delta,
        scenario_catalog=catalog,
    )


def test_generation_exact_selected_only_and_random_versions(tmp_path):
    current, state, snapshot = inputs()
    catalog = generate(current, state, snapshot)
    assert catalog.source_binding == state
    assert catalog.usage_scope == "approved_style_expressions_local_persistence"
    assert catalog.records[0].content == current.candidates[0].text
    assert catalog.records[0].candidate_index == 0
    assert catalog.records[0].confidence == 1.0
    assert "UNSELECTED_BODY_SENTINEL" not in catalog.model_dump_json()
    assert generate(current, state, snapshot).versions != catalog.versions
    gateway = Gateway(state)
    first = store(tmp_path, gateway)
    first.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=catalog)
    restarted = store(tmp_path, gateway)
    result = restarted.retrieve_evidence(user_id="owner-a", route=route(snapshot))
    assert result.status is StyleEvidenceRetrievalStatus.AVAILABLE
    assert result.matches[0].content == current.candidates[0].text
    assert "UNSELECTED_BODY_SENTINEL" not in result.model_dump_json()
    assert "content" not in result.source_manifest.model_dump_json()
    for path in first.artifact_paths[1:]:
        assert current.candidates[0].text not in path.read_text()


@pytest.mark.parametrize(
    "case",
    [
        "unapproved",
        "wrong_package",
        "revision",
        "legacy",
        "zero",
        "scene",
        "duplicate",
        "oversize",
        "naive",
        "early",
        "owner",
        "bool_index",
        "bounds",
        "too_many",
        "mutated_snapshot",
        "mutated_state",
    ],
)
def test_generation_rejects_invalid_inputs_with_fixed_error(case, recwarn, capsys):
    current, state, snapshot = inputs()
    overrides = {}
    if case == "unapproved":
        current, state, snapshot = inputs(indices=(1,))
    elif case == "wrong_package":
        current = package(["changed", "UNSELECTED_BODY_SENTINEL"])
    elif case == "revision":
        state = state.model_copy(update={"revision": "v1-" + "c" * 64})
    elif case == "legacy":
        snapshot = _snapshot()
    elif case == "zero":
        overrides["selections"] = ()
    elif case == "scene":
        overrides["selections"] = ((0, "unknown_scene"),)
    elif case == "duplicate":
        overrides["selections"] = ((0, "gratitude"), (0, "sleep_farewell"))
    elif case == "oversize":
        current, state, snapshot = inputs(["Z" * 161, "not selected"])
    elif case == "naive":
        overrides["observed_at"] = NOW.replace(tzinfo=None)
    elif case == "early":
        overrides["observed_at"] = NOW - timedelta(seconds=1)
    elif case == "owner":
        overrides["user_id"] = "other"
    elif case == "bool_index":
        overrides["selections"] = ((False, "gratitude"),)
    elif case == "bounds":
        overrides["selections"] = ((2, "gratitude"),)
    elif case == "too_many":
        overrides["selections"] = ((0, "gratitude"),) * 65
    elif case == "mutated_snapshot":
        snapshot = snapshot.model_copy(
            update={
                "consent": snapshot.consent.model_copy(
                    update={"status": "SYNTHETIC_INVALID_STATUS_SENTINEL"}
                )
            }
        )
    elif case == "mutated_state":
        state = state.model_copy(update={"approved_indices": (True,)})
    with pytest.raises(StyleValidationError) as caught:
        generate(current, state, snapshot, **overrides)
    assert caught.value.code == "style_evidence_generation_failed"
    assert "SENTINEL" not in str(caught.value)
    assert not recwarn
    assert "SENTINEL" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "change",
    [
        "batch_digest",
        "package_digest",
        "approved_indices",
        "revision",
        "missing",
        "error",
        "mutated",
    ],
)
def test_current_source_drift_rejects_retrieval_view_withdraw_but_delete_works(tmp_path, change):
    current, state, snapshot = inputs()
    gateway = Gateway(state)
    resource = store(tmp_path, gateway)
    catalog = generate(current, state, snapshot)
    resource.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=catalog)
    updates = {
        "batch_digest": "d" * 64,
        "package_digest": "e" * 64,
        "approved_indices": (0,),
        "revision": "v1-" + "f" * 64,
    }
    gateway.state = (
        None
        if change == "missing"
        else RuntimeError("private sentinel")
        if change == "error"
        else state.model_copy(update={"approved_indices": (True,)})
        if change == "mutated"
        else state.model_copy(update={change: updates[change]})
    )
    result = resource.retrieve_evidence(user_id="owner-a", route=route(snapshot))
    assert result.status is StyleEvidenceRetrievalStatus.CORRUPT
    assert not result.matches and result.source_manifest is None
    assert resource.view_approved_evidence("owner-a") is None
    with pytest.raises(StyleValidationError):
        resource.withdraw_evidence(
            user_id="owner-a",
            evidence_ref=catalog.records[0].evidence_ref,
            versions=catalog.versions,
            updated_at=NOW,
        )
    calls = len(gateway.calls)
    resource.delete_user_evidence("owner-a")
    assert len(gateway.calls) == calls
    assert all(not path.exists() for path in resource.artifact_paths)


def test_nonowner_and_open_set_do_not_probe_files_or_gateway(tmp_path, monkeypatch):
    _, state, snapshot = inputs()
    gateway = Gateway(state)
    resource = store(tmp_path, gateway)

    def forbidden(*args, **kwargs):
        pytest.fail("private probe")

    monkeypatch.setattr(resource, "_load_state", forbidden)
    for operation in [
        lambda: resource.retrieve_evidence(user_id="other", route=route(snapshot)),
        lambda: resource.view_approved_evidence("other"),
        lambda: resource.delete_user_evidence("other"),
        lambda: resource.publish_approved_evidence(
            user_id="other", snapshot=snapshot, catalog=None
        ),
        lambda: resource.withdraw_evidence(
            user_id="other", evidence_ref="any", versions=None, updated_at=NOW
        ),
    ]:
        with pytest.raises(StyleValidationError):
            operation()
    result = resource.retrieve_evidence(user_id="owner-a", route=route(snapshot, "计算量子公式"))
    assert result.status is StyleEvidenceRetrievalStatus.NOT_REQUESTED
    assert gateway.calls == []


def test_strict_requires_gateway_and_rejects_legacy_catalog(tmp_path):
    with pytest.raises(ValueError):
        store(tmp_path, None)
    _, state, snapshot = inputs()
    legacy = _catalog(_record("legacy-ref", "gratitude", "legacy text"))
    store(tmp_path, None, False).publish_approved_evidence(
        user_id="owner-a", snapshot=_snapshot(), catalog=legacy
    )
    strict = store(tmp_path, Gateway(state))
    assert strict.retrieve_evidence(user_id="owner-a", route=route(snapshot)).status is (
        StyleEvidenceRetrievalStatus.CORRUPT
    )
    with pytest.raises(StyleValidationError):
        strict.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=legacy)


def test_publication_rechecks_source_after_write(tmp_path, monkeypatch):
    current, state, snapshot = inputs()
    gateway = Gateway(state)
    resource = store(tmp_path, gateway)
    original = resource._publish_catalog

    def drift(catalog):
        original(catalog)
        gateway.state = state.model_copy(update={"revision": "v1-" + "d" * 64})

    monkeypatch.setattr(resource, "_publish_catalog", drift)
    with pytest.raises(StyleValidationError):
        resource.publish_approved_evidence(
            user_id="owner-a", snapshot=snapshot, catalog=generate(current, state, snapshot)
        )
    assert resource.view_approved_evidence("owner-a") is None


def test_selected_text_is_verbatim_and_private_binding_never_enters_directives(tmp_path):
    from companion.dialogue.personalization import (
        adapt_personalization_spec,
        compile_personalization_spec,
    )

    current, state, snapshot = inputs(["  synthetic selected phrase  ", "OTHER_BODY_SENTINEL"])
    legacy = _record("legacy-trim", "gratitude", current.candidates[0].text)
    assert legacy.content == current.candidates[0].text.strip()
    catalog = generate(current, state, snapshot)
    assert catalog.records[0].content == current.candidates[0].text
    gateway = Gateway(state)
    resource = store(tmp_path, gateway)
    resource.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=catalog)
    current_route = route(snapshot)
    result = resource.retrieve_evidence(user_id="owner-a", route=current_route)
    assert result.matches[0].content == current.candidates[0].text
    spec = compile_personalization_spec(
        user_id="owner-a", route=current_route, retrieval_result=result
    )
    directives = str(adapt_personalization_spec(user_id="owner-a", spec=spec))
    for private in (
        state.package_digest,
        state.batch_digest,
        state.revision,
        "candidate_index",
        "source_binding",
        "OTHER_BODY_SENTINEL",
    ):
        assert private not in directives


def test_withdraw_last_bound_record_deletes_four_artifacts(tmp_path):
    current, state, snapshot = inputs()
    resource = store(tmp_path, Gateway(state))
    catalog = generate(current, state, snapshot)
    resource.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=catalog)
    resource.withdraw_evidence(
        user_id="owner-a",
        evidence_ref=catalog.records[0].evidence_ref,
        versions=catalog.versions,
        updated_at=NOW,
    )
    assert all(not path.exists() for path in resource.artifact_paths)


def test_publish_source_or_snapshot_mismatch_creates_no_artifacts(tmp_path):
    current, state, snapshot = inputs()
    catalog = generate(current, state, snapshot)
    gateway = Gateway(state.model_copy(update={"revision": "v1-" + "f" * 64}))
    resource = store(tmp_path, gateway)
    with pytest.raises(StyleValidationError):
        resource.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=catalog)
    assert all(not path.exists() for path in resource.artifact_paths)
    gateway.state = state
    with pytest.raises(StyleValidationError):
        resource.publish_approved_evidence(user_id="owner-a", snapshot=_snapshot(), catalog=catalog)
    assert all(not path.exists() for path in resource.artifact_paths)


@pytest.mark.parametrize("change", ["scope", "binding", "index", "duplicate", "empty"])
def test_bound_catalog_pair_and_indices_are_deeply_validated(change):
    current, state, snapshot = inputs()
    catalog = generate(current, state, snapshot)
    payload = catalog.model_dump()
    if change == "scope":
        payload["usage_scope"] = "non_text_style_features_local_persistence"
    elif change == "binding":
        payload["source_binding"] = None
    elif change == "index":
        payload["records"][0]["candidate_index"] = True
    elif change == "duplicate":
        other = dict(payload["records"][0], evidence_ref="another-ref")
        payload["records"] += (other,)
    else:
        payload["records"] = ()
    with pytest.raises(ValidationError):
        style.StyleEvidenceCatalog.model_validate(payload)


def test_malformed_copied_snapshot_status_rejects_without_warning_or_output(capsys):
    import warnings

    current, state, snapshot = inputs()
    sentinel = "PUBLIC_SYNTHETIC_STATUS_SENTINEL"
    malformed = snapshot.model_copy(
        update={"consent": snapshot.consent.model_copy(update={"status": sentinel})}
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(StyleValidationError) as caught:
            generate(current, state, malformed)
    assert caught.value.code == "style_evidence_generation_failed"
    assert caught.value.__cause__ is None
    assert sentinel not in str(caught.value) and sentinel not in repr(caught.value)
    assert caught_warnings == []
    output = capsys.readouterr()
    assert output.out == "" and output.err == ""


@pytest.mark.parametrize("case", ["snapshot", "catalog", "gateway", "route", "versions"])
def test_malformed_store_inputs_fail_closed_without_warning_or_output(tmp_path, case, capsys):
    import warnings

    sentinel = "PUBLIC_SYNTHETIC_STORE_SENTINEL"
    current, state, snapshot = inputs()
    catalog = generate(current, state, snapshot)
    gateway = Gateway(state)
    resource = store(tmp_path, gateway)
    if case != "snapshot" and case != "catalog":
        resource.publish_approved_evidence(user_id="owner-a", snapshot=snapshot, catalog=catalog)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if case == "snapshot":
            malformed = snapshot.model_copy(
                update={"consent": snapshot.consent.model_copy(update={"status": sentinel})}
            )
            with pytest.raises(StyleValidationError) as caught:
                resource.publish_approved_evidence(
                    user_id="owner-a", snapshot=malformed, catalog=catalog
                )
        elif case == "catalog":
            malformed = catalog.model_copy(update={"usage_scope": sentinel})
            with pytest.raises(StyleValidationError) as caught:
                resource.publish_approved_evidence(
                    user_id="owner-a", snapshot=snapshot, catalog=malformed
                )
        elif case == "gateway":
            gateway.state = state.model_copy(update={"owner_id": [sentinel]})
            assert resource.view_approved_evidence("owner-a") is None
            caught = None
        elif case == "route":
            malformed = route(snapshot).model_copy(update={"route_kind": sentinel})
            with pytest.raises(StyleValidationError) as caught:
                resource.retrieve_evidence(user_id="owner-a", route=malformed)
        else:
            malformed = catalog.versions.model_copy(update={"evidence_version": [sentinel]})
            with pytest.raises(StyleValidationError) as caught:
                resource.withdraw_evidence(
                    user_id="owner-a",
                    evidence_ref=catalog.records[0].evidence_ref,
                    versions=malformed,
                    updated_at=NOW,
                )
    if caught is not None:
        assert sentinel not in str(caught.value) and sentinel not in repr(caught.value)
    assert caught_warnings == []
    output = capsys.readouterr()
    assert output.out == "" and output.err == ""
    if case in ("snapshot", "catalog"):
        assert all(not path.exists() for path in resource.artifact_paths)
