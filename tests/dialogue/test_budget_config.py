from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dialogue_support import synthetic_dialogue_config
from test_config import make_settings

from companion.dialogue import DialogueConfigError, load_budget_config


def test_budget_only_loader_ignores_environment_and_other_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    folder = settings.private_data_dir / "dialogue"
    folder.mkdir(parents=True)
    expected = synthetic_dialogue_config(tmp_path).budget
    (folder / "runtime-config.json").write_text(
        json.dumps(
            {
                "budget": expected.model_dump(mode="json"),
                "provider": "broken",
                "style": "synthetic-private-sentinel",
            }
        )
    )
    import companion.dialogue.config as config

    class NoEnvironment:
        def __getitem__(self, key):
            raise AssertionError("environment read")

        def __iter__(self):
            raise AssertionError("environment read")

    monkeypatch.setattr(
        config,
        "os",
        SimpleNamespace(
            environ=NoEnvironment(),
            open=os.open,
            close=os.close,
            fstat=os.fstat,
            fdopen=os.fdopen,
            O_RDONLY=os.O_RDONLY,
            O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 0),
        ),
    )
    assert load_budget_config(settings) == expected


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "json",
        "budget_missing",
        "budget_invalid",
        "file_link",
        "folder_link",
        "root_link",
        "ancestor_link",
    ],
)
def test_budget_only_loader_fixed_failure_and_symlink_refusal(tmp_path: Path, fault: str) -> None:
    settings = make_settings(tmp_path)
    folder = settings.private_data_dir / "dialogue"
    folder.mkdir(parents=True)
    path = folder / "runtime-config.json"
    budget = synthetic_dialogue_config(tmp_path).budget.model_dump(mode="json")
    payload = {"budget": budget}
    if fault == "budget_missing":
        payload = {"synthetic-private-sentinel": "irrelevant"}
    if fault == "budget_invalid":
        budget["monthly_cap_cny"] = "synthetic-private-sentinel"
    if fault != "missing":
        path.write_text("synthetic-private-sentinel" if fault == "json" else json.dumps(payload))
    if fault == "file_link":
        target = folder / "synthetic-other.json"
        path.rename(target)
        path.symlink_to(target)
    elif fault == "folder_link":
        target = settings.private_data_dir / "synthetic-other"
        folder.rename(target)
        folder.symlink_to(target, target_is_directory=True)
    elif fault in {"root_link", "ancestor_link"}:
        target = tmp_path / "synthetic-alias"
        target.symlink_to(settings.private_data_dir, target_is_directory=True)
        settings = settings.model_copy(
            update={"private_data_dir": target if fault == "root_link" else target / "child"}
        )
    with pytest.raises(
        DialogueConfigError, match=r"^dialogue configuration unavailable$"
    ) as caught:
        load_budget_config(settings)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    tb = caught.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_filename.endswith("dialogue/config.py"):
            assert not any(
                "synthetic-private-sentinel" in repr(v) or str(tmp_path) in repr(v)
                for v in tb.tb_frame.f_locals.values()
            )
        tb = tb.tb_next
