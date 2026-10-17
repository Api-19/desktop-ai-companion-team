import ssl
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from companion.bodylink.pairing import (
    PairingPaths,
    body_ssl_context,
    brain_ssl_context,
    generate_pairing,
)


def test_pairing_layout_and_permissions(pairing: PairingPaths) -> None:
    assert sorted(p.name for p in pairing.brain_dir.iterdir()) == [
        "brain.crt",
        "brain.key",
        "ca.crt",
    ]
    assert sorted(p.name for p in pairing.body_dir.iterdir()) == ["body.crt", "body.key", "ca.crt"]
    for directory in (pairing.brain_dir, pairing.body_dir):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        for path in directory.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # The CA private key is not kept anywhere in the output.
    root = pairing.brain_dir.parent
    assert sorted(p.name for p in root.iterdir()) == ["body", "brain"]


def test_contexts_require_tls13_and_verification(pairing: PairingPaths) -> None:
    server = brain_ssl_context(pairing.brain_dir)
    client = body_ssl_context(pairing.body_dir)
    for context in (server, client):
        assert context.minimum_version == ssl.TLSVersion.TLSv1_3
        assert context.verify_mode == ssl.CERT_REQUIRED
    assert client.check_hostname


def test_rejects_unsafe_body_id(tmp_path) -> None:
    with pytest.raises(ValueError):
        generate_pairing(tmp_path / "x", "body/CN=evil")


def test_refuses_to_overwrite_existing_pairing(pairing: PairingPaths) -> None:
    with pytest.raises(FileExistsError):
        generate_pairing(pairing.brain_dir.parent, "body-1")


def test_pair_script_writes_both_sides(tmp_path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "bodylink_pair.py"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path / "pair"), "--body-id", "pi-1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "brain files" in result.stdout
    assert (tmp_path / "pair" / "brain" / "brain.key").exists()
    assert (tmp_path / "pair" / "body" / "body.key").exists()
