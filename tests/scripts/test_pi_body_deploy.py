"""Deployment contract checks using dry-run and local synthetic commands only."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "pi_body"
DEPLOY = SCRIPTS / "deploy.sh"
WAIT = SCRIPTS / "wait_time_sync.sh"
TARGET = "kiki@kiki.local"


def fake_command(directory, name, body):
    path = directory / name
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body + "\n")
    path.chmod(0o755)
    return path


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pi-body-dry-run")
    home = tmp_path / "synthetic home 'quoted'"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "unexpected-command"
    for name in ("ssh", "rsync", "sudo", "mkdir", "chmod", "cat", "dirname", "pwd"):
        fake_command(bin_dir, name, 'printf called > "$UNEXPECTED_COMMAND"; exit 99')
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "UNEXPECTED_COMMAND": str(marker),
    }
    result = subprocess.run(
        ["/bin/bash", str(DEPLOY), "--dry-run", TARGET],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert not marker.exists(), "dry-run executed an external command"
    assert list(home.iterdir()) == [], "dry-run wrote files"
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert all(command[0] in {"ssh", "rsync", "printf"} for command in commands)
    return commands, home


def remote_commands(dry_run):
    commands, _ = dry_run
    return [command[-1] for command in commands if command[0] == "ssh"]


@pytest.mark.parametrize("script", [DEPLOY, WAIT], ids=["deploy", "wait"])
def test_shell_syntax(script):
    result = subprocess.run(["/bin/bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_generated_remote_shell_syntax(dry_run, tmp_path):
    # The third shell program is sent over SSH; never execute it on a real host.
    script = tmp_path / "remote.sh"
    script.write_text("\n".join(remote_commands(dry_run)))
    result = subprocess.run(["/bin/bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_order_and_ssh_options(dry_run):
    commands, home = dry_run
    assert commands[0][0] == commands[1][0] == "rsync"
    remote = remote_commands(dry_run)
    markers = [
        "chmod 0600",
        "apt-get install -y libportaudio2",
        "python3 -m venv --system-site-packages",
        "wait_time_sync.sh",
        "loginctl enable-linger",
        "systemctl --user daemon-reload",
        "sounddevice.query_devices()",
    ]
    assert len(remote) == len(markers)
    for command, marker in zip(remote, markers, strict=True):
        assert marker in command
    expected_ssh = ["ssh", "-i", str(home / ".ssh/id_ed25519"), "-o", "BatchMode=yes"]
    for command in commands:
        if command[0] == "ssh":
            assert command[:-2] == expected_ssh
            assert command[-2] == TARGET
        elif command[0] == "rsync":
            assert shlex.split(command[command.index("-e") + 1]) == expected_ssh
    assert DEPLOY.read_text().splitlines()[1] == "set -euo pipefail"
    assert commands[-1][0] == "printf"
    guidance = commands[-1][-1]
    for text in ("KIKI_BRAIN_HOST", "KIKI_INPUT_DEVICE", "KIKI_OUTPUT_DEVICE", "start"):
        assert text in guidance
    assert not any("systemctl --user start" in command for command in remote)


def test_only_source_and_body_certificates_are_synced(dry_run):
    commands, home = dry_run
    syncs = [command for command in commands if command[0] == "rsync"]
    assert len(syncs) == 2
    assert [Path(command[-2]).resolve() for command in syncs] == [
        ROOT / "src/companion",
        home / "Library/Application Support/kiki-companion/bodylink/body",
    ]
    assert syncs[0][-1] == f"{TARGET}:~/kiki-body/src/companion/"
    assert syncs[1][-1] == f"{TARGET}:~/.config/kiki-body/certs/"
    assert "--delete" in syncs[0]
    assert syncs[0][syncs[0].index("--exclude") + 1] == "__pycache__"
    for forbidden in (
        "private_data", "runtime_data", "test-results", "bodylink/brain", ".codex-tasks", ".venv"
    ):
        assert all(forbidden not in " ".join(command) for command in syncs)


def test_certificate_permissions(dry_run):
    commands, _ = dry_run
    cert_sync = commands[1]
    preparation = cert_sync[cert_sync.index("--rsync-path") + 1]
    assert 'chmod 0700 "$HOME/.config/kiki-body/certs"' in preparation
    permissions = remote_commands(dry_run)[0]
    assert 'chmod 0700 "$HOME/.config/kiki-body/certs"' in permissions
    assert '-type f -exec chmod 0600 {}' in permissions


def test_deployer_contains_no_api_key_transport():
    text = DEPLOY.read_text()
    for forbidden in ("security", "OPENAI", "KEY="):
        assert forbidden not in text


def test_environment_has_exact_public_fields():
    fields = dict(
        line.split("=", 1)
        for line in (SCRIPTS / "kiki-body.env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert fields == {
        "KIKI_BRAIN_HOST": "",
        "KIKI_BRAIN_PORT": "8765",
        "KIKI_BODY_ID": "kiki-body",
        "KIKI_INPUT_DEVICE": "",
        "KIKI_OUTPUT_DEVICE": "",
        "KIKI_HOSTAPI": "ALSA",
    }


def test_service_has_every_frozen_line():
    lines = (SCRIPTS / "kiki-body.service").read_text().splitlines()
    expected = [
        "[Unit]",
        "After=network-online.target",
        "[Service]",
        "EnvironmentFile=%h/.config/kiki-body/kiki-body.env",
        "Environment=PYTHONPATH=%h/kiki-body/src",
        "ExecStartPre=%h/kiki-body/wait_time_sync.sh",
        "ExecStart=%h/kiki-body/venv/bin/python -m companion.bodylink.body_main "
        "--brain-host ${KIKI_BRAIN_HOST} --brain-port ${KIKI_BRAIN_PORT} "
        "--body-id ${KIKI_BODY_ID} --cert-dir %h/.config/kiki-body/certs "
        "--input-device ${KIKI_INPUT_DEVICE} --output-device ${KIKI_OUTPUT_DEVICE} "
        "--hostapi ${KIKI_HOSTAPI} --button none",
        "Restart=on-failure",
        "RestartSec=5",
        "[Install]",
        "WantedBy=default.target",
    ]
    assert any(
        line.startswith("Description=") and line.removeprefix("Description=") for line in lines
    )
    assert all(line in lines for line in expected)
    # systemd's default 90-second start timeout must allow the full 120-second wait.
    assert "TimeoutStartSec=130" in lines


def test_venv_and_pinned_dependencies(dry_run):
    command = remote_commands(dry_run)[2]
    assert '[ ! -d "$HOME/kiki-body/venv" ]' in command
    assert "python3 -m venv --system-site-packages" in command
    assert '"$HOME/kiki-body/venv/bin/python" -m pip install' in command
    for package in (
        "pydantic==2.13.5", "websockets==15.0.1", "sounddevice==0.5.5", "samplerate==0.2.4"
    ):
        assert package in command


def test_remote_file_install_is_private_and_preserves_existing_env(dry_run, tmp_path):
    command = remote_commands(dry_run)[3]
    home = tmp_path / "remote home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home)}
    for existing in (False, True):
        if existing:
            (home / ".config/kiki-body/kiki-body.env").write_text("KIKI_BRAIN_HOST=synthetic\n")
        result = subprocess.run(
            ["/bin/bash", "-c", command], env=env, capture_output=True, text=True, timeout=2
        )
        assert result.returncode == 0, result.stderr
        for source, destination in (
            (WAIT, home / "kiki-body/wait_time_sync.sh"),
            (SCRIPTS / "kiki-body.service", home / ".config/systemd/user/kiki-body.service"),
        ):
            assert destination.read_text() == source.read_text()
        destination = home / ".config/kiki-body/kiki-body.env"
        assert destination.stat().st_mode & 0o777 == 0o600
        assert (home / "kiki-body/wait_time_sync.sh").stat().st_mode & 0o777 == 0o755
        expected = "KIKI_BRAIN_HOST=synthetic\n" if existing else (
            SCRIPTS / "kiki-body.env.example"
        ).read_text()
        assert destination.read_text() == expected


@pytest.mark.parametrize("succeeds", [False, True])
def test_portaudio_install_failure_has_exit_three(dry_run, succeeds):
    fake = 'sudo() { printf "%s\\n" "$*"; return "$SUDO_RESULT"; }; '
    result = subprocess.run(
        ["/bin/bash", "-c", fake + remote_commands(dry_run)[1]],
        env={
            **os.environ,
            "SUDO_RESULT": "0" if succeeds else "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if succeeds else 3)
    assert "-n apt-get install -y libportaudio2" in result.stdout
    if not succeeds:
        assert "手动" in result.stderr
        assert "sudo -n apt-get install -y libportaudio2" in result.stderr


@pytest.mark.parametrize("user_result,sudo_result,expected", [(0, 1, 0), (1, 0, 0), (1, 1, 4)])
def test_linger_fallback(dry_run, tmp_path, user_result, sudo_result, expected):
    log = tmp_path / "calls"
    # Local shell fakes avoid extra macOS process launches for these branch checks.
    fake = (
        'loginctl() { echo "loginctl $*" >> "$CALLS"; return "$USER_RESULT"; }; '
        'sudo() { echo "sudo $*" >> "$CALLS"; return "$SUDO_RESULT"; }; '
    )
    result = subprocess.run(
        ["/bin/bash", "-c", fake + remote_commands(dry_run)[4]],
        env={
            **os.environ,
            "USER": "synthetic-user",
            "CALLS": str(log),
            "USER_RESULT": str(user_result),
            "SUDO_RESULT": str(sudo_result),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected
    calls = ["loginctl enable-linger synthetic-user"]
    if user_result:
        calls.append("sudo -n loginctl enable-linger synthetic-user")
    assert log.read_text().splitlines() == calls
    if expected:
        assert "手动" in result.stderr


@pytest.mark.parametrize("yes_at,expected,calls", [(1, 0, 1), (3, 0, 3), (0, 1, None)])
def test_time_sync_with_fake_timedatectl(tmp_path, yes_at, expected, calls):
    counter = tmp_path / "counter"
    fake_command(
        tmp_path,
        "timedatectl",
        '[ "$*" = "show -p NTPSynchronized --value" ]\n'
        'count=0; if [ -f "$COUNTER" ]; then read -r count < "$COUNTER"; fi\n'
        'count=$((count + 1)); echo "$count" > "$COUNTER"\n'
        'if [ "$YES_AT" -gt 0 ] && [ "$count" -ge "$YES_AT" ]; '
        'then echo yes; else echo no; fi',
    )
    result = subprocess.run(
        ["/bin/bash", str(WAIT)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "COUNTER": str(counter),
            "YES_AT": str(yes_at),
            "KIKI_SYNC_TIMEOUT": "3" if yes_at else "1",
            "KIKI_SYNC_INTERVAL": "0.01",
        },
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert result.returncode == expected, result.stderr
    if calls:
        assert int(counter.read_text()) == calls
    else:
        assert int(counter.read_text()) >= 1
        assert len(result.stderr.splitlines()) == 1
        assert "超时" in result.stderr
    text = WAIT.read_text()
    assert "${KIKI_SYNC_TIMEOUT:-120}" in text
    assert "${KIKI_SYNC_INTERVAL:-2}" in text
