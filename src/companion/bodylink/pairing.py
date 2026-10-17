"""Fixed body<->brain pairing: a private CA plus one brain and one body certificate.

Certificates are generated with the system ``openssl`` command line tool. The CA private
key is deleted after signing, so re-pairing means generating a new set.
"""

import os
import re
import secrets
import shutil
import ssl
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

BRAIN_SERVER_NAME = "companion-brain"
CA_NAME = "companion-bodylink-ca"
CERT_DAYS = 3650
_BODY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_OPENSSL_CONFIG = f"""\
[req]
distinguished_name = dn

[dn]

[v3_ca]
basicConstraints = critical,CA:TRUE,pathlen:0
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash

[v3_brain]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = serverAuth
subjectAltName = DNS:{BRAIN_SERVER_NAME}
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid

[v3_body]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = clientAuth
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid
"""


@dataclass(frozen=True)
class PairingPaths:
    brain_dir: Path
    body_dir: Path


def generate_pairing(out_dir: Path, body_id: str, *, openssl: str = "openssl") -> PairingPaths:
    """Write ``out_dir/brain`` (ca.crt, brain.crt, brain.key) and ``out_dir/body``
    (ca.crt, body.crt, body.key). Directories are 0700, files 0600."""
    if not _BODY_ID_PATTERN.fullmatch(body_id):
        raise ValueError("body_id must match [A-Za-z0-9_-]{1,64}")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    brain_dir = out_dir / "brain"
    body_dir = out_dir / "body"
    brain_dir.mkdir(mode=0o700)
    body_dir.mkdir(mode=0o700)

    with tempfile.TemporaryDirectory(dir=out_dir) as scratch_name:
        scratch = Path(scratch_name)
        config = scratch / "openssl.cnf"
        config.write_text(_OPENSSL_CONFIG, encoding="ascii")

        def run(*args: str) -> None:
            subprocess.run([openssl, *args], check=True, capture_output=True)

        def new_key(path: Path) -> None:
            run("ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(path))

        ca_key, ca_crt = scratch / "ca.key", brain_dir / "ca.crt"
        new_key(ca_key)
        run(
            "req", "-x509", "-new", "-sha256", "-config", str(config),
            "-key", str(ca_key), "-subj", f"/CN={CA_NAME}", "-days", str(CERT_DAYS),
            "-extensions", "v3_ca", "-set_serial", str(_serial()), "-out", str(ca_crt),
        )  # fmt: skip

        def issue(name: str, cn: str, extensions: str, target: Path) -> None:
            key, csr = target / f"{name}.key", scratch / f"{name}.csr"
            new_key(key)
            run(
                "req", "-new", "-sha256", "-config", str(config),
                "-key", str(key), "-subj", f"/CN={cn}", "-out", str(csr),
            )  # fmt: skip
            run(
                "x509", "-req", "-sha256", "-in", str(csr),
                "-CA", str(ca_crt), "-CAkey", str(ca_key), "-set_serial", str(_serial()),
                "-days", str(CERT_DAYS), "-extfile", str(config), "-extensions", extensions,
                "-out", str(target / f"{name}.crt"),
            )  # fmt: skip

        issue("brain", BRAIN_SERVER_NAME, "v3_brain", brain_dir)
        issue("body", body_id, "v3_body", body_dir)
        shutil.copyfile(ca_crt, body_dir / "ca.crt")

    for directory in (brain_dir, body_dir):
        for path in directory.iterdir():
            path.chmod(0o600)
    return PairingPaths(brain_dir=brain_dir, body_dir=body_dir)


def _serial() -> int:
    return secrets.randbits(63) | 1


def brain_ssl_context(brain_dir: Path) -> ssl.SSLContext:
    """TLS 1.3 server context that requires a client certificate signed by the pairing CA."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=os.fspath(brain_dir / "ca.crt"))
    context.load_cert_chain(brain_dir / "brain.crt", brain_dir / "brain.key")
    return context


def body_ssl_context(body_dir: Path) -> ssl.SSLContext:
    """TLS 1.3 client context that verifies the brain against the pairing CA and
    ``BRAIN_SERVER_NAME``."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_verify_locations(cafile=os.fspath(body_dir / "ca.crt"))
    context.load_cert_chain(body_dir / "body.crt", body_dir / "body.key")
    return context
