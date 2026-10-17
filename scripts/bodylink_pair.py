"""Generate the fixed body<->brain pairing certificates (uses the system openssl CLI).

Usage: python scripts/bodylink_pair.py --out <new dir outside the repo> --body-id <id>

Writes <out>/brain (keep on the Mac) and <out>/body (copy to the Pi). The CA private key is
not kept; to re-pair, generate a fresh set and replace both sides.
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from companion.bodylink.pairing import generate_pairing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--body-id", required=True)
    args = parser.parse_args(argv)
    paths = generate_pairing(args.out, args.body_id)
    print(f"brain files: {paths.brain_dir}")
    print(f"body files:  {paths.body_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
