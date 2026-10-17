from __future__ import annotations

from pathlib import Path

from companion.dialogue import load_dialogue_config, preflight_style_refresh
from companion.runtime.settings import load_settings


def main() -> int:
    try:
        settings = load_settings(Path.cwd())
        config, _ = load_dialogue_config(settings)
        summary = preflight_style_refresh(settings, config)
    except Exception:
        print('{"status":"not_ready","provider_requests":0}')
        return 2
    print(summary.model_dump_json())
    return 0 if summary.status == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
