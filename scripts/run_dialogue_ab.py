from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from companion.dialogue import (
    BudgetExceeded,
    UsageLedgerError,
    UsageReviewRequired,
    build_real_orchestrator,
    load_ab_scenarios,
    load_dialogue_config,
    load_style_refresh_scenarios,
    run_ab_suite,
    run_baseline_smoke,
)
from companion.dialogue.ab import DIALOGUE_EVALUATION_USER_ID, load_framework20_scenarios
from companion.runtime.settings import load_settings


def main(
    suite: Literal["legacy", "style_refresh", "framework20"] = "legacy",
    *,
    enable_production_personalization: bool = False,
    preflight_only: bool = False,
) -> int:
    framework20 = suite == "framework20"
    if framework20 != enable_production_personalization or (preflight_only and not framework20):
        print('{"status":"configuration_unavailable"}')
        return 2
    try:
        settings = load_settings(Path.cwd())
        config, secrets = load_dialogue_config(settings)
        if framework20:
            owner = config.style_framework_owner_id
            if type(owner) is not str:
                raise ValueError
            orchestrator = build_real_orchestrator(
                settings,
                config,
                secrets,
                trusted_user_id=owner,
                enable_production_personalization=True,
            )
        else:
            orchestrator = build_real_orchestrator(
                settings,
                config,
                secrets,
                trusted_user_id=DIALOGUE_EVALUATION_USER_ID,
            )
    except Exception:
        print('{"status":"configuration_unavailable"}')
        return 2

    if framework20:
        try:
            scenarios = load_framework20_scenarios(Path("config/dialogue-ab-full-scenarios.json"))
            orchestrator.ensure_paid_capacity(41)
            orchestrator.preflight_framework20(
                scenarios,
                trusted_user_id=owner,
                results_dir=settings.test_results_dir,
            )
        except BudgetExceeded:
            print('{"status":"budget_exceeded"}')
            return 1
        except UsageReviewRequired:
            print('{"status":"usage_review_required"}')
            return 2
        except UsageLedgerError:
            print('{"status":"usage_unavailable"}')
            return 2
        except Exception:
            print('{"status":"preflight_unavailable"}')
            return 2
        if preflight_only:
            print('{"status":"preflight_ready","scenario_count":20,"provider_attempt_count":0}')
            return 0
        try:
            smoke = run_baseline_smoke(
                orchestrator,
                settings.test_results_dir,
                trusted_user_id=owner,
            )
            if not smoke.succeeded:
                print(smoke.model_dump_json())
                return 1
            summary = run_ab_suite(
                orchestrator,
                scenarios,
                settings.test_results_dir,
                trusted_user_id=owner,
            )
        except Exception:
            print('{"status":"run_failed"}')
            return 1
        print(summary.model_dump_json())
        return (
            0
            if (
                summary.scenario_count == 20
                and summary.paired_count == 20
                and summary.request_attempt_count == 40
                and summary.failed_count == summary.blocked_count == 0
            )
            else 1
        )

    try:
        orchestrator.ensure_paid_capacity(15 if suite == "style_refresh" else 13)
    except BudgetExceeded:
        print('{"status":"budget_exceeded"}')
        return 1
    except UsageReviewRequired:
        print('{"status":"usage_review_required"}')
        return 2
    except UsageLedgerError:
        print('{"status":"usage_unavailable"}')
        return 2
    except Exception:
        print('{"status":"usage_unavailable"}')
        return 2

    try:
        smoke = run_baseline_smoke(orchestrator, settings.test_results_dir)
        if not smoke.succeeded:
            print(smoke.model_dump_json())
            return 1
        if suite == "style_refresh":
            scenarios = load_style_refresh_scenarios(
                Path("config/dialogue-ab-full-scenarios.json"),
                Path("config/dialogue-ab-style-refresh-scenarios.json"),
            )
        else:
            scenarios = load_ab_scenarios(Path("config/dialogue-ab-scenarios.json"))
        summary = run_ab_suite(orchestrator, scenarios, settings.test_results_dir)
    except Exception:
        print('{"status":"run_failed"}')
        return 1

    print(summary.model_dump_json())
    return 0 if summary.failed_count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite", choices=("legacy", "style_refresh", "framework20"), nargs="?", default="legacy"
    )
    parser.add_argument("--enable-production-personalization", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(
        main(
            arguments.suite,
            enable_production_personalization=arguments.enable_production_personalization,
            preflight_only=arguments.preflight_only,
        )
    )
