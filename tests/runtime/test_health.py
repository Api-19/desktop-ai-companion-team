from pathlib import Path

from companion.runtime.health import check_health
from companion.runtime.settings import Settings


def test_health_status_contains_only_local_runtime_facts(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        log_level="INFO",
        private_data_dir=tmp_path / "private",
        runtime_data_dir=tmp_path / "runtime",
        test_results_dir=tmp_path / "results",
    )
    status = check_health(settings)
    assert status.status == "healthy"
    assert status.environment == "test"
    assert status.python_version
    assert list(tmp_path.iterdir()) == []
