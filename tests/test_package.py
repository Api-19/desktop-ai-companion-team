from companion import __version__


def test_package_exposes_development_version() -> None:
    assert __version__ == "0.0.0.dev0"
