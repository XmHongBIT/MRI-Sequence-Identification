from pathlib import Path


def test_repository_files_exist() -> None:
    root = Path(__file__).parents[1]
    assert (root / "README.md").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/mri_sequence_identification/cli.py").is_file()


def test_cli_help_is_import_light() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from mri_sequence_identification.cli import build_parser

    help_text = build_parser().format_help()
    assert "recognize" in help_text
    assert "convert" in help_text
