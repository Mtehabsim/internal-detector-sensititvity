"""Path options must be resolved where the command runs: mtkaudit.release.import_release() changes the working
directory, so a path interpreted after it would land inside the release's working copy."""
import os

from mtkaudit import release
from mtkaudit.cli import abs_path
from mtkaudit.paths import ROOT, display


def test_driver_out_is_absolute_before_the_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = release.parse_cli(["--model", "llama2", "--out", "rerun/llama2"])
    os.chdir("/")                                   # what import_release() does, in effect
    assert a.out.is_absolute()
    assert a.out == tmp_path / "rerun" / "llama2"


def test_driver_out_defaults_to_the_archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert release.parse_cli(["--model", "vicuna"]).out is None


def test_abs_path_uses_the_invocation_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = abs_path("fresh/cache")
    os.chdir("/")
    assert p == tmp_path / "fresh" / "cache"


def test_reports_record_repository_relative_paths(tmp_path):
    assert display(ROOT / "results" / "release_pipeline") == "results/release_pipeline"
    assert display(tmp_path) == str(tmp_path)          # outside the repository: kept as given
