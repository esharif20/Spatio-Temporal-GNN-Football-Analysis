"""Unit tests for utils/cache.py."""

from pathlib import Path
import tempfile

from pipeline import Mode
from utils.cache import clear_stubs, stub_path, stub_paths_for_mode


def test_stub_path_naming():
    path = stub_path("/tmp/Test1.mp4", Mode.BALL_DETECTION)
    assert path.name == "Test1_ball_tracks.pkl"


def test_stub_paths_for_all_mode():
    paths = stub_paths_for_mode("/tmp/Test1.mp4", Mode.ALL)
    names = {p.name for p in paths}
    assert names == {
        "Test1_people_tracks.pkl",
        "Test1_ball_tracks.pkl",
        "Test1_ball_tracks_full.pkl",
    }


def test_clear_stubs_deletes_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        stub = Path(tmpdir) / "temp_stub.pkl"
        stub.write_bytes(b"test")
        clear_stubs([stub])
        assert not stub.exists()
