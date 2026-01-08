"""
Unit tests for team_assigner/team_assigner.py.

These tests use a lightweight classifier stub to avoid downloading
SigLIP weights during CI.
"""
from __future__ import annotations

import numpy as np
import pytest
import supervision as sv


class DummyClassifier:
    """Simple classifier stub for unit tests."""

    def __init__(self, device: str | None = None, batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size
        self.fitted = False

    def fit(self, crops: list[np.ndarray]) -> None:
        if len(crops) == 0:
            raise ValueError("No crops provided")
        self.fitted = True

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        return np.array([idx % 2 for idx in range(len(crops))], dtype=int)


@pytest.fixture()
def team_assigner(monkeypatch):
    """Create TeamAssigner with a dummy classifier."""
    import team_assigner.team_assigner as ta

    monkeypatch.setattr(ta, "TeamClassifier", DummyClassifier)
    config = ta.TeamAssignerConfig(
        stride=1,
        batch_size=4,
        max_crops=50,
        min_crop_size=(5, 5),
    )
    return ta.TeamAssigner(config=config)


@pytest.fixture()
def sample_frame() -> np.ndarray:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:40, 10:30] = [0, 0, 255]   # red
    frame[10:40, 60:80] = [255, 0, 0]   # blue
    frame[50:80, 40:60] = [0, 255, 0]   # green goalkeeper
    return frame


def _make_tracks() -> dict:
    return {
        "players": [
            {
                1: {"bbox": [10, 10, 30, 40]},
                2: {"bbox": [60, 10, 80, 40]},
            }
        ],
        "goalkeepers": [
            {
                10: {"bbox": [40, 50, 60, 80]},
            }
        ],
    }


def test_collect_crops_respects_max(monkeypatch, sample_frame):
    import team_assigner.team_assigner as ta

    monkeypatch.setattr(ta, "TeamClassifier", DummyClassifier)
    config = ta.TeamAssignerConfig(stride=1, batch_size=4, max_crops=2, min_crop_size=(5, 5))
    assigner = ta.TeamAssigner(config=config)

    frames = [sample_frame] * 4
    tracks = {
        "players": [
            {
                1: {"bbox": [10, 10, 30, 40]},
                2: {"bbox": [60, 10, 80, 40]},
            }
            for _ in frames
        ]
    }

    crops_by_id = assigner._collect_crops_by_track(frames, tracks)
    total_crops = sum(len(crops) for crops in crops_by_id.values())
    assert total_crops <= 2


def test_fit_populates_track_team(team_assigner, sample_frame):
    frames = [sample_frame]
    tracks = _make_tracks()

    team_assigner.fit(frames, tracks)

    assert set(team_assigner.track_team.keys()) == {1, 2}
    assert set(team_assigner.track_team.values()) <= {0, 1}
    assert team_assigner.team_colors_bgr.get(0) is not None
    assert team_assigner.team_colors_bgr.get(1) is not None


def test_assign_teams_updates_tracks(team_assigner, sample_frame):
    frames = [sample_frame]
    tracks = _make_tracks()

    team_assigner.fit(frames, tracks)
    team_assigner.assign_teams(frames, tracks)

    for pid, info in tracks["players"][0].items():
        assert info.get("team_id") in (0, 1)
        assert "team_color" in info

    gk_info = tracks["goalkeepers"][0][10]
    assert gk_info.get("team_id") in (0, 1)
    assert "team_color" in gk_info


def test_resolve_goalkeepers_team_id_basic():
    from team_assigner.team_assigner import resolve_goalkeepers_team_id

    players = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [90, 0, 100, 10]], dtype=np.float32),
        confidence=np.array([1.0, 1.0], dtype=np.float32),
        class_id=np.array([0, 1], dtype=np.int32),
    )
    goalkeepers = sv.Detections(
        xyxy=np.array([[2, 0, 12, 10]], dtype=np.float32),
        confidence=np.array([1.0], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )

    team_ids = resolve_goalkeepers_team_id(players, goalkeepers)
    assert team_ids.tolist() == [0]
