"""
Tutorial-style team assignment using SigLIP embeddings + UMAP + KMeans.

This follows the full_pipeline.ipynb approach:
- collect player crops with a frame stride
- fit a TeamClassifier on all crops
- assign a team per track by majority vote
- assign goalkeepers by distance to team centroids
"""

from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    """Generate batches from a sequence with a specified batch size."""
    batch_size = max(batch_size, 1)
    current_batch: List[V] = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SigLIP model for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """

    def __init__(self, device: Optional[str] = None, batch_size: int = 32) -> None:
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH, use_fast=True)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device:
            return device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.empty((0, 0), dtype=np.float32)

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction"):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        if len(crops) == 0:
            raise ValueError("No crops provided for team classifier training.")

        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)


@dataclass
class TeamAssignerConfig:
    stride: int = 30
    batch_size: int = 32
    max_crops: int = 2000
    min_crop_size: Tuple[int, int] = (10, 6)


class TeamAssigner:
    """Tutorial-style team assignment wrapper for the pipeline."""

    def __init__(self, device: Optional[str] = None, config: Optional[TeamAssignerConfig] = None) -> None:
        self.config = config or TeamAssignerConfig()
        self.classifier = TeamClassifier(device=device, batch_size=self.config.batch_size)
        self.track_team: Dict[int, int] = {}
        self.team_colors_bgr: Dict[int, Tuple[int, int, int]] = {}

    def _crop(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        min_h, min_w = self.config.min_crop_size
        if crop.shape[0] < min_h or crop.shape[1] < min_w:
            return None
        return crop

    def _collect_crops_by_track(self, frames: List[np.ndarray], tracks: dict) -> Dict[int, List[np.ndarray]]:
        crops_by_id: Dict[int, List[np.ndarray]] = {}
        stride = max(1, int(self.config.stride))
        total_crops = 0

        for frame_idx in range(0, len(frames), stride):
            frame = frames[frame_idx]
            for pid, info in tracks["players"][frame_idx].items():
                crop = self._crop(frame, info["bbox"])
                if crop is None:
                    continue
                crops_by_id.setdefault(pid, []).append(crop)
                total_crops += 1
                if self.config.max_crops > 0 and total_crops >= self.config.max_crops:
                    return crops_by_id

        return crops_by_id

    @staticmethod
    def _flatten_crops(crops_by_id: Dict[int, List[np.ndarray]]) -> Tuple[List[np.ndarray], List[int]]:
        crops: List[np.ndarray] = []
        owners: List[int] = []
        for pid, track_crops in crops_by_id.items():
            for crop in track_crops:
                crops.append(crop)
                owners.append(pid)
        return crops, owners

    @staticmethod
    def _mean_crop_bgr(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        y1, y2 = int(h * 0.2), int(h * 0.6)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        region = crop[y1:y2, x1:x2]
        if region.size == 0:
            region = crop
        return region.reshape(-1, 3).mean(axis=0)

    def _compute_team_colors(
        self,
        crops_by_id: Dict[int, List[np.ndarray]],
        track_team: Dict[int, int],
    ) -> Dict[int, Tuple[int, int, int]]:
        colors_by_team: Dict[int, List[np.ndarray]] = {0: [], 1: []}
        for pid, crops in crops_by_id.items():
            team_id = track_team.get(pid)
            if team_id is None:
                continue
            for crop in crops:
                colors_by_team[int(team_id)].append(self._mean_crop_bgr(crop))

        team_colors = {}
        for team_id, samples in colors_by_team.items():
            if len(samples) == 0:
                continue
            median_color = np.median(np.vstack(samples), axis=0)
            team_colors[int(team_id)] = (int(median_color[0]), int(median_color[1]), int(median_color[2]))

        self.team_colors_bgr = team_colors
        return team_colors

    @staticmethod
    def _detections_from_tracks(
        frame_tracks: Dict[int, dict],
        class_from_team: bool,
    ) -> sv.Detections:
        boxes = []
        confs = []
        class_ids = []

        for _, info in frame_tracks.items():
            if class_from_team:
                team_id = info.get("team_id")
                if team_id not in (0, 1):
                    continue
                class_ids.append(int(team_id))
            else:
                class_ids.append(0)

            boxes.append(info["bbox"])
            confs.append(float(info.get("confidence", 1.0)))

        if len(boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.array(class_ids, dtype=np.int32),
        )

    def fit(self, frames: List[np.ndarray], tracks: dict) -> None:
        crops_by_id = self._collect_crops_by_track(frames, tracks)
        all_crops, owners = self._flatten_crops(crops_by_id)
        if len(all_crops) == 0:
            raise ValueError("No crops provided for team classifier training.")

        self.classifier.fit(all_crops)
        predictions = self.classifier.predict(all_crops)

        votes: Dict[int, np.ndarray] = {}
        for pid, pred in zip(owners, predictions):
            votes.setdefault(pid, np.zeros(2, dtype=int))
            votes[pid][int(pred)] += 1

        self.track_team = {pid: int(np.argmax(counts)) for pid, counts in votes.items()}
        self._compute_team_colors(crops_by_id, self.track_team)

    def assign_teams(self, frames: List[np.ndarray], tracks: dict) -> None:
        if len(self.track_team) == 0:
            raise RuntimeError("TeamAssigner.fit must be called before assign_teams.")

        for frame_idx in range(len(frames)):
            players = tracks["players"][frame_idx]
            for pid, info in players.items():
                team_id = self.track_team.get(pid)
                if team_id is None:
                    continue
                info["team_id"] = int(team_id)
                color = self.team_colors_bgr.get(team_id)
                if color is not None:
                    info["team_color"] = color
                    info["team_colour"] = color

        for frame_idx in range(len(frames)):
            players = tracks["players"][frame_idx]
            goalkeepers = tracks.get("goalkeepers", [{}] * len(frames))[frame_idx]

            players_det = self._detections_from_tracks(players, class_from_team=True)
            goalkeepers_det = self._detections_from_tracks(goalkeepers, class_from_team=False)

            if len(players_det) == 0 or len(goalkeepers_det) == 0:
                continue
            if not ((players_det.class_id == 0).any() and (players_det.class_id == 1).any()):
                continue

            gk_team_ids = resolve_goalkeepers_team_id(players_det, goalkeepers_det)
            for (gid, info), team_id in zip(goalkeepers.items(), gk_team_ids):
                info["team_id"] = int(team_id)
                color = self.team_colors_bgr.get(int(team_id))
                if color is not None:
                    info["team_color"] = color
                    info["team_colour"] = color
