"""People tracking using ByteTrack."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv


class PeopleTracker:
    """ByteTrack-based tracker for players, goalkeepers, and referees."""

    def __init__(
        self,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.70,
    ):
        """Initialize people tracker.

        Args:
            conf_threshold: Minimum confidence for detections.
            nms_threshold: IoU threshold for NMS.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        # Debug statistics
        self.debug_counts: List[Dict[str, Dict[str, int]]] = []

    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracker.reset()
        self.debug_counts = []

    def _empty_debug_counts(self) -> Dict[str, Dict[str, int]]:
        """Create empty debug count structure."""
        stages = ("raw", "post_thresh", "post_nms", "post_suppress", "final")
        return {stage: {"player": 0, "referee": 0, "goalkeeper": 0} for stage in stages}

    def track_frame(
        self,
        detections: sv.Detections,
        class_names: Dict[int, str],
        player_cls: Optional[int],
        goalkeeper_cls: Optional[int],
        referee_cls: Optional[int],
    ) -> Tuple[Dict[str, Dict], Dict[int, float], Dict[int, str]]:
        """Track people in a single frame.

        Args:
            detections: Supervision detections from YOLO.
            class_names: Mapping of class IDs to names.
            player_cls: Class ID for players.
            goalkeeper_cls: Class ID for goalkeepers.
            referee_cls: Class ID for referees.

        Returns:
            Tuple of (tracks_dict, confidences_dict, labels_dict).
            tracks_dict has keys 'players', 'goalkeepers', 'referees'.
        """
        debug = self._empty_debug_counts()

        tracks = {
            "players": {},
            "goalkeepers": {},
            "referees": {},
        }
        confidences = {}
        labels = {}

        if detections is None or len(detections) == 0:
            self.tracker.update_with_detections(sv.Detections.empty())
            self.debug_counts.append(debug)
            return tracks, confidences, labels

        people_classes = [c for c in (player_cls, goalkeeper_cls, referee_cls) if c is not None]
        if not people_classes:
            self.tracker.update_with_detections(sv.Detections.empty())
            self.debug_counts.append(debug)
            return tracks, confidences, labels

        # Count raw detections
        for name, cls_id in [("player", player_cls), ("referee", referee_cls), ("goalkeeper", goalkeeper_cls)]:
            if cls_id is not None:
                debug["raw"][name] = int(np.sum(detections.class_id == cls_id))

        # Filter by class and confidence
        keep = np.isin(detections.class_id, people_classes) & (detections.confidence >= self.conf_threshold)

        debug["post_thresh"]["player"] = int(np.sum(keep & (detections.class_id == player_cls))) if player_cls is not None else 0
        debug["post_thresh"]["referee"] = int(np.sum(keep & (detections.class_id == referee_cls))) if referee_cls is not None else 0
        debug["post_thresh"]["goalkeeper"] = int(np.sum(keep & (detections.class_id == goalkeeper_cls))) if goalkeeper_cls is not None else 0

        if not keep.any():
            self.tracker.update_with_detections(sv.Detections.empty())
            self.debug_counts.append(debug)
            return tracks, confidences, labels

        # Apply NMS
        people = detections[keep].with_nms(threshold=self.nms_threshold, class_agnostic=True)

        debug["post_nms"]["player"] = int(np.sum(people.class_id == player_cls)) if player_cls is not None else 0
        debug["post_nms"]["referee"] = int(np.sum(people.class_id == referee_cls)) if referee_cls is not None else 0
        debug["post_nms"]["goalkeeper"] = int(np.sum(people.class_id == goalkeeper_cls)) if goalkeeper_cls is not None else 0
        debug["post_suppress"] = debug["post_nms"].copy()

        # Store original class info before tracking (ByteTrack modifies class_id)
        orig_class_id = people.class_id.copy()
        orig_conf = people.confidence.copy()
        people.class_id[:] = 0  # ByteTrack needs uniform class

        # Run tracking
        tracked = self.tracker.update_with_detections(people)

        # Extract tracked results
        final_counts = {"player": 0, "referee": 0, "goalkeeper": 0}
        for tid, bbox, cid, conf in zip(tracked.tracker_id, tracked.xyxy, orig_class_id, orig_conf):
            if tid is None:
                continue

            tid = int(tid)
            cid = int(cid)
            confidences[tid] = float(conf)
            labels[tid] = class_names.get(cid, str(cid))

            if goalkeeper_cls is not None and cid == int(goalkeeper_cls):
                bucket = "goalkeepers"
                final_counts["goalkeeper"] += 1
            elif referee_cls is not None and cid == int(referee_cls):
                bucket = "referees"
                final_counts["referee"] += 1
            else:
                bucket = "players"
                final_counts["player"] += 1

            tracks[bucket][tid] = {
                "bbox": bbox.tolist(),
                "class_id": cid,
                "confidence": float(conf),
            }

        debug["final"] = final_counts
        self.debug_counts.append(debug)

        return tracks, confidences, labels

    def track_all_frames(
        self,
        all_detections: List[sv.Detections],
        class_names: Dict[int, str],
        player_cls: Optional[int],
        goalkeeper_cls: Optional[int],
        referee_cls: Optional[int],
    ) -> Tuple[Dict[str, List[dict]], Dict[int, Dict[int, float]], Dict[int, Dict[int, str]]]:
        """Track people across all frames.

        Args:
            all_detections: List of detections per frame.
            class_names: Mapping of class IDs to names.
            player_cls: Class ID for players.
            goalkeeper_cls: Class ID for goalkeepers.
            referee_cls: Class ID for referees.

        Returns:
            Tuple of (tracks, all_confidences, all_labels).
        """
        self.reset()

        tracks = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
        }
        all_confidences = {}
        all_labels = {}

        for frame_num, dets in enumerate(all_detections):
            frame_tracks, confs, labels = self.track_frame(
                dets, class_names, player_cls, goalkeeper_cls, referee_cls
            )

            tracks["players"].append(frame_tracks["players"])
            tracks["goalkeepers"].append(frame_tracks["goalkeepers"])
            tracks["referees"].append(frame_tracks["referees"])

            all_confidences[frame_num] = confs
            all_labels[frame_num] = labels

        return tracks, all_confidences, all_labels
