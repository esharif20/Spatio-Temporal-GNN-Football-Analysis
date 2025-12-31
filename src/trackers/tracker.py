"""
Object tracking module using YOLO and ByteTrack.
Detects and tracks players, referees, and the ball across video frames.
"""

import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox


class Tracker:
    """
    Tracks objects (players, referees, ball) across video frames.
    Uses YOLO for detection and ByteTrack for multi-object tracking.
    """
    
    def __init__(self, model_path):
        """
        Initialise tracker with YOLO model.
        
        Args:
            model_path: Path to YOLO model weights file
        """
        self.model = YOLO(model_path)
        self.byte_tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Run YOLO detection on all frames in batches.
        
        Args:
            frames: List of video frames (numpy arrays)
            
        Returns:
            List of YOLO detection results
        """
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model.predict(source=batch, conf=0.1, verbose=True)
            detections.extend(results)
            
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Generate or load object tracks for all frames.
        
        Args:
            frames: List of video frames
            read_from_stub: Whether to load cached tracks from disk
            stub_path: Path to cached tracks pickle file
            
        Returns:
            Dictionary containing tracks for players, referees, and ball
        """
        # Load cached tracks if available
        if read_from_stub and stub_path:
            abs_stub = os.path.abspath(stub_path)
            if os.path.exists(abs_stub):
                with open(abs_stub, "rb") as f:
                    tracks = pickle.load(f)
                print(f"[stub] Loaded tracks from {abs_stub}")
                return tracks
            else:
                print(f"[stub] Stub not found at {abs_stub}. Running detection")

        # Run detection on all frames
        detections = self.detect_frames(frames)
        
        # Initialise track storage
        tracks = {
            "players": [],
            "referee": [],
            "ball": []
        }

        # Build class name mappings
        cls_names = detections[0].names if detections else {}
        cls_names_inv = {v: k for k, v in cls_names.items()}

        # Process each frame's detections
        for frame_num, det in enumerate(detections):
            det_sv = sv.Detections.from_ultralytics(det)

            # Normalise goalkeeper class to player class
            for idx, cid in enumerate(det_sv.class_id):
                if cls_names.get(cid) == "goalkeeper":
                    det_sv.class_id[idx] = cls_names_inv.get("player", cid)

            # Update tracker with current frame detections
            det_trk = self.byte_tracker.update_with_detections(det_sv)

            # Initialise empty track dictionaries for this frame
            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            # Store tracked objects by class
            for i in range(len(det_trk)):
                bbox = det_trk.xyxy[i].tolist()
                cid = det_trk.class_id[i]
                tid = det_trk.tracker_id[i]
                
                if tid is None:
                    continue

                if cid == cls_names_inv.get("player"):
                    tracks["players"][frame_num][int(tid)] = {"bbox": bbox}
                    
                elif cid == cls_names_inv.get("referee"):
                    tracks["referee"][frame_num][int(tid)] = {"bbox": bbox}
                    
                elif cid == cls_names_inv.get("ball"):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracks to stub file for future use
        if stub_path:
            abs_stub = os.path.abspath(stub_path)
            os.makedirs(os.path.dirname(abs_stub), exist_ok=True)
            with open(abs_stub, "wb") as f:
                pickle.dump(tracks, f)
            print(f"[stub] Saved tracks to {abs_stub}")

        return tracks

    def draw_ellipse(self, frame, bbox, colour, track_id=None):
        """
        Draw ellipse at player/referee feet with optional track ID label.
        
        Args:
            frame: Video frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            colour: BGR colour tuple
            track_id: Optional track ID to display
            
        Returns:
            Modified frame with ellipse drawn
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse at bottom of bounding box
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw track ID label if provided
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            # Draw background rectangle
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                colour,
                cv2.FILLED
            )

            # Adjust text position for triple-digit IDs
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            # Draw track ID text
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, colour):
        """
        Draw triangle marker above ball position.
        
        Args:
            frame: Video frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            colour: BGR colour tuple
            
        Returns:
            Modified frame with triangle drawn
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Define triangle points
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        
        # Draw filled triangle with black outline
        cv2.drawContours(frame, [triangle_points], 0, colour, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """
        Draw tracking annotations on all video frames.
        
        Args:
            video_frames: List of video frames
            tracks: Dictionary of tracked objects from get_object_tracks()
            
        Returns:
            List of annotated video frames
        """
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            # Draw players with team colours (or default red)
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], colour, track_id)

            # Draw referees (yellow ellipses)
            for _, ref in referee_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 255, 255))

            # Draw ball (green triangle)
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames