"""
Object tracking module using YOLO and ByteTrack.
"""

import os
import pickle
import sys
import pandas as pd
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox


class Tracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to('mps')
        self.model.model.half()  # FP16 for speed
        self.byte_tracker = sv.ByteTrack()
        print(f"Model loaded on: {self.model.device}")

    def interpolate_ball_positions(self, ball_positions):
        # Track which frames had real detections
        detected_frames = set()
        for i, x in enumerate(ball_positions):
            if x.get(1, {}).get('bbox'):
                detected_frames.add(i)
        
        # Extract bboxes
        ball_positions_list = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        # Convert back with interpolation flag
        ball_positions = [
            {1: {"bbox": x, "interpolated": i not in detected_frames}} 
            for i, x in enumerate(df_ball_positions.to_numpy().tolist())
        ]
        
        return ball_positions

    def detect_frames(self, frames):
        detections = []
        
        for i, frame in enumerate(tqdm(frames, desc="Detecting objects")):
            results = self.model.predict(source=frame, conf=0.1, imgsz=1280, max_det=50, verbose=False)
            detections.append(results[0])
            
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path:
            abs_stub = os.path.abspath(stub_path)
            if os.path.exists(abs_stub):
                with open(abs_stub, "rb") as f:
                    tracks = pickle.load(f)
                print(f"[stub] Loaded tracks from {abs_stub}")
                return tracks
            else:
                print(f"[stub] Stub not found. Running detection...")

        tracks = {"players": [], "referee": [], "ball": []}

        cls_names = self.model.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        ball_cls_id = cls_names_inv.get("ball")
        player_cls_id = cls_names_inv.get("player")
        goalkeeper_cls_id = cls_names_inv.get("goalkeeper")
        referee_cls_id = cls_names_inv.get("referee")

        for frame_idx, frame in enumerate(tqdm(frames, desc="Detecting objects")):
            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})
            
            results = self.model.predict(source=frame, conf=0.01, imgsz=1280, max_det=50, verbose=False)
            det = results[0]
            
            # Extract ball (before ByteTrack)
            for i in range(len(det.boxes)):
                cid = int(det.boxes.cls[i])
                if cid == ball_cls_id:
                    bbox = det.boxes.xyxy[i].tolist()
                    conf = float(det.boxes.conf[i])
                    if 1 not in tracks["ball"][frame_idx]:
                        tracks["ball"][frame_idx][1] = {"bbox": bbox, "conf": conf}
                    elif conf > tracks["ball"][frame_idx][1].get("conf", 0):
                        tracks["ball"][frame_idx][1] = {"bbox": bbox, "conf": conf}
            
            # Players/refs through ByteTrack
            det_sv = sv.Detections.from_ultralytics(det)
            
            keep_mask = []
            for i, (cid, conf) in enumerate(zip(det_sv.class_id, det_sv.confidence)):
                is_person = cid in [player_cls_id, goalkeeper_cls_id, referee_cls_id]
                high_conf = conf >= 0.1
                keep_mask.append(is_person and high_conf)
            
            keep_mask = np.array(keep_mask)
            
            if keep_mask.any():
                det_sv_filtered = det_sv[keep_mask]
                
                for idx, cid in enumerate(det_sv_filtered.class_id):
                    if cid == goalkeeper_cls_id:
                        det_sv_filtered.class_id[idx] = player_cls_id

                det_trk = self.byte_tracker.update_with_detections(det_sv_filtered)

                for i in range(len(det_trk)):
                    bbox = det_trk.xyxy[i].tolist()
                    cid = det_trk.class_id[i]
                    tid = det_trk.tracker_id[i]
                    
                    if tid is None:
                        continue

                    if cid == player_cls_id:
                        tracks["players"][frame_idx][int(tid)] = {"bbox": bbox}
                    elif cid == referee_cls_id:
                        tracks["referee"][frame_idx][int(tid)] = {"bbox": bbox}
            else:
                empty_det = sv.Detections.empty()
                self.byte_tracker.update_with_detections(empty_det)

        # Stats
        ball_detected = sum(1 for b in tracks["ball"] if b.get(1))
        frames_with_players = sum(1 for p in tracks["players"] if len(p) > 0)
        frames_with_refs = sum(1 for r in tracks["referee"] if len(r) > 0)
        
        if stub_path:
            abs_stub = os.path.abspath(stub_path)
            os.makedirs(os.path.dirname(abs_stub), exist_ok=True)
            with open(abs_stub, "wb") as f:
                pickle.dump(tracks, f)
            print(f"[stub] Saved tracks to {abs_stub}")

        return tracks

    def draw_ellipse(self, frame, bbox, colour, track_id=None):
        """Draw ellipse at player feet with track ID box."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Main ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Track ID rectangle
        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            x2_rect = x_center + rect_w // 2
            y1_rect = (y2 - rect_h // 2) + 15
            y2_rect = (y2 + rect_h // 2) + 15

            # Background rectangle
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                colour,
                cv2.FILLED
            )

            # Text position
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            # Track ID text
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        return frame

    def draw_triangle(self, frame, bbox, colour, interpolated=False):
        """
        Draw triangle marker below ball position.
        Points down so ball is visible above.
        """
        y2 = int(bbox[3])  # Bottom of ball
        x, _ = get_center_of_bbox(bbox)

        # Colours
        if interpolated:
            colour = (0, 140, 255)  # Orange for interpolated
        else:
            colour = (0, 255, 0)    # Green for detected

        # Triangle pointing UP to ball (sits below ball)
        triangle_points = np.array([
            [x, y2 + 5],           # Top point (near ball)
            [x - 12, y2 + 25],     # Bottom left
            [x + 12, y2 + 25],     # Bottom right
        ])
        
        # Filled triangle with border
        cv2.drawContours(frame, [triangle_points], 0, colour, cv2.FILLED, cv2.LINE_AA)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """Draw annotations on all frames."""
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], colour, track_id)

            # Draw referees (yellow)
            for _, ref in referee_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                interpolated = ball.get("interpolated", False)
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0), interpolated)

            output_video_frames.append(frame)

        return output_video_frames