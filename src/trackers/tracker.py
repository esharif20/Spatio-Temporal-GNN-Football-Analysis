# src/trackers/tracker.py
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.byte_tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            # pass frames as 'source' to be explicit and avoid warnings
            results = self.model.predict(source=batch, conf=0.1, verbose=True)
            detections.extend(results)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Fast path: load cached tracks and return
        if read_from_stub and stub_path:
            abs_stub = os.path.abspath(stub_path)
            if os.path.exists(abs_stub):
                with open(abs_stub, "rb") as f:
                    tracks = pickle.load(f)
                print(f"[stub] Loaded tracks from {abs_stub}")
                return tracks
            else:
                print(f"[stub] Stub not found at {abs_stub}. Running detection")

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referee": [], "ball": []}

        cls_names = detections[0].names if detections else {}
        cls_names_inv = {v: k for k, v in cls_names.items()}

        for frame_num, det in enumerate(detections):
            det_sv = sv.Detections.from_ultralytics(det)

            # normalise goalkeeper to player
            for idx, cid in enumerate(det_sv.class_id):
                if cls_names.get(cid) == "goalkeeper":
                    det_sv.class_id[idx] = cls_names_inv.get("player", cid)

            det_trk = self.byte_tracker.update_with_detections(det_sv)

            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

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

        if stub_path:
            abs_stub = os.path.abspath(stub_path)
            os.makedirs(os.path.dirname(abs_stub), exist_ok=True)
            with open(abs_stub, "wb") as f:
                pickle.dump(tracks, f)
            print(f"[stub] Saved tracks to {abs_stub}")


        return tracks



    def draw_ellipse(self,frame,bbox,colour,track_id):
        y2 = int(bbox[3]) # bottom of bbox
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_center,y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = 45,
            endAngle = 235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame


    def draw_annotations(self, video_frames , tracks):
        output_video_frames = []
        for frame_num , frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            # Draw players
            for track_id , player in player_dict.items():
                frame = self.draw_ellipse(frame , player["bbox"], (0,0,255) , track_id)

            output_video_frames.append(frame)

        return output_video_frames

