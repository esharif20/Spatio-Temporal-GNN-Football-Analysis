# src/main.py
from pathlib import Path
from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker

ROOT = Path(__file__).resolve().parent                  # .../src
BEST = ROOT / "models" / "best.pt"
STUB = ROOT / "stubs" / "track_stubs.pkl"

if not BEST.exists():
    raise FileNotFoundError(f"Missing weights: {BEST}")

def main():
    in_video = ROOT / "input_videos" / "08fd33_4.mp4"
    out_video = ROOT / "output_videos" / "output_video.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    frames = read_video(str(in_video))
    tracker = Tracker(str(BEST))
    tracks = tracker.get_object_tracks(frames=frames, read_from_stub=True, stub_path=str(STUB))

    #draw output
    output_video_frames = tracker.draw_annotations(frames, tracks)
    save_video(output_video_frames, str(out_video), fps=24)

if __name__ == "__main__":
    main()
