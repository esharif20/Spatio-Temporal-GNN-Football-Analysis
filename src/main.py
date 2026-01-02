"""
Main entry point for football video analysis pipeline.
Processes input video, tracks objects, assigns teams, and generates annotated output.
"""

from pathlib import Path
import sys
import numpy as np

from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner


# Path configuration
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
STUB_PATH = ROOT / "stubs" / "track_stubs.pkl"
INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"


def validate_environment():
    """Validate required files and directories exist."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}\n"
            f"Please ensure best.pt is in the models directory."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_video_frames(video_path):
    """Load video frames from file."""
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    
    print(f"Loading video: {video_path.name}")
    frames = read_video(str(video_path))
    print(f"Loaded {len(frames)} frames")
    
    return frames


def track_objects(frames, use_cache=True):
    """Detect and track objects across video frames."""
    print("Initialising tracker...")
    tracker = Tracker(str(MODEL_PATH))
    
    print("Tracking objects...")
    tracks = tracker.get_object_tracks(
        frames=frames,
        read_from_stub=use_cache,
        stub_path=str(STUB_PATH)
    )
    
    return tracker, tracks


def print_detection_stats(tracks):
    """Print detection statistics."""
    total_frames = len(tracks["ball"])
    ball_detected = sum(1 for b in tracks["ball"] if b.get(1, {}).get("bbox"))
    frames_with_players = sum(1 for p in tracks["players"] if len(p) > 0)
    frames_with_refs = sum(1 for r in tracks["referee"] if len(r) > 0)
    avg_players = sum(len(p) for p in tracks["players"]) / total_frames

    print(f"\n=== Detection Stats ===")
    print(f"Ball detected: {ball_detected}/{total_frames} frames ({100*ball_detected/total_frames:.1f}%)")
    print(f"Frames with players: {frames_with_players}/{total_frames} ({100*frames_with_players/total_frames:.1f}%)")
    print(f"Frames with referees: {frames_with_refs}/{total_frames} ({100*frames_with_refs/total_frames:.1f}%)")
    print(f"Avg players per frame: {avg_players:.1f}")
    print("=" * 25 + "\n")


def assign_teams(frames, tracks):
    """Assign team colours to all tracked players."""
    print("Assigning teams...")
    team_assigner = TeamAssigner()
    
    team_assigner.assign_team_colour(frames[0], tracks['players'][0])
    
    total_frames = len(tracks['players'])
    for frame_num, player_track in enumerate(tracks['players']):
        if frame_num % 100 == 0:
            print(f"Processing frame {frame_num}/{total_frames}")
        
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_colour'] = (
                team_assigner.team_colours[team]
            )
    
    print(f"Team assignment complete. Teams: {list(team_assigner.team_colours.keys())}")
    return tracks


def assign_ball_to_players(tracks):
    """Assign ball possession to nearest player each frame."""
    print("Assigning ball possession...")
    player_ball_assigner = PlayerBallAssigner()

    team_ball_control = []
    last_team = None

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")

        # Clear possession
        for player_id in player_track.keys():
            player_track[player_id]["has_ball"] = False

        # Skip if no ball
        if ball_bbox is None:
            team_ball_control.append(last_team)
            continue

        assigned_player = player_ball_assigner.assign_ball_to_players(player_track, ball_bbox)

        if assigned_player != -1 and assigned_player in player_track:
            player_track[assigned_player]["has_ball"] = True
            last_team = player_track[assigned_player].get("team")

        team_ball_control.append(last_team)

    print("Ball possession assignment complete")
    return tracks, np.array(team_ball_control, dtype=object)


def generate_output_video(tracker, frames, tracks, team_ball_control, output_path):
    """Draw annotations and save output video."""
    print("Generating annotated video...")
    annotated_frames = tracker.draw_annotations(frames, tracks, team_ball_control)

    print(f"Saving output video: {output_path}")
    save_video(annotated_frames, str(output_path), fps=24)
    print("Output video saved successfully")


def main():
    """Run the complete football analysis pipeline."""
    try:
        validate_environment()
        
        input_video = INPUT_DIR / "08fd33_4.mp4"
        output_video = OUTPUT_DIR / "output_video.mp4"
        
        frames = load_video_frames(input_video)
        tracker, tracks = track_objects(frames, use_cache=True)
        print_detection_stats(tracks)

        # Interpolate ball positions (ByteTrack already filtered)
        tracks["ball"] = tracker.process_ball_tracks(tracks["ball"], smooth_sigma=2)
        
        tracks = assign_teams(frames, tracks)
        tracks, team_ball_control = assign_ball_to_players(tracks)

        generate_output_video(tracker, frames, tracks, team_ball_control, output_video)
        
        print("\nPipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
