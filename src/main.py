"""
Main entry point for football video analysis pipeline.

Processes input video, tracks objects, assigns teams, and generates annotated output.
Role stabilisation runs BEFORE team assignment to prevent referee contamination.
"""

# =============================================================================
# Imports
# =============================================================================

# Standard library
from pathlib import Path
import sys
from typing import List, Tuple

# Third-party
import cv2
import numpy as np

# Local modules
from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from trackers.track_stabilizer import stabilize_tracks
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
STUB_PATH = ROOT / "stubs" / "track_stubs.pkl"
INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"


# =============================================================================
# Helper Functions
# =============================================================================

def hsv_to_bgr(hsv: tuple) -> Tuple[int, int, int]:
    """Convert HSV colour to BGR for OpenCV.

    Args:
        hsv: Tuple of (H, S, V) values

    Returns:
        BGR tuple for OpenCV
    """
    h, s, v = hsv
    hsv_img = np.uint8([[[
        int(np.clip(h, 0, 179)),
        int(np.clip(s, 0, 255)),
        int(np.clip(v, 0, 255))
    ]]])
    bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def validate_environment() -> None:
    """Validate required files and directories exist."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}\n"
            f"Please ensure best.pt is in the models directory."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_video_frames(video_path: Path) -> List[np.ndarray]:
    """Load video frames from file.

    Args:
        video_path: Path to input video file

    Returns:
        List of video frames as numpy arrays
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    print(f"Loading video: {video_path.name}")
    frames = read_video(str(video_path))
    print(f"Loaded {len(frames)} frames")

    return frames


def print_detection_stats(tracks: dict) -> None:
    """Print detection statistics.

    Args:
        tracks: Dictionary containing player, referee, and ball tracks
    """
    total_frames = len(tracks["ball"])
    ball_detected = sum(1 for b in tracks["ball"] if b.get(1, {}).get("bbox"))
    frames_with_refs = sum(1 for r in tracks["referees"] if len(r) > 0)

    print(f"\nBall: {ball_detected}/{total_frames} ({100*ball_detected/total_frames:.1f}%)")
    print(f"Refs: {frames_with_refs}/{total_frames} ({100*frames_with_refs/total_frames:.1f}%)")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """Run the complete football analysis pipeline."""
    try:
        validate_environment()

        # Load video
        input_video = INPUT_DIR / "08fd33_4.mp4"
        frames = load_video_frames(input_video)
        frame_height, frame_width = frames[0].shape[:2]

        # Track objects
        print("\nTracking...")
        tracker = Tracker(str(MODEL_PATH))
        tracks = tracker.get_object_tracks(
            frames=frames,
            read_from_stub=True,
            stub_path=str(STUB_PATH),
        )

        print_detection_stats(tracks)

        # Interpolate ball positions (cubic spline + Gaussian smoothing)
        tracks["ball"] = tracker.process_ball_tracks(tracks["ball"], smooth_sigma=2)

        # =====================================================================
        # Role Stabilisation (Majority Voting) - FIRST
        # Fixes referee flickering by locking roles before team clustering
        # =====================================================================
        tracks, stable_roles = stabilize_tracks(tracks)

        # =====================================================================
        # Team Assignment (K-means Clustering) - SECOND
        # =====================================================================
        print("\n" + "=" * 50)
        print("TEAM ASSIGNMENT")
        print("=" * 50)

        team_assigner = TeamAssigner()

        # Step 1: Identify goalkeepers by position
        team_assigner.identify_goalkeepers(tracks, frame_width)

        # Step 2: Collect colour samples (refs now excluded)
        team_assigner.collect_colour_samples(frames, tracks)

        # Step 3: Cluster into 2 teams
        team_assigner.assign_teams_by_clustering()

        # Get actual kit colours from cluster centres
        centres_hsv = team_assigner.kmeans.cluster_centers_
        team_kit_colours = {
            1: hsv_to_bgr(centres_hsv[0]),
            2: hsv_to_bgr(centres_hsv[1]),
        }

        # Step 4: Assign goalkeeper teams
        team_assigner.assign_goalkeeper_teams(tracks)

        # =====================================================================
        # Apply Team Colours
        # =====================================================================
        print("\nApplying team colours...")

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(player_id)

                if team is not None:
                    tracks["players"][frame_num][player_id]["team"] = team
                    tracks["players"][frame_num][player_id]["team_color"] = team_kit_colours[team]
                else:
                    # Fallback: predict team from colour
                    hsv = team_assigner.get_player_colour_hsv(frames[frame_num], track["bbox"])
                    if hsv is not None and team_assigner.kmeans is not None:
                        predicted = team_assigner.kmeans.predict(
                            np.array(hsv, dtype=np.float64).reshape(1, -1)
                        )[0] + 1
                        tracks["players"][frame_num][player_id]["team"] = predicted
                        tracks["players"][frame_num][player_id]["team_color"] = team_kit_colours[predicted]
                    else:
                        tracks["players"][frame_num][player_id]["team"] = 1
                        tracks["players"][frame_num][player_id]["team_color"] = team_kit_colours[1]

        # Print final distribution
        team_counts = {1: 0, 2: 0}
        for pid, team in team_assigner.player_team_dict.items():
            team_counts[team] = team_counts.get(team, 0) + 1

        print("\n=== Final Distribution ===")
        print(f"Team 1: {team_counts[1]} players")
        print(f"Team 2: {team_counts[2]} players")
        print(f"Goalkeepers: {team_assigner.goalkeeper_ids}")

        # =====================================================================
        # Ball Possession (Industry Standard - only count actual possession)
        # =====================================================================
        print("\nAssigning ball possession...")
        player_ball_assigner = PlayerBallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")

            # Clear possession
            for player_id in player_track.keys():
                player_track[player_id]["has_ball"] = False

            if ball_bbox is None:
                team_ball_control.append(None)  # No ball detected - no possession
                continue

            assigned_player = player_ball_assigner.assign_ball_to_players(player_track, ball_bbox)

            if assigned_player != -1 and assigned_player in player_track:
                player_track[assigned_player]["has_ball"] = True
                current_team = player_track[assigned_player].get("team")
                team_ball_control.append(current_team)
            else:
                team_ball_control.append(None)  # Loose ball - no team possession

        team_ball_control = np.array(team_ball_control, dtype=object)

        # =====================================================================
        # Generate Output Video
        # =====================================================================
        print("\nGenerating video...")
        output = tracker.draw_annotations(frames, tracks, team_ball_control)

        output_path = OUTPUT_DIR / "output_video.mp4"
        print(f"Saving: {output_path}")
        save_video(output, str(output_path), fps=24)

        print("\nPipeline completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
