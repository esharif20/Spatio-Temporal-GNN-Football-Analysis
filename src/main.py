"""
Main entry point for football video analysis pipeline.
Processes input video, tracks objects, assigns teams, and generates annotated output.
"""

from pathlib import Path
import sys

from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner


# Path configuration
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
STUB_PATH = ROOT / "stubs" / "track_stubs.pkl"
INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"


def validate_environment():
    """
    Validate required files and directories exist.
    
    Raises:
        FileNotFoundError: If model weights are missing
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}\n"
            f"Please ensure best.pt is in the models directory."
        )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_video_frames(video_path):
    """
    Load video frames from file.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        List of video frames as numpy arrays
        
    Raises:
        FileNotFoundError: If video file does not exist
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    
    print(f"Loading video: {video_path.name}")
    frames = read_video(str(video_path))
    print(f"Loaded {len(frames)} frames")
    
    return frames


def track_objects(frames, use_cache=True):
    """
    Detect and track objects across video frames.
    
    Args:
        frames: List of video frames
        use_cache: Whether to use cached tracks if available
        
    Returns:
        Dictionary containing player, referee, and ball tracks
    """
    print("Initialising tracker...")
    tracker = Tracker(str(MODEL_PATH))
    
    print("Tracking objects...")
    tracks = tracker.get_object_tracks(
        frames=frames,
        read_from_stub=use_cache,
        stub_path=str(STUB_PATH)
    )
    
    return tracker, tracks


def assign_teams(frames, tracks):
    """
    Assign team colours to all tracked players.
    
    Args:
        frames: List of video frames
        tracks: Object tracking dictionary
        
    Returns:
        Updated tracks dictionary with team assignments
    """
    print("Assigning teams...")
    team_assigner = TeamAssigner()
    
    # Determine team colours from first frame
    team_assigner.assign_team_colour(frames[0], tracks['players'][0])
    
    # Assign teams to all players across all frames
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
    
    print(f"Team assignment complete. Teams: {team_assigner.team_colours.keys()}")
    return tracks


def generate_output_video(tracker, frames, tracks, output_path):
    """
    Draw annotations and save output video.
    
    Args:
        tracker: Tracker instance with draw methods
        frames: List of video frames
        tracks: Object tracking dictionary with team assignments
        output_path: Path to save output video
    """
    print("Generating annotated video...")
    annotated_frames = tracker.draw_annotations(frames, tracks)
    
    print(f"Saving output video: {output_path}")
    save_video(annotated_frames, str(output_path), fps=24)
    print("Output video saved successfully")


def main():
    """
    Run the complete football analysis pipeline.
    
    Pipeline steps:
    1. Validate environment
    2. Load video frames
    3. Track objects (players, referees, ball)
    4. Assign team colours
    5. Generate annotated output video
    """
    try:
        # Validate environment
        validate_environment()
        
        # Define paths
        input_video = INPUT_DIR / "08fd33_4.mp4"
        output_video = OUTPUT_DIR / "output_video.mp4"
        
        # Load video
        frames = load_video_frames(input_video)
        
        # Track objects
        tracker, tracks = track_objects(frames, use_cache=True)
        
        # Assign teams
        tracks = assign_teams(frames, tracks)
        
        # Generate output
        generate_output_video(tracker, frames, tracks, output_video)
        
        print("\nPipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()