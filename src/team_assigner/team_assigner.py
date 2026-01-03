"""
Generalizable team assignment using K-means clustering on multi-frame HSV samples.
Works for ANY two distinct jersey colours (not just green vs white).
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


class TeamAssigner:
    """
    Assigns players to teams using multi-frame colour sampling + K-means clustering.
    
    Generalizable approach:
    1. Collect HSV samples from multiple frames per player
    2. Cluster ALL samples into 2 groups using K-means
    3. Assign each player to cluster based on their median sample
    
    Works for any two distinct jersey colours.
    """
    
    def __init__(self):
        self.player_team_dict = {}
        self.goalkeeper_ids = set()
        self.frame_width = 1920
        
        # Store colour samples per player
        self.player_colour_samples = defaultdict(list)
        
        # K-means model fitted on all player colours
        self.kmeans = None
        self.cluster_centers = {}

    def get_player_colour_hsv(self, frame, bbox):
        """
        Extract HSV values from player's jersey area.
        Returns (H, S, V) tuple or None.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        image = frame[y1:y2, x1:x2]
        
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 6:
            return None
        
        # Use top half (jersey area)
        top_half = image[0:int(image.shape[0] // 2), :]
        
        if top_half.size == 0:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        
        # Use K-means to find dominant colour (exclude background)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        
        if len(pixels) < 10:
            return None
        
        # Cluster into 2: background and jersey
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        
        # Reshape to find corners (background)
        label_img = labels.reshape(top_half.shape[0], top_half.shape[1])
        corners = [
            label_img[0, 0],
            label_img[0, -1],
            label_img[-1, 0],
            label_img[-1, -1]
        ]
        bg_cluster = max(set(corners), key=corners.count)
        jersey_cluster = 1 - bg_cluster
        
        # Get jersey HSV
        jersey_hsv = kmeans.cluster_centers_[jersey_cluster]
        return tuple(jersey_hsv)

    def _is_dark(self, hsv):
        """Check if colour is very dark (goalkeeper or referee)."""
        if hsv is None:
            return True
        h, s, v = hsv
        return v < 80

    def _is_referee_colour(self, hsv):
        """Check if colour is likely referee (black kit)."""
        if hsv is None:
            return False
        h, s, v = hsv
        # Black: low saturation AND low value
        return s < 50 and v < 100

    def identify_goalkeepers(self, tracks, frame_width):
        """Identify goalkeepers by extreme positions."""
        self.frame_width = frame_width
        print("\n=== Identifying Goalkeepers ===")
        
        player_positions = {}
        for frame_track in tracks["players"]:
            for pid, player in frame_track.items():
                bbox = player["bbox"]
                x_center = (bbox[0] + bbox[2]) / 2
                if pid not in player_positions:
                    player_positions[pid] = []
                player_positions[pid].append(x_center)
        
        player_stats = []
        for pid, positions in player_positions.items():
            if len(positions) < 30:
                continue
            player_stats.append({
                "pid": pid,
                "avg_x": np.mean(positions),
                "frames": len(positions)
            })
        
        player_stats.sort(key=lambda x: x["avg_x"])
        
        if len(player_stats) < 2:
            print("Not enough players for GK detection")
            return
        
        print("Leftmost players:")
        for p in player_stats[:3]:
            print(f"  Player {p['pid']}: avg_x={p['avg_x']:.0f}")
        print("Rightmost players:")
        for p in player_stats[-3:]:
            print(f"  Player {p['pid']}: avg_x={p['avg_x']:.0f}")
        
        # Use relative thresholds based on player spread
        all_x = [p["avg_x"] for p in player_stats]
        min_x, max_x = min(all_x), max(all_x)
        spread = max_x - min_x
        
        left_threshold = min_x + spread * 0.15
        right_threshold = max_x - spread * 0.15
        
        for p in player_stats:
            if p["avg_x"] < left_threshold:
                self.goalkeeper_ids.add(p["pid"])
                print(f"LEFT GK: Player {p['pid']}")
                break
        
        for p in reversed(player_stats):
            if p["avg_x"] > right_threshold:
                self.goalkeeper_ids.add(p["pid"])
                print(f"RIGHT GK: Player {p['pid']}")
                break
        
        print(f"Goalkeeper IDs: {self.goalkeeper_ids}")

    def collect_colour_samples(self, frames, tracks):
        """
        Collect HSV colour samples from multiple frames per player.
        This ensures reliable team assignment.
        """
        print("\n=== Collecting Colour Samples ===")
        
        # Sample every 10th frame
        sample_indices = list(range(0, len(frames), 10))
        
        for frame_idx in sample_indices:
            frame = frames[frame_idx]
            for pid, player in tracks["players"][frame_idx].items():
                if pid in self.goalkeeper_ids:
                    continue
                
                hsv = self.get_player_colour_hsv(frame, player["bbox"])
                if hsv is None:
                    continue
                
                h, s, v = hsv
                
                # Skip dark (referee/GK)
                if v < 80:
                    continue
                
                # Store full HSV for clustering
                self.player_colour_samples[pid].append(hsv)
        
        print(f"Collected samples for {len(self.player_colour_samples)} players")

    def assign_teams_by_clustering(self):
        """
        Assign teams using K-means clustering on all colour samples.
        
        This is GENERALIZABLE - works for any two distinct jersey colours:
        - Green vs White
        - Red vs Blue  
        - Yellow vs Black
        - etc.
        """
        print("\n=== Clustering Player Colours ===")
        
        # Collect all samples with player IDs
        all_samples = []
        sample_to_player = []
        
        for pid, samples in self.player_colour_samples.items():
            for sample in samples:
                all_samples.append(sample)
                sample_to_player.append(pid)
        
        if len(all_samples) < 4:
            print("WARNING: Not enough samples for clustering")
            return
        
        all_samples = np.array(all_samples, dtype=np.float64)
        
        # Cluster into 2 teams
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        self.kmeans.fit(all_samples)
        
        self.cluster_centers[1] = self.kmeans.cluster_centers_[0]
        self.cluster_centers[2] = self.kmeans.cluster_centers_[1]
        
        print(f"Cluster 1 (Team 1): H={self.cluster_centers[1][0]:.0f}, S={self.cluster_centers[1][1]:.0f}, V={self.cluster_centers[1][2]:.0f}")
        print(f"Cluster 2 (Team 2): H={self.cluster_centers[2][0]:.0f}, S={self.cluster_centers[2][1]:.0f}, V={self.cluster_centers[2][2]:.0f}")
        
        # Assign each player based on MEDIAN of their samples
        print("\nAssigning players by median colour:")
        
        for pid, samples in self.player_colour_samples.items():
            if len(samples) == 0:
                continue
            
            # Calculate median HSV
            samples_arr = np.array(samples, dtype=np.float64)
            median_hsv = np.median(samples_arr, axis=0)
            
            # Predict cluster
            cluster = self.kmeans.predict(median_hsv.astype(np.float64).reshape(1, -1))[0]
            team = cluster + 1  # 0-indexed to 1-indexed
            
            self.player_team_dict[pid] = team
            
            print(f"  Player {pid}: median HSV=({median_hsv[0]:.0f}, {median_hsv[1]:.0f}, {median_hsv[2]:.0f}) → Team {team}")
        
        # Count teams
        team_counts = {1: 0, 2: 0}
        for pid, team in self.player_team_dict.items():
            team_counts[team] = team_counts.get(team, 0) + 1
        
        print(f"\nTeam 1: {team_counts[1]} players")
        print(f"Team 2: {team_counts[2]} players")

    def assign_goalkeeper_teams(self, tracks):
        """Assign goalkeepers to teams based on pitch side."""
        if not self.goalkeeper_ids:
            return
        
        print("\n=== Assigning Goalkeeper Teams ===")
        
        # Calculate team centroids
        team_positions = {1: [], 2: []}
        
        for frame_idx in range(0, len(tracks["players"]), 10):
            for pid, player in tracks["players"][frame_idx].items():
                if pid in self.goalkeeper_ids:
                    continue
                if pid not in self.player_team_dict:
                    continue
                
                team = self.player_team_dict[pid]
                x = (player["bbox"][0] + player["bbox"][2]) / 2
                team_positions[team].append(x)
        
        team_centroids = {
            1: np.mean(team_positions[1]) if team_positions[1] else self.frame_width/2,
            2: np.mean(team_positions[2]) if team_positions[2] else self.frame_width/2
        }
        
        print(f"Team centroids: Team1={team_centroids[1]:.0f}, Team2={team_centroids[2]:.0f}")
        
        for gk_id in self.goalkeeper_ids:
            gk_positions = []
            for frame_track in tracks["players"]:
                if gk_id in frame_track:
                    bbox = frame_track[gk_id]["bbox"]
                    gk_positions.append((bbox[0] + bbox[2]) / 2)
            
            if not gk_positions:
                self.player_team_dict[gk_id] = 1
                continue
            
            gk_avg_x = np.mean(gk_positions)
            
            # GK on same side as team centroid = that team
            if gk_avg_x < self.frame_width / 2:
                team = 1 if team_centroids[1] < team_centroids[2] else 2
            else:
                team = 1 if team_centroids[1] > team_centroids[2] else 2
            
            self.player_team_dict[gk_id] = team
            print(f"Goalkeeper {gk_id} (x={gk_avg_x:.0f}): Team {team}")

    def get_player_team(self, player_id):
        """Get team for a player."""
        return self.player_team_dict.get(player_id, None)
