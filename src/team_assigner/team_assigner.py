"""
Team assignment module using K-means clustering on jersey colours.
Assigns players to teams based on dominant shirt colour.
"""

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns players to teams based on jersey colour clustering.
    """
    
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}
        self.kmean = None

    def get_clustering_model(self, image):
        """
        Fit K-means clustering model to image pixels.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Fitted KMeans model with 2 clusters
        """
        # Reshape image to 2D array of pixels
        image_2d = image.reshape((-1, 3))

        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(image_2d)
        
        return kmeans

    def get_player_colour(self, frame, bbox):
        """
        Extract dominant jersey colour from player bounding box.
        
        Args:
            frame: Video frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            RGB colour array representing player's jersey
        """
        # Crop player from frame
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Use only top half (jersey area, avoid shorts/boots)
        top_half_image = image[0:int(image.shape[0] // 2), :]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to original image dimensions
        clustered_image = labels.reshape(
            top_half_image.shape[0],
            top_half_image.shape[1]
        )

        # Identify background cluster from image corners
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # Player cluster is the opposite cluster
        player_cluster = 1 - non_player_cluster
        player_colour = kmeans.cluster_centers_[player_cluster]
        
        return player_colour

    def assign_team_colour(self, frame, player_detections):
        """
        Assign team colours based on first frame player detections.
        
        Args:
            frame: First video frame
            player_detections: Dictionary of player detections in first frame
        """
        # Extract jersey colours for all players
        player_colours = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_colour = self.get_player_colour(frame, bbox)
            player_colours.append(player_colour)
        
        # Cluster players into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(player_colours)

        self.kmean = kmeans
        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Get team assignment for a player.
        
        Args:
            frame: Current video frame
            player_bbox: Player bounding box
            player_id: Unique player ID
            
        Returns:
            Team ID (1 or 2)
        """
        # Return cached team if already assigned
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extract player colour
        player_colour = self.get_player_colour(frame, player_bbox)

        # Predict team based on colour (FIX: remove extra list wrapping)
        team_id = self.kmean.predict(player_colour.reshape(1, -1))[0]
        team_id += 1  # Adjust to make team IDs 1 and 2

        # Cache result
        self.player_team_dict[player_id] = team_id
        
        return team_id