"""Homography-based view transformation between frame and pitch coordinates."""

from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt


class ViewTransformer:
    """Transform coordinates between camera frame and pitch plane using homography.

    This class computes a homography matrix from corresponding points in two
    coordinate systems (e.g., detected keypoints in a video frame and their
    known positions on the pitch) and uses it to transform arbitrary points
    between the two systems.
    """

    def __init__(
        self,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32]
    ) -> None:
        """Initialize the ViewTransformer with source and target points.

        Args:
            source: Source points array of shape (N, 2) - e.g., frame keypoints
            target: Target points array of shape (N, 2) - e.g., pitch coordinates

        Raises:
            ValueError: If source and target shapes don't match, aren't 2D,
                or if homography computation fails.
        """
        if source.shape != target.shape:
            raise ValueError(
                f"Source shape {source.shape} != target shape {target.shape}"
            )
        if len(source.shape) != 2 or source.shape[1] != 2:
            raise ValueError(
                f"Points must be 2D coordinates with shape (N, 2), got {source.shape}"
            )
        if source.shape[0] < 4:
            raise ValueError(
                f"Need at least 4 point pairs for homography, got {source.shape[0]}"
            )

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Use standard least-squares (no RANSAC) to match the blog exactly
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError(
                "Homography matrix could not be calculated. "
                "Check that points are not collinear."
            )

        self._inlier_count = source.shape[0]  # All points used without RANSAC

    @property
    def inlier_count(self) -> int:
        """Number of inlier points used in homography estimation."""
        return self._inlier_count

    def transform_points(
        self,
        points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Transform points using the homography matrix.

        Args:
            points: Points to transform, shape (N, 2)

        Returns:
            Transformed points, shape (N, 2)

        Raises:
            ValueError: If points are not 2D coordinates.
        """
        if points.size == 0:
            return points.copy()

        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError(
                f"Points must be 2D coordinates with shape (N, 2), got {points.shape}"
            )

        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_image(
        self,
        image: npt.NDArray[np.uint8],
        resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """Transform an image using the homography matrix.

        Args:
            image: Input image (grayscale or color)
            resolution_wh: Output resolution as (width, height)

        Returns:
            Warped image with the specified resolution.

        Raises:
            ValueError: If image is not 2D (grayscale) or 3D (color).
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError(
                f"Image must be grayscale (2D) or color (3D), got shape {image.shape}"
            )
        return cv2.warpPerspective(image, self.m, resolution_wh)

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """The 3x3 homography matrix."""
        return self.m.copy()

    @matrix.setter
    def matrix(self, value: npt.NDArray[np.float64]) -> None:
        """Set the homography matrix directly (e.g., for averaging)."""
        self.m = value.astype(np.float64)
