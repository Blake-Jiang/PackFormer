"""
Box class for 3D bin packing problem.
Implements basic box properties and operations including positioning, rotation, and splitting.
"""

import numpy as np
from typing import Tuple, List, Optional


class Box:
    """
    Represents a box in 3D space.

    Attributes:
        dimensions (Tuple[float, float, float]): The length, width, and height of the box
        position (Optional[Tuple[float, float, float]]): The (x, y, z) coordinates of the box in the container
        rotation (Optional[int]): The rotation state (0-5) representing 6 possible orientations
        id (Optional[int]): Unique identifier for tracking the box
    """

    def __init__(self, length: float, width: float, height: float):
        """
        Initialize a box instance.

        Args:
            length (float): Length of the box
            width (float): Width of the box
            height (float): Height of the box
        """
        self.dimensions = (length, width, height)
        self.position: Optional[Tuple[float, float, float]] = None
        self.rotation: Optional[int] = 0
        self.id = None

    def get_volume(self) -> float:
        """
        Calculate the volume of the box.

        Returns:
            float: The volume of the box
        """
        return np.prod(self.dimensions)

    def get_rotated_dimensions(self, rotation: int) -> Tuple[float, float, float]:
        """
        Get the dimensions of the box after applying a rotation.

        Args:
            rotation (int): Rotation state (0-5)

        Returns:
            Tuple[float, float, float]: The rotated dimensions (length, width, height)

        Notes:
            Rotation states represent:
            - 0: Original orientation
            - 1: 90° rotation around x-axis
            - 2: 90° rotation around y-axis
            - 3: 90° rotation around both x and y axes
            - 4: 90° rotation around z-axis
            - 5: 90° rotation around both z and x axes
        """
        l, w, h = self.dimensions
        rotations = [
            (l, w, h),
            (l, h, w),
            (w, l, h),
            (w, h, l),
            (h, l, w),
            (h, w, l),
        ]
        return rotations[rotation]

    def set_position(self, x: float, y: float, z: float) -> None:
        """
        Set the position of the box in the container.

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            z (float): z-coordinate
        """
        self.position = (x, y, z)

    def set_rotation(self, rotation: int) -> None:
        """
        Set the rotation state of the box.

        Args:
            rotation (int): Rotation state (0-5)

        Raises:
            AssertionError: If rotation value is not in range [0, 5]
        """
        assert 0 <= rotation <= 5, "Rotation value must be between 0 and 5"
        self.rotation = rotation

    def split(self, axis: int, position: float) -> Tuple["Box", "Box"]:
        """
        Split the box into two new boxes along the specified axis.
        Used primarily for generating cutting planes and space partitioning.

        Args:
            axis (int): Axis along which to split (0:x, 1:y, 2:z)
            position (float): Position along the axis where to split

        Returns:
            Tuple[Box, Box]: Two new boxes resulting from the split

        Notes:
            The resulting boxes will need their positions and rotations
            to be set separately as these are not inherited from the
            original box.
        """
        new_dims1 = list(self.dimensions)
        new_dims2 = list(self.dimensions)

        original_length = self.dimensions[axis]
        new_dims1[axis] = position
        new_dims2[axis] = original_length - position

        box1 = Box(*new_dims1)
        box2 = Box(*new_dims2)

        return box1, box2
