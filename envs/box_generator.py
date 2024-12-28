"""
Box generation module for 3D bin packing problem.
This module provides different strategies for generating boxes with various sizes and rotations.
"""

import numpy as np
from typing import List, Tuple, Optional
from .box import Box


class BoxGenerator:
    """
    A box generator that uses volume-based splitting method.
    This is the primary generator used for training, which creates boxes by recursively
    splitting a large container into smaller boxes.

    The algorithm ensures that:
    1. Generated boxes have reasonable size distributions
    2. All dimensions are integers
    3. Each box has a unique ID for tracking
    """

    def __init__(self, container_size: Tuple[float, float, float] = (100, 100, 100)):
        """
        Initialize the box generator.

        Args:
            container_size (Tuple[float, float, float]): Initial container dimensions (length, width, height).
                                                        Will be converted to integers.
        """

        self.container_size = tuple(int(x) for x in container_size)
        self._id_counter = 0

    def generate_boxes(self, n_min: int = 10, n_max: int = 50) -> List[Box]:
        """
        Generate a list of boxes using the volume-based splitting method (Algorithm 1).

        Args:
            n_min (int): Minimum number of boxes to generate
            n_max (int): Maximum number of boxes to generate

        Returns:
            List[Box]: List of generated boxes with random rotations and unique IDs

        Algorithm Steps:
        1. Start with a single box of container size
        2. Repeatedly split boxes until desired count is reached:
           - Select a box with probability proportional to its volume
           - Choose a splitting axis based on edge lengths
           - Split at a random position
           - Apply random rotations to resulting boxes
        """

        items = [Box(*self.container_size)]
        N = np.random.randint(n_min, n_max + 1)

        while len(items) < N:

            volumes = [item.get_volume() for item in items]
            probabilities = np.array(volumes) / sum(volumes)
            selected_idx = np.random.choice(len(items), p=probabilities)
            selected_box = items.pop(selected_idx)

            edge_lengths = selected_box.dimensions
            axis_probs = np.array(edge_lengths) / sum(edge_lengths)
            axis = np.random.choice(3, p=axis_probs)

            edge_length = edge_lengths[axis]
            min_split = max(1, int(0.1 * edge_length))
            max_split = min(edge_length - 1, int(0.9 * edge_length))

            if max_split <= min_split:
                items.append(selected_box)
                continue

            split_position = np.random.randint(min_split, max_split + 1)

            box1, box2 = selected_box.split(axis, split_position)

            if all(d > 0 for d in box1.dimensions) and all(
                d > 0 for d in box2.dimensions
            ):

                box1.set_rotation(np.random.randint(0, 6))
                box2.set_rotation(np.random.randint(0, 6))

                box1.id = self._get_next_id()
                box2.id = self._get_next_id()
                items.extend([box1, box2])
            else:
                items.append(selected_box)

        return items

    def _get_next_id(self) -> int:
        """
        Generate a unique ID for a box.

        Returns:
            int: A unique identifier
        """
        self._id_counter += 1
        return self._id_counter


class CoordinateBoxGenerator:
    """
    Alternative box generator using coordinate-based splitting method.
    Currently not in use, but kept for potential future use or comparison.

    This generator:
    1. Uses direct coordinate splitting rather than volume-based
    2. Maintains more uniform size distribution
    3. May be useful for specific test cases
    """

    def __init__(self, container_size: Tuple[float, float, float]):
        """
        Initialize the coordinate-based box generator.

        Args:
            container_size (Tuple[float, float, float]): Initial container dimensions (length, width, height)
        """
        self.container_size = tuple(int(x) for x in container_size)
        self.boxes = []

    def generate_boxes(self, n_min: int = 10, n_max: int = 50) -> List[Box]:
        """
        Generate boxes using coordinate-based splitting method.

        Args:
            n_min (int): Minimum number of boxes to generate
            n_max (int): Maximum number of boxes to generate

        Returns:
            List[Box]: List of generated boxes with random rotations

        Note:
            This method splits space more uniformly than volume-based splitting,
            but may not provide as good a distribution for training purposes.
        """
        meta_boxes = [self.container_size]
        target_n = np.random.randint(n_min, n_max + 1)

        while len(meta_boxes) < target_n:
            box_idx = np.random.randint(0, len(meta_boxes))
            current_dims = meta_boxes.pop(box_idx)

            axis = np.random.randint(0, 3)
            dim = current_dims[axis]

            split_pos = int(np.random.uniform(0.2 * dim, 0.8 * dim))

            dims1 = list(current_dims)
            dims2 = list(current_dims)
            dims1[axis] = split_pos
            dims2[axis] = dim - split_pos

            meta_boxes.extend(
                [tuple(int(x) for x in dims1), tuple(int(x) for x in dims2)]
            )

        boxes = []
        for dims in meta_boxes:
            box = Box(*dims)
            box.set_rotation(np.random.randint(0, 6))
            boxes.append(box)

        return boxes

    def _get_next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter
