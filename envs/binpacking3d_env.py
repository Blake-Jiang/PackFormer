import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List, Optional
from .box import Box
from .box_generator import BoxGenerator


class BinPacking3DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        container_size=(100, 100, 100),
        n_boxes_range=(10, 30),
        use_position_mask=False,
    ):
        super().__init__()
        self.container_size = container_size
        self.L, self.W, self.H = container_size
        self.height_map = np.zeros((self.L, self.W), dtype=np.int32)
        self.n_boxes_range = n_boxes_range
        self.box_generator = BoxGenerator(container_size)
        self.use_position_mask = use_position_mask

        self.boxes = []
        self.packed_mask = None

        self.max_steps = 100
        self.current_step = 0

        high_values = [self.H, self.L, self.W, self.L, self.W, self.L, self.W]

        self.observation_space = spaces.Dict(
            {
                "boxes": spaces.Box(
                    low=0,
                    high=max(container_size),
                    shape=(n_boxes_range[1], 3),
                    dtype=np.float32,
                ),
                "boxes_mask": spaces.Box(
                    low=0, high=1, shape=(n_boxes_range[1],), dtype=np.bool_
                ),
                "container_state": spaces.Box(
                    low=np.zeros((self.L, self.W, 7), dtype=np.float32),
                    high=np.full((self.L, self.W, 7), high_values, dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.MultiDiscrete(
            [self.n_boxes_range[1], 6, self.L, self.W]
        )

    def _get_container_state(self):
        container_state = np.zeros((self.L, self.W, 7), dtype=np.float32)

        container_state[:, :, 0] = self.height_map

        for i in range(self.L):
            for j in range(self.W):
                h_current = self.height_map[i, j]

                found = False
                for di in range(i - 1, -1, -1):
                    if self.height_map[di, j] != h_current:
                        container_state[i, j, 1] = i - di
                        found = True
                        break
                if not found:
                    container_state[i, j, 1] = i

                found = False
                for dj in range(j - 1, -1, -1):
                    if self.height_map[i, dj] != h_current:
                        container_state[i, j, 2] = j - dj
                        found = True
                        break
                if not found:
                    container_state[i, j, 2] = j

                found = False
                for di in range(i + 1, self.L):
                    if self.height_map[di, j] != h_current:
                        container_state[i, j, 3] = di - i
                        found = True
                        break
                if not found:
                    container_state[i, j, 3] = self.L - i - 1

                found = False
                for dj in range(j + 1, self.W):
                    if self.height_map[i, dj] != h_current:
                        container_state[i, j, 4] = dj - j
                        found = True
                        break
                if not found:
                    container_state[i, j, 4] = self.W - j - 1

                min_dist_l = float("inf")
                for di in range(self.L):
                    if self.height_map[di, j] > h_current:

                        left_edge = di
                        for d in range(di - 1, -1, -1):
                            if self.height_map[d, j] != self.height_map[di, j]:
                                left_edge = d + 1
                                break

                        right_edge = di
                        for d in range(di + 1, self.L):
                            if self.height_map[d, j] != self.height_map[di, j]:
                                right_edge = d - 1
                                break

                        dist = min(abs(i - left_edge), abs(i - right_edge))
                        min_dist_l = min(min_dist_l, dist)

                if min_dist_l != float("inf"):
                    container_state[i, j, 5] = min_dist_l

                min_dist_w = float("inf")
                for dj in range(self.W):
                    if self.height_map[i, dj] > h_current:

                        front_edge = dj
                        for d in range(dj - 1, -1, -1):
                            if self.height_map[i, d] != self.height_map[i, dj]:
                                front_edge = d + 1
                                break

                        back_edge = dj
                        for d in range(dj + 1, self.W):
                            if self.height_map[i, d] != self.height_map[i, dj]:
                                back_edge = d - 1
                                break

                        dist = min(abs(j - front_edge), abs(j - back_edge))
                        min_dist_w = min(min_dist_w, dist)

                if min_dist_w != float("inf"):
                    container_state[i, j, 6] = min_dist_w

        return container_state

    def _get_box_state(self):
        """获取所有未放置箱子的状态"""
        max_boxes = self.n_boxes_range[1]
        boxes = np.zeros((max_boxes, 3), dtype=np.float32)
        mask = np.zeros(max_boxes, dtype=np.bool_)

        box_idx = 0
        for i, box in enumerate(self.boxes):
            if not self.packed_mask[i]:
                if box_idx < max_boxes:
                    boxes[box_idx] = box.dimensions
                    mask[box_idx] = True
                    box_idx += 1

        return {"boxes": boxes, "mask": mask}

    def _get_observation(self):
        box_state = self._get_box_state()
        container_state = self._get_container_state()

        obs = {
            "boxes": box_state["boxes"],
            "boxes_mask": box_state["mask"],
            "container_state": container_state,
        }

        if self.use_position_mask:
            position_mask = self.get_position_mask()
            obs["position_mask"] = position_mask
        return obs

    def reset(self, seed=None, options=None):
        """重置环境"""

        super().reset(seed=seed)

        self.boxes = self.box_generator.generate_boxes(*self.n_boxes_range)

        self.packed_mask = np.zeros(len(self.boxes), dtype=bool)
        self.height_map.fill(0)
        self.current_step = 0

        obs = self._get_observation()
        return obs, {}

    def get_position_mask(self):
        """获取有效动作掩码 - 只保留位置掩码"""

        position_mask = np.ones(self.L * self.W, dtype=np.bool_)

        height_mask_2d = self.height_map >= self.H
        position_mask[height_mask_2d.reshape(-1)] = False

        return position_mask

    def _is_valid_placement_position(self, x, y, min_l, min_w):
        """检查位置是否有效"""

        if x + min_l > self.L or y + min_w > self.W:
            return False

        current_height = self.height_map[x, y]
        if current_height >= self.H:
            return False

        return True

    def step(self, action):
        """执行一步动作"""
        self.current_step += 1

        box_idx, orientation, x, y = action

        valid_boxes = np.where(~self.packed_mask)[0]
        if len(valid_boxes) == 0:
            return self._get_observation(), 0, True, False, {"error": "no valid boxes"}

        box_idx = valid_boxes[box_idx % len(valid_boxes)]

        selected_box = self.boxes[box_idx]
        l, w, h = selected_box.get_rotated_dimensions(orientation)

        if x + l <= self.L and y + w <= self.W:
            z = np.max(self.height_map[x : x + l, y : y + w])
        else:

            return (
                self._get_observation(),
                -1,
                True,
                False,
                {"error": "position out of bounds"},
            )

        if self._is_valid_placement(x, y, z, orientation, selected_box):

            self._place_box(box_idx, x, y, z, orientation)
            self.packed_mask[box_idx] = True

            reward = self._calculate_gap_reward(box_idx)

            info = {
                "utilization": self._calculate_utilization(),
                "packed_boxes": np.sum(self.packed_mask),
                "total_boxes": len(self.boxes),
                "max_height": np.max(self.height_map),
                "height_variance": np.var(self.height_map[self.height_map > 0]),
                "gap_ratio": 1.0 - self._calculate_utilization(),
                "action": {
                    "box_idx": box_idx,
                    "orientation": orientation,
                    "position": (x, y, z),
                    "dimensions": (l, w, h),
                },
            }

            done = np.all(self.packed_mask) or self.current_step >= self.max_steps

            return self._get_observation(), reward, done, False, info
        else:

            return (
                self._get_observation(),
                -1,
                True,
                False,
                {"error": "invalid placement"},
            )

    def _is_valid_placement(
        self, x: int, y: int, z: int, rotation: int, box: Box
    ) -> bool:
        l, w, h = box.get_rotated_dimensions(rotation)

        if x < 0 or y < 0 or x + l > self.L or y + w > self.W:
            return False

        placement_area = self.height_map[x : x + l, y : y + w]
        max_height = np.max(placement_area)

        if z != max_height:
            return False

        if z + h > self.H:
            return False

        return True

    def _place_box(self, box_idx: int, x: int, y: int, z: int, rotation: int) -> None:
        """在容器中放置箱子"""
        box = self.boxes[box_idx]
        l, w, h = box.get_rotated_dimensions(rotation)

        self.height_map[x : x + l, y : y + w] = z + h

        box.set_position(x, y, z)
        box.set_rotation(rotation)

    def _calculate_gap_reward(self, box_idx: int) -> float:
        """计算基于间隙率的奖励"""
        current_stacked_height = np.max(self.height_map)
        total_packed_volume = sum(
            np.prod(box.dimensions)
            for i, box in enumerate(self.boxes)
            if self.packed_mask[i]
        )

        used_space = self.L * self.W * current_stacked_height
        gap_ratio = 1.0 - (total_packed_volume / used_space) if used_space > 0 else 0

        reward = 1.0 + (1.0 - gap_ratio)

        height_penalty = -0.1 * (current_stacked_height / self.H)
        reward += height_penalty

        return reward

    def _calculate_utilization(self) -> float:
        """计算容器的空间利用率"""
        volume_used = sum(
            box.dimensions[0] * box.dimensions[1] * box.dimensions[2]
            for i, box in enumerate(self.boxes)
            if self.packed_mask[i]
        )

        total_volume = np.prod(self.container_size)
        return volume_used / total_volume

    def render(self, mode="human"):
        pass
