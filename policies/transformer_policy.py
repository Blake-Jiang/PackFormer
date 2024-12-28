import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import gymnasium as gym
from torch.distributions import Categorical
import math


class CombinedCategoricalDistribution(Distribution):
    """组合多个Categorical分布"""

    def __init__(
        self,
        action_dims: List[int],
        container_size: Tuple[int, int],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.action_dims = action_dims
        self.container_size = container_size
        self.device = device
        self.distributions = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """创建用于生成分布参数的网络"""
        return nn.ModuleList([nn.Linear(latent_dim, dim) for dim in self.action_dims])

    def proba_distribution(
        self, logits: Tuple[torch.Tensor, ...]
    ) -> "CombinedCategoricalDistribution":
        """根据logits设置分布"""
        self.distributions = [Categorical(logits=l) for l in logits]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率
        Args:
            actions: [batch_size, 4] (box_selection, orientation, x, y)
        """
        box_selection, orientation, x, y = actions.unbind(-1)
        W = self.container_size[1]
        position = x * W + y
        log_probs = []
        for dist, action in zip(
            self.distributions, [box_selection, orientation, position]
        ):
            log_probs.append(dist.log_prob(action))
        return torch.stack(log_probs).sum(dim=0)

    def entropy(self) -> torch.Tensor:
        """计算分布的熵"""
        return torch.stack([dist.entropy() for dist in self.distributions]).sum(dim=0)

    def sample(self) -> torch.Tensor:
        """采样动作"""
        samples = [dist.sample() for dist in self.distributions]
        position = samples[2]
        W = self.container_size[1]
        x = position // W
        y = position % W
        return torch.stack([samples[0], samples[1], x, y], dim=-1)

    def mode(self) -> torch.Tensor:
        """返回最可能的动作"""
        modes = [dist.probs.argmax(dim=-1) for dist in self.distributions]
        position = modes[2]
        W = self.action_dims[3]
        x = position // W
        y = position % W
        return torch.stack([modes[0], modes[1], x, y], dim=-1)

    def actions_from_params(self, *args, **kwargs) -> torch.Tensor:
        """从参数中获取动作（SB3要求的方法）"""
        logits = args[0]
        deterministic = kwargs.get("deterministic", False)
        self.proba_distribution(logits)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从参数中计算对数概率（SB3要求的方法）"""
        logits = args[0]
        actions = kwargs.get("actions", None)
        distribution = self.proba_distribution(logits)
        log_prob = self.log_prob(actions)
        entropy = self.entropy()
        return log_prob, entropy


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        返回:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        return x + self.pe[:, :seq_len, :]


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):

        attn_output, _ = self.cross_attn(
            query, key_value, key_value, key_padding_mask=key_padding_mask
        )
        query = query + self.dropout(attn_output)
        query = self.norm1(query)

        ff_output = self.feed_forward(query)
        query = query + self.dropout(ff_output)
        query = self.norm2(query)

        return query


class DummyMlpExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.latent_dim_pi = d_model
        self.latent_dim_vf = d_model

    def forward(self, features):
        return features, features


class CustomTransformerPolicy(ActorCriticPolicy):
    """
    Custom Transformer-based policy for PPO.

    :param observation_space: Observation space
    :param action_space: Action space (must be MultiDiscrete)
    :param lr_schedule: Learning rate schedule
    :param d_model: Dimension of transformer model
    :param nhead: Number of attention heads
    :param num_encoder_layers: Number of encoder layers
    :param num_decoder_layers: Number of decoder layers
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.MultiDiscrete,
        lr_schedule: callable,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        patch_size: int = 10,
        *args,
        **kwargs
    ):

        if "net_arch" in kwargs:
            del kwargs["net_arch"]

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.patch_size = patch_size
        self.container_size = observation_space["container_state"].shape[:2]
        self.max_boxes = observation_space["boxes"].shape[0]
        self.box_features = observation_space["boxes"].shape[1]
        self.action_dims = [
            action_space.nvec[0],
            action_space.nvec[1],
            action_space.nvec[2] * action_space.nvec[3],
        ]

        super().__init__(
            observation_space, action_space, lr_schedule, net_arch=[], *args, **kwargs
        )

        self.action_dist = CombinedCategoricalDistribution(
            self.action_dims, self.container_size, self.device
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """重写 _build 方法来创建 mlp_extractor 和优化器"""

        class DummyMlpExtractor(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.latent_dim_pi = self.d_model
                self_inner.latent_dim_vf = self.d_model

            def forward(self_inner, features):
                return features, features

            def forward_actor(self_inner, features):
                return features

            def forward_critic(self_inner, features):
                return features

        self.mlp_extractor = DummyMlpExtractor()

        self.box_feature_embedding = nn.Linear(1, self.d_model)
        self.container_encoder = nn.Sequential(
            nn.Linear(self.observation_space["container_state"].shape[2], self.d_model),
            nn.ReLU(),
        )

        self.pos_encoder = PositionalEncoding(self.d_model)
        self.box_transformer = nn.ModuleList(
            [
                TransformerEncoderBlock(self.d_model, self.nhead)
                for _ in range(self.num_encoder_layers)
            ]
        )

        self.container_transformer = nn.ModuleList(
            [
                TransformerEncoderBlock(self.d_model, self.nhead)
                for _ in range(self.num_encoder_layers)
            ]
        )

        self.position_decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.d_model, self.nhead)
                for _ in range(self.num_decoder_layers)
            ]
        )

        self.selection_decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.d_model, self.nhead)
                for _ in range(self.num_decoder_layers)
            ]
        )

        self.orientation_decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.d_model, self.nhead)
                for _ in range(self.num_decoder_layers)
            ]
        )

        self.position_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, np.prod(self.container_size)),
        )

        self.box_selection_head = nn.Sequential(
            nn.Linear(self.d_model, 512), nn.ReLU(), nn.Linear(512, self.max_boxes)
        )

        self.orientation_head = nn.Sequential(
            nn.Linear(self.d_model, 512), nn.ReLU(), nn.Linear(512, 6)
        )

        self.position_feature_extractor = nn.Linear(7, self.d_model)
        self.orientation_embedding = nn.Linear(3, self.d_model)

        self.value_net = ValueNetwork(self.d_model, self.nhead, self.num_encoder_layers)

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """获取构造函数参数，用于模型保存和加载"""
        data = super()._get_constructor_parameters()
        data.update(
            {
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
            }
        )
        return data

    def _downsample_container(self, container_state: torch.Tensor) -> torch.Tensor:
        """
        对容器状态进行下采样，选择每个patch中最重要的点
        container_state: [batch_size, L, W, 7]
        return: [batch_size, (L//patch_size)*(W//patch_size), 7]
        """
        batch_size, L, W, features = container_state.shape
        patch_size = self.patch_size

        patches = container_state.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )

        patches = patches.permute(0, 1, 2, 4, 5, 3)

        el = patches[..., 1]
        ew = patches[..., 2]

        importance = el * ew
        importance = importance.reshape(
            batch_size, L // patch_size, W // patch_size, -1
        )

        max_idx = importance.argmax(dim=-1)

        patches = patches.reshape(
            batch_size,
            L // patch_size,
            W // patch_size,
            patch_size * patch_size,
            features,
        )

        batch_indices = torch.arange(batch_size, device=patches.device)[:, None, None]
        l_indices = torch.arange(L // patch_size, device=patches.device)[None, :, None]
        w_indices = torch.arange(W // patch_size, device=patches.device)[None, None, :]

        downsampled = patches[batch_indices, l_indices, w_indices, max_idx]

        downsampled = downsampled.reshape(batch_size, -1, features)

        return downsampled

    def _extract_position_feature(
        self, container_state: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """从container_state中提取指定位置的特征"""
        batch_size = container_state.shape[0]
        L, W = self.container_size
        x = positions // W
        y = positions % W

        batch_idx = torch.arange(batch_size, device=positions.device)
        features = container_state[batch_idx, x, y]

        position_embedding = self.position_feature_extractor(features)
        return position_embedding

    def _generate_orientation_embedding(
        self, selected_box: torch.Tensor
    ) -> torch.Tensor:
        """生成选定箱子的六种朝向嵌入
        Args:
            selected_box: [batch_size, 3] 选中箱子的尺寸 (l,w,h)
        Returns:
            orientation_embedding: [batch_size, 6, d_model] 六种朝向的嵌入
        """
        batch_size = selected_box.shape[0]

        orientations = []
        for i in range(6):
            if i < 2:
                dims = selected_box[:, [0, 1, 2]]
            elif i < 4:
                dims = selected_box[:, [2, 1, 0]]
            else:
                dims = selected_box[:, [0, 2, 1]]
            orientations.append(dims)

        orientations = torch.stack(orientations, dim=1)

        orientation_embedding = self.orientation_embedding(orientations)

        return orientation_embedding

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        从策略网络的latent特征中获取动作分布
        """
        position_logits, box_selection_logits, orientation_logits = latent_pi
        return self.action_dist.proba_distribution(
            (box_selection_logits, orientation_logits, position_logits)
        )

    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Policy forward pass
        Args:
            obs: 包含以下键的字典:
                - boxes: [batch_size, max_boxes, box_features] 箱子状态
                - boxes_mask: [batch_size, max_boxes] 箱子的mask
                - container_state: [batch_size, L, W, features] 容器状态
        Returns:
            actions: [batch_size, 4] 动作
            values: [batch_size, 1] 值函数预测
            log_probs: [batch_size] 动作的对数概率
        """

        boxes = obs["boxes"]
        boxes_mask = obs["boxes_mask"].bool()
        container_state = obs["container_state"]

        attention_mask = ~boxes_mask

        box_features = self._encode_boxes(boxes, attention_mask)

        container_features = self._encode_container(container_state)

        position_logits, position_embedding = self._decode_position_and_get_embedding(
            container_state, container_features, box_features, attention_mask
        )

        box_selection_logits, orientation_embedding = self._decode_box_selection(
            box_features, position_embedding, obs["boxes"], attention_mask
        )

        orientation_logits = self._decode_orientation(
            orientation_embedding, position_embedding
        )

        values = self.value_net(box_features, container_features, attention_mask)

        distribution = self._get_action_dist_from_latent(
            (position_logits, box_selection_logits, orientation_logits)
        )

        actions = distribution.sample()

        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作的值、对数概率和熵
        """

        boxes = obs["boxes"]
        boxes_mask = obs["boxes_mask"].bool()
        container_state = obs["container_state"]

        attention_mask = ~boxes_mask

        box_features = self._encode_boxes(boxes, attention_mask)
        container_features = self._encode_container(container_state)

        position_logits, position_embedding = self._decode_position_and_get_embedding(
            container_state, container_features, box_features, attention_mask
        )
        box_selection_logits, orientation_embedding = self._decode_box_selection(
            box_features, position_embedding, obs["boxes"], attention_mask
        )
        orientation_logits = self._decode_orientation(
            orientation_embedding, position_embedding
        )

        values = self.value_net(box_features, container_features, attention_mask)

        distribution = self._get_action_dist_from_latent(
            (position_logits, box_selection_logits, orientation_logits)
        )
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def _predict(
        self,
        observation: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: Dictionary of observations
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Actions
        """

        position_mask = observation.get("position_mask", None)

        actions, _, _ = self.forward(observation)
        return actions

    def _build_mlp_extractor(self) -> None:
        """
        Overridden method from ActorCriticPolicy.
        Not used in this implementation.
        """
        pass

    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> Dict[str, Distribution]:
        """
        Get the current policy distribution.

        :param obs: Dictionary of observations
        :return: Dictionary of action distributions
        """
        actions, _, _ = self.forward(obs)
        position_logits, box_selection_logits, orientation_logits = actions

        return {
            "position": Categorical(logits=position_logits),
            "box_selection": Categorical(logits=box_selection_logits),
            "orientation": Categorical(logits=orientation_logits),
        }

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the estimated values according to the current policy.

        :param obs: Dictionary of observations
        :return: The estimated values
        """
        _, values, _ = self.forward(obs)
        return values

    def _encode_boxes(
        self, boxes: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """编码箱子特征
        Args:
            boxes: [batch_size, max_boxes, 3] 箱子尺寸
            attention_mask: [batch_size, max_boxes] 箱子的mask,True表示需要mask掉
        """
        box_dims = boxes.unsqueeze(-1)

        box_features = self.box_feature_embedding(box_dims)

        box_features = box_features.mean(dim=2)

        for transformer in self.box_transformer:
            box_features = transformer(box_features, mask=attention_mask)

        return box_features

    def _encode_container(self, container_state: torch.Tensor) -> torch.Tensor:
        """编码容器状态"""

        container_features = self._downsample_container(container_state)

        container_features = self.container_encoder(container_features)

        container_features = self.pos_encoder(container_features)

        for transformer in self.container_transformer:
            container_features = transformer(container_features)

        return container_features

    def _decode_position_and_get_embedding(
        self,
        container_state: torch.Tensor,
        container_features: torch.Tensor,
        box_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """解码位置并获取position embedding
        Args:
            container_state: [batch_size, L, W, 7] 原始容器状态
            container_features: [batch_size, L*W, d_model] 容器编码
            box_features: [batch_size, num_boxes, d_model] 箱子编码
            attention_mask: [batch_size, num_boxes] 箱子的mask
        Returns:
            position_logits: [batch_size, L*W] 位置logits
            position_embedding: [batch_size, L*W, d_model] 位置嵌入
        """

        position_query = container_features
        for decoder in self.position_decoder:
            position_query = decoder(
                query=position_query,
                key_value=box_features,
                key_padding_mask=attention_mask,
            )

        position_logits = self.position_head(position_query.mean(dim=1))

        if self.training:
            position = torch.multinomial(
                torch.softmax(position_logits, dim=-1), num_samples=1
            ).squeeze(-1)
        else:
            position = torch.argmax(position_logits, dim=-1)

        state_features = self._extract_position_feature(container_state, position)

        downsampled_position = self._map_to_downsampled_position(
            position, original_size=self.container_size
        )

        batch_indices = torch.arange(container_features.size(0), device=position.device)
        encoding_features = container_features[batch_indices, downsampled_position]

        combined_features = state_features + encoding_features

        position_embedding = container_features.clone()

        position_embedding[batch_indices, downsampled_position] = combined_features

        return position_logits, position_embedding

    def _decode_box_selection(
        self,
        box_features: torch.Tensor,
        position_embedding: torch.Tensor,
        boxes: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码箱子选择logits并生成orientation embedding
        Args:
            box_features: [batch_size, num_boxes, d_model] 箱子特征，作为query
            position_embedding: [batch_size, L*W, d_model] 位置嵌入，作为key-value
            boxes: [batch_size, num_boxes, 3] 原始箱子尺寸
            attention_mask: [batch_size, num_boxes] 箱子的mask
        Returns:
            box_selection_logits: [batch_size, num_boxes] 箱子选择的logits
            orientation_embedding: [batch_size, 6, d_model] 选中箱子的朝向嵌入
        """

        selection_query = box_features
        for decoder in self.selection_decoder:
            selection_query = decoder(
                query=selection_query,
                key_value=position_embedding,
            )

        box_selection_logits = self.box_selection_head(selection_query.mean(dim=1))

        box_selection_logits = box_selection_logits.masked_fill(
            attention_mask, float("-inf")
        )

        if self.training:
            selected_idx = torch.multinomial(
                torch.softmax(box_selection_logits, dim=-1), num_samples=1
            ).squeeze(-1)
        else:
            selected_idx = torch.argmax(box_selection_logits, dim=-1)

        batch_indices = torch.arange(boxes.size(0), device=boxes.device)
        selected_box = boxes[batch_indices, selected_idx]

        orientation_embedding = self._generate_orientation_embedding(selected_box)

        return box_selection_logits, orientation_embedding

    def _decode_orientation(
        self, orientation_embedding: torch.Tensor, position_embedding: torch.Tensor
    ) -> torch.Tensor:
        """解码朝向logits
        Args:
            orientation_embedding: [batch_size, 6, d_model] 箱子的6种朝向嵌入，作为query
            position_embedding: [batch_size, L*W, d_model] 位置嵌入，作为key-value
        Returns:
            orientation_logits: [batch_size, 6] 朝向选择的logits
        """

        orientation_query = orientation_embedding
        for decoder in self.orientation_decoder:
            orientation_query = decoder(
                query=orientation_query,
                key_value=position_embedding,
            )

        orientation_logits = self.orientation_head(orientation_query.mean(dim=1))

        return orientation_logits

    def extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        提取特征的方法，用于替代SB3默认的特征提取
        """

        boxes = obs["boxes"]
        boxes_mask = obs["boxes_mask"]
        container_state = obs["container_state"]

        attention_mask = ~boxes_mask

        box_features = self._encode_boxes(boxes, attention_mask)

        container_features = self._encode_container(container_state)

        combined_features = torch.cat(
            [box_features.mean(dim=1), container_features.mean(dim=1)], dim=-1
        )

        return combined_features

    def _get_latent(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取潜在特征，用于替代SB3默认的特征提取
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, latent_vf

    def _map_to_downsampled_position(
        self, position: torch.Tensor, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """将原始位置映射到下采样后的位置
        Args:
            position: [batch_size] 原始空间中的位置索引
            original_size: (L, W) 原始容器尺寸
        Returns:
            downsampled_position: [batch_size] 下采样空间中的位置索引
        """
        L, W = original_size
        patch_size = self.patch_size

        x = position // W
        y = position % W

        x_downsampled = x // patch_size
        y_downsampled = y // patch_size

        downsampled_W = W // patch_size
        downsampled_position = x_downsampled * downsampled_W + y_downsampled

        return downsampled_position


class ValueNetwork(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()

        self.box_transformer = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead) for _ in range(num_layers)]
        )

        self.container_transformer = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead) for _ in range(num_layers)]
        )

        self.cross_attention = TransformerDecoderBlock(d_model, nhead)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, box_features, container_features, attention_mask):

        for transformer in self.box_transformer:
            box_features = transformer(box_features, attention_mask)

        for transformer in self.container_transformer:
            container_features = transformer(container_features)

        value_features = self.cross_attention(
            container_features, box_features, attention_mask
        )

        value = self.value_head(value_features.mean(dim=1))
        return value
