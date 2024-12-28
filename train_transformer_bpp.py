import os
import sys
import torch
import numpy as np
import gymnasium as gym
import multiprocessing
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.utils import set_random_seed
from policies.transformer_policy import CustomTransformerPolicy
import envs
import datetime


def linear_schedule(initial_value: float):
    """
    线性学习率调度
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    创建环境的辅助函数
    """

    def _init():
        env = gym.make(
            env_id,
            container_size=(10, 10, 10),
            n_boxes_range=(5, 15),
            use_position_mask=False,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


class CustomMetricsCallback(BaseCallback):
    """用于记录训练指标的回调"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {"utilization": [], "gap_ratio": [], "max_height": []}

    def _on_step(self):
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            for key in self.metrics:
                if key in info:
                    self.metrics[key].append(info[key])
                    self.logger.record(f"metrics/{key}", info[key])
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """用于定期保存VecNormalize环境的回调"""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path, f"vec_normalize_{self.n_calls}_steps.pkl"
            )
            self.training_env.save(path)
            if self.verbose > 0:
                print(f"Saved environment checkpoint to {path}")
        return True


def main():

    parser = argparse.ArgumentParser(description="训练Transformer BPP模型")
    parser.add_argument(
        "--continue_training", action="store_true", help="是否继续之前的训练"
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="加载模型的路径（logs目录下的子目录名）",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="指定要使用的 GPU 设备编号（-1 表示使用 CPU）",
    )
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    env_id = "BinPacking3D-v0"
    seed = np.random.randint(0, 10000)
    print(f"Using seed: {seed}")

    n_envs = int(multiprocessing.cpu_count() * 1)
    print(f"Creating {n_envs} environments")

    env = SubprocVecEnv(
        [
            make_env(
                env_id=env_id,
                rank=i,
                seed=seed + i,
            )
            for i in range(n_envs)
        ]
    )

    if args.continue_training and args.load_path:
        log_dir = f"logs/{args.load_path}"
        if not os.path.exists(log_dir):
            raise ValueError(f"指定的日志目录不存在: {log_dir}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/transformer_{timestamp}/"

    tensorboard_log = f"{log_dir}/tensorboard/"
    os.makedirs(tensorboard_log, exist_ok=True)

    if args.continue_training and args.load_path:

        env = VecNormalize.load(f"{log_dir}/vec_normalize_transformer.pkl", env)

        env.training = True
        print(
            f"Loaded environment normalization from {log_dir}/vec_normalize_transformer.pkl"
        )
    else:

        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

    if args.continue_training and args.load_path:

        model_path = f"{log_dir}/final_model_transformer.zip"
        if not os.path.exists(model_path):

            checkpoints = sorted(
                [
                    f
                    for f in os.listdir(f"{log_dir}/checkpoints/")
                    if f.startswith("transformer_ppo_") and f.endswith(".zip")
                ]
            )
            if checkpoints:
                model_path = f"{log_dir}/checkpoints/{checkpoints[-1]}"
            else:
                raise ValueError(f"在指定目录下未找到可加载的模型: {log_dir}")

        print(f"Loading model from {model_path}")
        model = PPO.load(
            model_path,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": linear_schedule(3e-4),
                "policy_class": CustomTransformerPolicy,
            },
        )
    else:

        model = PPO(
            CustomTransformerPolicy,
            env,
            learning_rate=linear_schedule(3e-4),
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=tensorboard_log,
            policy_kwargs=dict(
                d_model=64,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                patch_size=1,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5),
            ),
            verbose=1,
            device=device,
        )

    eval_env = SubprocVecEnv([make_env(env_id, 0, seed + 1000)])
    if args.continue_training and args.load_path:

        eval_env = VecNormalize.load(
            f"{log_dir}/vec_normalize_transformer.pkl", eval_env
        )
        eval_env.training = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            training=False,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="transformer_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    save_env_callback = SaveVecNormalizeCallback(
        save_freq=10000, save_path=f"{log_dir}/env_checkpoints/", verbose=1
    )

    callbacks = [
        checkpoint_callback,
        eval_callback,
        CustomMetricsCallback(),
        save_env_callback,
    ]

    os.makedirs(f"{log_dir}/checkpoints/", exist_ok=True)
    os.makedirs(f"{log_dir}/env_checkpoints/", exist_ok=True)

    try:
        model.learn(
            total_timesteps=1e8,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not args.continue_training,
        )

        model.save(f"{log_dir}/final_model_transformer")
        env.save(f"{log_dir}/vec_normalize_transformer.pkl")

    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
