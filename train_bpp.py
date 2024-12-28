import os
import sys
import yaml
import torch
import numpy as np
import gymnasium as gym
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.utils import set_random_seed
import envs


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    创建环境的辅助函数
    """

    def _init():
        env = gym.make(env_id, container_size=(10, 10, 10), n_boxes_range=(10, 10))
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def load_hyperparameters(env_id: str) -> dict:
    """
    从yaml文件加载超参数
    """
    with open("configs/ppo.yml", "r") as f:
        hyperparams = yaml.safe_load(f)

    if env_id in hyperparams:
        params = hyperparams[env_id]

        params["n_envs"] = int(multiprocessing.cpu_count() * 2)

        params["batch_size"] = 2048

        params["n_steps"] = 2048

        params["n_epochs"] = 20

        params["policy_kwargs"] = (
            "dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]))"
        )

        params["learning_rate"] = 3e-4

        params["clip_range"] = 0.2
        params["ent_coef"] = 0.01
        params["gae_lambda"] = 0.95
        params["gamma"] = 0.99

        return params
    else:
        raise ValueError(f"Hyperparameters not found for {env_id}")


def train(
    env_id: str = "BinPacking3D-v0", seed: int = 0, continue_training: bool = False
):
    """
    训练函数
    """

    hyperparams = load_hyperparameters(env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

    n_envs = hyperparams.get("n_envs", multiprocessing.cpu_count() * 2)
    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])

    if hyperparams.get("normalize", True):
        env = VecNormalize(env)

    log_dir = "logs/"
    tensorboard_log = f"{log_dir}/tensorboard/"
    os.makedirs(tensorboard_log, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="ppo_binpacking3d",
    )

    eval_env = DummyVecEnv([make_env(env_id, 0, seed + 1000)])
    if hyperparams.get("normalize", True):
        eval_env = VecNormalize(eval_env, training=False, norm_reward=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    if continue_training:

        model = PPO.load(f"{log_dir}/final_model", env=env, device=device)

        if hyperparams.get("normalize", True):
            env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", env)

            env.training = True

            eval_env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", eval_env)
            eval_env.training = False
    else:

        model = PPO(
            policy=hyperparams.get("policy", "MultiInputPolicy"),
            env=env,
            learning_rate=hyperparams.get("learning_rate", 3e-4),
            n_steps=hyperparams.get("n_steps", 128),
            batch_size=hyperparams.get("batch_size", 1024),
            n_epochs=hyperparams.get("n_epochs", 10),
            gamma=hyperparams.get("gamma", 0.98),
            gae_lambda=hyperparams.get("gae_lambda", 0.95),
            clip_range=hyperparams.get("clip_range", 0.2),
            clip_range_vf=None,
            ent_coef=hyperparams.get("ent_coef", 0.01),
            vf_coef=hyperparams.get("vf_coef", 0.5),
            max_grad_norm=hyperparams.get("max_grad_norm", 0.5),
            use_sde=hyperparams.get("use_sde", False),
            sde_sample_freq=hyperparams.get("sde_sample_freq", -1),
            target_kl=hyperparams.get("target_kl", None),
            tensorboard_log=tensorboard_log,
            policy_kwargs=eval(hyperparams.get("policy_kwargs", "dict()")),
            verbose=1,
            seed=seed,
            device=device,
        )

    print("开始训练...")

    custom_callback = CustomCallback()
    save_callback = SaveModelCallback(save_freq=10000, save_path=log_dir)
    model.learn(
        total_timesteps=int(hyperparams.get("n_timesteps", 5e6)),
        callback=[checkpoint_callback, eval_callback, custom_callback, save_callback],
        progress_bar=True,
    )

    model.save(f"{log_dir}/final_model")

    if hyperparams.get("normalize", True):
        env.save(f"{log_dir}/vec_normalize.pkl")

    return model


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.utilizations = []
        self.gap_ratios = []
        self.max_heights = []

    def _on_step(self):
        try:
            infos = self.locals.get("infos")
            if infos and len(infos) > 0:

                for info in infos:
                    if info:
                        self.utilizations.append(info.get("utilization", 0))
                        self.gap_ratios.append(info.get("gap_ratio", 0))
                        self.max_heights.append(info.get("max_height", 0))

                if len(self.utilizations) >= 10000:
                    self.logger.record("utilization/mean", np.mean(self.utilizations))
                    self.logger.record("gap_ratio/mean", np.mean(self.gap_ratios))
                    self.logger.record("max_height/mean", np.mean(self.max_heights))

                    self.utilizations = []
                    self.gap_ratios = []
                    self.max_heights = []

                    self.logger.dump(self.num_timesteps)
        except Exception as e:
            print(f"Warning in CustomCallback: {e}")
        return True


class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq=10000, save_path="logs/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:

            self.model.save(f"{self.save_path}/latest_model")

            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{self.save_path}/latest_vec_normalize.pkl")
        return True


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="训练3D装箱PPO模型")
        parser.add_argument(
            "--continue_training", action="store_true", help="是否继续训练已有模型"
        )
        args = parser.parse_args()

        print(f"CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")

        seed = np.random.randint(0, 10000)
        print(f"Using random seed: {seed}")

        print("Starting training...")
        sys.stdout.flush()
        model = train(seed=seed, continue_training=args.continue_training)
        print("Training completed successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
