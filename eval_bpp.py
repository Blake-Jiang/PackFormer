import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import envs
import os
import datetime
import argparse


def evaluate_model(
    model_path, vec_normalize_path, env_id="BinPacking3D-v0", n_eval_episodes=10
):
    """
    评估训练好的模型
    """
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_results_{timestamp}.txt")

    def log_print(*args, **kwargs):
        """同时输出到控制台和文件"""
        print(*args, **kwargs)
        with open(output_file, "a", encoding="utf-8") as f:
            print(*args, **kwargs, file=f)

    log_print(f"评估时间: {timestamp}")
    log_print(f"模型路径: {model_path}")
    log_print(f"环境ID: {env_id}")
    log_print(f"评估回合数: {n_eval_episodes}")
    log_print("=" * 50)

    env = gym.make(env_id, container_size=(10, 10, 10), n_boxes_range=(10, 10))
    env = DummyVecEnv([lambda: env])

    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False

    unwrapped_env = env.unwrapped.envs[0].unwrapped
    model = PPO.load(model_path)

    episode_stats = []

    for episode in range(n_eval_episodes):
        log_print(f"\n====== Episode {episode+1} ======")
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        last_valid_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = action[0]

            box_idx = action[0]
            orientation = action[1]
            pos_x, pos_y = action[2], action[3]

            obs, reward, done, info = env.step([action])
            reward = reward[0]
            done = done[0]
            info = info[0]

            log_print(f"\n步骤 {episode_length + 1}:")
            log_print("Height Map:")
            log_print(unwrapped_env.height_map)

            if "action" in info:
                action_info = info["action"]
                log_print(f"放置箱子: Box {box_idx}")
                log_print(f"位置: ({pos_x}, {pos_y})")
                log_print(f"朝向: {orientation}")
                if "dimensions" in action_info:
                    log_print(f"箱子尺寸: {action_info['dimensions']}")

            log_print(f"奖励: {float(reward):.2f}")
            log_print(f"空间利用率: {info.get('utilization', 0):.2%}")
            log_print(f"间隙比率: {info.get('gap_ratio', 0):.2%}")
            log_print(f"最大高度: {info.get('max_height', 0)}")
            log_print("------------------------")

            total_reward += reward
            episode_length += 1

            if not done:
                last_valid_info = info.copy()

        final_info = last_valid_info if last_valid_info is not None else info

        episode_stats.append(
            {
                "reward": float(total_reward),
                "length": episode_length,
                "utilization": final_info.get("utilization", 0),
                "gap_ratio": final_info.get("gap_ratio", 0),
                "max_height": final_info.get("max_height", 0),
            }
        )

        log_print(f"\nEpisode {episode+1} 总结:")
        log_print(f"总奖励: {float(total_reward):.2f}")
        log_print(f"回合长度: {episode_length}")
        log_print(f"最终空间利用率: {final_info.get('utilization', 0):.2%}")
        log_print(f"最终间隙比率: {final_info.get('gap_ratio', 0):.2%}")
        log_print(f"最终最大高度: {final_info.get('max_height', 0)}")
        log_print("============================")

    log_print("\n=== 评估总结 ===")
    log_print(
        f"平均奖励: {np.mean([s['reward'] for s in episode_stats]):.2f} ± {np.std([s['reward'] for s in episode_stats]):.2f}"
    )
    log_print(
        f"平均空间利用率: {np.mean([s['utilization'] for s in episode_stats]):.2%} ± {np.std([s['utilization'] for s in episode_stats]):.2%}"
    )
    log_print(
        f"平均间隙比率: {np.mean([s['gap_ratio'] for s in episode_stats]):.2%} ± {np.std([s['gap_ratio'] for s in episode_stats]):.2%}"
    )
    log_print(
        f"平均最大高度: {np.mean([s['max_height'] for s in episode_stats]):.2f} ± {np.std([s['max_height'] for s in episode_stats]):.2f}"
    )

    log_print(f"\n评估结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估3D装箱模型")
    parser.add_argument("--logs_dir", type=str, default="logs", help="日志目录路径")
    args = parser.parse_args()

    model_path = os.path.join(args.logs_dir, "best_model/best_model.zip")
    vec_normalize_path = os.path.join(args.logs_dir, "vec_normalize.pkl")
    evaluate_model(model_path, vec_normalize_path)
