import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import envs
import os
import datetime
import argparse
from policies.transformer_policy import CustomTransformerPolicy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def evaluate_model(
    model_path, vec_normalize_path, policy_type="multi_input", env_id="BinPacking3D-v0", n_eval_episodes=10
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

    env = gym.make(
        env_id,
        container_size=(10, 10, 10),
        n_boxes_range=(5, 15),
        use_position_mask=False
    )
    env = DummyVecEnv([lambda: env])

    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False

    unwrapped_env = env.unwrapped.envs[0].unwrapped

    if policy_type == "transformer":
        model = PPO.load(
            model_path,
            custom_objects={
                "policy_class": CustomTransformerPolicy
            }
        )
    else:
        model = PPO.load(model_path)

    episode_stats = []

    for episode in range(n_eval_episodes):
        log_print(f"\n====== Episode {episode+1} ======")
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        last_valid_info = None
        height_maps = []  # 在每个 episode 开始时初始化列表

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
            height_map = unwrapped_env.height_map
            log_print(height_map)
            height_maps.append(height_map.copy())  # 收集每一步的 height map

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

        # episode 结束后生成动画
        visualize_height_map(height_maps, f"height_maps_episode_{episode}.gif")

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


def visualize_height_map(height_maps, save_path="height_map_animation.gif"):
    """将一个 episode 的 height maps 制作成动画"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 设置固定的值范围 (0-10)
    im = ax.imshow(height_maps[0], cmap='YlOrRd', vmin=0, vmax=10)
    plt.colorbar(im, ax=ax)
    
    def update(frame):
        ax.clear()
        # 保持相同的值范围
        im = ax.imshow(height_maps[frame], cmap='YlOrRd', vmin=0, vmax=10)
        ax.set_title(f'Step {frame + 1}')
        return [im]
        
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(height_maps),
        interval=1000,
        blit=True
    )
    
    anim.save(save_path, writer='pillow')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="评估3D装箱模型")
    parser.add_argument("--logs_dir", type=str, default="logs", help="日志目录路径")
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=["multi_input", "transformer"],
        default="multi_input",
        help="要评估的策略类型"
    )
    parser.add_argument('--visualize', action='store_true', 
                       help='是否生成装箱过程的可视化动画')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='输出目录的根路径')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录结构
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(args.output_dir, os.path.basename(args.logs_dir), timestamp)
    txt_dir = os.path.join(output_base, 'logs')  # 存放文本日志
    vis_dir = os.path.join(output_base, 'visualizations')  # 存放可视化结果
    
    os.makedirs(txt_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(txt_dir, f'eval_results.txt')
    
    def log_print(*args, **kwargs):
        """同时输出到控制台和文件"""
        print(*args, **kwargs)
        with open(output_file, "a", encoding="utf-8") as f:
            print(*args, **kwargs, file=f)
    
    # 记录基本信息
    log_print(f"评估时间: {timestamp}")
    log_print(f"模型路径: {args.logs_dir}")
    log_print("=" * 50)

    if args.policy_type == "transformer":
        model_path = os.path.join(args.logs_dir, "best_model/best_model.zip")
        vec_normalize_path = os.path.join(args.logs_dir, "env_checkpoints/vec_normalize_150000_steps.pkl")
    else:
        model_path = os.path.join(args.logs_dir, "best_model/best_model.zip")
        vec_normalize_path = os.path.join(args.logs_dir, "vec_normalize.pkl")

    env = gym.make(
        "BinPacking3D-v0",
        container_size=(10, 10, 10),
        n_boxes_range=(5, 15),
        use_position_mask=False
    )
    env = DummyVecEnv([lambda: env])

    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False

    unwrapped_env = env.unwrapped.envs[0].unwrapped

    if args.policy_type == "transformer":
        model = PPO.load(
            model_path,
            custom_objects={
                "policy_class": CustomTransformerPolicy
            }
        )
    else:
        model = PPO.load(model_path)

    episode_stats = []

    for episode in range(10):
        log_print(f"\n====== Episode {episode+1} ======")
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        last_valid_info = None
        height_maps = []  # 在每个 episode 开始时初始化列表

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
            height_map = unwrapped_env.height_map
            log_print(height_map)
            height_maps.append(height_map.copy())  # 收集每一步的 height map

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

        # 如果启用可视化，保存到可视化目录
        if args.visualize:
            vis_path = os.path.join(vis_dir, f'height_maps_episode_{episode}.gif')
            visualize_height_map(height_maps, save_path=vis_path)

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
    main()
