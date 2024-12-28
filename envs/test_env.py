import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from envs.binpacking3d_env import BinPacking3DEnv

def visualize_height_map(height_map, step, save_dir):
    """可视化高度图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(height_map, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'Height Map - Step {step}')
    plt.savefig(os.path.join(save_dir, f'height_map_step_{step}.png'))
    plt.close()

def visualize_box_selection_mask(mask, step, save_dir):
    """可视化箱子选择掩码"""
    plt.figure(figsize=(10, 2))
    plt.imshow([mask], aspect='auto', cmap='Blues')
    plt.title(f'Box Selection Mask - Step {step}')
    # 添加数值标注
    for i in range(len(mask)):
        plt.text(i, 0, str(int(mask[i])), ha='center', va='center')
    plt.xlabel('Box Index')
    plt.savefig(os.path.join(save_dir, f'box_selection_mask_step_{step}.png'))
    plt.close()

def visualize_orientation_mask(mask, step, save_dir):
    """可视化朝向掩码"""
    plt.figure(figsize=(8, 2))
    plt.imshow([mask], aspect='auto', cmap='Blues')
    plt.title(f'Orientation Mask - Step {step}')
    # 添加数值标注
    for i in range(len(mask)):
        plt.text(i, 0, str(int(mask[i])), ha='center', va='center')
    plt.xlabel('Orientation')
    plt.savefig(os.path.join(save_dir, f'orientation_mask_step_{step}.png'))
    plt.close()

def visualize_position_mask(mask, step, save_dir):
    """可视化位置掩码"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(mask.astype(int), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Position Mask - Step {step}')
    plt.savefig(os.path.join(save_dir, f'position_mask_step_{step}.png'))
    plt.close()

def test_env():
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"test_results/{timestamp}"
    viz_dir = f"{result_dir}/viz"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 创建环境
    env = BinPacking3DEnv(
        container_size=(100,100,100),  # 小型环境方便测试
        n_boxes_range=(10,30)          # 3-5个箱子
    )
    print("环境创建成功!")
    
    with open(f"{result_dir}/log.txt", 'w') as f:
        # 记录空间信息
        f.write("空间信息:\n")
        f.write(f"观察空间: {env.observation_space}\n")
        f.write(f"动作空间: {env.action_space}\n")
        print("\n空间信息:")
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 重置环境
        obs, info = env.reset()
        f.write("\n初始状态:\n")
        
        # 记录箱子信息
        f.write("\n可用箱子:\n")
        print("\n可用箱子:")
        for i, box in enumerate(env.boxes):
            box_info = f"箱子 {i}: 尺寸={box.dimensions}"
            f.write(box_info + '\n')
            print(box_info)
        
        # 记录并可视化初始高度图
        f.write("\n初始高度图:\n")
        np.savetxt(f, env.height_map, fmt='%3d')
        visualize_height_map(env.height_map, 'initial', viz_dir)
        print("\n初始高度图已保存")
        
        # 测试动作掩码
        action_mask = env.get_action_mask()
        f.write("\n动作掩码:\n")
        
        # 1. 箱子选择掩码
        f.write("可用箱子掩码:\n")
        np.savetxt(f, [action_mask['box_selection']], fmt='%d')
        print("\n可用箱子:", np.where(action_mask['box_selection'])[0])
        visualize_box_selection_mask(action_mask['box_selection'], 'initial', viz_dir)
        
        # 2. 朝向掩码
        f.write("\n朝向掩码:\n")
        np.savetxt(f, [action_mask['box_orientation']], fmt='%d')
        print("可用朝向:", np.where(action_mask['box_orientation'])[0])
        visualize_orientation_mask(action_mask['box_orientation'], 'initial', viz_dir)
        
        # 3. 位置掩码
        f.write("\n可用位置掩码:\n")
        np.savetxt(f, action_mask['position'].astype(int), fmt='%d')
        valid_positions = np.where(action_mask['position'])
        print(f"有效位置数量: {len(valid_positions[0])}")
        visualize_position_mask(action_mask['position'], 'initial', viz_dir)
        
        # 执行随机动作
        print("\n开始测试随机动作...")
        for i in range(3):  # 执行3步
            action_mask = env.get_action_mask()
            
            # 获取有效的箱子索引
            valid_box_indices = np.where(action_mask['box_selection'])[0]
            if len(valid_box_indices) == 0:
                msg = "没有可用的箱子!"
                f.write(msg + '\n')
                print(msg)
                break
            
            # 获取有效的位置
            valid_positions = np.where(action_mask['position'])
            if len(valid_positions[0]) == 0:
                msg = "没有有效的放置位置!"
                f.write(msg + '\n')
                print(msg)
                break
            
            # 随机选择有效动作
            box_idx = np.random.choice(valid_box_indices)
            pos_idx = np.random.randint(len(valid_positions[0]))
            x, y = valid_positions[0][pos_idx], valid_positions[1][pos_idx]
            orientation = np.random.randint(6)
            
            # 记录动作信息
            f.write(f"\nStep {i+1}:\n")
            box = env.boxes[box_idx]
            action_info = [
                f"选择的箱子 {box_idx}:",
                f"- 原始尺寸: {box.dimensions}",
                f"- 旋转后尺寸: {box.get_rotated_dimensions(orientation)}",
                f"- 放置位置: ({x}, {y})",
                f"- 当前位置高度: {env.height_map[x, y]}"
            ]
            for info in action_info:
                f.write(info + '\n')
                print(info)
            
            # 执行动作
            action = [box_idx, orientation, x, y]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录结果
            result_info = [
                f"\n执行结果:",
                f"- 奖励: {reward}",
                f"- 终止: terminated={terminated}, truncated={truncated}",
                f"- 信息: {info}"
            ]
            for info in result_info:
                f.write(info + '\n')
                print(info)
            
            # 记录并可视化更新后的高度图
            f.write("\n更新后的高度图:\n")
            np.savetxt(f, env.height_map, fmt='%3d')
            visualize_height_map(env.height_map, i+1, viz_dir)
            print(f"步骤 {i+1} 的高度图已保存")
            
            if terminated or truncated:
                msg = "回合结束!"
                f.write('\n' + msg + '\n')
                print(msg)
                break
        
        # 记录最终状态
        final_info = [
            "\n最终状态:",
            f"已放置箱子数量: {np.sum(env.packed_mask)}/{len(env.boxes)}",
            f"最大高度: {np.max(env.height_map)}"
        ]
        for info in final_info:
            f.write(info + '\n')
            print(info)
    
    env.close()
    print(f"\n测试完成! 结果保存在: {result_dir}")

if __name__ == "__main__":
    test_env()