# PackFormer: 基于 Transformer 的三维装箱问题求解器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

基于论文 [Solving 3D packing problem using Transformer network and reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0957417422021716) 的非官方实现。

## 项目结构

```
PackFormer/
├── configs/                    # 配置文件目录
│   └── ppo.yml                 # PPO 算法配置
│
├── envs/                      # 环境实现目录
│   ├── __init__.py          # 环境注册
│   ├── box.py               # 箱子类定义
│   ├── box_generator.py     # 箱子生成器
│   ├── binpacking3d_env.py  # 主要环境实现
│   └── test_env.py          # 环境测试
│
├── policies/                  # 策略网络目录
│   └── transformer_policy.py # Transformer 策略实现
│
├── visualization/            # 可视化工具目录
│   └── vis_raw_box.py       # 装箱结果可视化
│
├── train_transformer_bpp.py  # Transformer 训练脚本
├── train_bpp.py             # 基准模型训练脚本
├── eval_bpp.py              # 评估脚本
├── requirements.txt         # 项目依赖
└── README.md               # 项目文档
```

### 核心模块说明

#### 1. 环境模块 (`envs/`)

- `box.py`: 实现箱子的基本属性和操作，包括位置设置、旋转和分割等功能
- `box_generator.py`: 提供两种箱子生成策略：基于体积的分割和基于坐标的分割
- `binpacking3d_env.py`: 实现符合 Gymnasium 接口的 3D 装箱环境，包括：
  - 状态表示：高度图和平面特征
  - 动作空间：箱子选择、旋转和位置
  - 奖励设计：体积利用率和稳定性考虑

#### 2. 策略模块 (`policies/`)

- `transformer_policy.py`: 实现基于 Transformer 的策略网络，包括：
  - Box Encoder: 编码箱子特征
  - Container Encoder: 编码容器状态
  - 多头注意力机制的位置和方向预测

#### 3. 配置模块 (`configs/`)

- `ppo.yml`: PPO 算法的超参数配置

#### 4. 训练和评估

- `train_bpp.py`: MLP模型的训练流程
- `train_transformer_bpp.py`：Transformer模型的训练
- `eval_bpp.py`: MLP模型评估和性能测试
- `visualization/`: 数据集生产的可视化工具

## 算法说明

本项目使用 Transformer 网络结合强化学习来解决三维装箱问题（3D-BPP）。主要特点包括：

1. **状态表示**
   - 使用平面特征（plane feature）表示容器状态
   - 动态编码剩余箱子信息

2. **网络架构**
   - Box Encoder：编码箱子特征
   - Container Encoder：编码容器状态
   - 多头注意力机制的位置和方向预测

3. **训练策略**
   - 使用 PPO 算法进行策略优化
   - 采用多阶段奖励函数
   - 实现动作掩码确保可行性

## 环境要求

- Python =3.9
- PyTorch =2.5.1
- CUDA =11.8 (如果使用 GPU)

### 安装步骤

1. 创建并激活 Conda 环境

```bash
# 创建环境
conda create -n bpp_rl python=3.9

# 激活环境
conda activate bpp_rl
```

2. 安装 PyTorch (GPU 版本)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. 安装 Stable-Baselines3 及其依赖

```bash
pip install stable-baselines3[extra]
```



## Get Started

### 训练

#### MLP模型训练
在开始训练前，请先在 `configs/ppo.yml` 中设置训练参数。

新训练：
```bash
python train_bpp.py
```

继续训练：
```bash
python train_bpp.py --continue_training
```

参数说明：
- `--continue_training`: 继续训练已有模型
- 其他训练参数请在 `configs/ppo.yml` 中配置

#### Transformer Policy 训练

使用 Transformer 架构的模型训练：

新训练：
```bash
python train_transformer_bpp.py --gpu 0
```

继续训练：
```bash
python train_transformer_bpp.py --continue_training --load_path transformer_YYYYMMDD_HHMMSS --gpu 0
```

参数说明：
- `--continue_training`: 继续训练已有模型
- `--load_path`: 继续训练时，指定要加载的模型路径（logs目录下的子目录名）
- `--gpu`: 指定使用的 GPU 设备编号（-1 表示使用 CPU）

训练过程：
1. 每轮收集 512 步数据，存入 RolloutBuffer
2. 对收集的数据进行 10 轮训练，每轮使用 128 大小的批次
3. 每 50000 步进行一次评估
4. 每 10000 步保存一次检查点和环境参数

主要训练指标：
- `utilization`: 容器空间利用率
- `gap_ratio`: 箱子之间的空隙比率
- `max_height`: 当前堆叠的最大高度
- `ep_len_mean`: 每个 episode 的平均长度
- `ep_rew_mean`: 每个 episode 的平均奖励

### 评估

```bash
# 评估 MLP 模型（默认）
python eval_bpp.py --logs_dir your/custom/path --policy_type multi_input

# 评估 Transformer 模型
python eval_bpp.py --logs_dir your/custom/path --policy_type transformer

# 生成装箱过程可视化
python eval_bpp.py --logs_dir your/custom/path --visualize

# 指定评估结果保存路径
python eval_bpp.py --logs_dir your/custom/path --output_dir custom_results
```

参数说明：
- `--logs_dir`: 指定日志目录路径，默认为 "logs"
- `--policy_type`: 指定要评估的策略类型：
  - `multi_input`: MLP 策略（默认）
  - `transformer`: Transformer 策略
- `--visualize`: 是否生成装箱过程的可视化动画
- `--output_dir`: 指定输出目录的根路径，默认为 "test_results"

评估输出：
```
test_results/
└── your_model_name/
    └── 20240101_123456/          # 时间戳目录
        ├── logs/                 # 评估日志
        │   └── eval_results.txt  # 详细评估结果
        └── visualizations/       # 可视化结果（如果启用）
            ├── height_maps_episode_0.gif
            ├── height_maps_episode_1.gif
            └── ...
```

预训练模型：
- 您可以从项目的 [Releases](https://github.com/Blake-Jiang/PackFormer/releases) 页面下载预训练的模型检查点
- 下载后将文件解压到项目的 `logs` 目录下即可使用

注意：
- 对于 MLP 策略，需要确保指定目录下存在 `best_model/best_model.zip` 和 `vec_normalize.pkl` 文件
- 对于 Transformer 策略，需要确保指定目录下存在 `best_model/best_model.zip` 和 `env_checkpoints/vec_normalize_150000_steps.pkl` 文件，这里的 `150000` 是最后一个checkpoint 保存时的步数，需要根据实际情况调整



# Timeline

## 第一阶段：环境搭建

### 基础环境设置

- [x] 创建项目结构
- [x] 设置虚拟环境
- [x] 安装必要的依赖
- [x] 创建配置文件模板

### 核心类实现

- [x] 实现 Box 类
  - [x] 基本属性（尺寸）
  - [x] 位置和旋转方法
  - [x] 碰撞检测方法
- [x] 实现 BinPackingEnv 类
  - [x] 初始化方法
  - [x] reset 方法
  - [x] step 方法的基本框架
  - [x] 简单的奖励函数

### 基础功能实现

- [x] 实现箱子生成器
- [x] 实现基本的放置逻辑
- [x] 实现简单的可视化功能
- [x] 添加基本的日志记录

## 第二阶段：强化学习设计

### 观察空间设计

- [x] plane feature
- [x] 剩余box状态

### 动作空间设计

- [x] 实现位置选择（x,y,z）
- [x] 实现旋转选择
- [x] 添加动作有效性检查
- [x] 实现动作掩码

### 奖励函数设计

- [x] 实现体积利用率计算
- [x] 实现稳定性评估
- [x] 设计组合奖励函数
- [x] 优化奖励函数
  - [x] 重新设计即时奖励
  - [x] 添加稀疏奖励
  - [x] 优化终止奖励
- [x] 改进终止条件
  - [x] 添加提前终止逻辑
  - [x] 优化终止状态判断

## 第三阶段：训练优化

### 训练框架搭建

- [x] 设置 Tensorboard 监控
- [x] 实现模型保存和加载
- [x] 添加训练中断和恢复

### 模型架构实现

- [ ] Stable Baselines 3 MultiInputPolicy

  - [x] 固定箱子数量训练
  - [x] 评估

  - [ ] box 数量可变训练




- [x] Transformer Policy

  - [x] Encoder 实现
    - [x] BoxEncoder
      - [x] 箱子特征嵌入
      - [x] 自注意力编码
      - [x] 位置编码
    - [x] ContainerEncoder
      - [x] 容器状态下采样
      - [x] 特征映射
      - [x] 自注意力编码
  - [x] Decoder 实现
    - [x] Position Decoder
      - [x] 交叉注意力解码
      - [x] 位置特征提取
      - [x] 位置嵌入生成
    - [x] Box Selection Decoder
      - [x] 交叉注意力解码
      - [x] 箱子选择预测
      - [x] 动作掩码处理
    - [x] Orientation Decoder
      - [x] 朝向特征生成
      - [x] 交叉注意力解码
      - [x] 朝向预测
  - [x] Actor-Critic 架构
    - [x] 动作分布设计
      - [x] 组合多个 Categorical 分布
      - [x] 实现采样和概率计算
    - [x] Value Network
      - [x] 特征提取器
      - [x] 交叉注意力层
      - [x] 价值预测头
  - [x] 训练实现
    - [x] PPO 算法适配
    - [x] 并行环境设置
    - [x] 训练参数优化
    - [x] 模型保存和加载



### 参数优化

- [ ] 测试不同的网络结构
- [ ] 优化学习率和批量大小
- [ ] 调整奖励权重



## 第四阶段：评估和改进

### 评估系统

- [x] 实现评估指标计算

- [x] 添加可视化工具

- [ ] 创建测试数据集

  

### 文档和部署

- [x] 编写使用文档

## Acknowledgements

本项目基于以下开源项目和研究工作：

- [Solving 3D packing problem using Transformer network and reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0957417422021716) - 提供了核心算法思路
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 提供了强化学习算法框架
- [PyTorch](https://pytorch.org/) - 深度学习框架支持
- [Gymnasium](https://gymnasium.farama.org/) - 强化学习环境接口

感谢所有为开源社区做出贡献的开发者。

  