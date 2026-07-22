# FSSH — Fewest-Switches Surface Hopping 的 Python 实现

## 项目简介

本项目是经典非绝热动力学算法 **FSSH（Fewest-Switches Surface Hopping）** 的 Python 实现，用于模拟两态体系中电子-核耦合的量子经典动力学过程。代码在王林军课题组科研训练期间编写，复现了 Tully 1990 年的经典论文结果。

核心物理场景：粒子从左侧入射，经过两态 avoided crossing 区域，最终可能 **透射** 或 **反射**，并停留在 **高能态** 或 **低能态**。程序统计四种结果概率（TL / RL / TH / RH）随入射动量 k 的变化。

配套的 C 语言高性能版本见同目录下的 `FSSH_by_C` 项目。

## 物理模型

当前激活的是 **模型一**（Tully Model I — 简单避免交叉），另有模型二、模型三以注释形式保留：

| 模型 | 描述 | 参数 |
|------|------|------|
| 模型一（当前） | Tully Model I — 简单避免交叉 | A=0.01, B=1.6, C=0.005, D=1.0 |
| 模型二 | 双势阱型避免交叉 | A=0.1, B=0.28, E0=0.05, C=0.015, D=0.06 |
| 模型三 | 非对称耦合模型 | A=6e-4, B=0.1, C=0.9 |

透热哈密顿矩阵（模型一）：

```
V11 = ±A·(1 - exp(∓Bx)),  V22 = -V11,  V12 = V21 = C·exp(-D·x²)
```

## 算法流程

每条 FSSH 轨迹的单步演化分为三个阶段：

1. **Step II — 核动力学 + 电子振幅演化**
   - Velocity-Verlet 更新核坐标 `x` 和动量 `k`
   - 刷新 H_di、U、H_ad、d12、H_eff
   - RKG 四阶方法积分电子振幅向量 `C(t)`

2. **Step III — 跃迁判定**
   - 计算跃迁概率 g12（低→高）或 g21（高→低）
   - 生成随机数 ξ，若 g > ξ 则发生面跃迁

3. **Step IV — 跃迁修正**
   - 若发生跃迁：计算跃迁后动量 `k_aft`，动能不足则撤销跃迁
   - 更新核受力 F 和体系总能量

**终止条件**：核坐标超出 ±X_Max 边界，或步数超过 Step_Max。

## 项目结构

```
FSSH/
├── main.py                     # 程序入口
├── fssh_class.py               # FSSH 类定义（单条轨迹 + 批量模拟）
├── math1.py                    # 物理量计算函数库
├── __init__.py                  # Python 包标识
├── test_input                   # 动量输入文件
├── test_out                     # 轨迹输出数据（大文件）
├── other/                       # 辅助工具脚本
│   ├── draw.py                  # 从粘贴数据绘图
│   ├── Ensemble-sta.py          # 系综统计（固定 sigma）
│   ├── Ensemble-sta-2.py        # 系综统计（k-dependent sigma）
│   ├── Landau-Zener.py          # Landau-Zener 公式解析计算
│   ├── sat_input                # 系综统计输入数据
│   └── sat_out                  # 系综统计输出数据
└── .vscode/                     # VS Code 配置
```

## 模块说明

### math1.py — 物理量计算函数库

项目的核心数学/物理模块，提供所有量子力学量的计算函数：

- `H_di(x)`：构造透热哈密顿矩阵（2×2 numpy 数组）
- `U(H)` / `H_ad(H)`：通过 `np.linalg.eigh` 计算酉变换矩阵和绝热特征值
- `stable_eigh_first(H)`：相位一致的 eigh，保证特征向量符号统一
- `dH_di_dx(x)`：中心差分计算 dH/dx
- `d12(x)`：非绝热耦合常数 = [U†·dH/dx·U]₀₁ / (E₂-E₁)
- `F(x, C)`：绝热态势能面上的核受力（中心差分求势能导数取负）
- `x_change_by_t` / `k_change_by_t`：Velocity-Verlet 核动力学更新
- `E_total(k, m, C, H_ad)`：体系总能量（动能 + 势能）
- `H_eff(H_ad, k, m, d12)`：有效哈密顿量 H_eff = H_ad - iħ·v·d12
- `C_change_by_t(H_eff, C_t, del_t)`：矩阵指数法积分电子振幅（`scipy.linalg.expm`）
- `C_change_by_t_RKG(...)`：RKG 四阶方法积分电子振幅 ODE
- `g12` / `g21`：FSSH 跃迁概率
- `k_aft(k, m, H_ad, C)`：跃迁后动量修正，动能不足返回 -10086

### fssh_class.py — FSSH 类

面向对象封装，`FSSH` 类包含：

**类变量（全局超参数）**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| Step_Max | 400000 | 单轨迹最大步数 |
| X_Max | 5 | 坐标逃逸边界 ±5 |
| Number_of_simulations | 3000 | 每个动量下模拟轨迹数 |
| M_Quality | 2000 | 粒子质量 |
| Dleta_T | 0.1 | 时间步长 |

**实例属性**：电子态归属、振幅向量、核坐标/动量、哈密顿量系列、受力、跃迁概率、总能量等。

**核心方法**：
- `__init__(k_Momentum, ...)`：初始化一条轨迹
- `step_II()` / `step_III()` / `step_IV()`：三阶段单步演化
- `start_simulation()`：循环演化至终止，同时写入轨迹数据到 `test_out`
- `if_stop_simulation()`：终止判定
- `simulation_result()`：返回 [低能/高能, 透射/反射]

**批量方法（多进程并行）**：
- `Create_simulations_k(k)`：单动量批量模拟，使用 `ProcessPoolExecutor` 拉满 CPU
- `Create_simulations_from_k1_to_k2(k1, k2, gap)`：动量区间遍历
- `Create_simulations_from_input()`：从 `test_input` 文件读取动量列表

**参数调节**：`Set_Step_Max`、`Set_X_Max`、`Set_Number_of_simulations`

**绘图**：`Draw_Simulation_Results_Plot()` — 绘制 TL / RL / TH / RH 曲线

### other/ — 辅助工具

#### draw.py
从标准输入粘贴矩阵数据，自动清洗、还原、排序后绘图。用于快速可视化 `Simulation_Results_Plot` 的打印输出。

#### Ensemble-sta.py
系综统计工具（固定 sigma=1.4）。对 FSSH 输出的概率数据做高斯波包加权平均，模拟波包入射的物理场景。3σ 原则选取邻域，归一化权重后输出到 `sat_out`。

#### Ensemble-sta-2.py
改进版系综统计（sigma 与动量 k 相关：σ = k/20）。每个动量点根据自身展宽独立计算加权窗口，边界处自动截断。

#### Landau-Zener.py
Landau-Zener 公式解析计算。在 avoided crossing 点 (x=0) 处计算跃迁概率：

```
P_LZ = exp(-2π·V₁₂² / (ΔF·v))
```

用于与 FSSH 模拟结果对比验证。

## 运行方式

### 依赖

```
numpy
matplotlib
scipy
tqdm
```

### 快速运行

```bash
python main.py
```

当前 `main.py` 配置：从 `test_input` 读取动量值，每个动量模拟 1 条轨迹，输出结果并绘图。

`test_input` 文件格式：每行一个动量值（浮点数），支持 `#` 开头的注释行。

### 自定义配置

```python
FSSH.Set_Number_of_simulations(3000)           # 调整轨迹数
FSSH.Set_Step_Max(200000)                       # 调整最大步数
FSSH.Create_simulations_from_k1_to_k2(5, 35, 0.5)  # 区间遍历
FSSH.Draw_Simulation_Results_Plot()             # 绘图
```

### 输出

- `test_out`：每条轨迹的逐步演化数据（步数、振幅、坐标、态归属、受力、动量、g12）
- `Simulation_Results_Plot`：类属性，6 列 numpy 数组 [k, TL, RL, TH, RH, other]
- `sat_input` / `sat_out`：系综统计的输入/输出数据文件

## 与 C 版本的关系

本 Python 版本是原型实现，侧重可读性和调试方便性。对应的 C 语言高性能版本（`FSSH_by_C/FSSH_c/`）从本项目重写，引入 pthread 线程池实现多核并行，性能提升数十倍。两个版本的物理模型和算法完全一致，可交叉验证。

## 技术细节

- **矩阵对角化**：使用 `np.linalg.eigh` + `stable_eigh_first` 统一特征向量相位
- **电子振幅积分**：同时提供矩阵指数法（`expm`）和 RKG 四阶方法，RKG 为默认
- **多进程并行**：`ProcessPoolExecutor` + `multiprocessing.cpu_count()`，Windows 兼容
- **能量守恒**：跃迁时检查动能是否足以支付势能差，不足则拒绝跃迁（返回 -10086 标记）
- **随机数**：`np.random.rand()`，多进程下每个进程独立随机状态
