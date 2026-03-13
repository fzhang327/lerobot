# ESOAdmittanceController 使用说明（中文）

`ESOAdmittanceController` 是放在 **VLA 策略输出** 和 **关节伺服控制器** 之间的中间件。

它完成两件事：

1. **轨迹平滑**：将低频（例如 5Hz）的阶跃关节目标平滑成更连续的命令，降低抖动与机械冲击。
2. **柔顺控制**：使用基于 LESO 的扰动估计 + 导纳模型，在无力传感器场景下估计外力并做顺应。

---

## 核心模型

### 1) LESO 物理观测器

- 误差：`e = q_meas - z1`
- 状态更新（欧拉离散）：
  - `z1_dot = z2 + beta1 * e`
  - `z2_dot = z3 + b0 * u + beta2 * e`
  - `z3_dot = beta3 * e`
- 带宽参数化：
  - `beta1 = 3 * omega_o`
  - `beta2 = 3 * omega_o^2`
  - `beta3 = omega_o^3`

其中 `b0` 是物理系统增益（与伺服刚度/惯量倒数相关），必须可调。

### 2) 外力估计

- `tau_ext_hat = z3 / b0 - gravity(q)`
- 支持 `gravity_fn(q)` 回调作为重力补偿接口。
- 通过 `force_deadband` 做小扰动死区过滤，减少摩擦噪声影响。

### 3) 虚拟导纳模型

- `M_d * qdd_c + D_d * qd_c + K_d * (q_c - q_vla) = tau_ext_hat`
- `M_d` 默认可设为 `1.0`（归一化惯量）
- `K_d` 可调（越小越柔顺）
- `D_d` 默认自动设置为临界阻尼：`2 * sqrt(M_d * K_d)`

### 4) 安全保护

当任一关节的 `|tau_ext_hat| > force_limit` 时，控制器冻结运动并输出当前测量位置。

---

## 快速上手示例

```python
import numpy as np
from lerobot.utils.eso_admittance_controller import ESOAdmittanceController


def gravity_comp(q: np.ndarray) -> np.ndarray:
    # 示例：先用零重力补偿，后续替换为机器人动力学模型
    return np.zeros_like(q)


controller = ESOAdmittanceController(
    dof=6,
    b0=1.2,
    omega_o=30.0,
    k_d=np.array([15, 15, 12, 8, 6, 5], dtype=np.float64),
    m_d=1.0,
    force_deadband=0.05,
    force_limit=8.0,
    target_cutoff_hz=3.0,
    gravity_fn=gravity_comp,
)

# 首次可显式 reset
q_measured = np.zeros(6, dtype=np.float64)
controller.reset(q_measured)

# 控制循环（例如 100Hz）
dt = 0.01
while True:
    q_vla_target = np.zeros(6, dtype=np.float64)  # 来自 VLA 的关节目标
    q_measured = np.zeros(6, dtype=np.float64)    # 来自编码器的关节反馈

    q_compliant_cmd = controller.update(
        q_vla_target=q_vla_target,
        q_measured=q_measured,
        dt=dt,
    )

    # 将 q_compliant_cmd 下发到伺服位置环
```

---

## 参数调参建议（SO-100 经验方向）

- `omega_o`：先从 `20~40` 开始。过大可能放大噪声，过小会导致扰动估计滞后。
- `k_d`：先从中小刚度开始，确认安全后再逐步增大。
- `force_deadband`：用来屏蔽静摩擦和编码器微抖，先试 `0.02~0.1`。
- `force_limit`：建议保守设置，先小后大，保证碰撞时能及时冻结。
- `target_cutoff_hz`：5Hz VLA 的情况下可尝试 `2~4Hz`，平衡响应和光滑度。

