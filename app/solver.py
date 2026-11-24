"""竖屏 9:16 裁切轨迹求解器。

问题模型：
- 每帧裁切窗高 = 原图高 H，宽 = 9H/16，垂直位置固定，仅横坐标 x 可变。
- x 的可行集合：窗口必须覆盖本帧全部必留框（以及用户锁定）。
- 相邻帧约束：|x_{i+1} - x_i| <= 8。
- 二阶约束：|(x_{i+1} - x_i) - (x_i - x_{i-1})| <= 2。
- 目标：最大化 sum_i energy_i(x_i)，energy 为窗口内横竖相邻像素灰度差绝对值之和。

用二阶动态规划求解：状态 (x_{i-1}, x_i)，向量化转移。
若无可行轨迹，返回从首帧起最早接不上的帧下标。
"""

from __future__ import annotations

import numpy as np

NEG = -1.0e15  # 不可行标记（远小于任何真实能量累计值）

MAX_SHIFT = 8        # 相邻帧横移上限
MAX_ACCEL = 2        # 连续两次有向横移之差的上限


def crop_width(height: int) -> int:
    """9:16 竖屏裁切窗宽度（高为 16 的倍数，故必为整数）。"""
    return height * 9 // 16


def window_energies(gray: np.ndarray) -> np.ndarray:
    """计算每个窗口横坐标 x 的能量。

    gray: (H, W) 灰度数组。返回长度 X = W - cw + 1 的 float64 数组，
    energy[x] = 窗口 [0:H, x:x+cw] 内水平相邻与竖直相邻像素灰度差绝对值之和。
    """
    g = gray.astype(np.float32)
    H, W = g.shape
    cw = crop_width(H)
    X = W - cw + 1
    if X < 1:
        raise ValueError("原图宽度不足以裁出 9:16 竖屏窗口")

    # 水平相邻差（每行内）：hd[y, j] = |g[y, j+1] - g[y, j]|，j = 0..W-2
    hd = np.abs(np.diff(g, axis=1)).sum(axis=0)          # 按列求和 -> (W-1,)
    # 竖直相邻差（每列内）：vd[y, j] = |g[y+1, j] - g[y, j]|，y = 0..H-2
    vd = np.abs(np.diff(g, axis=0)).sum(axis=0)          # 按列求和 -> (W,)

    # 窗口 x 覆盖水平差列 x .. x+cw-2（共 cw-1 列）
    ch = np.concatenate(([0.0], np.cumsum(hd, dtype=np.float64)))
    hsum = ch[cw - 1:cw - 1 + X] - ch[0:X]
    # 窗口 x 覆盖竖直差列 x .. x+cw-1（共 cw 列）
    cv = np.concatenate(([0.0], np.cumsum(vd, dtype=np.float64)))
    vsum = cv[cw:cw + X] - cv[0:X]
    return hsum + vsum


def feasible_mask(boxes: list, width: int, height: int) -> np.ndarray:
    """由必留框得到可行的窗口横坐标布尔掩码（长度 X = W - cw + 1）。

    boxes: [[x0, y0, x1, y1], ...]，像素区间 [x0, x1) x [y0, y1)。
    """
    H = height
    cw = crop_width(H)
    X = width - cw + 1
    mask = np.ones(X, dtype=bool)
    if boxes:
        min_x0 = min(b[0] for b in boxes)
        max_x1 = max(b[2] for b in boxes)
        # 窗口左缘 x 须满足 x <= min_x0 且 x + cw >= max_x1
        lo = int(np.ceil(max_x1 - cw))
        hi = int(np.floor(min_x0))
        lo = max(0, lo)
        hi = min(width - cw, hi)
        mask[:] = False
        if lo <= hi:
            mask[lo:hi + 1] = True
    return mask


def solve(energies: list[np.ndarray],
          feasibles: list[np.ndarray],
          locks: dict[int, int] | None = None):
    """求解最优轨迹。

    energies:  每帧 energy 数组（长度 X）
    feasibles: 每帧可行布尔掩码（长度 X）
    locks:     {帧号: 锁定 x}

    返回 (trajectory, fail_frame, total_energy)：
    - 成功：trajectory 为长度 N 的 x 列表，fail_frame 为 None；
    - 失败：trajectory 为 None，fail_frame 为从首帧起最早接不上的帧下标。
    """
    N = len(energies)
    X = energies[0].shape[0]
    locks = locks or {}

    # 合并锁定约束
    F = [f.copy() for f in feasibles]
    for i, x in locks.items():
        m = np.zeros(X, dtype=bool)
        if 0 <= int(x) < X:
            m[int(x)] = True
        F[i] &= m

    # E[i][x]：不可行位置能量置为 NEG
    E = [np.where(F[i], energies[i].astype(np.float64), NEG) for i in range(N)]

    if not F[0].any():
        return None, 0, None
    if N == 1:
        x = int(np.argmax(E[0]))
        return [x], None, float(E[0][x])

    B, C = np.indices((X, X))
    pair_ok = (np.abs(B - C) <= MAX_SHIFT)

    # 初始状态：帧 0、1 组成的对 (a, b)
    dp = np.where(pair_ok, E[0][B] + E[1][C], NEG)
    if dp.max() <= NEG / 2:
        return None, 1, None

    # parents[i][b, c] = 转移到帧 i 状态 (b, c) 时的前一横坐标 a（-1 表示不可达）
    parents: list[np.ndarray | None] = [None, None]

    for i in range(2, N):
        ndp = np.full((X, X), NEG, dtype=np.float64)
        par = np.full((X, X), -1, dtype=np.int32)
        # 转移：a = 2b - c + k，k = (c-b)-(b-a) ∈ [-MAX_ACCEL, MAX_ACCEL]
        for k in range(-MAX_ACCEL, MAX_ACCEL + 1):
            A = 2 * B - C + k
            valid = (A >= 0) & (A < X)
            Ac = np.clip(A, 0, X - 1)
            vals = np.where(valid, dp[Ac, B], NEG)
            better = vals > ndp
            ndp = np.where(better, vals, ndp)
            par = np.where(better, np.where(valid, Ac, -1), par)
        # 相邻帧横移约束 + 本帧能量
        ndp = np.where(pair_ok, ndp + E[i][C], NEG)
        par = np.where(pair_ok, par, -1)
        if ndp.max() <= NEG / 2:
            return None, i, None
        dp = ndp
        parents.append(par)

    # 回溯
    flat = int(np.argmax(dp))
    a, b = flat // X, flat % X
    total = float(dp[a, b])
    traj = [0] * N
    traj[N - 2], traj[N - 1] = int(a), int(b)
    for i in range(N - 1, 1, -1):
        b_, c_ = traj[i - 1], traj[i]
        traj[i - 2] = int(parents[i][b_, c_])
    return traj, None, total
