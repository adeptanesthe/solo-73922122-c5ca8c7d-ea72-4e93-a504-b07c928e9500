"""竖屏 9:16 裁切轨迹求解器。

问题模型：
- 每帧裁切窗高 = 原图高 H，宽 = 9H/16，垂直位置固定，仅横坐标 x 可变。
- x 的可行集合：窗口必须覆盖本帧全部必留框（以及用户锁定）。
- 相邻帧约束：|x_{i+1} - x_i| <= 8。
- 二阶约束：|(x_{i+1} - x_i) - (x_i - x_{i-1})| <= 2。
- 目标：最大化 sum_i energy_i(x_i)，energy 为窗口内横竖相邻像素灰度差绝对值之和。

评分流程（本版）：
1. 只用可行掩码与锁定，做布尔可达性 DP，判断整段能否接成一条完整合法轨迹，
   并求出每帧“能出现在某条完整合法轨迹上”的窗口集合（可达窗）。
2. 无解：返回从首帧起最早接不上的帧下标，不触碰灰度数据、不计算任何灰度差。
3. 有解：只给可达窗评分。某个水平/竖直相邻像素对，只有同时落在某个可达窗内
   才计算一次灰度差（同一轮、同一帧每对像素最多算一次），再用前缀和把
   列差聚合成各窗能量；不可达窗能量置 0，不参与选择。
4. 二阶数值动态规划在评分后的能量上求全局最优（同分选法与旧版完全一致）。
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
    """计算每个窗口横坐标 x 的能量（全量评分，不考虑可达性）。

    gray: (H, W) 灰度数组。返回长度 X = W - cw + 1 的 float64 数组，
    energy[x] = 窗口 [0:H, x:x+cw] 内水平相邻与竖直相邻像素灰度差绝对值之和。

    保留用于对照测试与旧 API；plan() 走按可达范围评分的新路径。
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


# ------------------------------------------------------------ 可达性

def _merge_locks(feasibles: list[np.ndarray], locks: dict[int, int] | None):
    """把锁定合并进可行掩码，返回新的掩码列表与横坐标数 X。"""
    X = feasibles[0].shape[0]
    locks = locks or {}
    F = [f.copy() for f in feasibles]
    for i, x in locks.items():
        m = np.zeros(X, dtype=bool)
        if 0 <= int(x) < X:
            m[int(x)] = True
        F[i] &= m
    return F, X


def reachable_masks(feasibles: list[np.ndarray],
                    locks: dict[int, int] | None = None):
    """布尔可达性 DP：每帧哪些窗口横坐标能出现在一条完整合法轨迹上。

    与数值 DP 同构但只记录“可达/不可达”，全程不使用灰度数据。

    返回 (masks, fail_frame)：
    - 成功：masks 为每帧长度 X 的 bool 数组（True = 该 x 在某条完整合法轨迹上），
      fail_frame 为 None；
    - 失败：masks 为 None，fail_frame 为从首帧起最早接不上的帧下标。
    """
    N = len(feasibles)
    F, X = _merge_locks(feasibles, locks)

    if not F[0].any():
        return None, 0
    if N == 1:
        return [F[0].copy()], None

    B, C = np.indices((X, X))
    pair_ok = np.abs(B - C) <= MAX_SHIFT

    # 正向：fwd[i] 对应相邻对 P_i = (x_i, x_{i+1})，i = 0..N-2，
    # 表示该对能否由首帧沿约束到达。
    fwd: list[np.ndarray | None] = [None] * (N - 1)
    fwd[0] = pair_ok & F[0][B] & F[1][C]
    if not fwd[0].any():
        return None, 1
    for i in range(1, N - 1):
        nxt = np.zeros((X, X), dtype=bool)
        # P_{i-1} = (a, b) -> P_i = (b, c)，a = 2b - c + k
        for k in range(-MAX_ACCEL, MAX_ACCEL + 1):
            A = 2 * B - C + k
            valid = (A >= 0) & (A < X)
            Ac = np.clip(A, 0, X - 1)
            nxt |= np.where(valid, fwd[i - 1][Ac, B], False)
        nxt &= pair_ok & F[i + 1][C]
        if not nxt.any():
            return None, i + 1
        fwd[i] = nxt

    # 反向：back[i] 表示 P_i 能否沿约束走到末帧。
    back: list[np.ndarray | None] = [None] * (N - 1)
    back[N - 2] = pair_ok & F[N - 2][B] & F[N - 1][C]
    for i in range(N - 3, -1, -1):
        nxt = np.zeros((X, X), dtype=bool)
        # P_i = (a, b) -> P_{i+1} = (b, c)，c = 2b - a + k
        for k in range(-MAX_ACCEL, MAX_ACCEL + 1):
            Cc = 2 * C - B + k
            valid = (Cc >= 0) & (Cc < X)
            Ccc = np.clip(Cc, 0, X - 1)
            nxt |= np.where(valid, back[i + 1][C, Ccc], False)
        nxt &= pair_ok & F[i][B]
        back[i] = nxt

    # 同时在“首帧可达”与“可走到末帧”的对上出现的横坐标，即完整轨迹上的窗。
    masks = [np.zeros(X, dtype=bool) for _ in range(N)]
    masks[0] = (fwd[0] & back[0]).any(axis=1)
    for j in range(1, N - 1):
        masks[j] = (fwd[j - 1] & back[j - 1]).any(axis=0)
    masks[N - 1] = (fwd[N - 2] & back[N - 2]).any(axis=0)
    return masks, None


# ------------------------------------------------------------ 按可达范围评分

def scored_window_energies(gray: np.ndarray, reachable: np.ndarray):
    """只给可达窗评分，返回 (energies, h_pairs, v_pairs)。

    - 水平相邻像素对 (y, j)-(y, j+1)：仅当某可达窗同时含两列 j、j+1 时才算，
      即 j 落在可达窗并集 ⋃[x, x+cw-2] 内；
    - 竖直相邻像素对 (y, j)-(y+1, j)：仅当列 j 落在并集 ⋃[x, x+cw-1] 内时才算。
    - 同一轮、同一帧每对像素只计算一次，列差经前缀和复用给各窗。

    energies 长度 X，不可达窗置 0.0；可达窗的数值与全量评分完全一致。
    h_pairs/v_pairs 为本帧实际计算灰度差的水平/竖直相邻像素对数量。
    """
    g = gray.astype(np.float32)
    H, W = g.shape
    cw = crop_width(H)
    X = W - cw + 1
    reach = np.asarray(reachable, dtype=bool).reshape(-1)

    # 差分标记可达窗覆盖到的列（并集），避免对每列重复判定。
    h_mark = np.zeros(W, dtype=np.int32)       # 水平差列 0..W-2
    v_mark = np.zeros(W + 1, dtype=np.int32)   # 竖直差列 0..W-1
    for x in np.flatnonzero(reach):
        h_mark[int(x)] += 1
        h_mark[int(x) + cw - 1] -= 1           # 覆盖 x .. x+cw-2
        v_mark[int(x)] += 1
        v_mark[int(x) + cw] -= 1               # 覆盖 x .. x+cw-1
    h_elig = np.cumsum(h_mark)[:W - 1] > 0
    v_elig = np.cumsum(v_mark)[:W] > 0

    h_cols = np.flatnonzero(h_elig)
    v_cols = np.flatnonzero(v_elig)
    h_pairs = int(h_cols.size) * H
    v_pairs = int(v_cols.size) * (H - 1)

    # 只对合格列计算灰度差（每对像素在本帧本轮只出现一次）。
    hd = np.zeros(W - 1, dtype=np.float32)
    if h_cols.size:
        hd[h_cols] = np.abs(g[:, h_cols + 1] - g[:, h_cols]).sum(axis=0)
    vd = np.zeros(W, dtype=np.float32)
    if v_cols.size:
        vd[v_cols] = np.abs(g[1:, v_cols] - g[:-1, v_cols]).sum(axis=0)

    # 前缀和把列差聚合成窗口能量：窗 x 的水平差列 x..x+cw-2、竖直差列 x..x+cw-1。
    ch = np.concatenate(([0.0], np.cumsum(hd, dtype=np.float64)))
    hsum = ch[cw - 1:cw - 1 + X] - ch[0:X]
    cv = np.concatenate(([0.0], np.cumsum(vd, dtype=np.float64)))
    vsum = cv[cw:cw + X] - cv[0:X]

    energies = hsum + vsum
    energies[~reach] = 0.0
    return energies, h_pairs, v_pairs


# ------------------------------------------------------------ 数值 DP（保留）

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

    # E[i][x]：不可行位置能量置 NEG
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


# ------------------------------------------------------------ 整轮编排

def full_pair_count(height: int, width: int, n_frames: int) -> int:
    """旧全量评分一轮会计算的相邻像素对总数（对照基准）。"""
    return n_frames * (height * (width - 1) + (height - 1) * width)


def plan(grays: list[np.ndarray],
         feasibles: list[np.ndarray],
         locks: dict[int, int] | None = None) -> dict:
    """一轮求解的完整编排：先判可行、再按可达范围评分、最后数值 DP。

    返回 dict：
    - 成功：{ok: True, trajectory, total_energy, reachable(masks),
             pair_count, h_pairs, v_pairs, full_pair_count, fail_frame: None}
    - 失败：{ok: False, fail_frame, pair_count: 0, reachable: None, ...}
      失败路径不访问灰度数组，整轮灰度差计算量为 0。
    """
    n = len(grays)
    H, W = grays[0].shape
    base = full_pair_count(H, W, n)

    masks, fail = reachable_masks(feasibles, locks)
    if fail is not None:
        return {
            "ok": False, "fail_frame": int(fail),
            "trajectory": None, "total_energy": None, "reachable": None,
            "pair_count": 0, "h_pairs": 0, "v_pairs": 0,
            "full_pair_count": base,
        }

    energies: list[np.ndarray] = []
    h_total = v_total = 0
    for i in range(n):
        e, hp, vp = scored_window_energies(grays[i], masks[i])
        energies.append(e)
        h_total += hp
        v_total += vp

    traj, fail2, total = solve(energies, feasibles, locks)
    if fail2 is not None:  # 理论不可达：可达掩码已保证存在完整轨迹
        return {
            "ok": False, "fail_frame": int(fail2),
            "trajectory": None, "total_energy": None, "reachable": None,
            "pair_count": 0, "h_pairs": 0, "v_pairs": 0,
            "full_pair_count": base,
        }
    return {
        "ok": True, "fail_frame": None,
        "trajectory": traj, "total_energy": total, "reachable": masks,
        "pair_count": h_total + v_total,
        "h_pairs": h_total, "v_pairs": v_total,
        "full_pair_count": base,
    }
