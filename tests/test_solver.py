"""求解器测试：与暴力枚举对照最优性，并验证约束、失败检测与可达范围评分。

运行：python3 tests/test_solver.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import solver  # noqa: E402


def brute_force(energies, feasibles, locks):
    N = len(energies)
    X = len(energies[0])
    locks = locks or {}
    best, best_val = None, None
    for traj in itertools.product(range(X), repeat=N):
        ok = True
        for i, x in enumerate(traj):
            if not feasibles[i][x] or (i in locks and locks[i] != x):
                ok = False
                break
        if not ok:
            continue
        for i in range(N - 1):
            if abs(traj[i + 1] - traj[i]) > solver.MAX_SHIFT:
                ok = False
                break
        if not ok:
            continue
        for i in range(N - 2):
            d1 = traj[i + 1] - traj[i]
            d2 = traj[i + 2] - traj[i + 1]
            if abs(d2 - d1) > solver.MAX_ACCEL:
                ok = False
                break
        if not ok:
            continue
        val = sum(energies[i][traj[i]] for i in range(N))
        if best_val is None or val > best_val:
            best_val, best = val, traj
    return best, best_val


def all_legal_trajectories(feasibles, locks):
    """枚举全部完整合法轨迹，返回每帧“出现在某条合法轨迹上”的横坐标集合。"""
    N = len(feasibles)
    X = len(feasibles[0])
    locks = locks or {}
    seen = [set() for _ in range(N)]
    for traj in itertools.product(range(X), repeat=N):
        ok = True
        for i, x in enumerate(traj):
            if not feasibles[i][x] or (i in locks and locks[i] != x):
                ok = False
                break
        if not ok:
            continue
        for i in range(N - 1):
            if abs(traj[i + 1] - traj[i]) > solver.MAX_SHIFT:
                ok = False
                break
        if not ok:
            continue
        for i in range(N - 2):
            d1 = traj[i + 1] - traj[i]
            d2 = traj[i + 2] - traj[i + 1]
            if abs(d2 - d1) > solver.MAX_ACCEL:
                ok = False
                break
        if not ok:
            continue
        for i, x in enumerate(traj):
            seen[i].add(x)
    return seen


def check_traj(traj, energies, feasibles, locks):
    N = len(energies)
    assert len(traj) == N
    for i, x in enumerate(traj):
        assert feasibles[i][x], f"帧 {i} 位置 {x} 不可行"
        if i in locks:
            assert locks[i] == x, f"帧 {i} 未按锁定位置"
    for i in range(N - 1):
        assert abs(traj[i + 1] - traj[i]) <= solver.MAX_SHIFT, "横移超限"
    for i in range(N - 2):
        d1 = traj[i + 1] - traj[i]
        d2 = traj[i + 2] - traj[i + 1]
        assert abs(d2 - d1) <= solver.MAX_ACCEL, "二阶横移超限"
    return sum(energies[i][traj[i]] for i in range(N))


def test_random_vs_brute():
    rng = np.random.RandomState(0)
    for trial in range(30):
        N = int(rng.randint(2, 7))
        X = int(rng.randint(2, 9))
        energies = [rng.rand(X) * 100 for _ in range(N)]
        feasibles = []
        for _ in range(N):
            f = rng.rand(X) < 0.8
            f[rng.randint(X)] = True  # 保证至少一个可行位
            feasibles.append(f)
        locks = {}
        if rng.rand() < 0.5:
            i = int(rng.randint(N))
            cand = np.flatnonzero(feasibles[i])
            locks[i] = int(cand[rng.randint(len(cand))])
        traj, fail, total = solver.solve(energies, feasibles, locks)
        bf_traj, bf_val = brute_force(energies, feasibles, locks)
        if bf_traj is None:
            assert traj is None, f"trial {trial}: DP 有解而暴力无解"
            assert fail is not None and 0 <= fail < N
        else:
            assert traj is not None, f"trial {trial}: DP 无解而暴力有解 {bf_traj}"
            val = check_traj(traj, energies, feasibles, locks)
            assert abs(val - bf_val) < 1e-6, f"trial {trial}: 最优性不符 {val} vs {bf_val}"
            assert abs(total - bf_val) < 1e-6
    print("随机对照暴力：30 例全部一致")


def test_failure_earliest():
    # 3 帧，帧1 与帧2 的可行区间相距 >8 -> 应在帧2 失败
    X = 30
    energies = [np.ones(X) for _ in range(3)]
    f0 = np.zeros(X, bool); f0[0] = True
    f1 = np.zeros(X, bool); f1[5] = True
    f2 = np.zeros(X, bool); f2[20] = True
    traj, fail, _ = solver.solve(energies, [f0, f1, f2], {})
    assert traj is None and fail == 2, f"期望帧2 失败，得到 {fail}"
    # 帧0 无可行 -> 帧0 失败
    traj, fail, _ = solver.solve(energies, [np.zeros(X, bool), f1, f2], {})
    assert traj is None and fail == 0
    print("失败检测：最早接不上帧定位正确")


def test_lock_infeasible():
    X = 20
    energies = [np.ones(X) for _ in range(3)]
    feas = [np.ones(X, bool) for _ in range(3)]
    # 锁定位置超出可行集合 -> 该帧失败
    f1 = np.zeros(X, bool); f1[3] = True
    feas[1] = f1
    traj, fail, _ = solver.solve(energies, feas, {1: 10})
    assert traj is None and fail == 1
    print("锁定与必留框冲突：正确定位失败帧")


def test_energy_and_feasible():
    rng = np.random.RandomState(1)
    g = rng.randint(0, 256, (64, 100), dtype=np.uint8)
    e = solver.window_energies(g)
    cw = solver.crop_width(64)
    assert cw == 36 and len(e) == 100 - 36 + 1
    gf = g.astype(np.float64)
    for x in [0, 7, 64]:
        win = gf[:, x:x + cw]
        expect = (np.abs(np.diff(win, axis=1)).sum()
                  + np.abs(np.diff(win, axis=0)).sum())
        assert abs(e[x] - expect) < 1e-3, f"能量不符 x={x}"
    m = solver.feasible_mask([[10, 0, 30, 10], [40, 5, 44, 20]], 100, 64)
    lo, hi = np.flatnonzero(m)[0], np.flatnonzero(m)[-1]
    assert lo == max(0, 44 - 36) and hi == min(100 - 36, 10)
    # 两框跨度超过窗宽 -> 无可行位置
    m2 = solver.feasible_mask([[10, 0, 30, 10], [40, 5, 50, 20]], 100, 64)
    assert not m2.any()
    print("能量与可行区间：数值正确")


# ------------------------------------------------------------ 新增：可达性

def test_reachable_masks_brute():
    """可达掩码与暴力枚举的“完整合法轨迹横坐标集合”逐位一致；失败帧一致。"""
    rng = np.random.RandomState(2)
    for trial in range(60):
        N = int(rng.randint(1, 7))
        X = int(rng.randint(2, 10))
        feas = [rng.rand(X) < 0.75 for _ in range(N)]
        for f in feas:
            f[rng.randint(X)] = True
        locks = {}
        if rng.rand() < 0.4:
            i = int(rng.randint(N))
            cand = np.flatnonzero(feas[i])
            locks[i] = int(cand[rng.randint(len(cand))])

        masks, fail = solver.reachable_masks(feas, locks)
        seen = all_legal_trajectories(feas, locks)
        if all(s for s in seen):
            assert fail is None, f"trial {trial}: 暴力有解却报失败"
            assert masks is not None and len(masks) == N
            for i in range(N):
                got = set(np.flatnonzero(masks[i]).tolist())
                assert got == seen[i], f"trial {trial} 帧{i}: {got} != {seen[i]}"
        else:
            assert fail is not None, f"trial {trial}: 暴力无解却报有解"
            assert masks is None
    print("可达掩码：60 例与暴力枚举完全一致")


def test_reachable_failure_frame_matches_dp():
    """布尔可达 DP 的失败帧与原数值 DP 的失败帧一致。"""
    rng = np.random.RandomState(3)
    for _ in range(40):
        N = int(rng.randint(1, 8))
        X = int(rng.randint(2, 12))
        feas = [rng.rand(X) < 0.5 for _ in range(N)]
        energies = [rng.rand(X) for _ in range(N)]
        locks = {}
        if rng.rand() < 0.4:
            i = int(rng.randint(N))
            locks[i] = int(rng.randint(X))
        _, fail_b = solver.reachable_masks(feas, locks)
        _, fail_dp, _ = solver.solve(energies, feas, locks)
        assert fail_b == fail_dp, f"{fail_b} != {fail_dp}"
    print("失败帧：布尔 DP 与数值 DP 一致")


# ------------------------------------------------------------ 新增：按可达范围评分

def test_scored_energy_values_and_pairs():
    """可达窗能量与全量评分一致；像素对只在“被某个可达窗覆盖”时计算。"""
    rng = np.random.RandomState(4)
    H, W = 48, 80
    cw = solver.crop_width(H)
    X = W - cw + 1
    g = rng.randint(0, 256, (H, W), dtype=np.uint8)
    full = solver.window_energies(g)

    # 取一段不连续的可达窗集合，验证并非只有连续区间才正确。
    reach = np.zeros(X, dtype=bool)
    reach[[2, 3, 4, 10, 11, 20]] = True
    e, hp, vp = solver.scored_window_energies(g, reach)

    # 不可达窗能量必须为 0；可达窗与全量评分逐位相等。
    assert np.all(e[~reach] == 0.0)
    for x in np.flatnonzero(reach):
        assert abs(e[x] - full[x]) < 1e-6, f"可达窗 {x} 能量被改变"

    # 手工核算合格列与像素对数量。
    h_cols, v_cols = set(), set()
    for x in np.flatnonzero(reach):
        h_cols.update(range(x, x + cw - 1))   # 水平差列 x..x+cw-2
        v_cols.update(range(x, x + cw))       # 竖直差列 x..x+cw-1
    assert hp == len(h_cols) * H
    assert vp == len(v_cols) * (H - 1)

    # 逐像素对验证：只有合格列上的差才计入，且每对最多一次（按列去重天然满足）。
    gf = g.astype(np.float64)
    hd = np.abs(np.diff(gf, axis=1)).sum(axis=0)
    vd = np.abs(np.diff(gf, axis=0)).sum(axis=0)
    for x in np.flatnonzero(reach):
        expect = sum(hd[j] for j in range(x, x + cw - 1)) \
            + sum(vd[j] for j in range(x, x + cw))
        assert abs(e[x] - expect) < 1e-6, f"窗 {x} 列差聚合错误"

    # 全部窗可达时，像素对数量等于全量，能量与全量评分完全一致。
    e2, hp2, vp2 = solver.scored_window_energies(g, np.ones(X, dtype=bool))
    assert np.allclose(e2, full)
    assert hp2 == H * (W - 1) and vp2 == (H - 1) * W
    print("可达范围评分：能量、合格像素对与去重全部正确")


def test_plan_end_to_end_and_pair_count():
    """plan() 全流程：有解轨迹/总能量与旧路径一致；无解不算灰度差。"""
    rng = np.random.RandomState(5)
    N, H, W = 6, 48, 80
    cw = solver.crop_width(H)
    X = W - cw + 1
    grays = [rng.randint(0, 256, (H, W), dtype=np.uint8) for _ in range(N)]
    feas = [rng.rand(X) < 0.8 for _ in range(N)]
    for f in feas:
        f[rng.randint(X)] = True
    locks = {2: int(np.flatnonzero(feas[2])[0])}

    res = solver.plan(grays, feas, locks)
    assert res["ok"] and res["fail_frame"] is None
    assert len(res["trajectory"]) == N

    # 对照旧路径：全量能量 + 数值 DP，轨迹与总能量（同分选法）一致。
    full_e = [solver.window_energies(g) for g in grays]
    old_traj, old_fail, old_total = solver.solve(full_e, feas, locks)
    assert old_fail is None
    assert res["trajectory"] == old_traj, "同分选法应与旧版一致"
    assert abs(res["total_energy"] - old_total) < 1e-6

    # 轨迹上的每个窗都在对应可达掩码内；掩码外窗能量为 0。
    for i, x in enumerate(res["trajectory"]):
        assert res["reachable"][i][x]
    # 可达窗数不超过全部窗，故计算量不超过全量基准。
    assert 0 < res["pair_count"] <= res["full_pair_count"]
    assert res["pair_count"] == res["h_pairs"] + res["v_pairs"]
    assert res["full_pair_count"] == solver.full_pair_count(H, W, N)

    # 无解场景：不可接续且整轮灰度差为 0，不返回轨迹与掩码。
    g0 = np.zeros(X, bool); g0[0] = True
    g1 = np.zeros(X, bool); g1[5] = True
    g2 = np.zeros(X, bool); g2[25] = True
    bad = [g0, g1, g2] + [np.ones(X, bool) for _ in range(N - 3)]
    res2 = solver.plan(grays, bad, {})
    assert not res2["ok"] and res2["fail_frame"] == 2
    assert res2["pair_count"] == res2["h_pairs"] == res2["v_pairs"] == 0
    assert res2["trajectory"] is None and res2["reachable"] is None
    print("plan 全流程：轨迹/能量同旧版，无解轮灰度差为 0")


def test_plan_pair_count_shrinks_with_locks():
    """锁得越死，参与评分的可达窗与像素对越少（可达范围随锁变化）。"""
    rng = np.random.RandomState(6)
    N, H, W = 8, 32, 64
    cw = solver.crop_width(H)
    X = W - cw + 1
    grays = [rng.randint(0, 256, (H, W), dtype=np.uint8) for _ in range(N)]
    feas = [np.ones(X, bool) for _ in range(N)]
    mid = X // 2

    free = solver.plan(grays, feas, {})
    locked = solver.plan(grays, feas, {0: mid, N - 1: mid})
    assert free["ok"] and locked["ok"]
    free_windows = sum(int(m.sum()) for m in free["reachable"])
    lock_windows = sum(int(m.sum()) for m in locked["reachable"])
    assert lock_windows < free_windows
    assert locked["pair_count"] < free["pair_count"]
    # 中间帧可达集合应围绕锁定的 mid 收缩（不再包含边缘窗）。
    assert not locked["reachable"][N // 2][0]
    print("改锁重评：可达范围与像素差计算量随锁定收缩")


if __name__ == "__main__":
    test_energy_and_feasible()
    test_random_vs_brute()
    test_failure_earliest()
    test_lock_infeasible()
    test_reachable_masks_brute()
    test_reachable_failure_frame_matches_dp()
    test_scored_energy_values_and_pairs()
    test_plan_end_to_end_and_pair_count()
    test_plan_pair_count_shrinks_with_locks()
    print("全部通过")
