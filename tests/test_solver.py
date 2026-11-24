"""求解器测试：与暴力枚举对照最优性，并验证约束与失败检测。

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


if __name__ == "__main__":
    test_energy_and_feasible()
    test_random_vs_brute()
    test_failure_earliest()
    test_lock_infeasible()
    print("全部通过")
