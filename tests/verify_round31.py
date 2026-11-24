#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十一轮验证：polyfit_normal_core（np.polyfit 正规方程快路径）。

1. 值语义：deg1/deg2 精确还原（降序）、2-D y 回退原路径、
   NaN → LinAlgError、秩亏回退 SVD（RankWarning）
2. 性能：np.polyfit deg2 100K
"""
import sys
import time
import warnings

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402

PASS = [0]
FAIL = [0]


def check(name, ok):
    if ok:
        PASS[0] += 1
    else:
        FAIL[0] += 1
        print(f"FAIL: {name}")


def main():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 6.0, 17.0, 34.0, 57.0])  # 1+2x+3x²
    # ---- deg2 精确（降序 [3,2,1]）----
    c = np.polyfit(x, y, 2)
    check("deg2 exact", all(abs(float(c[i]) - [3.0, 2.0, 1.0][i]) < 1e-9
                            for i in range(3)))
    # ---- deg1 ----
    c1 = np.polyfit(x, np.array([1.0, 3.0, 5.0, 7.0, 9.0]), 1)
    check("deg1", all(abs(float(c1[i]) - [2.0, 1.0][i]) < 1e-9
                      for i in range(2)))
    # ---- 2-D y（回退 lstsq 原路径）----
    y2 = np.array([[1.0, 2.0], [6.0, 7.0], [17.0, 18.0],
                   [34.0, 35.0], [57.0, 58.0]])
    c2 = np.polyfit(x, y2, 2)
    _exp2 = [[3.0, 3.0], [2.0, 2.0], [1.0, 2.0]]
    check("2-D y", all(abs(float(c2[i][j]) - _exp2[i][j]) < 1e-9
                       for i in range(3) for j in range(2)))
    # ---- NaN → LinAlgError（快路径检查）----
    from rsarray.linalg import LinAlgError as _LAE
    try:
        np.polyfit(np.array([0.0, float('nan'), 2.0]),
                   np.array([1.0, 2.0, 3.0]), 1)
        check("nan raises", False)
    except _LAE:
        check("nan raises", True)
    # ---- 秩亏回退 SVD：3 点拟合 deg2（n == order → 欠定）----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cu = np.polyfit(np.array([0.0, 1.0, 2.0]),
                        np.array([1.0, 6.0, 17.0]), 2)
    check("underdetermined exact",
          abs(float(cu[0]) - 3.0) < 1e-9 and abs(float(cu[2]) - 1.0) < 1e-9)
    # ---- 性能 ----
    import random
    random.seed(31)
    N = 100_000
    xb = np.array([random.uniform(-2, 2) for _ in range(N)])
    yb = np.array([random.uniform(-2, 2) for _ in range(N)])
    np.polyfit(xb, yb, 2)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.polyfit(xb, yb, 2)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH np.polyfit deg2 100K: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
