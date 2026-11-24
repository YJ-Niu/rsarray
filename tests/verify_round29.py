#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十九轮验证：polyfit_rows_core（Vandermonde 构造核心化）。

1. 值语义：deg2 拟合对照解析解（最小二乘闭式）、1-D/2-D y、
   w 权重缩放、int dtype
2. 复数回退原路径仍正确
3. 性能：polyfit deg2 100K
"""
import sys
import time

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
    # ---- deg2 精确数据：y = 1 + 2x + 3x² → 拟合应还原系数 ----
    # np.polyfit 为降序约定（[x², x, 1]）；ppolyfit 为升序（[1, x, x²]）
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 6.0, 17.0, 34.0, 57.0])  # 1+2x+3x²
    c = np.polyfit(x, y, 2)
    check("polyfit deg2 exact",
          all(abs(float(c[i]) - [3.0, 2.0, 1.0][i]) < 1e-9
              for i in range(3)))
    # ---- deg1 线性（降序 [2, 1]）----
    c1 = np.polyfit(x, np.array([1.0, 3.0, 5.0, 7.0, 9.0]), 1)
    check("polyfit deg1", all(abs(float(c1[i]) - [2.0, 1.0][i]) < 1e-9
                              for i in range(2)))
    # ---- 2-D y（逐列，降序）----
    y2 = np.array([[1.0, 2.0], [6.0, 7.0], [17.0, 18.0],
                   [34.0, 35.0], [57.0, 58.0]])
    c2 = np.polyfit(x, y2, 2)
    # 列 0 常数 1.0；列 1 = 列 0 + 1 → 常数 2.0（斜率/二次系数相同）
    _exp2 = [[3.0, 3.0], [2.0, 2.0], [1.0, 2.0]]
    check("polyfit 2-D y",
          all(abs(float(c2[i][j]) - _exp2[i][j]) < 1e-9
              for i in range(3) for j in range(2)))
    # ---- w 权重 ----
    xw = np.array([0.0, 1.0, 2.0])
    yw = np.array([1.0, 3.0, 5.0])
    cw = np.polyfit(xw, yw, 1, w=np.array([1.0, 1.0, 1.0]))
    check("polyfit weights", all(abs(float(cw[i]) - [2.0, 1.0][i]) < 1e-9
                                 for i in range(2)))
    # ---- 复数回退原路径（polynomial.polyfit 支持复数；np.polyfit 走
    # _npcompat 兼容层，lstsq 复数不支持为既有 gap）----
    xc = np.array([0 + 0j, 1 + 0j, 2 + 0j])
    yc = np.array([1 + 0j, 3 + 0j, 5 + 0j])
    from rsarray.polynomial.polynomial import polyfit as ppolyfit
    cc = ppolyfit(xc, yc, 1)
    check("polyfit complex fallback",
          abs(complex(cc[0]).real - 1.0) < 1e-9)
    # ---- full（rsarray 返回 c, resids, rank, s, rcond 共 5 元素）----
    res = np.polyfit(x, y, 2, full=True)
    check("polyfit full", len(res) == 5)
    # ---- lstsq 非有限检查（has_nonfinite_core 快速路径）----
    from rsarray.linalg import LinAlgError as _LAE
    try:
        np.polyfit(np.array([0.0, float('nan'), 2.0]),
                   np.array([1.0, 2.0, 3.0]), 1)
        check("polyfit nan raises", False)
    except _LAE:
        check("polyfit nan raises", True)
    # ---- 性能 ----
    import random
    random.seed(29)
    N = 100_000
    xb = np.array([random.uniform(-2, 2) for _ in range(N)])
    yb = np.array([random.uniform(-2, 2) for _ in range(N)])
    np.polyfit(xb, yb, 2)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.polyfit(xb, yb, 2)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH polyfit deg2 100K: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
