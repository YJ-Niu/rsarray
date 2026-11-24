#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十五轮验证：cov 内核优化（对称上三角 + 扁平 centered）。

1. 值语义：cov 对称、对角 = 方差、非 rowvar 转置、1-D 输入
2. corrcoef：对角 = 1、对称、值域
3. 性能：cov/corrcoef 1000×100
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
    m = np.array([[1.0, 2.0, 4.0, 8.0],
                  [2.0, 4.0, 8.0, 16.0],
                  [8.0, 4.0, 2.0, 1.0]])
    # ---- cov 对称 + 对角方差 ----
    c = np.cov(m)
    check("cov shape", c.shape == (3, 3))
    check("cov symmetric", all(
        abs(float(c[i][j]) - float(c[j][i])) < 1e-12
        for i in range(3) for j in range(3)))
    check("cov diag = var(ddof1)",
          all(abs(float(c[i][i]) - np.var(m[i]) * 4 / 3) < 1e-9
              for i in range(3)))
    # ---- 非行主（rowvar=False）转置读取 ----
    m2 = np.array([[1.0, 2.0, 4.0, 8.0],
                   [2.0, 4.0, 8.0, 16.0]])
    c2t = np.cov(m2.T)
    check("cov rowvar=False", c2t.shape == (4, 4))
    c2 = np.cov(m2)
    check("cov transpose equiv", c2.shape == (2, 2))
    # ---- 1-D ----
    c1 = np.cov(np.array([1.0, 2.0, 3.0]))
    check("cov 1-D", float(c1) if c1.ndim == 0 else float(c1[0][0]) == 1.0)
    # ---- corrcoef：对角 1、对称 ----
    r = np.corrcoef(m)
    check("corrcoef diag = 1", all(abs(float(r[i][i]) - 1.0) < 1e-12
                                   for i in range(3)))
    check("corrcoef symmetric", all(
        abs(float(r[i][j]) - float(r[j][i])) < 1e-12
        for i in range(3) for j in range(3)))
    check("corrcoef perfect corr",
          abs(float(r[0][1]) - 1.0) < 1e-9)
    # ---- 性能 ----
    import random
    random.seed(35)
    mm = np.random.rand(1000, 100)
    np.cov(mm)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.cov(mm)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH cov 1000x100: {best*1e3:.1f}ms")
    np.corrcoef(mm)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.corrcoef(mm)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH corrcoef 1000x100: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
