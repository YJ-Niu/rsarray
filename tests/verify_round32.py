#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十二轮验证：np.where 标量分支广播（numpy 兼容性修复）。

1. 标量 x/y：np.where(cond, a, 0.0)（此前抛 TypeError）
2. 双标量、bool 标量、int 标量 dtype
3. 回归：arr/arr 路径不变
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
    x = np.array([0.0, 1.0, 2.0, 3.0])
    # ---- 标量 y（此前抛 TypeError）----
    r = np.where(x > 1, x, 0.0)
    check("scalar y", r.tolist() == [0.0, 0.0, 2.0, 3.0])
    # ---- 标量 x/y 混合 ----
    # ---- 标量 x（cond False 处取 y=x）----
    r2 = np.where(x > 1, -1.0, x)
    check("scalar x", r2.tolist() == [0.0, 1.0, -1.0, -1.0])
    # ---- 双标量 ----
    r3 = np.where(x > 1, 1, 0)
    check("scalar both int",
          r3.tolist() == [0, 0, 1, 1] and str(r3.dtype) == 'int64')
    # ---- bool 标量 ----
    rb = np.where(x > 1, True, False)
    check("scalar both bool", rb.tolist() == [False, False, True, True])
    # ---- arr/arr 路径不变 ----
    ra = np.where(x > 1, x, x * 2)
    check("arr/arr unchanged", ra.tolist() == [0.0, 2.0, 2.0, 3.0])
    # ---- 性能（1M 标量分支）----
    N = 1_000_000
    a = np.arange(N, dtype='float64')
    np.where(a > 5, a, 0.0)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.where(a > 5, a, 0.0)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH where(scalar y) 1M: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
