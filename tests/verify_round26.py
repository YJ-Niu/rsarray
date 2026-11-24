#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十六轮验证：nanarg_core（nanargmax/nanargmin 核心化）。

1. 值语义：首个最大/最小值原索引、NaN 过滤
2. 全 NaN/空 → ValueError（Python 语义）
3. 复数 ((real, imag) 字典序)
4. 性能：nanargmax 1M
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
    # ---- 值语义：NaN 过滤 + 首个最大/最小 ----
    a = np.array([1.0, float('nan'), 3.0, 3.0, 0.5])
    check("nanargmax basic", int(np.nanargmax(a)) == 2)
    check("nanargmin basic", int(np.nanargmin(a)) == 4)
    # ---- 全 NaN → ValueError ----
    allnan = np.array([np.nan, np.nan])
    try:
        np.nanargmax(allnan)
        check("nanargmax all-nan raises", False)
    except ValueError:
        check("nanargmax all-nan raises", True)
    try:
        np.nanargmin(allnan)
        check("nanargmin all-nan raises", False)
    except ValueError:
        check("nanargmin all-nan raises", True)
    # ---- 无 NaN ----
    check("nanargmax no nan", int(np.nanargmax(np.array([3.0, 1.0, 2.0]))) == 0)
    # ---- 复数（(real, imag) 字典序）----
    ac = np.array([1 + 0j, 2 + 0j, 1 + 5j])
    check("nanargmax complex", int(np.nanargmax(ac)) == 1)
    check("nanargmin complex", int(np.nanargmin(ac)) == 0)
    # ---- 性能 ----
    N = 1_000_000
    a = np.arange(N, dtype='float64')
    anan = a.copy()
    anan[::10] = np.nan
    np.nanargmax(anan)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.nanargmax(anan)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH nanargmax 1M (partial nan): {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
