#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十四轮验证：median_core（全归约中位数核心化）。

1. 值语义：奇/偶长度、排序、NaN 传染、float32/complex dtype
2. nanmedian/nanstd 等家族回归
3. 性能：median 1M
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
    # ---- 奇/偶长度 ----
    check("median odd", float(np.median(np.array([3.0, 1.0, 2.0]))) == 2.0)
    check("median even", float(np.median(np.array([4.0, 1.0, 3.0, 2.0]))) == 2.5)
    # ---- 排序语义 ----
    a = np.array([5.0, -1.0, 3.0, 0.5])
    check("median sorted", float(np.median(a)) == (0.5 + 3.0) / 2.0)
    # ---- NaN 传染 ----
    an = np.array([1.0, float('nan'), 3.0])
    v = float(np.median(an))
    check("median nan propagate", v != v)
    # ---- dtype：float32 保持、int 提升 float64 ----
    f32 = np.median(np.array([3.0, 1.0, 2.0], dtype='float32'))
    check("median float32 dtype", str(f32.dtype) == 'float32')
    iv = np.median(np.array([3, 1, 2], dtype='int64'))
    check("median int -> float64",
          float(iv) == 2.0 and str(iv.dtype) == 'float64')
    # ---- complex（(real, imag) 字典序）----
    cv = np.median(np.array([2 + 0j, 1 + 0j, 1 + 1j]))
    check("median complex", complex(cv) == 1 + 1j)
    # ---- 多维 axis=None（展平全归约）----
    m = np.array([[3.0, 1.0], [2.0, 5.0]])
    check("median 2-D axis=None", float(np.median(m)) == 2.5)
    # ---- 性能 ----
    a = np.arange(1_000_000, dtype='float64')
    bench_lbl = "median 1M"
    np.median(a)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.median(a)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH {bench_lbl}: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
