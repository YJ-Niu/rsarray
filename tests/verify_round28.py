#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十八轮验证：percentile_core（linear 百分位数核心化）。

1. 值语义：linear 插值（q=50 ≡ median）、NaN 过滤（nanpercentile）/
   传染（percentile）、q=0/100 边界
2. dtype：float32 保持、int → float64
3. 性能：percentile 1M、nanpercentile 1M
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
    a = np.arange(1_000_001, dtype='float64')  # 0..1M，median = 500000
    # ---- 基础值语义 ----
    check("p50 ≡ median",
          float(np.percentile(a, 50)) == float(np.median(a)))
    check("p0 = min", float(np.percentile(a, 0)) == 0.0)
    check("p100 = max", float(np.percentile(a, 100)) == 1000000.0)
    check("p25 linear",
          abs(float(np.percentile(a, 25)) - 250000.0) < 1e-9)
    # ---- nanpercentile：过滤 NaN ----
    an = a.copy()
    an[::10] = np.nan
    vn = float(np.nanpercentile(an, 50))
    check("nanp50 filters nan",
          abs(vn - 500000.0) < 1.0)
    check("nanp0 = 1.0 (0 被替换为 nan)",
          float(np.nanpercentile(an, 0)) == 1.0)
    # ---- percentile NaN 传染 ----
    v = float(np.percentile(an, 50))
    check("percentile nan propagate", v != v)
    # ---- dtype ----
    f32 = np.percentile(np.array([3.0, 1.0, 2.0], dtype='float32'), 50)
    check("float32 dtype", str(f32.dtype) == 'float32')
    iv = np.percentile(np.array([3, 1, 2], dtype='int64'), 50)
    check("int -> float64", float(iv) == 2.0)
    # ---- 性能 ----
    for label, fn in (("percentile 1M", lambda: np.percentile(a, 50)),
                      ("nanpercentile 1M", lambda: np.nanpercentile(an, 50))):
        fn()
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        print(f"BENCH {label}: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
