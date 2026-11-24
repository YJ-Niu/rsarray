#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十五轮验证：median_core(nan_filter=True)——nanmedian 快路径。

1. 值语义：nanmedian 忽略 NaN（部分 NaN/无 NaN/全 NaN → NaN）
2. 警告语义：全 NaN 数组触发 "All-NaN slice encountered"
3. 性能：nanmedian 1M 含 NaN
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
    # ---- 值语义：部分 NaN（忽略 NaN）----
    an = np.array([1.0, float('nan'), 3.0])
    v = float(np.nanmedian(an))
    check("nanmedian partial", v == 2.0)
    # ---- 无 NaN（与 median 一致）----
    check("nanmedian no nan", float(np.nanmedian(np.array([3.0, 1.0, 2.0]))) == 2.0)
    # ---- 全 NaN → NaN + 警告 ----
    allnan = np.array([np.nan, np.nan])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = float(np.nanmedian(allnan))
        check("nanmedian all-nan = nan", v != v)
        check("nanmedian all-nan warns", len(w) >= 1)
    # ---- 偶数长度 ----
    ae = np.array([4.0, 1.0, float('nan'), 3.0, 2.0, 5.0])
    check("nanmedian even", float(np.nanmedian(ae)) == 3.0)
    # ---- 性能 ----
    N = 1_000_000
    a = np.arange(N, dtype='float64')
    anan = a.copy()
    anan[::10] = np.nan
    np.nanmedian(anan)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.nanmedian(anan)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH nanmedian 1M (partial nan): {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
