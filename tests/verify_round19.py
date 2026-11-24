#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十九轮验证：nan 聚合快路径放开（含 NaN 输入）。

1. 值语义：nansum/nanprod/nanmean/nanmin/nanmax/nanstd/nanvar
   含 NaN / 全 NaN 输入
2. 警告语义：全 NaN 数组 nanmean/nanmin/nanmax 触发 RuntimeWarning
   （慢路径保留）；部分 NaN 无警告
3. 性能：nansum/nanmean 1M 含 NaN
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


def bench(label, fn, n=3):
    fn()
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH {label}: {best*1e3:.1f}ms")


def main():
    N = 1_000_000
    a = np.arange(N, dtype='float64')
    anan = a.copy()
    anan[::1000] = np.nan  # 部分 NaN

    # ---- 值语义：部分 NaN ----
    expected = sum(x for x in anan.tolist() if x == x)
    check("nansum partial",
          abs(float(np.nansum(anan)) - expected)
          <= 1e-9 * max(1.0, abs(expected)))
    check("nanmean partial",
          abs(float(np.nanmean(anan)) - 500000.0) < 1.0)
    check("nanmin partial", float(np.nanmin(anan)) == 1.0)
    check("nanmax partial", float(np.nanmax(anan)) == 999999.0)
    # ---- 全 NaN 数组 ----
    allnan = np.array([np.nan, np.nan])
    check("nansum all-nan = 0", float(np.nansum(allnan)) == 0.0)
    check("nanprod all-nan = 1", float(np.nanprod(allnan)) == 1.0)
    check("nanmean all-nan = nan",
          float(np.nanmean(allnan)) != float(np.nanmean(allnan)))
    # ---- 警告语义：全 NaN 触发 RuntimeWarning ----
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.nanmean(allnan)
        check("nanmean all-nan warns", len(w) >= 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.nanmin(allnan)
        check("nanmin all-nan warns", len(w) >= 1)
    # 部分无警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.nanmean(anan)
        check("nanmean partial no warn", len(w) == 0)
    # ---- nanstd/nanvar 值 ----
    check("nanstd partial",
          abs(float(np.nanstd(np.array([1.0, 2.0, 3.0]))) - 0.81649658)
          < 1e-6)
    # ---- axis 归约含 NaN（nan_groups_allnan 判定放开）----
    m = np.array([[1.0, np.nan], [3.0, 4.0]])
    check("nansum axis=0", np.nansum(m, axis=0).tolist() == [4.0, 4.0])
    check("nansum axis=1", np.nansum(m, axis=1).tolist() == [1.0, 7.0])
    check("nanmin axis=0",
          np.nanmin(m, axis=0).tolist() == [1.0, 4.0])
    # 无全 NaN 组 → 无警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.nanmin(m, axis=0)
        check("nanmin axis no all-nan warn", len(w) == 0)
    # 存在全 NaN 组（axis=1 第二列全 NaN）→ 警告 + nan
    m2 = np.array([[1.0, np.nan], [3.0, np.nan]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r2 = np.nanmin(m2, axis=0)
        check("nanmin axis all-nan group warns", len(w) >= 1)
        check("nanmin axis all-nan group nan",
              r2.tolist()[1] != r2.tolist()[1])
    # ---- 性能 ----
    bench("nansum 1M (partial nan)", lambda: np.nansum(anan))
    bench("nanmean 1M (partial nan)", lambda: np.nanmean(anan))

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
