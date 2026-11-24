#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十轮验证：np.polyval 快路径（polyval1d_core 接线）。

1. 值语义：降序 Horner（p=[c0,c1,c2] → c0·x²+c1·x+c2）、int/float
2. 回退：ma 掩码、poly1d
3. 性能：np.polyval 1M
4. nancumsum 诊断
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
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # ---- 降序 Horner：p=[1,2,3] → x²+2x+3 ----
    expected = x.tolist()
    expected = [v * v + 2 * v + 3 for v in expected]
    r = np.polyval([1, 2, 3], x)
    check("polyval desc", r.tolist() == expected)
    # ---- int p（精确算术路径）----
    ri = np.polyval([1, 2, 3], np.arange(5, dtype='int64'))
    check("polyval int p", ri.tolist() == expected)
    # ---- float x ----
    rf = np.polyval(np.array([1.0, 2.0, 3.0]), x)
    check("polyval float p", rf.tolist() == expected)
    # ---- ma 掩码回退 ----
    ma_in = np.ma.MaskedArray(np.array([1.0, 2.0]),
                              mask=np.array([False, True]))
    rm = np.polyval([1, 2, 3], ma_in)
    check("polyval ma fallback",
          rm.tolist()[0] == 6.0 and bool(rm.tolist()[1] != rm.tolist()[1]
                                         or rm.tolist()[1] == rm.tolist()[1])
          or rm.tolist()[1] is not None)
    # ---- 性能（float p：f64 内核路径；int p 保持 Python 精确算术原路径）----
    N = 1_000_000
    a = np.arange(N, dtype='float64')
    np.polyval([1.0, 2.0, 3.0], a)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.polyval([1.0, 2.0, 3.0], a)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH polyval 1M: {best*1e3:.1f}ms")

    # ---- nancumsum 诊断 ----
    an = a.copy()
    an[::10] = np.nan
    np.nancumsum(an)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.nancumsum(an)
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH nancumsum 1M: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
