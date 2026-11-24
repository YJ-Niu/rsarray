#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十八轮验证：has_nan/has_negative/round 三个快速内核。

1. sqrt：负数输入触发 invalid（errstate raise）；正常不触发
2. round：half-even 语义（6.35→6.4）、负 decimals、NaN 传递、
   float32 dtype 保持
3. nansum/nanmean：有 NaN 数组走 Rust 快路径（原 _has_nan_arr 拦截）
4. 性能：sqrt/round/nansum 1M
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

    # ---- sqrt errstate 语义 ----
    neg = np.array([-1.0, 4.0])
    try:
        with np.errstate(invalid='raise'):
            np.sqrt(neg)
        check("sqrt neg raises", False)
    except FloatingPointError:
        check("sqrt neg raises", True)
    check("sqrt pos no raise",
          not raises_fp(lambda: np.sqrt(np.array([4.0, 9.0]))))
    # ---- round half-even 语义 ----
    r = np.round(np.array([6.35, 2.5, 3.5, -2.5]), 0)
    check("round half-even",
          r.tolist() == [6.0, 2.0, 4.0, -2.0])
    r2 = np.round(np.array([123.456]), 2)
    check("round decimals=2", r2.tolist() == [123.46])
    r3 = np.round(np.array([1234.0]), -2)
    check("round decimals=-2", r3.tolist() == [1200.0])
    rn = np.round(np.array([1.0, float('nan'), 2.0]), 1)
    check("round nan passthrough",
          rn.tolist()[0] == 1.0 and rn.tolist()[1] != rn.tolist()[1])
    f32 = np.array([1.234, 5.678], dtype='float32')
    r4 = np.round(f32, 1)
    check("round float32 dtype", str(r4.dtype) == 'float32')
    # ---- nansum/nanmean 快路径（有 NaN 数组）----
    an = np.array([1.0, float('nan'), 3.0])
    check("nansum basic", np.nansum(an).tolist() == 4.0
          or float(np.nansum(an)) == 4.0)
    check("nanmean basic",
          float(np.nanmean(an)) == 2.0)
    # ---- 性能 ----
    bench("sqrt 1M", lambda: np.sqrt(a))
    bench("round 1M", lambda: np.round(a, 2))
    bench("nansum 1M (no nan)", lambda: np.nansum(a))
    anan = a.copy()
    anan[::1000] = np.nan  # 1000 个 NaN
    bench("nansum 1M (with nan)", lambda: np.nansum(anan))

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


def raises_fp(fn):
    try:
        with np.errstate(invalid='raise'):
            fn()
        return False
    except FloatingPointError:
        return True
    except Exception:
        return False


if __name__ == '__main__':
    main()
