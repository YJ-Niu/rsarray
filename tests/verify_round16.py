#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十六轮验证：fp_arith_check_core（errstate 触发快速路径）。

errstate 语义回归（raise 模式最可靠）：
1. overflow：有限大数相加/相乘 → ±inf → FloatingPointError
2. invalid：inf-inf / 0×inf → nan → FloatingPointError
3. underflow：multiply 有限非零 → 0/次正规 → FloatingPointError
4. int/bool/complex 输入不触发（原语义）
5. 性能：a+b、a*b、a-b 1M 元素（旧 = Python 逐元素检查）
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


def raises_fp_error(fn):
    try:
        with np.errstate(over='raise', invalid='raise', under='raise'):
            fn()
        return False
    except FloatingPointError:
        return True
    except Exception:
        return False


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
    b = np.arange(N, 0, -1, dtype='float64')
    # ---- errstate 语义回归 ----
    big = np.array([1e308, 1e308])
    check("overflow add raises",
          raises_fp_error(lambda: big + big))
    check("overflow mul raises",
          raises_fp_error(lambda: big * big))
    check("invalid inf-inf raises",
          raises_fp_error(lambda: np.array([np.inf]) - np.array([np.inf])))
    check("invalid 0*inf raises",
          raises_fp_error(lambda: np.array([0.0]) * np.array([np.inf])))
    tiny = np.array([1e-200, 1e-200])
    check("underflow mul raises",
          raises_fp_error(lambda: tiny * tiny))
    # int 数组：无浮点触发
    check("int add no raise",
          not raises_fp_error(
              lambda: np.array([2**62], dtype='int64')
              + np.array([2**62], dtype='int64')))
    # complex：原语义不触发
    check("complex no raise",
          not raises_fp_error(
              lambda: np.array([1e308 + 0j]) + np.array([1e308 + 0j])))
    # 正常加法不触发
    check("normal add no raise",
          not raises_fp_error(lambda: a + b))
    # ---- divide errstate 语义（第二十二轮接线）----
    try:
        with np.errstate(divide='raise'):
            np.array([1.0, 2.0]) / np.array([1.0, 0.0])
        check("divide by zero raises", False)
    except FloatingPointError:
        check("divide by zero raises", True)
    check("divide normal no raise",
          not raises_fp_error(lambda: np.array([4.0, 8.0]) / np.array([2.0, 4.0])))
    # ---- 性能（1M 元素二元算术）----
    bench("add 1M", lambda: a + b)
    bench("mul 1M", lambda: a * b)
    bench("sub 1M", lambda: a - b)

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
