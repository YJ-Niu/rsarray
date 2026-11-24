#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十七轮验证：astype 数值目标核心化。

1. 语义：int64（NaN→0、trunc、位宽补码回绕）、uint64 高位回落、
   bool 0/1（NaN→1）、float32 舍入与 dtype 保持
2. 性能：astype int64 1M、bincount(astype) 1M
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
    # ---- int64：NaN→0、trunc、回绕 ----
    a = np.array([1.7, -1.7, float('nan'), 2.0])
    r = a.astype('int64')
    check("int64 values", r.tolist() == [1, -1, 0, 2])
    # 位宽回绕：2^63 → -2^63（补码）
    big = np.array([2.0 ** 63])
    check("int64 wraparound", big.astype('int64').tolist() == [-(2 ** 63)])
    # ---- uint64 高位回落补码负值 ----
    # ---- uint64 高位回落补码负值（存储）；tolist 按无符号重解释显示 ----
    check("uint64 high-bit wraparound",
          big.astype('uint64').tolist() == [2 ** 63])
    # ---- bool：0/1（NaN→1）----
    rb = a.astype('bool')
    check("bool 0/1", rb.tolist() == [True, True, True, True])
    check("bool zero", np.array([0.0, 1.0]).astype('bool').tolist() == [False, True])
    # ---- float32：舍入 + dtype ----
    r32 = np.array([1.2345678, 2.0]).astype('float32')
    check("float32 dtype", str(r32.dtype) == 'float32')
    check("float32 value", abs(float(r32.tolist()[0]) - 1.2345678) < 1e-6)
    # ---- float64 不变（无 NaN 数组；nan != nan 会使列表比较 False）----
    check("float64 identity",
          big.astype('float64').tolist() == big.tolist())
    # ---- bincount 全链（astype + 内核）----
    bc_in = np.arange(100000, dtype='float64')
    r = np.bincount(bc_in.astype('int64'))
    check("bincount chain", int(r.sum()) == 100000)

    # ---- 性能 ----
    N = 1_000_000
    big = np.arange(N, dtype='float64')
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        big.astype('int64')
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH astype int64 1M: {best*1e3:.1f}ms")
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.bincount(big[:500000].astype('int64'))
        best = min(best, time.perf_counter() - t0)
    print(f"BENCH bincount(astype) 500K: {best*1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
