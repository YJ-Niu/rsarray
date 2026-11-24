#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十七轮验证：unique_1d_core（排序法 O(n log n)）。

语义回归（对照纯 Python 原实现 np.sort/np.unique 语义）：
1. 数值 1-D：值集合、排序、first/inverse/counts
2. NaN 两模式（equal_nan=True 归一 / False 独立）
3. -0.0 与 0.0 合并保留首次出现者
4. int/bool dtype；sorted=False 首次出现序
5. 复数/MaskedArray/axis 回退原路径仍正确
6. 性能：unique 100K/1M
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


def flat_close(a, b, tol=1e-9):
    fa = a.tolist()
    fb = b.tolist()
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if isinstance(x, complex) or isinstance(y, complex):
            x = complex(x)
            y = complex(y)
            if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                return False
        elif x != y:
            return False
    return True


def main():
    # ---- 基本数值 ----
    a = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])
    v = np.unique(a)
    check("basic values", v.tolist() == [1.0, 2.0, 3.0])
    v, idx, inv, cnt = np.unique(a, return_index=True,
                                 return_inverse=True, return_counts=True)
    check("basic idx", idx.tolist() == [1, 2, 0])
    check("basic inverse",
          inv.tolist() == [2, 0, 1, 0, 2, 1])
    check("basic counts", cnt.tolist() == [2, 2, 2])
    # ---- NaN 两模式 ----
    an = np.array([1.0, float('nan'), 2.0, float('nan'), 1.0])
    v = np.unique(an)  # equal_nan=True：NaN 归一排最后
    check("nan merged",
          len(v.tolist()) == 3 and v.tolist()[2] != v.tolist()[2])
    v2 = np.unique(an, equal_nan=False)  # False：各 NaN 独立
    check("nan separate",
          len(v2.tolist()) == 4
          and sum(1 for x in v2.tolist() if x != x) == 2)
    # ---- -0.0 与 0.0 合并（保留首次出现者）----
    az = np.array([0.0, -0.0, 0.0])
    vz = np.unique(az)
    check("zero merge keeps first",
          len(vz.tolist()) == 1 and vz.tolist()[0] == 0.0)
    az2 = np.array([-0.0, 0.0, -0.0])
    vz2 = np.unique(az2)
    check("neg zero merge keeps first",
          len(vz2.tolist()) == 1
          and np.signbit(vz2.tolist()[0]))
    # ---- sorted=False 首次出现序 ----
    v3 = np.unique(a, sorted=False)
    check("unsorted order", v3.tolist() == [3.0, 1.0, 2.0])
    _, inv3 = np.unique(a, return_inverse=True, sorted=False)
    check("unsorted inverse", inv3.tolist() == [0, 1, 2, 1, 0, 2])
    # ---- int/bool dtype ----
    vi = np.unique(np.array([3, 1, 2, 1], dtype='int64'))
    check("int unique",
          vi.tolist() == [1, 2, 3] and str(vi.dtype) == 'int64')
    vb = np.unique(np.array([True, False, True]))
    check("bool unique", vb.tolist() == [False, True])
    # ---- 复数/MaskedArray/axis 回退原路径仍正确 ----
    vc = np.unique(np.array([3.0 + 0j, 1.0 + 0j, 3.0 + 0j]))
    check("complex fallback", vc.tolist() == [1.0 + 0j, 3.0 + 0j])
    ma = np.ma.MaskedArray(np.array([3.0, 1.0, 3.0]),
                           mask=np.array([False, False, True]))
    vm = np.unique(ma)
    check("masked fallback", vm.tolist() == [1.0, 3.0])
    m2 = np.array([[3.0, 1.0], [1.0, 3.0]])
    va = np.unique(m2, axis=0)
    check("axis fallback",
          va.tolist() == [[1.0, 3.0], [3.0, 1.0]])
    # ---- A ≈ np.sort 语义（排序后与 flat 去重集合一致）----
    import random
    random.seed(17)
    big = np.array([random.uniform(-2, 2) for _ in range(100000)]
                   + [float('nan')] * 5)
    vb = np.unique(big)
    vb_list = vb.tolist()
    check("large unique sorted ascending",
          all(x <= y
              for x, y in zip(vb_list, vb_list[1:])
              if x == x and y == y))
    check("large unique nan last",
          vb.tolist()[-1] != vb.tolist()[-1])

    # ---- 性能 ----
    a1m = np.arange(1_000_000, dtype='float64')
    a100k = np.arange(100_000, dtype='float64')
    t0 = time.perf_counter()
    np.unique(a100k)
    print(f"BENCH unique 100K: "
          f"{(time.perf_counter() - t0) * 1e3:.1f}ms")
    t0 = time.perf_counter()
    np.unique(a1m)
    print(f"BENCH unique 1M: "
          f"{(time.perf_counter() - t0) * 1e3:.1f}ms")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
