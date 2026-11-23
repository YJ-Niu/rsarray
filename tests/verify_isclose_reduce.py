# -*- coding: utf-8 -*-
"""验证 Rust isclose 快路径与复数 sum/mean 轴归约的正确性 + 性能。

非门禁（无 test_ 前缀）：python3 tests/verify_isclose_reduce.py
"""
import sys
import time

sys.path.insert(0, 'python')

import rsarray as np
from rsarray.math_functions import _isclose_arrays

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f" FAIL {name}")


def close_list(a, b):
    """逐元素近似比较两个扁平列表（支持 complex）。"""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != x and y != y:
            continue
        if abs(x - y) > 1e-9 * max(1.0, abs(x), abs(y)):
            return False
    return True


def flat(x):
    out = []
    for w in (x if isinstance(x, (list, tuple)) else [x]):
        if isinstance(w, (list, tuple)):
            out.extend(flat(w))
        else:
            out.append(w)
    return out


def ref_sum_axis(vals, axis, depth=0):
    """纯 Python 参考：嵌套列表沿 axis 求和。"""
    if depth == axis:
        if not vals:
            return []
        if isinstance(vals[0], (list, tuple)):
            # 沿 axis 折叠：对每个列位置跨行求和
            n = len(vals)
            return [sum(vals[i][j] for i in range(n))
                    for j in range(len(vals[0]))]
        return sum(vals)
    return [ref_sum_axis(v, axis, depth + 1) for v in vals]


print("=== isclose 正确性（对照 _isclose_arrays 参考） ===")
rng = np.random.default_rng(42)

a = rng.standard_normal(1000)
b = rng.standard_normal(1000) * 1.001
r_new = np.isclose(a, b, rtol=1e-3, atol=1e-8).tolist()
r_ref = _isclose_arrays(a, b, 1e-3, 1e-8, False).tolist()
check("isclose 实数 1D 同形状", r_new == r_ref)

a2 = rng.standard_normal((3, 4))
b2 = a2 * (1 + rng.standard_normal((3, 4)) * 1e-4)
check("isclose 实数 2D 同形状",
      np.isclose(a2, b2).tolist() ==
      _isclose_arrays(a2, b2, 1e-5, 1e-8, False).tolist())

check("isclose 数组 vs 标量",
      np.isclose(a, 0.0, atol=5.0).tolist() ==
      _isclose_arrays(a, np.array([0.0]), 1e-5, 5.0, False).tolist())
check("isclose 标量 vs 数组",
      np.isclose(0.0, a, atol=5.0).tolist() ==
      _isclose_arrays(np.array([0.0]), a, 1e-5, 5.0, False).tolist())

# 特值：NaN / inf / 相等
an = np.array([float('nan'), float('nan'), 1.0, float('inf'),
               float('-inf'), 2.0, 2.0])
bn = np.array([float('nan'), 1.0, 1.0, float('inf'),
               float('-inf'), 3.0, 2.0])
check("isclose NaN 对 (equal_nan=False)",
      np.isclose(an, bn).tolist() ==
      _isclose_arrays(an, bn, 1e-5, 1e-8, False).tolist())
check("isclose NaN 对 (equal_nan=True)",
      np.isclose(an, bn, equal_nan=True).tolist() ==
      _isclose_arrays(an, bn, 1e-5, 1e-8, True).tolist())

# 复数
ac = np.array([1 + 2j, 3 - 4j, 0.5 + 0.5j, 10 + 0j], dtype='complex128')
bc = np.array([1 + 2j, 3 - 4.001j, 0.6 + 0.5j, 11 + 1j], dtype='complex128')
check("isclose 复数 vs 复数",
      np.isclose(ac, bc).tolist() ==
      _isclose_arrays(ac, bc, 1e-5, 1e-8, False).tolist())
check("isclose 实数 vs 复数",
      np.isclose(np.array([1.0, 3.0, 0.5, 10.0]), bc).tolist() ==
      _isclose_arrays(np.array([1.0, 3.0, 0.5, 10.0]), bc,
                      1e-5, 1e-8, False).tolist())
_nan = float('nan')
acn = np.array([complex(_nan, 1.0), complex(1.0, _nan), 2 + 0j],
               dtype='complex128')
bcn = np.array([complex(1.0, _nan), complex(_nan, 1.0), 2 + 0j],
               dtype='complex128')
check("isclose 复数 NaN 对 (equal_nan=True)",
      np.isclose(acn, bcn, equal_nan=True).tolist() ==
      _isclose_arrays(acn, bcn, 1e-5, 1e-8, True).tolist())

# 广播回退路径
c1 = np.array([[1.0], [2.0], [3.0]])
c2 = np.array([[1.0, 2.0, 3.0]])
check("isclose (3,1)x(1,3) 广播回退",
      np.isclose(c1, c2).tolist() ==
      _isclose_arrays(c1, c2, 1e-5, 1e-8, False).tolist())
try:
    np.isclose(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
               np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    check("isclose (2,3)x(3,2) 应报错", False)
except ValueError:
    check("isclose (2,3)x(3,2) 应报错", True)

# allclose 一致性
check("allclose 一致", np.allclose(a, b, rtol=1e-3) ==
      bool(np.isclose(a, b, rtol=1e-3).all()))
check("allclose 复数", np.allclose(ac, ac) is True)

print("=== 复数 sum（对照纯 Python 参考） ===")
z = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
vals = z.tolist()

for axis in (0, 1, -1):
    ax = axis % 2
    got = z.sum(axis=axis)
    ref = ref_sum_axis(vals, ax)
    check(f"sum(axis={axis})", flat(got.tolist()) == ref or
          close_list(flat(got.tolist()), flat(ref)))
    got_k = z.sum(axis=axis, keepdims=True)
    check(f"sum(axis={axis}, keepdims=True) shape",
          got_k.shape == (1, 4) if ax == 0 else (3, 1))

s = z.sum()
ref_s = sum(flat(vals))
check("sum(axis=None) 复数标量", abs(s - ref_s) < 1e-9)
check("sum(axis=None) 返回复数", isinstance(s, complex))

# 0-d 短路
z0 = np.array([1 + 2j]).reshape(())
check("sum 0-d 复数短路", abs(z0.sum() - (1 + 2j)) < 1e-12)

# 空数组
ze = np.array([], dtype='complex128')
se = ze.sum()
check("sum 空复数", abs(se - 0j) < 1e-12)

print("=== 复数 mean（此前 Rust 丢虚部，重点回归） ===")
m = z.mean()
ref_m = ref_s / 12
check("mean(axis=None) 复数", abs(m - ref_m) < 1e-9)
for axis in (0, 1):
    gm = z.mean(axis=axis)
    ref = [v / (3 if axis == 0 else 4) for v in ref_sum_axis(vals, axis)]
    check(f"mean(axis={axis}) 复数", close_list(flat(gm.tolist()), ref))
    gmk = z.mean(axis=axis, keepdims=True)
    check(f"mean(axis={axis}, keepdims=True) shape",
          gmk.shape == (1, 4) if axis == 0 else (3, 1))

print("=== 实数 sum 无回归 ===")
ra = rng.standard_normal((3, 4))
check("实数 sum(axis=1)",
      close_list(flat(ra.sum(axis=1).tolist()),
                 ref_sum_axis(ra.tolist(), 1)))
ri = np.array([[1, 2, 3], [4, 5, 6]])
check("整型 sum", close_list(flat(ri.sum(axis=0).tolist()), [5, 7, 9]))

print()
print(f"正确性: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)

print("\n=== 性能对比（release） ===")


def t(f, *args, **kw):
    t0 = time.perf_counter()
    r = f(*args, **kw)
    return (time.perf_counter() - t0) * 1000, r


big = rng.standard_normal((1000, 1000))
big2 = big * 1.0005
t_new, _ = t(np.isclose, big, big2, rtol=1e-5, atol=1e-8)
t_old, _ = t(_isclose_arrays, big, big2, 1e-5, 1e-8, False)
print(f"isclose 实数 (1000,1000):  新 {t_new:8.1f} ms | 旧 {t_old:8.1f} ms"
      f"  → {t_old / t_new:.1f}x")

cbig = rng.standard_normal((500, 1000)) + 1j * rng.standard_normal((500, 1000))
cbig2 = cbig * (1 + 5e-4)
t_new, _ = t(np.isclose, cbig, cbig2)
t_old, _ = t(_isclose_arrays, cbig, cbig2, 1e-5, 1e-8, False)
print(f"isclose 复数 (500,1000):   新 {t_new:8.1f} ms | 旧 {t_old:8.1f} ms"
      f"  → {t_old / t_new:.1f}x")


def old_complex_sum_axis(arr, axis):
    """旧实现逻辑（tolist + flatten + 三重循环），仅用于基准。"""
    v = arr.tolist()
    fv = flat(v)
    shape = list(arr.shape)
    ax = axis % len(shape)
    size_ax = shape[ax]
    lead = 1
    for d in shape[:ax]:
        lead *= d
    trail = 1
    for d in shape[ax + 1:]:
        trail *= d
    res = []
    for li in range(lead):
        for ti in range(trail):
            s = 0j
            for ai in range(size_ax):
                s += fv[(li * size_ax + ai) * trail + ti]
            res.append(s)
    return res


zbig = rng.standard_normal((1000, 1000)) + 1j * rng.standard_normal((1000, 1000))
t_new, _ = t(zbig.sum, 1)
t_old, _ = t(old_complex_sum_axis, zbig, 1)
print(f"sum 复数 (1000,1000) ax1: 新 {t_new:8.1f} ms | 旧 {t_old:8.1f} ms"
      f"  → {t_old / t_new:.1f}x")

t_new, _ = t(zbig.mean, 1)
t_old, _ = t(old_complex_sum_axis, zbig, 1)
print(f"mean 复数 (1000,1000) ax1: 新 {t_new:8.1f} ms | 旧(Python) {t_old:8.1f} ms"
      f"  → {t_old / t_new:.1f}x")
print("\n完成")
