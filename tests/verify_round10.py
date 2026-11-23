#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十轮验证：gj_solve_core（高斯-约当消元核心化：inv + solve）。

独立参考：
1. 模块内纯 Python _gauss_jordan_inv/_gauss_jordan_solve（兜底路径，仍在）
2. _solve_pinv 伪逆兜底（特征值路径，独立算法）——奇异 solve 对照
3. 残差 A@x - b ≈ 0（嵌套列表独立 matmul）
4. 形状 / dtype / 异常类型（LinAlgError）
"""
import random
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402
from rsarray.linalg import (LinAlgError, _gauss_jordan_inv,  # noqa: E402
                            _gauss_jordan_solve, _solve_pinv,
                            linalg_module)

PASS = [0]
FAIL = [0]


def check(name, ok):
    if ok:
        PASS[0] += 1
    else:
        FAIL[0] += 1
        print(f"FAIL: {name}")


def flat(v):
    out = []
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(reversed(x))
        else:
            out.append(x)
    return out


def close(a, b, tol=1e-7):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, complex) or isinstance(y, complex):
            x = complex(x)
            y = complex(y)
            d = abs(x - y)
            s = max(1.0, abs(x), abs(y))
        else:
            x = float(x)
            y = float(y)
            d = abs(x - y)
            s = max(1.0, abs(x), abs(y))
        if d > tol * s:
            return False
    return True


def matmul_ref(A, B):
    m = len(A)
    n = len(A[0]) if m else 0
    p = len(B[0]) if B else 0
    return [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(p)]
            for i in range(m)]


def rand_mat(m, n, complex_=False):
    # 对角占优 → 良态，残差对照稳定
    rows = []
    for i in range(m):
        row = []
        for j in range(n):
            v = random.uniform(-1.0, 1.0) + (5.0 if i == j else 0.0)
            if complex_:
                v = v + 1j * random.uniform(-1.0, 1.0)
            row.append(v)
        rows.append(row)
    return rows


def main():
    random.seed(42)
    # ---- inv 2-D ----
    for n in (2, 3, 4, 7, 16):  # 2/3 走既有 Rust 快路径，其余走 core
        A = rand_mat(n, n)
        got = linalg_module.inv(np.array(A))
        ref = _gauss_jordan_inv(A)
        check(f"inv {n}x{n} values", close(flat(got.tolist()), flat(ref)))
        R = matmul_ref(A, got.tolist())
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        check(f"inv {n}x{n} residual", close(flat(R), flat(I), 1e-9))
    got = linalg_module.inv(np.array([[5.0]]))
    check("inv 1x1", close(flat(got.tolist()), [0.2]))
    A = rand_mat(5, 5, complex_=True)
    got = linalg_module.inv(np.array(A))
    check("inv complex 5x5",
          close(flat(got.tolist()), flat(_gauss_jordan_inv(A))))
    try:
        linalg_module.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
        check("inv singular raises", False)
    except LinAlgError:
        check("inv singular raises", True)
    got = linalg_module.inv(np.array([[2, 0], [0, 4]], dtype='int64'))
    check("inv int dtype",
          str(got.dtype) == 'float64'
          and close(flat(got.tolist()), [0.5, 0.0, 0.0, 0.25]))
    # ---- inv stacked ----
    A = [rand_mat(3, 3) for _ in range(6)]
    got = linalg_module.inv(np.array(A))
    check("inv stacked shape", tuple(got.shape) == (6, 3, 3))
    check("inv stacked values",
          all(close(flat(got[i].tolist()), flat(_gauss_jordan_inv(A[i])))
              for i in range(6)))
    Ac = [rand_mat(2, 2, complex_=True) for _ in range(3)]
    got = linalg_module.inv(np.array(Ac))
    check("inv stacked complex",
          all(close(flat(got[i].tolist()), flat(_gauss_jordan_inv(Ac[i])))
              for i in range(3)))
    # ---- solve 2-D ----
    for n in (4, 10):
        A = rand_mat(n, n)
        bv = [row[0] for row in rand_mat(n, 1)]
        got = linalg_module.solve(np.array(A), np.array(bv))
        ref = _gauss_jordan_solve(A, [[v] for v in bv])
        gx = flat(got.tolist())
        check(f"solve b1d {n}",
              tuple(got.shape) == (n,) and close(gx, flat(ref)))
        R = [sum(A[i][j] * gx[j] for j in range(n)) - bv[i]
             for i in range(n)]
        check(f"solve b1d {n} residual", close(R, [0.0] * n, 1e-9))
    A = rand_mat(6, 6)
    B = rand_mat(6, 3)
    got = linalg_module.solve(np.array(A), np.array(B))
    check("solve b (m,k)",
          tuple(got.shape) == (6, 3)
          and close(flat(got.tolist()), flat(_gauss_jordan_solve(A, B))))
    A = rand_mat(5, 5, complex_=True)
    B = rand_mat(5, 2, complex_=True)
    got = linalg_module.solve(np.array(A), np.array(B))
    check("solve complex",
          tuple(got.shape) == (5, 2)
          and close(flat(got.tolist()), flat(_gauss_jordan_solve(A, B))))
    # ---- solve 奇异 → pinv 兜底（无异常，对照独立算法）----
    A = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]
    b = [[1.0], [2.0], [3.0]]
    got = linalg_module.solve(np.array(A), np.array(b))
    check("solve singular pinv fallback",
          close(flat(got.tolist()), flat(_solve_pinv(A, b)), 1e-6))
    # ---- solve stacked ----
    A = [rand_mat(3, 3) for _ in range(4)]
    bv = [row[0] for row in rand_mat(3, 1)]  # (3,) 一维 → 广播过全部批次
    got = linalg_module.solve(np.array(A), np.array(bv))
    check("solve stacked b1d",
          tuple(got.shape) == (4, 3)
          and all(close(flat(got[i].tolist()),
                        flat(_gauss_jordan_solve(A[i], [[v] for v in bv])))
                  for i in range(4)))
    Bm = [rand_mat(3, 2) for _ in range(4)]
    got = linalg_module.solve(np.array(A), np.array(Bm))
    check("solve stacked b (lead,m,k)",
          tuple(got.shape) == (4, 3, 2)
          and all(close(flat(got[i].tolist()),
                        flat(_gauss_jordan_solve(A[i], Bm[i])))
                  for i in range(4)))
    Bb = rand_mat(3, 2)
    got = linalg_module.solve(np.array(A), np.array(Bb))
    check("solve stacked b broadcast",
          tuple(got.shape) == (4, 3, 2)
          and all(close(flat(got[i].tolist()),
                        flat(_gauss_jordan_solve(A[i], Bb)))
                  for i in range(4)))
    As = [rand_mat(3, 3) for _ in range(4)]
    As[2] = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]
    got = linalg_module.solve(np.array(As), np.array(bv))
    check("solve stacked singular sub", tuple(got.shape) == (4, 3))
    got = linalg_module.solve(
        np.array([[2, 0], [0, 4]], dtype='int64'), np.array([[1], [2]]))
    check("solve int dtype", str(got.dtype) == 'float64')

    # ---- 性能对照（旧 = 纯 Python GJ，新 = core）----
    A100 = rand_mat(100, 100)
    t0 = time.perf_counter()
    _gauss_jordan_inv(A100)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.inv(np.array(A100))
    t_new = time.perf_counter() - t0
    print(f"BENCH inv 100x100 float: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    B4 = rand_mat(100, 4)
    t0 = time.perf_counter()
    _gauss_jordan_solve(A100, B4)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.solve(np.array(A100), np.array(B4))
    t_new = time.perf_counter() - t0
    print(f"BENCH solve 100x100 k=4: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    A60c = rand_mat(60, 60, complex_=True)
    t0 = time.perf_counter()
    _gauss_jordan_inv(A60c)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.inv(np.array(A60c))
    t_new = time.perf_counter() - t0
    print(f"BENCH inv 60x60 complex: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    A32 = [rand_mat(20, 20) for _ in range(32)]
    t0 = time.perf_counter()
    [_gauss_jordan_inv(s) for s in A32]
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.inv(np.array(A32))
    t_new = time.perf_counter() - t0
    print(f"BENCH inv stacked (32,20,20): py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
