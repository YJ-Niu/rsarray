#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十一轮验证：svd_jacobi_sig/uv（Jacobi SVD 核心化）。

独立参考：
1. 模块内纯 Python _svd_jacobi（兜底路径，仍在）
2. 重构残差 U S Vh ≈ A、正交性 U^T U ≈ I / Vh Vh^T ≈ I、sig 降序
   （数学上独立于消元实现）
3. lstsq 对照正规方程解（_gauss_jordan_solve(A^T A, A^T b)，独立路径）
4. pinv 对照纯 Python SVD 重构 V diag(1/s) U^T
5. 形状 / dtype / 复数与空矩阵兜底
"""
import random
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402
from rsarray.linalg import (_gauss_jordan_solve, _svd_jacobi,  # noqa: E402
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


def close(a, b, tol=1e-8):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, complex) or isinstance(y, complex):
            x = complex(x)
            y = complex(y)
        else:
            x = float(x)
            y = float(y)
        d = abs(x - y)
        s = max(1.0, abs(x), abs(y))
        if d > tol * s:
            return False
    return True


def mm(A, B):
    m = len(A)
    n = len(A[0]) if m else 0
    p = len(B[0]) if B and B[0] else 0
    return [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(p)]
            for i in range(m)]


def transpose(A):
    if not A:
        return []
    return [[row[j] for row in A] for j in range(len(A[0]))]


def rand_mat(m, n, complex_=False):
    rows = []
    for _ in range(m):
        row = []
        for _ in range(n):
            v = random.uniform(-2.0, 2.0)
            if complex_:
                v = v + 1j * random.uniform(-2.0, 2.0)
            row.append(v)
        rows.append(row)
    return rows


def svd_ref(A, full=True):
    u, s, vh = _svd_jacobi(A, full, True)
    return (u.tolist(), [float(v) for v in s.tolist()], vh.tolist())


def sig_ref(M):
    return [abs(v) for v in _svd_jacobi(M, False, False).tolist()]


def main():
    random.seed(7)
    # ---- svd 2-D 实数：m<n / m>n / 方阵，full/reduced ----
    for (m, n) in ((3, 4), (4, 3), (5, 5), (2, 5)):
        for full in (True, False):
            A = rand_mat(m, n)
            r = linalg_module.svd(np.array(A), full_matrices=full,
                                  compute_uv=True)
            u = r.U.tolist()
            s = [float(v) for v in r.S.tolist()]
            vh = r.Vh.tolist()
            ref_u, ref_s, ref_vh = svd_ref(A, full)
            k = min(m, n)
            u_shape = (m, m) if full else (m, k)
            v_shape = (n, n) if full else (k, n)
            tag = f"svd {m}x{n} full={full}"
            check(tag + " shape",
                  tuple(r.U.shape) == u_shape
                  and tuple(r.Vh.shape) == v_shape
                  and tuple(r.S.shape) == (k,))
            check(tag + " sig", close(s, ref_s))
            # 重构 A = U S Vh：full 用 (m,m) U 与 (m,n) 嵌入 Σ；
            # reduced 用 U_k (m,k)、对角 (k,k)、Vh_k (k,n)
            uk = [[row[j] for j in range(k)] for row in u]
            vk = vh[:k]
            if full:
                Sigma = [[s[i] if i == j else 0.0 for j in range(n)]
                         for i in range(m)]
                R = mm(u, mm(Sigma, vh))
            else:
                Sdiag = [[s[i] if i == j else 0.0 for j in range(k)]
                         for i in range(k)]
                R = mm(uk, mm(Sdiag, vk))
            check(tag + " reconstruct", close(flat(R), flat(A), 1e-8))
            I1 = [[1.0 if i == j else 0.0 for j in range(k)]
                  for i in range(k)]
            check(tag + " u ortho",
                  close(flat(mm(transpose(uk), uk)), flat(I1), 1e-7))
            check(tag + " vh ortho",
                  close(flat(mm(vk, transpose(vk))), flat(I1), 1e-7))
            check(tag + " sig desc",
                  all(s[i] >= s[i + 1] for i in range(len(s) - 1)))
    # ---- compute_uv=False ----
    A = rand_mat(6, 4)
    s = linalg_module.svd(np.array(A), compute_uv=False)
    check("svd sig only",
          tuple(s.shape) == (4,)
          and close([float(v) for v in s.tolist()], sig_ref(A)))
    # ---- 复数（Python 兜底回归）----
    A = rand_mat(4, 3, complex_=True)
    r = linalg_module.svd(np.array(A))
    ref_u, ref_s, ref_vh = svd_ref(A, True)
    check("svd complex",
          close([float(v) for v in r.S.tolist()], ref_s, 1e-7)
          and close(flat(r.U.tolist()), flat(ref_u), 1e-7)
          and close(flat(r.Vh.tolist()), flat(ref_vh), 1e-7))
    # ---- 2x2 实数（既有快路径回归）+ 整型输入 ----
    A = rand_mat(2, 2)
    r = linalg_module.svd(np.array(A))
    _, ref_s, _ = svd_ref(A, True)
    check("svd 2x2 fast path",
          close([float(v) for v in r.S.tolist()], ref_s, 1e-6))
    r = linalg_module.svd(np.array([[3, 4], [0, 5]], dtype='int64'))
    ref_s = sig_ref([[3.0, 4.0], [0.0, 5.0]])
    check("svd int input",
          close([float(v) for v in r.S.tolist()], ref_s, 1e-6)
          and str(r.S.dtype) == 'float64')
    # ---- 空矩阵（Python 路径）----
    z1 = linalg_module.svd(np.zeros((1, 0)))
    check("svd empty (1,0)", tuple(z1.S.shape) == (0,))
    z2 = linalg_module.svd(np.zeros((0, 3)))
    check("svd empty (0,3)", tuple(z2.S.shape) == (0,))
    # ---- norm ord=2/-2/'nuc' ----
    for (m, n) in ((4, 3), (3, 5), (4, 4)):
        M = rand_mat(m, n)
        ref_s = sig_ref(M)
        for o, ref in ((2, ref_s[0]), (-2, ref_s[-1]),
                       ('nuc', sum(ref_s))):
            got = float(linalg_module.norm(np.array(M), o))
            check(f"norm {m}x{n} ord={o}",
                  abs(got - ref) <= 1e-7 * max(1.0, abs(ref)))
    # ---- norm stacked ord=2（axis 批量路径）----
    Ms = [rand_mat(5, 3) for _ in range(6)]
    got = linalg_module.norm(np.array(Ms), 2, axis=(-2, -1))
    refs = [sig_ref(M)[0] for M in Ms]
    check("norm stacked ord=2",
          tuple(got.shape) == (6,)
          and close([float(v) for v in got.tolist()], refs, 1e-7))
    # ---- cond ----
    A = rand_mat(5, 5)
    ref_s = sig_ref(A)
    for p, ref in ((None, ref_s[0] / ref_s[-1]),
                   (2, ref_s[0] / ref_s[-1]),
                   (-2, ref_s[-1] / ref_s[0])):
        got = float(linalg_module.cond(np.array(A), p))
        check(f"cond p={p}",
              abs(got - ref) <= 1e-6 * max(1.0, abs(ref)))
    # ---- pinv（对照纯 Python SVD 重构）----
    for (m, n) in ((4, 3), (3, 5)):
        A = rand_mat(m, n)
        P = linalg_module.pinv(np.array(A)).tolist()
        u_l, s_l, vh_l = svd_ref(A, True)
        k = min(m, n)
        inv_s = [1.0 / v if v > 1e-15 * s_l[0] else 0.0 for v in s_l]
        VT = transpose(vh_l)
        Vk = [[row[j] for j in range(k)] for row in VT]
        D = [[inv_s[i] if i == j else 0.0 for j in range(k)]
             for i in range(k)]
        Uk = [[row[j] for j in range(k)] for row in u_l]
        refP = mm(Vk, mm(D, transpose(Uk)))
        check(f"pinv {m}x{n}", close(flat(P), flat(refP), 1e-7))
        check(f"pinv {m}x{n} APA",
              close(flat(mm(A, mm(P, A))), flat(A), 1e-6))
    # ---- lstsq（对照正规方程解，独立路径）----
    A = rand_mat(5, 3)
    b = rand_mat(5, 1)
    out = linalg_module.lstsq(np.array(A), np.array(b))
    xflat = [v[0] for v in out[0].tolist()]
    AN = transpose(A)
    xref = _gauss_jordan_solve(mm(AN, A), mm(AN, b))
    check("lstsq vs normal solve",
          close(xflat, flat(xref), 1e-7))

    # ---- 性能对照（旧 = 纯 Python Jacobi，新 = core）----
    A = rand_mat(100, 100)
    t0 = time.perf_counter()
    _svd_jacobi(A, True, True)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.svd(np.array(A))
    t_new = time.perf_counter() - t0
    print(f"BENCH svd 100x100 full uv: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    t0 = time.perf_counter()
    _svd_jacobi(A, False, False)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.svd(np.array(A), compute_uv=False)
    t_new = time.perf_counter() - t0
    print(f"BENCH svd 100x100 sig: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    A2 = rand_mat(60, 90)
    t0 = time.perf_counter()
    _svd_jacobi(A2, True, True)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.svd(np.array(A2))
    t_new = time.perf_counter() - t0
    print(f"BENCH svd 60x90 full uv: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    Ms = [rand_mat(40, 40) for _ in range(25)]
    # 基线 = 纯 Python Jacobi（旧 _norm_mat 路径的消元成本）
    t0 = time.perf_counter()
    [_svd_jacobi(M, False, False) for M in Ms]
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.norm(np.array(Ms), 2, axis=(-2, -1))
    t_new = time.perf_counter() - t0
    print(f"BENCH norm ord=2 stacked (25,40,40): py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
