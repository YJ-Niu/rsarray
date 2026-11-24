#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十二轮验证：复数 Jacobi SVD 核心化（svd_jacobi_impl_c）。

独立参考：
1. 模块内纯 Python _svd_jacobi 复数分支（兜底路径，仍在）
2. 重构残差 U S Vh ≈ A（复数，full/reduced 分支公式）+ 正交性
   U^H U ≈ I / Vh Vh^H ≈ I + sig 降序（数学上独立于消元实现）
3. pinv 对照纯 Python SVD 重构 V_k D U_k^H；lstsq 对照复数正规
   方程解（_gauss_jordan_solve(A^H A, A^H b)，独立路径）
4. 形状 / dtype / 2x2 快路径回归
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
        x = complex(x)
        y = complex(y)
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


def ch(A):
    """共轭转置 A^H"""
    return [[v.conjugate() for v in row] for row in transpose(A)]


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


def phase_equiv(mat1, mat2, axis, tol=1e-6):
    """SVD 非唯一性检法：mat1 与 mat2 逐"线"（axis=0 列 / axis=1 行）
    至多差一个模 1 的标量因子（奇异三元组相位自由度）。"""
    if axis == 0:
        nlines = len(mat1[0])
        get = lambda l, i: mat1[i][l]
        len0 = len(mat1)
    else:
        nlines = len(mat1)
        get = lambda l, i: mat1[l][i]
        len0 = len(mat1[0])
    for l in range(nlines):
        a = None
        for i in range(len0):
            if abs(get(l, i)) > 1e-10:
                a = i
                break
        if a is None:
            continue
        phi = get(l, a) / mat2[a if axis == 0 else l][l if axis == 0 else a] \
            if axis == 0 else get(l, a) / mat2[l][a]
        for i in range(len0):
            ref = mat2[i][l] if axis == 0 else mat2[l][i]
            cur = get(l, i)
            if abs(cur - phi * ref) > tol * max(1.0, abs(cur)):
                return False
    return True


def reconstruct(A, u, s, vh, m, n, full):
    """full: A = u (m,m) Σ (m,n) vh (n,n)；reduced: uk (m,k) D (k,k) vk (k,n)"""
    k = min(m, n)
    if full:
        Sigma = [[s[i] if i == j else 0.0 for j in range(n)]
                 for i in range(m)]
        return mm(u, mm(Sigma, vh))
    uk = [[row[j] for j in range(k)] for row in u]
    vk = vh[:k]
    D = [[s[i] if i == j else 0.0 for j in range(k)] for i in range(k)]
    return mm(uk, mm(D, vk))


def main():
    random.seed(11)
    # ---- 复数 svd 2-D：m<n / m>n / 方阵，full/reduced ----
    for (m, n) in ((3, 4), (4, 3), (5, 5), (2, 5)):
        for full in (True, False):
            A = rand_mat(m, n, complex_=True)
            r = linalg_module.svd(np.array(A), full_matrices=full,
                                  compute_uv=True)
            u = r.U.tolist()
            s = [float(v) for v in r.S.tolist()]
            vh = r.Vh.tolist()
            ref_u, ref_s, ref_vh = svd_ref(A, full)
            k = min(m, n)
            u_shape = (m, m) if full else (m, k)
            v_shape = (n, n) if full else (k, n)
            tag = f"csvd {m}x{n} full={full}"
            check(tag + " shape",
                  tuple(r.U.shape) == u_shape
                  and tuple(r.Vh.shape) == v_shape
                  and tuple(r.S.shape) == (k,))
            check(tag + " sig", close(s, ref_s))
            # u/vh 与参考至多差逐奇异三元组相位（SVD 固有自由度，
            # 同算法不同实现的收敛 sweep 边界可差一次旋转）
            check(tag + " u phase-equiv", phase_equiv(u, ref_u, axis=0))
            check(tag + " vh phase-equiv", phase_equiv(vh, ref_vh, axis=1))
            R = reconstruct(A, u, s, vh, m, n, full)
            check(tag + " reconstruct", close(flat(R), flat(A), 1e-8))
            uk = [[row[j] for j in range(k)] for row in u]
            vk = vh[:k]
            I1 = [[1.0 if i == j else 0.0 for j in range(k)]
                  for i in range(k)]
            check(tag + " u ortho",
                  close(flat(mm(ch(uk), uk)), flat(I1), 1e-7))
            check(tag + " vh ortho",
                  close(flat(mm(vk, ch(vk))), flat(I1), 1e-7))
            check(tag + " sig desc",
                  all(s[i] >= s[i + 1] for i in range(len(s) - 1)))
    # ---- 复数 sig only ----
    A = rand_mat(6, 4, complex_=True)
    s = linalg_module.svd(np.array(A), compute_uv=False)
    check("csvd sig only",
          tuple(s.shape) == (4,)
          and close([float(v) for v in s.tolist()], sig_ref(A)))
    # ---- 2x2 复数（原 Python 路径，现 core）----
    A = rand_mat(2, 2, complex_=True)
    r = linalg_module.svd(np.array(A))
    ref_u, ref_s, ref_vh = svd_ref(A, True)
    check("csvd 2x2",
          close([float(v) for v in r.S.tolist()], ref_s, 1e-7)
          and close(flat(r.U.tolist()), flat(ref_u), 1e-7)
          and close(flat(r.Vh.tolist()), flat(ref_vh), 1e-7))
    # ---- 复数 norm ord=2/-2/'nuc' ----
    for (m, n) in ((4, 3), (3, 5), (4, 4)):
        M = rand_mat(m, n, complex_=True)
        ref_s = sig_ref(M)
        for o, ref in ((2, ref_s[0]), (-2, ref_s[-1]),
                       ('nuc', sum(ref_s))):
            got = float(linalg_module.norm(np.array(M), o))
            check(f"cnorm {m}x{n} ord={o}",
                  abs(got - ref) <= 1e-7 * max(1.0, abs(ref)))
    # ---- 复数 cond ----
    A = rand_mat(5, 5, complex_=True)
    ref_s = sig_ref(A)
    for p, ref in ((None, ref_s[0] / ref_s[-1]),
                   (2, ref_s[0] / ref_s[-1]),
                   (-2, ref_s[-1] / ref_s[0])):
        got = float(linalg_module.cond(np.array(A), p))
        check(f"ccond p={p}",
              abs(got - ref) <= 1e-6 * max(1.0, abs(ref)))
    # ---- 复数 pinv（对照纯 Python SVD 重构 V_k D U_k^H）----
    for (m, n) in ((4, 3), (3, 5)):
        A = rand_mat(m, n, complex_=True)
        P = linalg_module.pinv(np.array(A)).tolist()
        u_l, s_l, vh_l = svd_ref(A, True)
        k = min(m, n)
        V = ch(vh_l)
        U = u_l
        Vk = [[row[j] for j in range(k)] for row in V]
        Uk = [[row[j] for j in range(k)] for row in U]
        # 注意：链式条件表达式 (A if C1 else (B if C2 else C)) 在 C1
        # 成立时不检查 C2——必须显式 and i == j，否则 D 整行被填 1/s
        D = [[(1.0 / s_l[i]) if (s_l[i] > 1e-15 * s_l[0] and i == j)
              else 0.0 for j in range(k)] for i in range(k)]
        refP = mm(Vk, mm(D, ch(Uk)))
        check(f"cpinv {m}x{n}", close(flat(P), flat(refP), 1e-7))
        check(f"cpinv {m}x{n} APA",
              close(flat(mm(A, mm(P, A))), flat(A), 1e-6))
    # ---- 复数 lstsq（对照复数正规方程解，独立路径）----
    A = rand_mat(5, 3, complex_=True)
    b = rand_mat(5, 1, complex_=True)
    out = linalg_module.lstsq(np.array(A), np.array(b))
    xflat = [v[0] for v in out[0].tolist()]
    AH = ch(A)
    xref = _gauss_jordan_solve(mm(AH, A), mm(AH, b))
    check("clstsq vs normal solve", close(xflat, flat(xref), 1e-7))

    # ---- 性能对照（基线 = 纯 Python 复数 Jacobi）----
    A = rand_mat(80, 80, complex_=True)
    t0 = time.perf_counter()
    _svd_jacobi(A, True, True)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.svd(np.array(A))
    t_new = time.perf_counter() - t0
    print(f"BENCH csvd 80x80 full uv: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    A2 = rand_mat(40, 60, complex_=True)
    t0 = time.perf_counter()
    _svd_jacobi(A2, False, False)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.svd(np.array(A2), compute_uv=False)
    t_new = time.perf_counter() - t0
    print(f"BENCH csvd 40x60 sig: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
