#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十三轮验证：svd_assemble_core（pinv / lstsq 组装核心化）。

独立参考：
1. pinv：从公共 svd（core）结果以 V_k·D⁺·U_k^H 重构（相位无关，应
   ~1e-12 一致）+ APA ≈ A
2. lstsq：x 对照正规方程解 _gauss_jordan_solve(A^H A, A^H b)（独立
   GJ 路径）；residuals 对照嵌套列表独立计算；rank/s 对照
3. 形状 / dtype / rcond 变体 / 欠定与秩亏行为
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


def close(a, b, tol=1e-9):
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


def pinv_ref(A, rcond=1e-15):
    """独立参考：公共 svd（core）→ V_k D⁺ U_k^H（相位无关）"""
    m = len(A)
    n = len(A[0])
    r = linalg_module.svd(np.array(A), full_matrices=False)
    u = r.U.tolist()
    s = [float(v) for v in r.S.tolist()]
    vh = r.Vh.tolist()
    k = min(m, n)
    Vk = [[row[j] for j in range(k)] for row in ch(vh)]
    Uk = [[row[j] for j in range(k)] for row in u]
    D = [[(1.0 / s[i]) if (s[i] > rcond * s[0] and i == j) else 0.0
          for j in range(k)] for i in range(k)]
    return mm(Vk, mm(D, ch(Uk)))


def old_lstsq_assemble(A, b_rows, rcond=1e-15):
    """旧纯 Python 组装（被替换代码的复刻，用于 bench 与数值对照）"""
    m = len(A)
    n = len(A[0])
    r = linalg_module.svd(np.array(A), full_matrices=False)
    u = r.U.tolist()
    s = [float(v) for v in r.S.tolist()]
    vt = r.Vh.tolist()
    k = min(m, n)
    s_max = max(s) if s else 0.0
    cutoff = rcond * s_max
    rank = sum(1 for v in s if v > cutoff)

    def _conj(z):
        return z.conjugate() if isinstance(z, complex) else z

    s_inv = [(1.0 / v) if v > cutoff else 0.0 for v in s]
    n_rhs = len(b_rows[0])
    x_rows = [[0.0] * n_rhs for _ in range(n)]
    for c in range(n_rhs):
        uhb = [0.0] * k
        for j in range(k):
            acc = 0
            for i in range(m):
                acc += _conj(u[i][j]) * b_rows[i][c]
            uhb[j] = acc
        for i in range(n):
            acc = 0
            for j in range(k):
                acc += _conj(vt[j][i]) * s_inv[j] * uhb[j]
            x_rows[i][c] = acc
    return x_rows, rank, s


def main():
    random.seed(13)
    # ---- pinv 2-D 实/复数 ----
    for (m, n) in ((4, 3), (3, 5), (5, 5), (2, 2), (1, 1)):
        for cx in (False, True):
            A = rand_mat(m, n, complex_=cx)
            P = linalg_module.pinv(np.array(A)).tolist()
            refP = pinv_ref(A)
            tol = 1e-9
            check(f"pinv {m}x{n} cx={cx}",
                  close(flat(P), flat(refP), tol))
            check(f"pinv {m}x{n} cx={cx} APA",
                  close(flat(mm(A, mm(P, A))), flat(A), 1e-6))
    # ---- pinv rcond 变体 ----
    A = rand_mat(4, 4)
    P0 = linalg_module.pinv(np.array(A), rcond=0.0).tolist()
    check("pinv rcond=0", close(flat(P0), flat(pinv_ref(A, 0.0))))
    P1 = linalg_module.pinv(np.array(A), rcond=1.0).tolist()
    check("pinv rcond=1 all-zero",
          all(abs(v) < 1e-30 for v in flat(P1)))
    # ---- pinv stacked ----
    As = [rand_mat(4, 3) for _ in range(3)]
    P = linalg_module.pinv(np.array(As))
    check("pinv stacked shape", tuple(P.shape) == (3, 3, 4))
    check("pinv stacked values",
          all(close(flat(P[i].tolist()), flat(pinv_ref(As[i])))
              for i in range(3)))
    # ---- lstsq 2-D：满秩 / 欠定 / 秩亏 ----
    for (m, n) in ((5, 3), (3, 5), (5, 5), (1, 1)):
        A = rand_mat(m, n)
        b = rand_mat(m, 1)
        x, res, rank, s = linalg_module.lstsq(np.array(A), np.array(b))
        xflat = [row[0] for row in x.tolist()]
        if m >= n:
            # 满秩：正规方程独立解（A^T A x = A^T b，GJ 路径）
            AN = transpose(A)
            xref = _gauss_jordan_solve(mm(AN, A), mm(AN, b))
        else:
            # 欠定：正规方程奇异，最小范数解 = A⁺ b（pinv 已独立验证）
            xref = [row[0] for row in mm(pinv_ref(A), b)]
        check(f"lstsq {m}x{n} x", close(xflat, flat(xref), 1e-7))
        ref_x_rows, ref_rank, ref_s = old_lstsq_assemble(A, b)
        check(f"lstsq {m}x{n} rank", rank == ref_rank)
        check(f"lstsq {m}x{n} s",
              close([float(v) for v in s.tolist()], ref_s, 1e-12))
        if m > n and rank == n:
            # 残差独立计算
            res_ref = []
            for c in range(1):
                acc = 0.0
                for i in range(m):
                    rr = b[i][0]
                    for kk in range(n):
                        rr -= A[i][kk] * xflat[kk]
                    acc += abs(rr) ** 2
                res_ref.append(acc)
            check(f"lstsq {m}x{n} residuals",
                  tuple(res.shape) == (1,)
                  and close([float(v) for v in res.tolist()], res_ref,
                            1e-9))
        else:
            check(f"lstsq {m}x{n} residuals empty",
                  tuple(res.shape) == (0,))
    # b (m,k)
    A = rand_mat(5, 3)
    B = rand_mat(5, 2)
    x, res, rank, s = linalg_module.lstsq(np.array(A), np.array(B))
    check("lstsq b(m,k) shape", tuple(x.shape) == (3, 2))
    AN = transpose(A)
    xref = _gauss_jordan_solve(mm(AN, A), mm(AN, B))
    check("lstsq b(m,k) x",
          close(flat(x.tolist()), flat(xref), 1e-7))
    # 复数
    A = rand_mat(5, 3, complex_=True)
    b = rand_mat(5, 1, complex_=True)
    x, res, rank, s = linalg_module.lstsq(np.array(A), np.array(b))
    AH = ch(A)
    xref = _gauss_jordan_solve(mm(AH, A), mm(AH, b))
    check("clstsq x",
          close([v[0] for v in x.tolist()], flat(xref), 1e-7)
          and str(x.dtype) == 'complex128')
    # dtype
    x, res, rank, s = linalg_module.lstsq(
        np.array([[2, 0], [0, 4]], dtype='int64'),
        np.array([[1], [2]], dtype='int64'))
    check("lstsq int dtype",
          str(x.dtype) == 'float64' and rank == 2
          and close(flat(x.tolist()), [0.5, 0.5]))

    # ---- 性能对照（基线 = core svd + 纯 Python 组装，即被替换的旧路径）----
    A = rand_mat(100, 100)
    b = rand_mat(100, 4)
    t0 = time.perf_counter()
    old_lstsq_assemble(A, b)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.lstsq(np.array(A), np.array(b))
    t_new = time.perf_counter() - t0
    print(f"BENCH lstsq 100x100 k=4: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")
    A = rand_mat(80, 80, complex_=True)
    t0 = time.perf_counter()
    pinv_ref(A)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.pinv(np.array(A))
    t_new = time.perf_counter() - t0
    print(f"BENCH pinv 80x80 complex: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
