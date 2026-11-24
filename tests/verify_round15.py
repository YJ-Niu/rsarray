#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十五轮验证：eigvals_qr_core（Hessenberg + 双移位 QR 迭代核心化）。

独立参考（特征值顺序任意，同 LAPACK 语义）：
1. 排序后对照 Python _eigvals_qr（同算法不同实现）
2. 数学性质：Σλ ≈ trace、∏λ ≈ det（乘积对照，与实现独立）
3. eigvals/eig 公共 API 输出形状 / dtype / stacked
"""
import cmath
import random
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402
from rsarray.linalg import linalg_module  # noqa: E402
from rsarray.linalg._pylinalg import _eigvals_qr  # noqa: E402

PASS = [0]
FAIL = [0]


def check(name, ok):
    if ok:
        PASS[0] += 1
    else:
        FAIL[0] += 1
        print(f"FAIL: {name}")


def sorted_cmp(a, b, tol=1e-6):
    """特征值集合匹配：贪心最近邻（共轭对的虚部符号排列受浮点噪声
    影响会交错，字典序排序不可靠）"""
    if len(a) != len(b):
        return False
    bb = [complex(v) for v in b]
    used = [False] * len(bb)
    for v in a:
        cv = complex(v)
        best, bi = None, None
        for i, w in enumerate(bb):
            if used[i]:
                continue
            d = abs(cv - w)
            if best is None or d < best:
                best, bi = d, i
        if best is None or best > tol * max(1.0, abs(cv)):
            return False
        used[bi] = True
    return True


def rand_mat(n, complex_=False):
    rows = []
    for _ in range(n):
        row = []
        for _ in range(n):
            v = random.uniform(-2.0, 2.0)
            if complex_:
                v = v + 1j * random.uniform(-2.0, 2.0)
            row.append(v)
        rows.append(row)
    return rows


def py_det(mat):
    """嵌套列表复/实行列式（列主元消元）"""
    n = len(mat)
    w = [list(row) for row in mat]
    det = 1.0 + 0j
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(w[r][col]))
        if p != col:
            w[col], w[p] = w[p], w[col]
            det = -det
        pv = w[col][col]
        if abs(pv) == 0:
            return 0.0 + 0j
        det *= pv
        for r in range(col + 1, n):
            f = w[r][col] / pv
            for c in range(col, n):
                w[r][c] -= f * w[col][c]
    return det


def main():
    random.seed(15)
    # ---- eigvals 2-D：n = 9..12（core）与 n = 4（Python 分支回归）----
    for n in (4, 9, 10, 12):
        for cx in (False, True):
            A = rand_mat(n, complex_=cx)
            got = linalg_module.eigvals(np.array(A)).tolist()
            ref = _eigvals_qr(A)
            tag = f"eigvals {n} cx={cx}"
            check(tag + " vs ref", sorted_cmp(got, ref))
            # 数学性质：Σλ ≈ trace、∏λ ≈ det
            tr = sum(A[i][i] for i in range(n))
            det = py_det(A)
            p = 1.0 + 0j
            for v in got:
                p *= complex(v)
            check(tag + " trace",
                  abs(sum(complex(v) for v in got) - tr)
                  <= 1e-6 * max(1.0, abs(tr)))
            check(tag + " det",
                  abs(p - det) <= 1e-5 * max(1.0, abs(det)))
    # ---- 大矩阵（n=30，core 迭代收敛性）----
    A = rand_mat(30)
    got = linalg_module.eigvals(np.array(A)).tolist()
    ref = _eigvals_qr(A)
    check("eigvals 30 vs ref", sorted_cmp(got, ref, 1e-5))
    # ---- stacked ----
    As = [rand_mat(10, complex_=True) for _ in range(3)]
    got = linalg_module.eigvals(np.array(As))
    check("eigvals stacked shape", tuple(got.shape) == (3, 10))
    ok = all(sorted_cmp(got[i].tolist(), _eigvals_qr(As[i]))
             for i in range(3))
    check("eigvals stacked values", ok)
    # ---- eig 公共函数（对称/非对称，走 core 或 fallback 均须正确）----
    A = rand_mat(10, complex_=True)
    evals, evecs = linalg_module.eig(np.array(A))
    ref = _eigvals_qr(A)
    check("eig 10 evals", sorted_cmp(evals.tolist(), ref))
    # ---- dtype ----
    got = linalg_module.eigvals(np.array(rand_mat(10)))
    check("eigvals dtype", str(got.dtype) == 'complex128')

    # ---- bench：eigvals 60×60 complex（旧 = 纯 Python QR 迭代）----
    A = rand_mat(60, complex_=True)
    t0 = time.perf_counter()
    _eigvals_qr(A)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.eigvals(np.array(A))
    t_new = time.perf_counter() - t0
    print(f"BENCH eigvals 60x60 complex: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
