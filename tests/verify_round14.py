#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十四轮验证：2x2 svd 快路径修复 + qr_householder_core（QR 核心化）。

独立参考：
1. svd 2x2：重构 U S Vh ≈ A、U^H U ≈ I、s 降序（修复前对角阵
   U/Vh 与 s 排列不一致）
2. QR：Q R 对照纯 Python _qr_householder（兜底路径，仍在）+
   A ≈ Q R、Q^H Q ≈ I、R 上三角（数学独立）
3. 形状 / dtype / mode 变体 / stacked
"""
import random
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402
from rsarray.linalg import linalg_module  # noqa: E402

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


def main():
    random.seed(14)
    # ---- 2x2 svd 快路径修复（含对角情形）----
    cases = [
        [[2.0, 0.0], [0.0, 4.0]],
        [[4.0, 0.0], [0.0, 2.0]],
        [[-3.0, 0.0], [0.0, 1.0]],
        [[0.0, 2.0], [4.0, 0.0]],
    ]
    cases += [rand_mat(2, 2) for _ in range(3)]
    for idx, A in enumerate(cases):
        r = linalg_module.svd(np.array(A))
        u = r.U.tolist()
        s = [float(v) for v in r.S.tolist()]
        vh = r.Vh.tolist()
        # 重构 U S Vh ≈ A（对角 Σ）
        R = [[s[j] * u[i][j] for j in range(2)] for i in range(2)]
        R = mm(u, mm([[s[0] if i == j else 0.0 for j in range(2)]
                      for i in range(2)], vh)) if False else \
            [[sum(u[i][t] * s[t] * vh[t][j] for t in range(2))
              for j in range(2)] for i in range(2)]
        check(f"svd2x2 case{idx} reconstruct",
              close(flat(R), flat(A), 1e-9))
        # 正交性
        I2 = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
        check(f"svd2x2 case{idx} ortho",
              close(flat(mm(ch(u), u)), flat(I2), 1e-9))
        check(f"svd2x2 case{idx} desc",
              s[0] >= s[1] and s[0] > 0)
    # ---- qr 2-D 实/复数，m>n / m<n / 方阵 ----
    for (m, n) in ((5, 3), (3, 5), (4, 4), (1, 1)):
        for cx in (False, True):
            for mode in ('reduced', 'complete'):
                A = rand_mat(m, n, complex_=cx)
                r = linalg_module.qr(np.array(A), mode=mode)
                q, rr = r.Q.tolist(), r.R.tolist()
                ref_q, ref_r = linalg_module._qr_householder(
                    np.array(A), mode=mode)
                tag = f"qr {m}x{n} cx={cx} {mode}"
                check(tag + " vs ref",
                      close(flat(q), flat(ref_q.tolist()), 1e-9)
                      and close(flat(rr), flat(ref_r.tolist()), 1e-9))
                # A ≈ Q R
                prod = mm(q, rr)
                check(tag + " reconstruct",
                      close(flat(prod), flat(A), 1e-8))
                # Q^H Q ≈ I（列数 = reduced: k, complete: m）
                qc = len(q[0]) if q else 0
                I = [[1.0 if i == j else 0.0 for j in range(qc)]
                     for i in range(qc)]
                check(tag + " q ortho",
                      close(flat(mm(ch(q), q)), flat(I), 1e-8))
                # R 上三角
                kexp = min(m, n)
                rrows = len(rr)
                upper = all(rr[i][j] == 0
                            for i in range(rrows)
                            for j in range(min(i, len(rr[0]))))
                check(tag + " r upper", upper)
    # ---- mode='r' 与 'raw' ----
    A = rand_mat(5, 3)
    rr = linalg_module.qr(np.array(A), mode='r')
    full = linalg_module.qr(np.array(A))
    check("qr mode=r",
          tuple(rr.shape) == (3, 3)
          and close(flat(rr.tolist()), flat(full.R.tolist())))
    h, tau = linalg_module.qr(np.array(A), mode='raw')
    check("qr mode=raw shape",
          tuple(h.shape) == (3, 5) and tuple(tau.shape) == (3,))
    # ---- qr stacked ----
    As = [rand_mat(4, 3) for _ in range(3)]
    r = linalg_module.qr(np.array(As))
    check("qr stacked shape",
          tuple(r.Q.shape) == (3, 4, 3) and tuple(r.R.shape) == (3, 3, 3))
    ok = True
    for i in range(3):
        ref_q, ref_r = linalg_module._qr_householder(
            np.array(As[i]), mode='reduced')
        ok = ok and close(flat(r.Q[i].tolist()), flat(ref_q.tolist()), 1e-9)
    check("qr stacked values", ok)
    # ---- dtype ----
    x = linalg_module.qr(np.array([[2, 1], [1, 3]], dtype='int64'))
    check("qr int dtype", str(x.Q.dtype) == 'float64')

    # ---- bench：复数 QR 40×40（旧 = 纯 Python Householder）----
    A = rand_mat(40, 40, complex_=True)
    t0 = time.perf_counter()
    linalg_module._qr_householder(np.array(A), mode='reduced')
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.qr(np.array(A), mode='reduced')
    t_new = time.perf_counter() - t0
    print(f"BENCH qr 40x40 complex: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
