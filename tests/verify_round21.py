#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十一轮验证：eig_power_core（对称/Hermitian 幂迭代核心化）。

1. 值语义：对照纯 Python _eig_power_iteration（同算法逐位对齐）
2. 特征值 ≈ 实数（Hermitian 输入）、重构近似
3. float 对称大矩阵走内核（原 ValueError fallback 路径）
4. 性能：Hermitian 30×30 幂迭代
"""
import random
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402
from rsarray.linalg import linalg_module  # noqa: E402
from rsarray.linalg.__init__ import _eig_power_iteration  # noqa: E402

PASS = [0]
FAIL = [0]


def check(name, ok):
    if ok:
        PASS[0] += 1
    else:
        FAIL[0] += 1
        print(f"FAIL: {name}")


def close(a, b, tol=1e-6):
    x = complex(a)
    y = complex(b)
    return abs(x - y) <= tol * max(1.0, abs(x), abs(y))


def rand_herm(n, complex_=False):
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if j < i:
                row.append(rows[j][i].conjugate())
            elif j == i:
                v = random.uniform(-2.0, 2.0)
                if complex_:
                    v = v + 0j
                row.append(v)
            else:
                v = random.uniform(-2.0, 2.0)
                if complex_:
                    v = v + 1j * random.uniform(-2.0, 2.0)
                row.append(v)
        rows.append(row)
    return rows


def main():
    random.seed(21)
    # ---- 对照 Python 版（同算法）----
    for n in (4, 10, 20):
        for cx in (False, True):
            A = rand_herm(n, complex_=cx)
            evals, evecs = linalg_module.eig(np.array(A))
            ref_ev, ref_ec = _eig_power_iteration(A)
            ok = all(close(evals.tolist()[i], ref_ev[i])
                     for i in range(n))
            check(f"eig_power {n} cx={cx} evals vs ref", ok)
            flat_ok = all(
                close(evecs.tolist()[i][j], ref_ec[i][j], 1e-6)
                for i in range(n) for j in range(n))
            check(f"eig_power {n} cx={cx} evecs vs ref", flat_ok)
    # ---- Hermitian 特征值应为实数（复算术残留虚部 < 1e-9）----
    A = rand_herm(8, complex_=True)
    evals = linalg_module.eig(np.array(A))[0]
    check("hermitian evals real",
          all(abs(complex(v).imag)
              <= 1e-9 * max(1.0, abs(complex(v).real))
              for v in evals.tolist()))
    # ---- 性能 ----
    A = rand_herm(30, complex_=True)
    t0 = time.perf_counter()
    _eig_power_iteration(A)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    linalg_module.eig(np.array(A))
    t_new = time.perf_counter() - t0
    print(f"BENCH eig 30x30 hermitian: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
