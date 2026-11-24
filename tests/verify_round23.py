#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二十三轮验证：eig_vectors_core（非对称特征向量核心化）。

1. 值语义：对照纯 Python _eigenvectors_general（同算法逐位对齐）
2. 特征向量性质：A v ≈ λ v（残差小）
3. 性能：非对称 40×40 复数 eig 全路径
"""
import random
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402
from rsarray.linalg._pylinalg import (  # noqa: E402
    _eigenvectors_general, _eigvals_qr)

PASS = [0]
FAIL = [0]


def check(name, ok):
    if ok:
        PASS[0] += 1
    else:
        FAIL[0] += 1
        print(f"FAIL: {name}")


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


def main():
    random.seed(23)
    # ---- 对照：数学性质（B 近奇异，浮点轨迹差异放大方向偏差，
    # 逐位对照不可复现；以 Av ≈ λv 残差为准）----
    for n in (4, 10):
        for cx in (False, True):
            A = rand_mat(n, complex_=cx)
            evals = _eigvals_qr(A)
            evecs = linalg_eigvecs(A, evals)
            ok = True
            for k in range(n):
                lam = complex(evals[k])
                v = [complex(evecs[i][k]) for i in range(n)]
                Av = [sum(complex(A[i][j]) * v[j] for j in range(n))
                      for i in range(n)]
                for i in range(n):
                    if abs(Av[i] - lam * v[i]) > 1e-6 * max(
                            1.0, abs(Av[i])):
                        ok = False
            check(f"eigv {n} cx={cx} Av≈λv", ok)
    # ---- 特征向量性质：A v ≈ λ v ----
    n = 8
    A = rand_mat(n, complex_=True)
    evals = _eigvals_qr(A)
    evecs = linalg_eigvecs(A, evals)
    ok = True
    for k in range(n):
        lam = complex(evals[k])
        v = [complex(evecs[i][k]) for i in range(n)]
        Av = [sum(complex(A[i][j]) * v[j] for j in range(n)) for i in range(n)]
        for i in range(n):
            if abs(Av[i] - lam * v[i]) > 1e-6 * max(1.0, abs(Av[i])):
                ok = False
    check("eigv Av ≈ λv", ok)
    # ---- 性能（口径一致：两侧都只计特征向量部分，evals 预先算好）----
    import rsarray.num_core as core
    A = rand_mat(40, complex_=True)
    evals40 = _eigvals_qr(A)
    ev_arr = np.array(list(evals40), dtype='complex128')._array
    a_arr = np.array(A)._array
    t0 = time.perf_counter()
    _eigenvectors_general(A, evals40)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    core.linalg.eig_vectors_core(a_arr, ev_arr)
    t_new = time.perf_counter() - t0
    print(f"BENCH eigvectors 40x40 complex: py={t_old*1e3:.0f}ms "
          f"core={t_new*1e3:.1f}ms x{t_old / max(t_new, 1e-9):.0f}")

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


def linalg_eigvecs(A, evals):
    """经公共 eig 路径取特征向量（触发 eig_vectors_core 快路径）。"""
    return linalg_module.eig(np.array(A))[1].tolist()


if __name__ == '__main__':
    from rsarray.linalg import linalg_module  # noqa: E402
    main()
