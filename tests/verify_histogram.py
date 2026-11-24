# -*- coding: utf-8 -*-
"""第七轮热点下放验证：histogram / histogram2d / histogramdd 分箱内核。

纯 Python 独立参考实现对照（numpy bin 语义：末 bin 闭区间、越界/NaN 不计）
+ 性能基准。
非门禁脚本：PYTHONPATH=python .venv/bin/python tests/verify_histogram.py
"""
import bisect
import builtins
import math
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
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


def ref_hist(cols, edges, w):
    """独立参考：C 序 flat，首维 bin 最快；NaN/越界跳过；末 bin 闭区间。"""
    d = len(cols)
    n = len(cols[0])
    nbins = [len(e) - 1 for e in edges]
    size = 1
    for nb in nbins:
        size *= nb
    H = [0.0] * size
    strides = [1] * d
    for i in range(d - 2, -1, -1):
        strides[i] = strides[i + 1] * nbins[i + 1]
    for s in range(n):
        flat_i = 0
        ok = True
        for i in range(d):
            v = cols[i][s]
            if v != v:  # NaN
                ok = False
                break
            e = edges[i]
            nb = nbins[i]
            bi = bisect.bisect_right(e, v) - 1
            if bi == nb and v == e[nb]:
                bi -= 1
            if bi < 0 or v < e[0] or v > e[nb]:
                ok = False
                break
            flat_i += bi * strides[i]
        if ok:
            H[flat_i] += w[s] if w is not None else 1.0
    return H, nbins


def edges_of(vals, nb=10, rng=None):
    lo, hi = (float(rng[0]), float(rng[1])) if rng else (min(vals), max(vals))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    step = (hi - lo) / nb
    e = [lo + step * i for i in range(nb + 1)]
    e[-1] = hi
    return e


def close(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
            return False
    return True


# ===================== histogram (1-D) =====================

def test_histogram():
    vals = [float((i * 37) % 97) / 10.0 for i in range(500)]
    # 越界值 + 末边缘闭合 + NaN
    vals += [-5.0, 100.0, 9.7, float('nan')]

    # 均匀 bins（int）
    got, edges = np.histogram(np.array(vals), bins=12)
    e = edges_of(vals, 12)
    ref, _ = ref_hist([vals], [e], None)
    check("hist uniform", close(flat(got.tolist()), ref)
          and close(flat(edges.tolist()), e))
    check("hist uniform dtype", str(getattr(got, '_dtype', '')) == 'int64')

    # 显式 edges（array）
    xe = [i * 1.0 for i in range(0, 10)]
    got, edges = np.histogram(np.array(vals), bins=np.array(xe))
    ref, _ = ref_hist([vals], [xe], None)
    check("hist explicit edges", close(flat(got.tolist()), ref)
          and close(flat(edges.tolist()), xe))

    # range
    got, edges = np.histogram(np.array(vals), bins=8, range=(2.0, 6.0))
    e = edges_of(vals, 8, (2.0, 6.0))
    ref, _ = ref_hist([vals], [e], None)
    check("hist range", close(flat(got.tolist()), ref)
          and close(flat(edges.tolist()), e))

    # weights（float → float64；int → int64）
    w = [0.5 * (i % 3 + 1) for i in range(504)]
    got, _ = np.histogram(np.array(vals), bins=12, weights=np.array(w))
    e = edges_of(vals, 12)
    ref, _ = ref_hist([vals], [e], w)
    check("hist weights float", close(flat(got.tolist()), ref)
          and str(getattr(got, '_dtype', '')) == 'float64')

    wi = [i % 3 + 1 for i in range(504)]
    got, _ = np.histogram(np.array(vals), bins=12, weights=np.array(wi))
    ref, _ = ref_hist([vals], [e], wi)
    check("hist weights int", close(flat(got.tolist()), ref)
          and str(getattr(got, '_dtype', '')) == 'int64')

    # density
    got, edges = np.histogram(np.array(vals), bins=12, density=True)
    e = edges_of(vals, 12)
    ref, _ = ref_hist([vals], [e], None)
    total = builtins.sum(float(c) for c in ref)
    refd = [r / (total * (e[i + 1] - e[i])) for i, r in enumerate(ref)]
    check("hist density", close(flat(got.tolist()), refd))


# ===================== histogram2d =====================

def test_histogram2d():
    N = 800
    xa = [float((i * 61) % 100) / 10.0 for i in range(N)]
    ya = [float((i * 29) % 80) / 10.0 - 4.0 for i in range(N)]
    xa += [float('nan'), 99.0]
    ya += [3.0, float('nan')]
    N += 2

    # 标量 bins
    H, xed, yed = np.histogram2d(np.array(xa), np.array(ya), bins=10)
    ex, ey = edges_of(xa, 10), edges_of(ya, 10)
    ref, (nbx, nby) = ref_hist([xa, ya], [ex, ey], None)
    check("hist2d scalar bins",
          close(flat(H.tolist()), ref) and tuple(H.shape) == (nbx, nby))

    # 每维 bins（list）+ weights + range
    w = [1.0 + (i % 4) * 0.25 for i in range(N)]
    H, xed, yed = np.histogram2d(np.array(xa), np.array(ya), bins=(6, 4),
                                 range=((0.0, 8.0), (-4.0, 4.0)),
                                 weights=np.array(w))
    ex = edges_of(xa, 6, (0.0, 8.0))
    ey = edges_of(ya, 4, (-4.0, 4.0))
    ref, (nbx, nby) = ref_hist([xa, ya], [ex, ey], w)
    check("hist2d per-dim+range+weights",
          close(flat(H.tolist()), ref) and tuple(H.shape) == (nbx, nby))

    # 显式 edges（每维 array）
    H, xed, yed = np.histogram2d(
        np.array(xa), np.array(ya),
        bins=(np.array([i * 2.0 for i in range(5)]),
              np.array([-4.0 + i * 2.0 for i in range(4)])))
    ex = [i * 2.0 for i in range(5)]
    ey = [-4.0 + i * 2.0 for i in range(4)]
    ref, (nbx, nby) = ref_hist([xa, ya], [ex, ey], None)
    check("hist2d explicit edges", close(flat(H.tolist()), ref))

    # density（无权重 → total = N × 1.0）
    H, xed, yed = np.histogram2d(np.array(xa), np.array(ya), bins=10,
                                 density=True)
    ex, ey = edges_of(xa, 10), edges_of(ya, 10)
    ref, (nbx, nby) = ref_hist([xa, ya], [ex, ey], None)
    total = float(N)
    refd = []
    for i in range(nbx):
        for j in range(nby):
            refd.append(ref[i * nby + j]
                        / (total * (ex[i + 1] - ex[i]) * (ey[j + 1] - ey[j])))
    check("hist2d density", close(flat(H.tolist()), refd))


# ===================== histogramdd =====================

def test_histogramdd():
    N = 600
    c0 = [float((i * 41) % 50) / 5.0 for i in range(N)]
    c1 = [float((i * 23) % 40) / 4.0 - 2.0 for i in range(N)]
    c2 = [float((i * 7) % 30) / 6.0 for i in range(N)]
    c0 += [float('nan'), 99.0, 0.0]
    c1 += [1.0, float('nan'), -1.5]
    c2 += [0.5, 2.9, float('nan')]
    N += 3

    # 2-D 输入矩阵形态 (N, 2)
    sample = np.array([[c0[i], c1[i]] for i in range(N)])
    H, edges = np.histogramdd(sample, bins=8)
    e0, e1 = edges_of(c0, 8), edges_of(c1, 8)
    ref, (n0, n1) = ref_hist([c0, c1], [e0, e1], None)
    check("histdd matrix 2d", close(flat(H.tolist()), ref)
          and tuple(H.shape) == (n0, n1))

    # 3-D（序列 of arrays）+ 每维 bins list + range
    sample3 = (np.array(c0), np.array(c1), np.array(c2))
    H, edges = np.histogramdd(sample3, bins=(4, 3, 2),
                              range=((0.0, 8.0), (-2.0, 8.0), (0.0, 5.0)))
    e0 = edges_of(c0, 4, (0.0, 8.0))
    e1 = edges_of(c1, 3, (-2.0, 8.0))
    e2 = edges_of(c2, 2, (0.0, 5.0))
    ref, (n0, n1, n2) = ref_hist([c0, c1, c2], [e0, e1, e2], None)
    check("histdd 3d per-dim+range",
          close(flat(H.tolist()), ref) and tuple(H.shape) == (n0, n1, n2))

    # weights
    w = [1.0 + (i % 5) * 0.5 for i in range(N)]
    H, _ = np.histogramdd(sample3, bins=4, weights=np.array(w))
    e0, e1, e2 = edges_of(c0, 4), edges_of(c1, 4), edges_of(c2, 4)
    ref, (n0, n1, n2) = ref_hist([c0, c1, c2], [e0, e1, e2], w)
    check("histdd weights", close(flat(H.tolist()), ref))

    # 显式 edges（每维 array）
    H, edges = np.histogramdd(
        sample3, bins=(np.array([i * 2.0 for i in range(4)]),
                       np.array([-2.0 + i * 2.0 for i in range(3)]),
                       np.array([0.0, 2.0, 5.0])))
    e0 = [i * 2.0 for i in range(4)]
    e1 = [-2.0 + i * 2.0 for i in range(3)]
    e2 = [0.0, 2.0, 5.0]
    ref, (n0, n1, n2) = ref_hist([c0, c1, c2], [e0, e1, e2], None)
    check("histdd explicit edges", close(flat(H.tolist()), ref))

    # density
    H, edges = np.histogramdd(sample3, bins=(4, 3, 2), density=True)
    e0, e1, e2 = edges_of(c0, 4), edges_of(c1, 3), edges_of(c2, 2)
    ref, (n0, n1, n2) = ref_hist([c0, c1, c2], [e0, e1, e2], None)
    total = builtins.sum(ref)
    refd = []
    for fi in range(n0 * n1 * n2):
        vol = 1.0
        rem = fi
        for i in (2, 1, 0):
            vol *= (edges[i][rem % (n0, n1, n2)[i] + 1]
                    - edges[i][rem % (n0, n1, n2)[i]])
            rem //= (n0, n1, n2)[i]
        refd.append(ref[fi] / (total * vol) if vol else ref[fi])
    check("histdd density", close(flat(H.tolist()), refd))

    # 极端 bin 规模 → ValueError("too many bins")
    big = tuple(np.array([0.0, 1.0]) for _ in range(10))
    try:
        np.histogramdd(sample3, bins=[2 ** 100 // 2] * 2)
        check("histdd too many bins", False)
    except (ValueError, MemoryError):
        check("histdd too many bins", True)


# ===================== 性能 =====================

def bench_histogram():
    n = 1_000_000
    x = np.random.rand(n)
    t0 = time.perf_counter()
    got, _ = np.histogram(x, bins=100)
    t_new = time.perf_counter() - t0
    vals = flat(x.tolist())
    e = edges_of(vals, 100)
    ref, _ = ref_hist([vals], [e], None)
    check("hist bench values", close(flat(got.tolist()), ref))
    t1 = time.perf_counter()
    counts = [0] * 100
    for v in vals:
        b = bisect.bisect_right(e, v) - 1
        if b == 100 and v == e[-1]:
            b = 99
        if 0 <= b < 100:
            counts[b] += 1
    t_old = time.perf_counter() - t1
    print(f"BENCH histogram n={n} bins=100: rust={t_new*1e3:.1f}ms "
          f"python={t_old*1e3:.1f}ms speedup={t_old/max(t_new,1e-9):.1f}x")


def bench_histogram2d():
    N = 300_000
    x = np.random.rand(N)
    y = np.random.rand(N)
    t0 = time.perf_counter()
    H, _, _ = np.histogram2d(x, y, bins=50)
    t_new = time.perf_counter() - t0
    check("hist2d bench shape", tuple(H.shape) == (50, 50)
          and builtins.sum(flat(H.tolist())) == float(N))
    print(f"BENCH histogram2d N={N} bins=50: rust={t_new*1e3:.1f}ms")


def bench_histogramdd():
    n = 150_000
    sample = np.random.rand(n, 3)
    t0 = time.perf_counter()
    H, _ = np.histogramdd(sample, bins=20)
    t_new = time.perf_counter() - t0
    check("histdd bench shape", tuple(H.shape) == (20, 20, 20)
          and builtins.sum(flat(H.tolist())) == float(n))
    print(f"BENCH histogramdd n={n} d=3 bins=20: rust={t_new*1e3:.1f}ms")


def main():
    test_histogram()
    test_histogram2d()
    test_histogramdd()
    bench_histogram()
    bench_histogram2d()
    bench_histogramdd()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
