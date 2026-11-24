# -*- coding: utf-8 -*-
"""第六轮热点下放验证：polyvander / polyvander2d / polyvander3d / _polyvalnd。

纯 Python 独立参考实现对照（不依赖被测代码）+ 性能基准。
非门禁脚本：PYTHONPATH=python .venv/bin/python tests/verify_polyval_vander.py
"""
import builtins
import itertools
import sys
import time

sys.path.insert(0, 'python')
import rsarray as np
from rsarray.polynomial import polynomial as P

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


# ===================== 独立参考实现 =====================

def ref_polyvalnd(pts_flat, c_nested, out_shape):
    """Σ c[idx] Π p_a[k]^idx[a]（C 序系数迭代，零系数跳过）。
    pts_flat: 每点的 flat 值列表（已按 out_shape 对齐，同长）。
    c_nested: 嵌套系数列表（首维为各点幂次轴）。
    """
    cshape = list(np.array(c_nested).shape) if c_nested else []

    def get(vals, idx):
        v = vals
        for i in idx:
            v = v[i]
        return v

    def prod_shape(sh):
        n = 1
        for s in sh:
            n *= s
        return n

    npts = prod_shape(out_shape)
    a = len(pts_flat)
    res = [0.0] * npts
    for idx in itertools.product(*[range(d) for d in cshape]):
        cv = get(c_nested, idx)
        if cv == 0:
            continue
        for k in range(npts):
            term = cv
            for p in range(a):
                e = idx[p]
                if e:
                    term = term * (pts_flat[p][k] ** e)
            res[k] += term
    return res


def ref_polyvander(xv, deg):
    return [v ** i for v, i in itertools.product(xv, range(deg + 1))]


def ref_polyvander2d(xv, yv, d0, d1):
    out = []
    for x, y in zip(xv, yv):
        for i in range(d0):
            for j in range(d1):
                out.append(x ** i * y ** j)
    return out


def ref_polyvander3d(xv, yv, zv, d0, d1, d2):
    out = []
    for x, y, z in zip(xv, yv, zv):
        for i in range(d0):
            for j in range(d1):
                for t in range(d2):
                    out.append(x ** i * y ** j * z ** t)
    return out


def grids_match(got, ref_vals, ref_shape, tol=1e-9):
    if tuple(got.shape) != tuple(ref_shape):
        return False
    gf = flat(got.tolist())
    if len(gf) != len(ref_vals):
        return False
    for x, y in zip(gf, ref_vals):
        if abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
            return False
    return True


# ===================== polyvander 家族 =====================

def test_polyvander():
    x1 = np.array([0.0, 1.0, -2.5, 3.0, 1e-3, 10.0])
    got = P.polyvander(x1, 5)
    check("vander 1d", grids_match(
        got, ref_polyvander(flat(x1.tolist()), 5), (6, 6)))

    # 2-D 输入
    x2 = np.array([[1.5, -0.5], [2.0, 0.25]])
    got = P.polyvander(x2, 3)
    check("vander 2d", grids_match(
        got, ref_polyvander(flat(x2.tolist()), 3), (2, 2, 4)))

    # 0-d 输入
    xs = np.array(2.0)
    got = P.polyvander(xs, 2)
    check("vander 0d", grids_match(
        got, ref_polyvander([2.0], 2), (3,)))

    # deg=0
    got = P.polyvander(x1, 0)
    check("vander deg0", grids_match(got, [1.0] * 6, (6, 1)))

    # 整型输入 → int64 输出
    xi = np.array([1, 2, 3])
    got = P.polyvander(xi, 3)
    check("vander int dtype",
          str(getattr(got, '_dtype', '')) == 'int64' and
          grids_match(got, ref_polyvander([1, 2, 3], 3), (3, 4)))


def test_polyvander2d():
    x = np.array([[1.0], [2.0], [-0.5]])          # (3,1)
    y = np.array([[0.5, 2.0, -1.0, 3.0]])         # (1,4)
    xv, yv = bcast_vals(x, (3, 4)), bcast_vals(y, (3, 4))
    got = P.polyvander2d(x, y, (3, 2))
    check("vander2d broadcast", grids_match(
        got, ref_polyvander2d(xv, yv, 4, 3), (3, 4, 12)))

    # 同形状
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    y2 = np.array([[0.5, -1.0], [2.0, 0.25]])
    got = P.polyvander2d(x2, y2, (2, 2))
    check("vander2d sameshape", grids_match(
        got, ref_polyvander2d(flat(x2.tolist()), flat(y2.tolist()), 3, 3),
        (2, 2, 9)))

    # 整型输入
    xi = np.array([[1], [2]])
    yi = np.array([[1, 3]])
    got = P.polyvander2d(xi, yi, (2, 1))
    check("vander2d int dtype",
          str(getattr(got, '_dtype', '')) == 'int64' and grids_match(
              got, ref_polyvander2d([1, 1, 2, 2], [1, 3, 1, 3], 3, 2),
              (2, 2, 6)))


def test_polyvander3d():
    x = np.array([[1.0], [2.0]])
    y = np.array([[0.5, 2.0, -1.0]])
    z = np.array([[0.25, 3.0, 1.0], [-2.0, 0.5, 2.0]])
    bs = (2, 3)
    xv, yv, zv = bcast_vals(x, bs), bcast_vals(y, bs), bcast_vals(z, bs)
    got = P.polyvander3d(x, y, z, (2, 2, 2))
    check("vander3d broadcast", grids_match(
        got, ref_polyvander3d(xv, yv, zv, 3, 3, 3), bs + (27,)))


# ===================== _polyvalnd 家族 =====================

def bcast_vals(arr, bshape):
    """纯 Python 广播展平（与内核输入对齐）。"""
    shp = list(np.asarray(arr).shape)
    nd = builtins.len(bshape)
    pad = [1] * (nd - builtins.len(shp)) + shp
    vals = flat(np.asarray(arr).tolist())
    out = []
    for k in range(_prod(bshape)):
        q = []
        r = k
        for s in reversed(bshape):
            q.append(r % s)
            r //= s
        q.reverse()
        sidx = 0
        for a in range(nd - builtins.len(shp), nd):
            s = int(q[a] % pad[a])
            sidx = sidx * builtins.int(pad[a]) + s
        out.append(vals[builtins.int(sidx)])
    return out


def _prod(shp):
    n = 1
    for s in shp:
        n *= builtins.int(s)
    return n


def test_polyval2d():
    x = np.array([[1.0], [2.0], [-0.5], [3.0]])   # (4,1)
    y = np.array([[0.25, -1.0, 4.0, 0.0, 2.0, 0.5, 1.5, -2.0]])  # (1,8)
    c = np.array([[1.0, -2.0, 0.5, 3.0],
                  [0.0, 1.5, -1.0, 0.25],
                  [2.0, 0.0, 4.0, -0.75]])        # (3,4) 含零系数
    bshape = (4, 8)
    ref = ref_polyvalnd([bcast_vals(x, bshape), bcast_vals(y, bshape)],
                        c.tolist(), bshape)
    got = P.polyval2d(x, y, c)
    check("polyval2d broadcast+zeros", grids_match(got, ref, bshape))
    check("polyval2d dtype", str(getattr(got, '_dtype', '')) == 'float64')


def test_polyval3d():
    x = np.array([[1.0, 2.0], [3.0, -1.0]])       # (2,2)
    y = np.array([[0.5, 1.5]])                    # (1,2)
    z = np.array([[2.0], [0.25]])                 # (2,1)
    c = [[[1.0, 0.0, 2.0], [0.5, -1.0, 0.0]],
         [[0.0, 3.0, 1.5], [2.0, 0.0, -1.0]]]    # (2,2,2) 含零系数
    bshape = (2, 2)
    ref = ref_polyvalnd([bcast_vals(x, bshape), bcast_vals(y, bshape),
                         bcast_vals(z, bshape)], c, bshape)
    got = P.polyval3d(x, y, z, np.array(c))
    check("polyval3d broadcast+zeros", grids_match(got, ref, bshape))


def test_polyvalnd_direct():
    x = np.array([[1.0, -2.0], [0.5, 3.0], [2.0, 1.0], [-1.0, 0.25]])
    y = np.array([[0.5, 1.0], [-1.0, 2.0], [3.0, -0.5], [0.0, 1.5]])
    c = np.array([[1.0, 2.0, -1.0, 0.5, 0.0, 3.0],
                  [-2.0, 0.0, 1.5, -0.25, 4.0, 1.0],
                  [0.75, 3.0, -1.0, 2.0, -0.5, 0.0]])   # (3,3)
    got = P.polyvalnd((x, y), c)
    ref = ref_polyvalnd([flat(x.tolist()), flat(y.tolist())],
                        c.tolist(), (4, 2))
    check("polyvalnd direct", grids_match(got, ref, (4, 2)))


def test_polygrid2d():
    x1 = np.array([1.0, 2.0, -0.5, 3.0, 0.25])
    x2 = np.array([0.5, 1.0, 2.0, 3.0, -1.0, 0.75, 2.5])
    c = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]])  # (3,2)
    # res[i, j] = Σ_a,b c[a, b] x1[i]^a x2[j]^b
    c2 = c.tolist()
    ref = []
    for i in range(5):
        for j in range(7):
            v = 0.0
            for a in range(len(c2)):
                for b in range(len(c2[0])):
                    if c2[a][b] == 0:
                        continue
                    v += c2[a][b] * (x1.tolist()[i] ** a) \
                        * (x2.tolist()[j] ** b)
            ref.append(v)
    got = P.polygrid2d(x1, x2, c)
    check("polygrid2d", grids_match(got, ref, (5, 7)))


# ===================== 性能 =====================

def test_polyval_1d():
    def ref_h(cvals, xvals):
        out = []
        for v in xvals:
            y = 0.0
            for p in reversed(cvals):
                y = y * v + p
            out.append(y)
        return out
    cvals = [1.0, -2.0, 0.5, 3.0]
    c = np.array(cvals)
    xvals = [0.0, 1.0, -1.5, 2.0, 10.0]
    x = np.array(xvals)
    # 1-D float
    got = P.polyval(x, c)
    check("polyval 1d", grids_match(got, ref_h(cvals, xvals), (5,)))
    # 2-D x（形状保持，C 序展平对照）
    got = P.polyval(np.array([[1.0, 2.0], [3.0, -1.0]]), c)
    check("polyval 1d 2d-x", grids_match(
        got, ref_h(cvals, [1.0, 2.0, 3.0, -1.0]), (2, 2)))
    # 0-D x
    got = P.polyval(np.array(2.5), c)
    y0 = 0.0
    for p in reversed(cvals):
        y0 = y0 * 2.5 + p
    check("polyval 1d 0d-x", abs(float(got.tolist()) - y0) < 1e-12)
    # int 输入 → 原路径（Python int 精确算术）
    got = P.polyval(np.array([1, 2, 3, 10]), np.array([1, 2, 3]))
    refi = [1 + 2 * v + 3 * v * v for v in (1, 2, 3, 10)]
    check("polyval 1d int", grids_match(got, refi, (4,)))
    # deg 0 / deg 1
    got = P.polyval(x, np.array([7.0]))
    check("polyval 1d deg0", grids_match(got, [7.0] * 5, (5,)))
    got = P.polyval(x, np.array([3.0, 2.0]))
    check("polyval 1d deg1",
          grids_match(got, [3.0 + 2.0 * v for v in xvals], (5,)))


def bench_polyvalnd():
    n = 200
    x = np.random.rand(n, n)
    y = np.random.rand(n, 1)      # 广播 (n,n)
    c = np.random.rand(6, 5)
    t0 = time.perf_counter()
    got = P.polyval2d(x, y, c)
    t_new = time.perf_counter() - t0
    # 旧实现（逐系数整数组幂+乘+加，30 次数组运算）
    NX = np

    def old_polyvalnd(pts, c):
        shapes = [tuple(p.shape) for p in pts]
        m = builtins.max(builtins.len(s) for s in shapes)
        padded = [list(reversed(s)) + [1] * (m - builtins.len(s)) for s in shapes]
        bshape = [max(p[d] for p in padded) for d in range(m)]
        bshape = list(reversed(bshape))
        cvals = c.tolist()
        res = NX.zeros(bshape)
        for idx in itertools.product(*[range(d) for d in c.shape]):
            cv = cvals[idx[0]][idx[1]]
            if cv == 0:
                continue
            term = NX.array(cv)
            for a, p in enumerate(pts):
                if idx[a]:
                    term = term * (p ** idx[a])
            res = res + term
        return res

    t1 = time.perf_counter()
    old = old_polyvalnd((x, y), c)
    t_old = time.perf_counter() - t1
    ok = all(abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
             for a, b in zip(flat(got.tolist()), flat(old.tolist())))
    check("polyvalnd bench values", ok)
    print(f"BENCH polyval2d (200x200) x (200x1) c(6,5): "
          f"rust={t_new*1e3:.1f}ms old={t_old*1e3:.1f}ms "
          f"speedup={t_old/max(t_new,1e-9):.1f}x")


def bench_polyvander():
    n = 20000
    x = np.random.rand(n)
    t0 = time.perf_counter()
    got = P.polyvander(x, 30)
    t_new = time.perf_counter() - t0

    def old(x):
        xv = x.tolist()
        def rec(v):
            if isinstance(v, (list, tuple)):
                return [rec(u) for u in v]
            return [v ** i for i in range(31)]
        return np.array(rec(xv))

    t1 = time.perf_counter()
    old = old(x)
    t_old = time.perf_counter() - t1
    ok = grids_match(got, ref_polyvander(flat(x.tolist()), 30), (n, 31))
    check("vander bench values", ok)
    print(f"BENCH polyvander (20000, deg 30): "
          f"rust={t_new*1e3:.1f}ms old={t_old*1e3:.1f}ms "
          f"speedup={t_old/max(t_new,1e-9):.1f}x")


def bench_polyvander2d():
    n = 400
    x = np.random.rand(n, n)
    y = np.random.rand(n, n)
    t0 = time.perf_counter()
    got = P.polyvander2d(x, y, (5, 5))
    t_new = time.perf_counter() - t0
    xv, yv = flat(x.tolist()), flat(y.tolist())
    ref = ref_polyvander2d(xv, yv, 6, 6)
    check("vander2d bench values", grids_match(got, ref, (n, n, 36)))
    print(f"BENCH polyvander2d (400x400, deg (5,5)): "
          f"rust={t_new*1e3:.1f}ms ref={len(ref)} values")


def bench_polyval():
    import random
    n = 1_000_000
    xs = [random.random() for _ in range(n)]
    cs = [random.random() for _ in range(199)]
    x = np.array(xs)
    c = np.array(cs)
    t0 = time.perf_counter()
    got = P.polyval(x, c)
    t_new = time.perf_counter() - t0
    check("polyval bench shape", tuple(got.shape) == (n,))
    ok = True
    for k in (0, 12345, n // 2, n - 1):
        y = 0.0
        for p in reversed(cs):
            y = y * xs[k] + p
        if abs(float(got[k]) - y) > 1e-9 * max(1.0, abs(y)):
            ok = False
    check("polyval bench values", ok)
    print(f"BENCH polyval n={n} deg=198: rust={t_new*1e3:.1f}ms")


def main():
    test_polyvander()
    test_polyval_1d()
    test_polyvander2d()
    test_polyvander3d()
    test_polyval2d()
    test_polyval3d()
    test_polyvalnd_direct()
    test_polygrid2d()
    bench_polyval()
    bench_polyvalnd()
    bench_polyvander()
    bench_polyvander2d()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
