# -*- coding: utf-8 -*-
"""第五轮热点下放验证：intersect1d / kron。

纯 Python 独立参考实现对照（不依赖被测代码）+ 性能基准。
非门禁脚本：PYTHONPATH=python .venv/bin/python tests/verify_setops_kron.py
"""
import builtins
import itertools
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


# ===================== 独立参考实现 =====================

def ref_intersect1d(x, y):
    """旧纯 Python 语义参考（O(n²) 去重 + NaN 视为相等 + 升序）。"""
    fa = flat(x.tolist()) if hasattr(x, 'tolist') else list(x)
    fb = flat(y.tolist()) if hasattr(y, 'tolist') else list(y)

    def _eq(v, w):
        if v is w:
            return True
        try:
            if v == w:
                return True
        except Exception:
            pass
        return (isinstance(v, float) and isinstance(w, float)
                and v != v and w != w)

    def _dedup_first(vals):
        seen, out_v = [], []
        for v in vals:
            if any(_eq(v, w) for w in seen):
                continue
            seen.append(v)
            out_v.append(v)
        return out_v

    va, vb = _dedup_first(fa), _dedup_first(fb)
    common = [v for v in va if any(_eq(v, w) for w in vb)]
    try:
        common.sort()
    except TypeError:
        pass
    return common


def ref_kron(a, b):
    """按定义独立实现：C 序下 out[c] = a[ca] * b[cb]。"""
    ash = list(a.shape)
    bsh = list(b.shape)
    n2 = max(len(ash), len(bsh))
    ash = [1] * (n2 - len(ash)) + ash
    bsh = [1] * (n2 - len(bsh)) + bsh
    out_shape = [x * y for x, y in zip(ash, bsh)]
    af = flat(a.tolist())
    bf = flat(b.tolist())
    sa = [1] * n2
    sb = [1] * n2
    for i in range(n2 - 2, -1, -1):
        sa[i] = sa[i + 1] * ash[i + 1]
        sb[i] = sb[i + 1] * bsh[i + 1]
    out = []
    for coord in itertools.product(*[range(s) for s in out_shape]):
        ia = sum((c // bd) * sd for c, sd, bd in zip(coord, sa, bsh))
        ib = sum((c % bd) * sd for c, sd, bd in zip(coord, sb, bsh))
        out.append(af[ia] * bf[ib])
    return out, tuple(out_shape)


def grids_match(got, ref_vals, ref_shape, tol=1e-9):
    if tuple(got.shape) != ref_shape:
        return False
    gf = flat(got.tolist())
    if len(gf) != len(ref_vals):
        return False
    for x, y in zip(gf, ref_vals):
        if abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
            return False
    return True


def lists_match(a, b, tol=1e-12):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, float) or isinstance(y, float):
            # NaN 对 NaN 视为相等
            if x != x and y != y:
                continue
            if abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
                return False
        elif x != y:
            return False
    return True


# ===================== intersect1d =====================

def test_intersect1d():
    a1 = np.array([1.0, 3.0, 5.0, 7.0, 3.0])
    b1 = np.array([2.0, 3.0, 5.0, 9.0])
    check("intersect basic", lists_match(
        flat(np.intersect1d(a1, b1).tolist()), ref_intersect1d(a1, b1)))

    # 空输入
    e = np.array([])
    check("intersect empty", flat(np.intersect1d(e, b1).tolist()) == [])

    # 无交集
    check("intersect disjoint", flat(
        np.intersect1d(np.array([1.0, 2.0]), np.array([3.0])).tolist()) == [])

    # -0.0 / 0.0 归一（交集只有 0）
    got = flat(np.intersect1d(np.array([-0.0, 1.0]), np.array([0.0, 2.0])).tolist())
    check("intersect -0.0", lists_match(got, [0.0]))

    # NaN 视为相等（交集只有 NaN）
    nan = float('nan')
    got = flat(np.intersect1d(np.array([nan, 1.0]), np.array([nan, 2.0])).tolist())
    check("intersect nan", lists_match(got, [nan]))

    # 整型（dtype 保留 int64）
    ai = np.array([1, 2, 3, 4])
    bi = np.array([4, 3, 2, 9])
    got = np.intersect1d(ai, bi)
    check("intersect int values", lists_match(flat(got.tolist()), ref_intersect1d(ai, bi)))
    check("intersect int dtype", str(getattr(got, '_dtype', '')) == 'int64')

    # 2D 输入（ravel 语义）
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b2 = np.array([4.0, 3.0, 1.0])
    check("intersect 2d", lists_match(
        flat(np.intersect1d(a2, b2).tolist()), ref_intersect1d(a2, b2)))

    # return_indices=True 走纯 Python 路径（值仍正确）
    r = np.intersect1d(a1, b1, return_indices=True)
    res, i1, i2 = r
    check("intersect return_indices values", lists_match(
        flat(res.tolist()), ref_intersect1d(a1, b1)))
    fa = flat(a1.tolist())
    fb = flat(b1.tolist())
    check("intersect return_indices valid",
          builtins.all(fa[x] == y or (fa[x] != fa[x] and y != y)
                       for x, y in zip(flat(i1.tolist()), flat(res.tolist())))
          and builtins.all(fb[x] == y or (fb[x] != fb[x] and y != y)
                            for x, y in zip(flat(i2.tolist()), flat(res.tolist()))))

    # 字符串输入走纯 Python 路径
    sa = np.array(['a', 'b', 'c'])
    sb = np.array(['b', 'c', 'd'])
    got = np.intersect1d(sa, sb)
    check("intersect strings", flat(got.tolist()) in (['b', 'c'], [b'b', b'c']))


# ===================== kron =====================

def test_kron():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([0.5, -1.0])
    got = np.kron(v1, v2)
    rv, rs = ref_kron(v1, v2)
    check("kron 1d x 1d", grids_match(got, rv, rs))

    m2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    got = np.kron(v1, m2)
    rv, rs = ref_kron(v1, m2)
    check("kron 1d x 2d", grids_match(got, rv, rs))

    got = np.kron(m2, m2)
    rv, rs = ref_kron(m2, m2)
    check("kron 2d x 2d", grids_match(got, rv, rs))

    m3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    got = np.kron(m3, v2)
    rv, rs = ref_kron(m3, v2)
    check("kron 3d x 1d", grids_match(got, rv, rs))

    # 标量 (0-d)
    s = np.array(2.0)
    got = np.kron(s, m2)
    rv, rs = ref_kron(s, m2)
    check("kron 0d x 2d", grids_match(got, rv, rs))
    got = np.kron(s, s)
    rv, rs = ref_kron(s, s)
    check("kron 0d x 0d", grids_match(got, rv, rs))

    # 零尺寸：(1,0) x (2,2) → (2,0)
    e = np.array([[]])
    got = np.kron(e, m2)
    check("kron zero-size", got.size == 0 and tuple(got.shape) == (2, 0))

    # 整型
    gi = np.array([[1, 2], [3, 4]])
    got = np.kron(gi, gi)
    rv, rs = ref_kron(gi, gi)
    check("kron int", grids_match(got, rv, rs) and
          str(getattr(got, '_dtype', '')) == 'int64')


# ===================== 性能 =====================

def bench_intersect1d():
    n = 4000
    a = np.array([float((i * 7919) % 9000) for i in range(n)])
    b = np.array([float((i * 104729) % 9000) for i in range(n)])
    t0 = time.perf_counter()
    got = np.intersect1d(a, b)
    t_rust = time.perf_counter() - t0
    # 旧实现参考（O(n²)）：仅在小规模上做全量对照已在正确性部分完成，
    # 此处用 800 规模测旧实现时间并线性外推仅作量级参照
    m = 800
    a_s = np.array(flat(a.tolist())[:m])
    b_s = np.array(flat(b.tolist())[:m])
    t1 = time.perf_counter()
    ref_intersect1d(a_s, b_s)
    t_py_small = time.perf_counter() - t1
    t_py_est = t_py_small * (n / m) ** 2
    ref = ref_intersect1d(a, b)
    check("intersect bench values", lists_match(flat(got.tolist()), ref))
    print(f"BENCH intersect1d n={n}: rust={t_rust*1e3:.1f}ms "
          f"python~{t_py_est:.2f}s(外推) speedup~{t_py_est/max(t_rust,1e-9):.0f}x")


def bench_kron():
    A = np.random.rand(60, 60)
    B = np.random.rand(50, 50)
    t0 = time.perf_counter()
    got = np.kron(A, B)
    t_rust = time.perf_counter() - t0
    rv, rs = ref_kron(A, B)
    check("kron bench values", grids_match(got, rv, rs))
    # 参考实现计时（1/4 规模，内容任意——值已在全规模验证）
    A_s = np.random.rand(30, 30)
    B_s = np.random.rand(25, 25)
    t1 = time.perf_counter()
    ref_kron(A_s, B_s)
    t1 = time.perf_counter() - t1
    print(f"BENCH kron (60x60) x (50x50): rust={t_rust*1e3:.1f}ms "
          f"ref(python定义)1/4规模={t1*1e3:.1f}ms")


def main():
    test_intersect1d()
    test_kron()
    bench_intersect1d()
    bench_kron()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
