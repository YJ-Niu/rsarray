# -*- coding: utf-8 -*-
"""第四轮热点下放验证：einsum / ravel_multi_index / meshgrid。

纯 Python 独立参考实现对照（不依赖被测代码）+ 性能基准。
非门禁脚本：PYTHONPATH=python .venv/bin/python tests/verify_einsum_rmi_meshgrid.py
"""
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


def close(a, b, tol=1e-10):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def _nested_get(l, idx):
    v = l
    for i in idx:
        v = v[i]
    return v


# ===================== 独立参考实现 =====================

def ref_einsum(subscripts, operands):
    sub = subscripts.replace(' ', '')
    if '->' in sub:
        ins, out = sub.split('->')
    else:
        ins = sub
        from collections import Counter
        cnt = Counter(ins.replace(',', ''))
        out = ''.join(sorted(l for l, c in cnt.items() if c == 1))
    terms = ins.split(',')
    dims = {}
    for term, op in zip(terms, operands):
        for k, lab in enumerate(term):
            dims[lab] = op.shape[k]
    labels = list(dims)
    lists = [op.tolist() for op in operands]
    acc = {}
    for combo in itertools.product(*[range(dims[l]) for l in labels]):
        p = 1.0
        for term, l in zip(terms, lists):
            p *= _nested_get(l, tuple(combo[labels.index(x)] for x in term))
        key = tuple(combo[labels.index(x)] for x in out)
        acc[key] = acc.get(key, 0.0) + p
    if not out:
        return acc.get((), 0.0)
    shape = tuple(dims[x] for x in out)

    def build(pre, d):
        if d == len(out):
            return acc.get(pre, 0.0)
        return [build(pre + (i,), d + 1) for i in range(shape[d])]

    return build((), 0)


def ref_rmi(multi_index, dims, mode='raise', order='C'):
    dims = list(dims)
    ndim = len(dims)
    modes = [mode] * ndim if isinstance(mode, str) else list(mode)
    idxs = [np._extra._flat(np.asarray(m)) for m in multi_index]
    n_ref = len(idxs[0])
    idxs = [x * n_ref if len(x) == 1 else x for x in idxs]
    strides = []
    if order == 'C':
        acc = 1
        for i in range(ndim - 1, -1, -1):
            strides.insert(0, acc)
            acc *= dims[i]
    else:
        acc = 1
        for i in range(ndim):
            strides.append(acc)
            acc *= dims[i]
    n = len(idxs[0])
    res = []
    for i in range(n):
        v = 0
        for d in range(ndim):
            x = int(idxs[d][i])
            dd = dims[d]
            if modes[d] == 'wrap':
                x = x % dd
            elif modes[d] == 'clip':
                x = max(0, min(x, dd - 1))
            else:
                if x < 0 or x >= dd:
                    raise ValueError('oob')
            v += x * strides[d]
        res.append(v)
    return res


def ref_meshgrid(xi, sparse=False, indexing='xy'):
    n = len(xi)
    shapes = [len(a) for a in xi]
    if indexing == 'xy' and n >= 2:
        perm = [1, 0] + list(range(2, n))
    else:
        perm = list(range(n))
    out_shape = tuple(shapes[k] for k in perm)
    res = []
    for i in range(n):
        pos = perm.index(i)
        if sparse:
            shp = [1] * n
            shp[pos] = shapes[i]
            res.append(([float(v) for v in xi[i]], tuple(shp)))
        else:
            vals = [xi[i][combo[pos]]
                    for combo in itertools.product(*[range(s) for s in out_shape])]
            res.append(([float(v) for v in vals], out_shape))
    return res


def grids_equal(got, ref):
    if len(got) != len(ref):
        return False
    for g, (rvals, rshape) in zip(got, ref):
        if tuple(g.shape) != rshape:
            return False
        gf = np._extra._flat(g)
        if len(gf) != len(rvals):
            return False
        for a, b in zip(gf, rvals):
            if abs(float(a) - float(b)) > 1e-12:
                return False
    return True


# ===================== einsum =====================

def test_einsum():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    T1 = np.random.rand(2, 3, 4)
    T2 = np.random.rand(3, 4, 5)
    bA = np.random.rand(3, 4, 5)
    bB = np.random.rand(3, 5, 6)

    cases = [
        ('ij,jk->ik', (A, B)),
        ('i,i->', (v1, v2)),
        ('ii->i', (M,)),
        ('ii->', (M,)),
        ('i,j->ij', (v1, v2)),
        ('bij,bjk->bik', (bA, bB)),
        ('abc,abc->a', (T1, T1)),
        ('ij->ji', (A,)),
        ('ij->', (A,)),
        ('ij->i', (A,)),
        ('abc,bcd->ad', (T1, T2)),
        ('ij,jk', (A, B)),          # 无箭头 → 默认 ik
        ('ij,ij->i', (A, A)),       # 逐元素乘积按行求和
        ('ik,ik->', (A, A)),        # 标量输出（2 维操作数）
    ]
    for sub, ops in cases:
        ref = ref_einsum(sub, ops)
        got = np.einsum(sub, *ops)
        got_v = got if isinstance(got, float) else got.tolist()
        check(f"einsum {sub!r}", got_v == ref or
              (isinstance(ref, float) and close(got_v, ref)) or
              _deep_close(got_v, ref))

    # 整型输入（f64 存储）
    Ai = np.array([[1, 2], [3, 4]])
    Bi = np.array([[5, 6], [7, 8]])
    check("einsum int dtype", _deep_close(
        np.einsum('ij,jk->ik', Ai, Bi).tolist(),
        ref_einsum('ij,jk->ik', (Ai, Bi))))

    # complex 输入（回退纯 Python 路径）
    Ac = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    Bc = np.array([[1 + 1j, 0], [0, 2 + 2j]])
    gotc = np.einsum('ij,jk->ik', Ac, Bc).tolist()
    refc = ref_einsum('ij,jk->ik', (Ac, Bc))
    ok = True
    for g, r in zip(_flat_ref(gotc), _flat_ref(refc)):
        if abs(complex(g) - complex(r)) > 1e-10:
            ok = False
    check("einsum complex fallback", ok)

    # label 冲突 → ValueError
    try:
        np.einsum('i,i->', np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        check("einsum label conflict raises", False)
    except ValueError:
        check("einsum label conflict raises", True)


def _deep_close(a, b, tol=1e-10):
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False
        return all(_deep_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return close(a, b, tol)
    return a == b


def _flat_ref(v):
    out = []
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(reversed(x))
        else:
            out.append(x)
    return out


# ===================== ravel_multi_index =====================

def test_rmi():
    dims = (4, 6)
    i0 = [0, 1, 3]
    i1 = [2, 5, 0]
    for order in ('C', 'F'):
        got = np.ravel_multi_index((i0, i1), dims, order=order)
        ref = ref_rmi((i0, i1), dims, order=order)
        check(f"rmi list {order}", list(np._extra._flat(got)) == ref)
    # 数组输入（含广播：(3,) 与 (1,)）
    a0 = np.array([0, 1, 2])
    a1 = np.array([3])
    got = np.ravel_multi_index((a0, a1), (4, 6))
    ref = ref_rmi((a0, a1), (4, 6))
    check("rmi array broadcast", list(np._extra._flat(got)) == ref)
    # 标量 → int
    check("rmi scalar C", np.ravel_multi_index((2, 3), (4, 6)) == 2 * 6 + 3)
    check("rmi scalar F", np.ravel_multi_index((2, 3), (4, 6), order='F') == 2 + 3 * 4)
    # clip（ref_rmi 返回 list，标量结果取 [0]）
    ref = ref_rmi((-1, 99), (4, 6), mode='clip')
    check("rmi clip scalar", np.ravel_multi_index((-1, 99), (4, 6), mode='clip') == ref[0])
    # wrap（含负索引）
    got = np.ravel_multi_index((-1, -1), (4, 6), mode='wrap')
    check("rmi wrap scalar", got == ref_rmi((-1, -1), (4, 6), mode='wrap')[0])
    # 逐维 mode 元组
    got = np.ravel_multi_index((-1, 99), (4, 6), mode=('wrap', 'clip'))
    ref = ref_rmi((-1, 99), (4, 6), mode=('wrap', 'clip'))
    check("rmi per-dim mode", got == ref[0])
    # raise 越界
    try:
        np.ravel_multi_index((5, 0), (4, 6))
        check("rmi raise oob", False)
    except ValueError:
        check("rmi raise oob", True)
    # 大数组
    n = 200000
    big0 = np.array([i % 5 for i in range(n)])
    big1 = np.array([(i * 7) % 11 for i in range(n)])
    t0 = time.perf_counter()
    got = np.ravel_multi_index((big0, big1), (5, 11))
    t_rust = time.perf_counter() - t0
    check("rmi big", list(np._extra._flat(got))[:100] == ref_rmi((big0, big1), (5, 11))[:100])
    t1 = time.perf_counter()
    ref_rmi((np._extra._flat(big0), np._extra._flat(big1)), (5, 11))
    t_py = time.perf_counter() - t1
    print(f"BENCH ravel_multi_index n={n}: rust={t_rust*1e3:.3f}ms python={t_py*1e3:.3f}ms speedup={t_py/max(t_rust,1e-9):.1f}x")


# ===================== meshgrid =====================

def test_meshgrid():
    xs = [np.array([0.0, 1.0, 2.5]), np.array([10.0, 20.0])]
    for indexing in ('xy', 'ij'):
        got = np.meshgrid(*xs, indexing=indexing)
        ref = ref_meshgrid([np._extra._flat(a) for a in xs], indexing=indexing)
        check(f"meshgrid 2D {indexing}", grids_equal(got, ref))
        got_s = np.meshgrid(*xs, sparse=True, indexing=indexing)
        ref_s = ref_meshgrid([np._extra._flat(a) for a in xs], sparse=True, indexing=indexing)
        check(f"meshgrid 2D sparse {indexing}", grids_equal(got_s, ref_s))
    # 3 维 / 4 维
    xs3 = [np.array([0.0, 1.0]), np.array([2.0, 3.0, 4.0]), np.array([5.0])]
    for indexing in ('xy', 'ij'):
        got = np.meshgrid(*xs3, indexing=indexing)
        ref = ref_meshgrid([np._extra._flat(a) for a in xs3], indexing=indexing)
        check(f"meshgrid 3D {indexing}", grids_equal(got, ref))
    xs4 = [np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([4.0]), np.array([5.0, 6.0])]
    got = np.meshgrid(*xs4, indexing='ij')
    ref = ref_meshgrid([np._extra._flat(a) for a in xs4], indexing='ij')
    check("meshgrid 4D ij", grids_equal(got, ref))
    got = np.meshgrid(*xs4, indexing='xy', sparse=True)
    ref = ref_meshgrid([np._extra._flat(a) for a in xs4], indexing='xy', sparse=True)
    check("meshgrid 4D xy sparse", grids_equal(got, ref))
    # 单输入（旧实现会 IndexError）
    got = np.meshgrid(np.array([1.0, 2.0, 3.0]))
    check("meshgrid 1D", tuple(got[0].shape) == (3,) and
          np._extra._flat(got[0]) == [1.0, 2.0, 3.0])
    # 浮点值保留（旧实现 int() 截断 2.5→2）
    got = np.meshgrid(np.array([0.5, 1.5]), np.array([10.0]))
    flat = np._extra._flat(got[0])
    check("meshgrid float preserved", abs(flat[0] - 0.5) < 1e-12)
    # 性能
    xa = np.array([float(i) for i in range(400)])
    xb = np.array([float(i) for i in range(300)])
    t0 = time.perf_counter()
    np.meshgrid(xa, xb)
    t_rust = time.perf_counter() - t0
    t1 = time.perf_counter()
    _py_meshgrid(xa.tolist(), xb.tolist())
    t_py = time.perf_counter() - t1
    print(f"BENCH meshgrid 400x300: rust={t_rust*1e3:.3f}ms python={t_py*1e3:.3f}ms speedup={t_py/max(t_rust,1e-9):.1f}x")


def _py_meshgrid(a, b):
    """旧实现风格的嵌套列表构造（性能对照用）。"""
    sa, sb = len(a), len(b)
    g0 = [[a[k] for k in range(sa)] for _ in range(sb)]
    g1 = [[b[j] for _ in range(sa)] for j in range(sb)]
    return g0, g1


# ===================== einsum 性能 =====================

def bench_einsum():
    n = 120
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    t0 = time.perf_counter()
    got = np.einsum('ij,jk->ik', A, B)
    t_rust = time.perf_counter() - t0
    ops = (A, B)
    t1 = time.perf_counter()
    ref_einsum('ij,jk->ik', ops)
    t_py = time.perf_counter() - t1
    check("einsum bench correctness", _deep_close(
        got.tolist(), ref_einsum('ij,jk->ik', ops)))
    print(f"BENCH einsum ij,jk->ik n={n}: rust={t_rust*1e3:.3f}ms python={t_py*1e3:.3f}ms speedup={t_py/max(t_rust,1e-9):.1f}x")


def main():
    test_einsum()
    test_rmi()
    test_meshgrid()
    bench_einsum()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
