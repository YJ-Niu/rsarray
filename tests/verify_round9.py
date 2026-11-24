# -*- coding: utf-8 -*-
"""第九轮热点下放验证：unpackbits / tile / broadcast_to。

纯 Python 独立参考实现对照 + 性能基准。
非门禁脚本：PYTHONPATH=python .venv/bin/python tests/verify_round9.py
"""
import builtins
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


def close(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
            return False
    return True


# ===================== unpackbits =====================

def ref_unpack(val, order='big'):
    if order == 'big':
        return [(val >> (7 - j)) & 1 for j in range(8)]
    return [(val >> j) & 1 for j in range(8)]


def test_unpackbits_1d():
    vals = [0x00, 0x01, 0x7F, 0x80, 0xFF, 0x55, 0x3C, 0xA5]
    a = np.array(vals, dtype='uint8')
    ref_big = [b for v in vals for b in ref_unpack(v, 'big')]
    # big / little
    got = np.unpackbits(a)
    check("unpack 1d big", close(flat(got.tolist()), ref_big)
          and str(got.dtype) == 'uint8')
    got = np.unpackbits(a, bitorder='little')
    ref = [b for v in vals for b in ref_unpack(v, 'little')]
    check("unpack 1d little", close(flat(got.tolist()), ref))
    # count 正数（截断 / 零填充）
    got = np.unpackbits(a, count=20)
    check("unpack 1d count trunc",
          close(flat(got.tolist()), ref_big[:20]))
    got = np.unpackbits(a, count=80)
    check("unpack 1d count pad",
          close(flat(got.tolist()), ref_big + [0] * 16))
    # count 负数（收缩）
    got = np.unpackbits(a, count=-24)
    check("unpack 1d count neg", close(flat(got.tolist()), ref_big[:40]))
    # count 0 → 空
    got = np.unpackbits(a, count=0)
    check("unpack 1d count 0", tuple(got.shape) == (0,) and got.size == 0)
    # count 过负 → ValueError
    try:
        np.unpackbits(a, count=-100)
        check("unpack 1d count invalid", False)
    except ValueError:
        check("unpack 1d count invalid", True)
    # int64 输入（core 路径）
    ai = np.array(vals, dtype='int64')
    got = np.unpackbits(ai)
    check("unpack 1d int64", close(flat(got.tolist()), ref_big))
    # 2-d 输入 axis=None → 展平
    a2 = np.array([[0x01, 0x80], [0xFF, 0x00]], dtype='uint8')
    got = np.unpackbits(a2)
    ref = [b for v in [0x01, 0x80, 0xFF, 0x00] for b in ref_unpack(v, 'big')]
    check("unpack 2d flatten", close(flat(got.tolist()), ref))


def test_unpackbits_axis():
    a = np.array([[0x01, 0x80], [0xFF, 0x00], [0x55, 0xA5]], dtype='uint8')
    av = flat(a.tolist())
    # axis=0：位轴替换第 0 轴 → (24, 2)，got[bit, j] = a[bit//8, j] 的 bit 位
    got = np.unpackbits(a, axis=0)
    ref = []
    for bit in range(24):
        for j in range(2):
            ref.append(ref_unpack(int(av[(bit // 8) * 2 + j]), 'big')[bit % 8])
    check("unpack axis=0", tuple(got.shape) == (24, 2)
          and close(flat(got.tolist()), ref))
    # axis=1 → (3, 16)
    got = np.unpackbits(a, axis=1)
    ref = []
    for i in range(3):
        for j in range(2):
            ref.extend(ref_unpack(int(av[i * 2 + j]), 'big'))
    check("unpack axis=1", tuple(got.shape) == (3, 16)
          and close(flat(got.tolist()), ref))
    # 3-d axis=1 / axis=2
    a3 = np.array([[[0x01, 0x80, 0xFF], [0x00, 0x55, 0xA5]],
                   [[0x3C, 0x02, 0x40], [0x7F, 0x03, 0x10]]], dtype='uint8')
    av3 = flat(a3.tolist())
    got = np.unpackbits(a3, axis=1)
    # 位轴在中间：元素 (i0, bit, i2) = a3[i0, bit//8, i2] 的第 bit%8 位
    ref = []
    for i0 in range(2):
        for bit in range(16):
            for i2 in range(3):
                i1, bp = divmod(bit, 8)
                ref.append(ref_unpack(
                    int(av3[(i0 * 2 + i1) * 3 + i2]), 'big')[bp])
    check("unpack 3d axis=1", tuple(got.shape) == (2, 16, 3)
          and close(flat(got.tolist()), ref))
    got = np.unpackbits(a3, axis=2)
    ref = []
    for i0 in range(2):
        for i1 in range(2):
            for i2 in range(3):
                ref.extend(ref_unpack(int(av3[(i0 * 2 + i1) * 3 + i2]), 'big'))
    check("unpack 3d axis=2", tuple(got.shape) == (2, 2, 24)
          and close(flat(got.tolist()), ref))
    # count + axis（截断 / 零填充）：count 作用于整行展开位（2 字节 = 16 位）
    got = np.unpackbits(a, axis=1, count=10)
    ref = []
    for i in range(3):
        bits_row = []
        for j in range(2):
            bits_row.extend(ref_unpack(int(av[i * 2 + j]), 'big'))
        ref.extend(bits_row[:10])
    check("unpack axis+count trunc", tuple(got.shape) == (3, 10)
          and close(flat(got.tolist()), ref))
    got = np.unpackbits(a, axis=1, count=12)
    ref = []
    for i in range(3):
        bits_row = []
        for j in range(2):
            bits_row.extend(ref_unpack(int(av[i * 2 + j]), 'big'))
        ref.extend((bits_row + [0] * 8)[:12])
    check("unpack axis+count pad", tuple(got.shape) == (3, 12)
          and close(flat(got.tolist()), ref))
    # 负 axis
    got = np.unpackbits(a, axis=-1)
    check("unpack axis neg", tuple(got.shape) == (3, 16))


# ===================== tile =====================

def test_tile():
    a = np.array([1.0, 2.0, 3.0])
    # reps int / list / r=1 / r=0
    got = np.tile(a, 3)
    check("tile 1d int", close(flat(got.tolist()), [1, 2, 3] * 3))
    got = np.tile(a, [2, 2])
    check("tile 1d list-2", tuple(got.shape) == (2, 6)
          and close(flat(got.tolist()), [1, 2, 3] * 4))
    got = np.tile(a, 1)
    check("tile r1", close(flat(got.tolist()), [1, 2, 3]))
    got = np.tile(a, 0)
    check("tile r0", tuple(got.shape) == (0,) and got.size == 0)
    # 2-d
    m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    got = np.tile(m, (2, 3))
    # reps (2,3)：每行横向重复 3 次、整块纵向重复 2 次
    ref = flat([row for _ in range(2)
                for row in [[v for _ in range(3) for v in r] for r in m.tolist()]])
    check("tile 2d", tuple(got.shape) == (6, 6)
          and close(flat(got.tolist()), ref))
    # reps 短于 ndim（前置补 1 → 沿末轴重复）
    got = np.tile(m, 2)
    ref = flat([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    check("tile 2d reps-short", tuple(got.shape) == (3, 4)
          and close(flat(got.tolist()), ref))
    # reps 长于 ndim（前导补新轴）：m→(1,3,2)，[2,1,2] → (2,3,4)，
    # 元素 (i0,i1,i2) = m[i1, i2%2]，每块为行内平铺（非 flat 整块重复）
    got = np.tile(m, [2, 1, 2])
    ref = []
    for _ in range(2):
        for r in m.tolist():
            ref.extend([v for _ in range(2) for v in r])
    check("tile 2d reps-long", tuple(got.shape) == (2, 3, 4)
          and close(flat(got.tolist()), ref))
    # int 输入 → 原路径（保 int64 dtype）
    mi = np.array([[1, 2], [3, 4]])
    got = np.tile(mi, 2)
    check("tile int dtype", str(got.dtype) == 'int64'
          and close(flat(got.tolist()), [1, 2, 1, 2, 3, 4, 3, 4]))
    # 空输入
    e = np.array([])
    got = np.tile(e, 3)
    check("tile empty", tuple(got.shape) == (0,))


# ===================== broadcast_to =====================

def test_broadcast_to():
    a = np.array([1.0, 2.0, 3.0])  # (3,)
    got = np.broadcast_to(a, (2, 3))
    check("bcast real", tuple(got.shape) == (2, 3)
          and close(flat(got.tolist()), [1, 2, 3, 1, 2, 3]))
    got = np.broadcast_to(np.array(5.0), ())
    check("bcast 0d", abs(float(flat(got.tolist())[0]) - 5.0) < 1e-12)
    m = np.array([[10.0], [20.0]])
    got = np.broadcast_to(m, (2, 4))
    check("bcast 2d", tuple(got.shape) == (2, 4)
          and close(flat(got.tolist()), [10, 10, 10, 10, 20, 20, 20, 20]))
    # 不兼容 → 报错
    try:
        np.broadcast_to(a, (2, 2))
        check("bcast incompatible", False)
    except Exception:
        check("bcast incompatible", True)
    # 复数：实/虚双通道同步广播
    c = np.array([1.0 + 2.0j, 3.0 - 4.0j])  # (2,)
    got = np.broadcast_to(c, (3, 2))
    vals = flat(got.tolist())
    ok = (tuple(got.shape) == (3, 2)
          and abs(complex(vals[0]) - (1 + 2j)) < 1e-12
          and abs(complex(vals[1]) - (3 - 4j)) < 1e-12
          and abs(complex(vals[2]) - (1 + 2j)) < 1e-12
          and abs(complex(vals[5]) - (3 - 4j)) < 1e-12)
    check("bcast complex", ok)
    c1 = np.array([2.0 + 1.0j])
    got = np.broadcast_to(c1, (4, 3))
    vals = flat(got.tolist())
    check("bcast complex (1,)→(4,3)",
          tuple(got.shape) == (4, 3) and all(
              abs(complex(v) - (2 + 1j)) < 1e-12 for v in vals))
    # 复数不兼容 → numpy 广播语义报错（非旧版 % 回绕）
    try:
        np.broadcast_to(c, (3, 3))
        check("bcast complex incompatible", False)
    except Exception:
        check("bcast complex incompatible", True)


# ===================== 性能 =====================

def bench_unpackbits():
    import random
    random.seed(9)
    n = 1_000_000
    vals = [random.randrange(256) for _ in range(n)]
    a = np.array(vals, dtype='uint8')
    t0 = time.perf_counter()
    got = np.unpackbits(a)
    t_new = time.perf_counter() - t0
    check("unpack bench shape", tuple(got.shape) == (8 * n,))
    # 抽点对照参考
    ok = all(int(flat(got.tolist())[k]) == ref_unpack(vals[k // 8], 'big')[k % 8]
             for k in (0, 1, 7, 8, 12345, 8 * n - 1))
    check("unpack bench values", ok)
    print(f"BENCH unpackbits n={n}: rust={t_new*1e3:.1f}ms")


def bench_tile():
    m = np.random.rand(200, 300)
    t0 = time.perf_counter()
    got = np.tile(m, (3, 2))
    t_new = time.perf_counter() - t0
    check("tile bench shape", tuple(got.shape) == (600, 600))
    ok = all(abs(float(got[i, j]) - float(m[i % 200, j % 300])) < 1e-12
             for i in (0, 1, 199, 200, 599) for j in (0, 299, 300, 599))
    check("tile bench values", ok)
    print(f"BENCH tile (200x300, reps 3x2): concatenate链={t_new*1e3:.1f}ms")


def bench_broadcast():
    c = np.array([1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j])
    t0 = time.perf_counter()
    got = np.broadcast_to(c, (100000, 3))
    t_new = time.perf_counter() - t0
    refs = [1 + 2j, 2 + 3j, 3 + 4j]
    ok = all(abs(complex(got[r, k]) - refs[k]) < 1e-12
             for r in (0, 1, 50000, 99999) for k in range(3))
    check("bcast bench complex", ok)
    print(f"BENCH broadcast_to complex (3,)→(100000,3): "
          f"rust={t_new*1e3:.1f}ms")


def main():
    test_unpackbits_1d()
    test_unpackbits_axis()
    test_tile()
    test_broadcast_to()
    bench_unpackbits()
    bench_tile()
    bench_broadcast()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
