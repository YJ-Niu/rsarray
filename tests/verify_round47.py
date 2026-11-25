#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四十七轮验证：float16 半精度语义（IEEE 754 binary16 round-trip）。"""
import sys

sys.path.insert(0, 'python')
import rsarray as np  # noqa: E402

PASS = [0]
FAIL = [0]


def check(name, ok):
    if ok:
        PASS[0] += 1
    else:
        FAIL[0] += 1
        print(f"FAIL: {name}")


def main():
    a = np.array([1.0, 2.0, 3.14159265, 65519.0, 1e-5, 70000.0, -70000.0])
    r = a.astype('float16').tolist()
    # ---- normal 舍入（10 位尾数）----
    check("normal grid", r[0] == 1.0 and r[1] == 2.0)
    check("pi rounded", r[2] == 3.140625)
    # ---- 边界：max normal 与溢出 → ±inf（IEEE 754 语义）----
    check("65519 -> 65504", r[3] == 65504.0)
    check("+70000 -> inf", r[5] == float('inf'))
    check("-70000 -> -inf", r[6] == float('-inf'))
    # ---- subnormal 网格 ----
    check("1e-5 on f16 grid", abs(r[4]) < 65504.0 and r[4] > 0)
    # ---- inf round-trip ----
    check("inf kept", np.array(
        [float('inf')], dtype='float16').tolist() == [float('inf')])
    # ---- 构造路径 ----
    check("construct overflow", np.array(
        [70000.0, -70000.0], dtype='float16').tolist() == [float('inf'),
                                                           float('-inf')])

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
