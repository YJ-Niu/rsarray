#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十三轮验证：ma 掩码感知聚合（sum/mean/prod）。

1. 部分掩码：masked 元素不计入
2. 全掩码：sum→0.0、prod→1.0、mean→masked
3. 2-D axis 归约、扫描回归（切片/索引/dtype 边界）
"""
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
    # ---- 部分掩码：masked 不计入 ----
    check("sum partial",
          float(np.ma.sum(np.ma.MaskedArray(
              [1.0, 2.0, 8.0], mask=[False, True, False]))) == 9.0)
    check("mean partial",
          float(np.ma.mean(np.ma.MaskedArray(
              [1.0, 2.0, 8.0], mask=[False, True, False]))) == 3.0)
    # ---- 全掩码：sum/prod/mean ----
    check("sum all-masked = 0",
          float(np.ma.sum(np.ma.MaskedArray(
              [1.0, 2.0], mask=[True, True]))) == 0.0)
    check("prod all-masked = 1",
          float(np.ma.prod(np.ma.MaskedArray(
              [3.0, 4.0], mask=[True, True]))) == 1.0)
    m = np.ma.mean(np.ma.MaskedArray([1.0, 2.0], mask=[True, True]))
    check("mean all-masked = masked",
          bool(getattr(m, 'mask', False) is not False
               and np.ma.MaskedArray is not None)
          and float(np.ma.getdata(m)) == 0.0)
    # ---- 2-D axis 归约 ----
    s2 = np.ma.sum(np.ma.MaskedArray(
        [[1.0, 2.0], [4.0, 8.0]], mask=[[True, False], [False, True]]), axis=1)
    check("2-D axis1", s2.tolist() == [2.0, 4.0])
    # ---- 兼容性扫描（第 33 轮）回归：切片/索引/dtype/广播 ----
    a = np.arange(12).reshape(3, 4)
    check("slice a[1:,::2]", a[1:, ::2].tolist() == [[4, 6], [8, 10]])
    check("bool mask idx", a[a > 5].tolist() == [6, 7, 8, 9, 10, 11])
    check("int64 overflow wrap",
          np.array([2**63]).astype('int64').tolist() == [-9223372036854775808])
    check("broadcast (3,1)*(1,4)",
          (np.ones((3, 1)) * np.ones((1, 4))).shape == (3, 4))
    # ---- 第 32 轮 where 标量回归 ----
    r = np.where(x_gt1(np.arange(4.0)), np.arange(4.0), 0.0)
    check("where scalar y (r32)", r.tolist() == [0.0, 0.0, 2.0, 3.0])

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


def x_gt1(a):
    return a > 1


if __name__ == '__main__':
    main()
