#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十四轮验证：ma 深层修复（MaskedArray 方法 + ma.clip + anom）。

1. MaskedArray 方法：sum/mean/prod/count/argmax
2. ma.clip（掩码保持）、anom 掩码感知、masked_all 构造器
3. 回归：第 33 轮 ma 聚合 + 切片/索引
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
    ma = np.ma
    m1 = np.ma.MaskedArray([1.0, 2.0, 3.0, 4.0],
                           mask=[False, True, False, False])
    # ---- MaskedArray 方法 ----
    check("m1.sum()", float(m1.sum()) == 8.0)
    check("m1.mean()", abs(float(m1.mean()) - 8.0 / 3.0) < 1e-9)
    check("m1.count()", int(m1.count()) == 3)
    check("m1.argmax()", int(m1.argmax()) == 3)
    # ---- ma.clip（掩码保持）----
    rc = ma.clip(m1, 2.0, 4.0)
    check("ma.clip values", rc.tolist()[0] == 2.0
          and rc.tolist()[2] == 3.0 and rc.tolist()[3] == 4.0)
    check("ma.clip mask kept", bool(rc.mask[1]))
    # ---- anom 掩码感知（x - mean，均值忽略 masked）----
    ra = ma.anom(m1)
    check("anom masked-aware",
          abs(float(ra.tolist()[0]) - (1.0 - 8.0 / 3.0)) < 1e-9)
    # ---- masked_all 构造器语义 ----
    mall = ma.masked_all((2, 2))
    check("masked_all shape", mall.shape == (2, 2))
    check("masked_all count", int(mall.count()) == 0)
    # ---- 回归：第 33 轮模块级聚合 ----
    check("ma.sum partial (r33)",
          float(np.ma.sum(np.ma.MaskedArray(
              [1.0, 2.0, 8.0], mask=[False, True, False]))) == 9.0)
    check("ma.mean all-masked (r33) = masked",
          bool(getattr(np.ma.mean(np.ma.MaskedArray(
              [1.0, 2.0], mask=[True, True])), 'mask', False) is not False))
    # ---- 回归：切片/索引/where 标量（第 32 轮）----
    a = np.arange(12).reshape(3, 4)
    check("slice (r33)", a[1:, ::2].tolist() == [[4, 6], [8, 10]])
    r = np.where(a.reshape(-1) > 1, a.reshape(-1), 0.0)
    check("where scalar y (r32)", r.tolist()[:3] == [0.0, 0.0, 2.0])

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
