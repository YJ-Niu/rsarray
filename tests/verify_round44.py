#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四十四轮验证：ma.sort/cumsum/is_masked + char.strlen + 字符串加法。

1. ma.sort 掩码感知（masked 排最后、掩码同步重排）
2. MaskedArray.cumsum / ma.is_masked
3. char.strlen 别名、字符串数组 + 拼接
4. 回归：第 33/37 轮 ma 聚合/where 标量
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
    # ---- ma.sort 掩码感知（masked 排最后，掩码同步重排）----
    s = np.ma.sort(m1)
    check("ma.sort values", s.tolist()[0] == 1.0 and s.tolist()[-1] == 2.0)
    check("ma.sort mask kept", bool(s.mask[-1]))
    # ---- MaskedArray.cumsum（masked 视为 0）----
    check("m1.cumsum()",
          m1.cumsum().tolist() == [1.0, 1.0, 4.0, 8.0])
    # ---- ma.is_masked ----
    check("is_masked True",
          np.ma.is_masked(np.ma.MaskedArray([1.0], mask=[True])))
    check("is_masked False",
          not np.ma.is_masked(np.ma.MaskedArray([1.0], mask=[False])))
    # ---- char.strlen ----
    check("char.strlen",
          np.char.strlen(np.array(['abc', 'a'])).tolist() == [3, 1])
    # ---- 字符串数组 + 拼接（numpy 语义：逐元素 char.add）----
    a = np.array(['a', 'bb'])
    b = np.array(['c', 'dd'])
    check("str array add", (a + b).tolist() == ['ac', 'bbdd'])
    # ---- 回归：第 33 轮 ma 聚合 ----
    check("ma.sum partial (r33)",
          float(np.ma.sum(np.ma.MaskedArray(
              [1.0, 2.0, 8.0], mask=[False, True, False]))) == 9.0)
    # ---- 回归：第 32 轮 where 标量 ----
    r = np.where(np.arange(4.0) > 1, np.arange(4.0), 0.0)
    check("where scalar y (r32)", r.tolist() == [0.0, 0.0, 2.0, 3.0])

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
