#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三十七轮验证：fft2/ifft2 + ma.corrcoef 双参数 + ma.unique。

1. fft2/ifft2：值、roundtrip、__all__ 导出
2. ma.corrcoef 双参数（x, y）完全相关
3. ma.unique 掩码感知 + counts
4. 回归：第 32-34 轮（where 标量/ma 聚合/ma 深层）
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
    # ---- fft2/ifft2 ----
    f2 = np.fft.fft2(np.ones((2, 2)))
    check("fft2 values", f2.tolist()[0][0] == complex(4, 0))
    rt = np.fft.ifft2(f2)
    check("ifft2 roundtrip", abs(rt[0][0].real - 1.0) < 1e-12)
    check("fft2 in __all__",
          'fft2' in np.fft.__all__ and 'ifft2' in np.fft.__all__)
    # ---- ma.corrcoef 双参数 ----
    rc = ma.corrcoef(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    check("ma.corrcoef (x,y)", rc.tolist() == [[1.0, 1.0], [1.0, 1.0]])
    # ---- ma.unique 掩码感知 ----
    ru = ma.unique(np.ma.MaskedArray([1.0, 2.0, 2.0],
                                     mask=[False, False, False]))
    check("ma.unique values", ru.tolist() == [1.0, 2.0])
    _, cnt = ma.unique(np.ma.MaskedArray([1.0, 2.0, 2.0]),
                       return_counts=True)
    check("ma.unique counts", cnt.tolist() == [1, 2])
    # ---- 回归：第 33 轮 ma 聚合 ----
    check("ma.sum partial (r33)",
          float(np.ma.sum(np.ma.MaskedArray(
              [1.0, 2.0, 8.0], mask=[False, True, False]))) == 9.0)
    # ---- 回归：第 32 轮 where 标量 ----
    r = np.where(np.arange(4.0) > 1, np.arange(4.0), 0.0)
    check("where scalar y (r32)", r.tolist() == [0.0, 0.0, 2.0, 3.0])
    # ---- 回归：切片/dtype 边界（第 33 轮）----
    a = np.arange(12).reshape(3, 4)
    check("slice (r33)", a[1:, ::2].tolist() == [[4, 6], [8, 10]])
    check("int64 overflow (r33)",
          np.array([2**63]).astype('int64').tolist() == [-9223372036854775808])

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
