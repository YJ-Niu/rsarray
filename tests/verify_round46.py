#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四十六轮验证：datetime64 三件套（构造解析/格式化/算术）。

1. 构造解析：字符串日期 → epoch 数值（D/s 单位）
2. tolist/repr 格式化：epoch → ISO 字符串
3. 算术：± int 按单位平移、datetime − datetime → 天差、max/min
4. NaT 哨兵与 timedelta64
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
    # ---- 构造解析 + 格式化 ----
    d = np.array(['2024-01-01', '2024-06-15'], dtype='datetime64[D]')
    check("storage epoch", d._array.tolist() == [19723.0, 19889.0])
    check("tolist ISO", d.tolist() == ['2024-01-01', '2024-06-15'])
    check("dtype label", d._dtype == 'M8[D]')
    # ---- 算术：± int 按单位平移 ----
    check("d + 1", (d + 1).tolist() == ['2024-01-02', '2024-06-16'])
    check("d - 1", (d - 1).tolist() == ['2023-12-31', '2024-06-14'])
    # ---- datetime − datetime → 天差 ----
    diff = (np.array(['2024-06-15'], dtype='datetime64[D]')
            - np.array(['2024-01-01'], dtype='datetime64[D]'))
    check("diff days", diff.tolist() == [166])
    # ---- max/min 格式化（0-d tolist → 标量，numpy 语义）----
    check("max ISO", np.max(d).tolist() == '2024-06-15')
    check("min ISO", np.min(d).tolist() == '2024-01-01')
    # ---- 秒级单位 ----
    dt = np.array(['2024-01-01T12:30:00'], dtype='datetime64[s]')
    check("s unit tolist", dt.tolist() == ['2024-01-01T12:30:00'])
    # ---- NaT 哨兵 ----
    check("NaT", np.array([None], dtype='datetime64[D]').tolist() == ['NaT'])
    # ---- timedelta64 构造 ----
    td = np.array([5, 10], dtype='timedelta64[D]')
    check("timedelta64", td.tolist() == [5, 10])

    print()
    print(f"{PASS[0]} passed, {FAIL[0]} failed")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == '__main__':
    main()
