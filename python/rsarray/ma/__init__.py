# SPDX-License-Identifier: MIT
"""掩码数组（ma 兼容层）。

子模块布局与 一致：

- ``rsarray.ma.core``：核心（MaskedArray / 构造器 / 掩码辅助）
- ``rsarray.ma.extras``：统计与集合扩展
- ``rsarray.ma.mrecords``：掩码记录数组
"""

from . import core
from . import extras
from . import mrecords
from .core import (  # noqa: F401
    MAError,
    MaskError,
    MaskType,
    MaskedArray,
    MaskedArrayFutureWarning,
    anom,
    array,
    asarray,
    common_fill_value,
    concatenate,
    count,
    default_fill_value,
    empty,
    empty_like,
    fix_invalid,
    flatten_mask,
    flatten_structured_array,
    filled,
    full,
    getdata,
    getmask,
    getmaskarray,
    isMaskedArray,
    make_mask,
    make_mask_descr,
    masked,
    masked_array,
    masked_equal,
    masked_greater,
    masked_greater_equal,
    masked_inside,
    masked_invalid,
    masked_less,
    masked_less_equal,
    masked_not_equal,
    masked_object,
    masked_outside,
    masked_print_option,
    masked_values,
    masked_where,
    mask_or,
    maximum_fill_value,
    minimum_fill_value,
    minmax,
    mvoid,
    nomask,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .mrecords import (  # noqa: F401
    MaskedRecords,
    mrecarray,
)

# ma 顶层暴露 core 的全部公共名（test_old_ma 按此导入）
for _n in dir(core):
    if not _n.startswith('_') and _n not in globals():
        globals()[_n] = getattr(core, _n)
del _n

# ma 顶层亦暴露 extras 的常用名
for _n in ('atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'clump',
           'clump_masked', 'clump_unmasked', 'compress_nd', 'compress_rowcols',
           'corrcoef', 'cov', 'diagflat', 'ediff1d', 'flatnotmasked_contiguous',
           'in1d', 'intersect1d', 'isin', 'masked_all', 'masked_all_like',
           'mask_rowcols', 'mask_rows', 'mask_cols', 'median', 'mr_', 'ndenumerate', 'notmasked_contiguous',
           'notmasked_edges', 'polyfit', 'setdiff1d', 'setxor1d', 'stack',
           'union1d', 'unique', 'unwrap', 'vstack', 'apply_along_axis',
           'apply_over_axes'):
    if hasattr(extras, _n):
        globals()[_n] = getattr(extras, _n)
del _n

__all__ = ['core', 'extras', 'mrecords', 'MaskedArray', 'MaskedRecords',
           'masked', 'nomask', 'array', 'asarray', 'masked_array', 'mrecarray',
           'MAError', 'MaskError', 'MaskType', 'MaskedArrayFutureWarning']


# ---- 掩码感知聚合（numpy ma 语义：masked 视为 0/1；全掩码 → 0/1/masked）。
# 覆盖此前从 statistics 泄漏进 ma 命名空间的无掩码感知实现。
def _ma_is_masked(m):
    return m is not nomask and m is not False and m is not None


def sum(a, axis=None, dtype=None, out=None, keepdims=None):
    """掩码感知求和：masked 元素视为 0（全掩码 → 0.0）。"""
    a = asarray(a)
    m = getmask(a)
    filled_a = filled(a, 0) if _ma_is_masked(m) else getdata(a)
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    from rsarray.statistics import sum as _stat_sum
    return _stat_sum(filled_a, **kwargs)


def prod(a, axis=None, dtype=None, out=None, keepdims=None):
    """掩码感知求积：masked 元素视为 1（全掩码 → 1.0）。"""
    a = asarray(a)
    m = getmask(a)
    filled_a = filled(a, 1) if _ma_is_masked(m) else getdata(a)
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    return filled_a.prod(**kwargs)


def mean(a, axis=None, dtype=None, out=None, keepdims=None):
    """掩码感知均值：masked 元素不计入分母；全掩码 → masked。"""
    import rsarray as _np
    a = asarray(a)
    m = getmask(a)
    filled_a = filled(a, 0) if _ma_is_masked(m) else getdata(a)
    cnt = int(a.size)
    if _ma_is_masked(m):
        cnt -= int(_np.asarray(m).astype('int64').sum())
    if cnt == 0:
        return MaskedArray(_np.array(0.0), mask=[True])
    from rsarray.statistics import mean as _stat_mean
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    return _stat_mean(filled_a, **kwargs)


def concatenate(arrays, axis=0, out=None):  # noqa: F811
    """掩码感知拼接：任一输入为 MaskedArray 时 data 拼接 + 掩码重建。"""
    arrays = list(arrays)
    if any(isinstance(a, MaskedArray) for a in arrays):
        import rsarray as _np_mod
        datas = [core.getdata(a) for a in arrays]
        masks = [core.getmaskarray(a) for a in arrays]
        out_d = _np_mod.concatenate(datas, axis=axis)
        out_m = _np_mod.concatenate(masks, axis=axis)
        return MaskedArray(out_d, mask=out_m)
    return core.concatenate(arrays, axis=axis, out=out)


def sort(a, axis=-1, kind=None, order=None):
    """掩码感知排序（ma.sort 兼容）：masked 元素排最后，掩码同步重排。

    覆盖 statistics.sort 的裸绑定（后者对 MaskedArray 走 core sort，
    负轴/掩码语义均缺失）；非 MaskedArray 输入回退普通 sort。
    """
    if isinstance(a, MaskedArray):
        return a.sort(axis=axis, kind=kind, order=order)
    from rsarray.statistics import sort as _stat_sort
    return _stat_sort(a, axis=axis, kind=kind, order=order)
