#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rsarray as np


def t(label, fn):
    try:
        r = fn()
        s = repr(r)[:110]
        print('OK   %s: %s' % (label, s), flush=True)
    except Exception as e:
        print('ERR  %s: %s: %s' % (label, type(e).__name__, e)[:110],
              flush=True)


print('== mrecords ==', flush=True)
mr = np.ma.mrecords
t('fromarrays', lambda: mr.fromarrays(
    [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    names=['a', 'b']).a.tolist())
t('fromarrays mask', lambda: mr.fromarrays(
    [np.ma.MaskedArray([1.0, 2.0], mask=[False, True])],
    names=['a']).a.tolist())
t('mrecarray view', lambda: type(mr.fromarrays(
    [np.array([1.0]), np.array([2.0])], names=['x', 'y'])).__name__)
print('== ma 深层 ==', flush=True)
m1 = np.ma.MaskedArray([[1.0, 2.0], [3.0, 4.0]],
                       mask=[[False, True], [False, False]])
t('filled default', lambda: np.ma.filled(m1, -1).tolist())
t('compressed', lambda: m1.compressed().tolist())
t('masked_where', lambda: np.ma.masked_where(
    np.array([True, False, False]), np.array([9.0, 8.0, 7.0])).tolist())
t('ma.concatenate', lambda: np.ma.concatenate(
    [np.ma.MaskedArray([1.0, 2.0]), np.ma.MaskedArray([3.0, 4.0])]).tolist())
t('anom axis', lambda: np.ma.anom(
    np.ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, True, False])).tolist())
t('minmax', lambda: np.ma.minmax(m1))
t('median ma', lambda: np.ma.median(m1))
t('ma.getmaskarray', lambda: np.ma.getmaskarray(np.array([1.0, 2.0])).tolist())
t('apply_along_axis', lambda: np.ma.apply_along_axis(
    lambda col: col.sum(), 0, m1).tolist())
print('== MaskedArray 构造边界 ==', flush=True)
t('0-d masked', lambda: np.ma.MaskedArray(np.array(5.0), mask=True).data)
t('int masked sum', lambda: np.ma.MaskedArray(
    [1, 2, 3], mask=[False, False, True]).sum())
t('nomask sum', lambda: np.ma.MaskedArray([1.0, 2.0], mask=False).sum())
print('DONE', flush=True)
