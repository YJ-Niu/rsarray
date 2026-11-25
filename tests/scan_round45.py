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


print('== datetime64 ==', flush=True)
d = np.array(['2024-01-01', '2024-06-15'], dtype='datetime64[D]')
t('datetime repr', lambda: repr(d)[:80])
t('datetime + 1 day', lambda: (d + 1).tolist())
t('datetime diff', lambda: np.array(d[1] - d[0]))
t('datetime max', lambda: np.max(d).tolist())
print('== poly 深层 ==', flush=True)
t('polydiv', lambda: np.polydiv(np.array([3.0, 2.0, 1.0]),
                                np.array([1.0, 1.0]))[0].tolist())
t('polyder', lambda: np.polyder(np.array([3.0, 2.0, 1.0])).tolist())
t('polyint', lambda: np.polyint(np.array([3.0, 2.0])).tolist())
t('poly1d call', lambda: np.poly1d([1.0, 2.0])(np.array([0.0, 1.0])).tolist())
t('polyfit full', lambda: len(np.polyfit(
    np.array([0.0, 1.0, 2.0]), np.array([1.0, 4.0, 9.0]), 2, full=True)))
print('== masked extras ==', flush=True)
m1 = np.ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, True, False])
t('ma.average weights', lambda: ma_average(m1))
t('ma.notmasked_contiguous', lambda: np.ma.notmasked_contiguous(m1))
t('ma.count_masked', lambda: np.ma.count_masked(m1))
t('ma masked_invalid', lambda: np.ma.masked_invalid(
    np.array([1.0, np.nan, 3.0])).mask.tolist())
t('ma.arange flat', lambda: np.ma.flatnotmasked_contiguous(m1))
print('== IO ==', flush=True)


def _io():
    np.save('/tmp/rs_io_test.npy', np.arange(4))
    return np.load('/tmp/rs_io_test.npy').tolist()
t('save/load npy', lambda: _io())
print('== 2-D 字符串 ==', flush=True)
s2 = np.array([['a', 'bb'], ['ccc', 'd']])
t('2-D upper', lambda: np.char.upper(s2).tolist())
t('2-D strlen', lambda: np.char.strlen(s2).tolist())
t('2-D str add', lambda: (s2 + s2).tolist())
print('DONE', flush=True)
