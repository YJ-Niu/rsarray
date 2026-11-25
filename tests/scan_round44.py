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


m1 = np.ma.MaskedArray([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
m2 = np.ma.MaskedArray([10.0, 20.0, 30.0, 40.0], mask=[False, False, True, False])
print('== ma 算术掩码传播 ==', flush=True)
t('m1 + m2 (掩码 OR)', lambda: (m1 + m2).tolist())
t('m1 - m2', lambda: (m1 - m2).tolist())
t('m1 * m2', lambda: (m1 * m2).tolist())
t('m1 / m2', lambda: (m1 / m2).tolist())
t('m1 ** 2', lambda: (m1 ** 2).tolist())
t('abs(m1)', lambda: np.abs(m1).tolist())
t('m1 == m2', lambda: (m1 == m2).tolist())
print('== ma 聚合/排序 ==', flush=True)
t('argsort', lambda: m1.argsort().tolist())
t('sort', lambda: np.ma.sort(m1).tolist())
t('ma.max', lambda: np.ma.max(m1))
t('min/max axis', lambda: (m1.min(), m1.max()))
t('cumsum', lambda: m1.cumsum().tolist())
print('== ma 构造/查询 ==', flush=True)
t('masked_where', lambda: np.ma.masked_where([True, False], [1, 2]).tolist())
t('count non-masked', lambda: np.ma.count(m1))
t('is_masked', lambda: np.ma.is_masked(m1))
t('fill_value default', lambda: repr(m1.fill_value)[:40])
t('ma.arange? asarray 2d', lambda: np.ma.asarray(np.arange(4)).tolist())
print('== 字符串函数 ==', flush=True)
s = np.array(['abc', 'Abc', 'a b'])
t('upper', lambda: np.char.upper(s).tolist())
t('lower', lambda: np.char.lower(np.array(['ABC'])).tolist())
t('strip', lambda: np.char.strip(np.array([' a b '])).tolist())
t('replace', lambda: np.char.replace(s, 'a', 'X').tolist())
t('startswith', lambda: np.char.startswith(s, 'a').tolist())
t('split', lambda: np.char.split(np.array(['a,b,c']), sep=',').tolist())
t('find', lambda: np.char.find(s, 'b').tolist())
t('strlen', lambda: np.char.strlen(s).tolist())
t('add str', lambda: (np.array(['a']) + np.array(['b'])).tolist())
print('DONE', flush=True)
