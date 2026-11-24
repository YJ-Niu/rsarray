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


a = np.arange(12).reshape(3, 4)
print('== setitem ==', flush=True)
t('basic a[0,0]=9', lambda: (a.__setitem__((0, 0), 9), a[0, 0]))
t('slice a[1:]=0', lambda: (a.__setitem__(slice(1, None), 0), a[1].tolist()))
t('bool mask set', lambda: a[a > 5].tolist())
t('fancy set', lambda: (a.__setitem__((np.array([0, 2]), 1), [7, 8]),
                        [a[0, 1], a[2, 1]]))
print('== fft ==', flush=True)
t('fft 4', lambda: np.fft.fft([1.0, 2.0, 3.0, 4.0]).tolist())
t('ifft roundtrip', lambda: [complex(v).real for v in
                             np.fft.ifft(np.fft.fft([1.0, 2.0, 3.0, 4.0]))])
t('fft2', lambda: np.fft.fft2(np.ones((2, 2))).shape)
t('rfft 4', lambda: np.fft.rfft([1.0, 2.0, 3.0, 4.0]).tolist())
print('== random ==', flush=True)
t('rand shape', lambda: np.random.rand(2, 3).shape)
t('seeded randn', lambda: [round(v, 4) for v in
                           np.random.RandomState(42).randn(3)])
t('randint', lambda: np.random.randint(0, 10, 5).tolist())
t('choice', lambda: np.random.choice(5, 3).tolist())
print('== ma extras ==', flush=True)
t('ma corrcoef', lambda: np.ma.corrcoef(np.array([1.0, 2.0, 3.0]),
                                        np.array([2.0, 4.0, 6.0])).tolist())
t('ma vstack', lambda: np.ma.vstack(
    [np.ma.MaskedArray([1.0, 2.0]),
     np.ma.MaskedArray([3.0, 4.0])]).tolist())
t('ma.unique', lambda: np.ma.unique(np.ma.MaskedArray(
    [1.0, 2.0, 2.0], mask=[False, False, False])).tolist())
print('DONE', flush=True)
