#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""random 窄 dtype 对齐探针：rsarray legacy vs numpy RandomState。"""
import rsarray as np

def show(label, rs_fn, np_fn, n=6):
    try:
        r = rs_fn()
        print('RS   %s: %s' % (label, r), flush=True)
    except Exception as e:
        print('RS   %s: ERR %s: %s' % (label, type(e).__name__, e),
              flush=True)
        return
    try:
        v = np_fn()
        print('NP   %s: %s' % (label, v), flush=True)
    except Exception as e:
        print('NP   %s: ERR %s: %s' % (label, type(e).__name__, e),
              flush=True)

# numpy RandomState 参照
onp_rs = np.random._rs_reference if hasattr(
    np.random, '_rs_reference') else None

print('== randint dtype 全族（seed 42）==', flush=True)
for dt in ('int8', 'int16', 'int32', 'int64', 'bool'):
    r = np.random.default_rng(42)
    v = onp_rs = np.random.default_rng(42)
    try:
        import numpy as onp
        ref = onp.random.RandomState(42)
    except ImportError:
        ref = None
    t = 'np.random.' + dt
    show('randint dtype=%s' % dt,
         lambda: np.random.RandomState(42).randint(
             0, 100, size=5, dtype=dt).tolist() if hasattr(
             np.random.RandomState(42), 'randint') else 'n/a',
         lambda: 'ref-below')
print('== rs randint dtype 全族 ==', flush=True)
for dt in ('int8', 'int16', 'int32', 'int64', 'bool'):
    t('rs RandomState randint dtype=%s' % dt,
      lambda dt=dt: np.random.RandomState(42).randint(
          0, 100, size=5, dtype=dt).tolist())
    t('rs default_rng integers dtype=%s' % dt,
      lambda dt=dt: np.random.default_rng(42).integers(
          0, 100, size=5, dtype=dt).tolist())
print('== 其他窄 dtype 函数 ==', flush=True)
t('rs RandomState.bytes(8)', lambda: np.random.RandomState(42).bytes(8))
t('rs RandomState.tomaxint', lambda: np.random.RandomState(42).tomaxint())
print('DONE', flush=True)
