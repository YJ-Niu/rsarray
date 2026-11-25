#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rsarray as np


def t(label, fn):
    try:
        r = fn()
        s = repr(r)[:100]
        print('OK   %s: %s' % (label, s), flush=True)
    except Exception as e:
        print('ERR  %s: %s: %s' % (label, type(e).__name__, e)[:100],
              flush=True)


a = np.array([1.0, 2.0, 3.0, 4.0])
an = a.copy(); an[1] = np.nan
m2 = np.arange(12, dtype='float64').reshape(3, 4)
m2[1, 2] = np.nan
print('== 基础 nan 聚合 ==', flush=True)
t('nanmean', lambda: np.nanmean(an))
t('nanvar', lambda: round(float(np.nanvar(an)), 6))
t('nanstd', lambda: round(float(np.nanstd(an)), 6))
t('nansum', lambda: np.nansum(an))
t('nansum axis=0', lambda: np.nansum(m2, axis=0).tolist())
t('nanmean axis=0', lambda: np.nanmean(m2, axis=0).tolist())
t('nanmean axis=1', lambda: np.nanmean(m2, axis=1).tolist())
t('nanmedian axis=1', lambda: np.nanmedian(m2, axis=1).tolist())
print('== 边界 ==', flush=True)
t('nanmean all-nan', lambda: np.nanmean(np.array([np.nan, np.nan])))
t('nanmin all-nan', lambda: np.nanmin(np.array([np.nan, np.nan])))
t('nanargmax all-nan', lambda: np.nanargmax(np.array([np.nan, np.nan])))
t('nancumsum', lambda: np.nancumsum(an).tolist())
t('nanpercentile 50', lambda: np.nanpercentile(an, 50))
t('nanstd ddof=1', lambda: round(float(np.nanstd(an, ddof=1)), 6))
t('nanvar axis=1', lambda: np.nanvar(m2, axis=1).tolist())
print('== nan 语义对照（numpy 行为）==', flush=True)
print('nanmean [1,2,3,4] with nan@1: 期望 2.666.. (1,3,4)')
print('nanmean axis=0 列 2 masked: 期望列 2 只算非 nan', flush=True)
print('DONE', flush=True)
