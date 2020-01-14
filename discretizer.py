import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from pandas.core.dtypes.common import is_integer, is_string_dtype
from pandas.core.reshape.tile import _format_labels


def equifrequency_cutpoints(score, q=10, winsor=True, append=False, de_dup=True):
    score = pd.Series(score)
#     bins = pd.qcut(score, q=q, retbins=True, duplicates='drop', precision=8)[1]

    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q

    bins = algos.quantile(score, quantiles)

    if winsor:
        bins[-1] = np.inf
        bins[0] = -np.inf
    if append is not False:
        bins = np.append(bins, append)
        bins.sort()
    if de_dup:
        bins = np.unique(bins)

    labels = _format_labels(bins, precision=8)

    return bins, labels


def x_encoder(score, splitter='qcut', q=10, order=True, fillna=-1):
    score = pd.Series(score)

    if splitter == 'qcut':
        bins, _ = equifrequency_cutpoints(score, q=q, winsor=True, 
                                          append=False, de_dup=True)
        intervals = len(bins)-1
    elif splitter == 'cut':
        bins = intervals = q
    elif splitter == 'cluster':
        raise ValueError('splitter is not defined.')
    else:
        raise ValueError('splitter is not defined.')
    
    labels = range(intervals, 0, -1) if order else None

    levels = pd.cut(score, bins=bins, labels=labels)
    
    if fillna is not False:
        levels.cat.add_categories([fillna], inplace=True)
        levels.fillna(fillna, inplace=True)

    return levels
