import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from pandas.core.dtypes.common import is_integer, is_string_dtype
from pandas.core.reshape.tile import _format_labels
from sklearn.model_selection import StratifiedKFold
from itertools import product


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


def describer(df, y, bins_dict, sort_index=False):
    pivot = pd.DataFrame()
    for i in set(df.columns) - set([y]):
        bins = bins_dict.get(i)
        if len(bins) > 1:
            df['interval_{}'.format(i)] = pd.cut(df[i], bins=bins, labels=None)

            pivot_tmp = df[['interval_{}'.format(i),
                            y]].groupby(['interval_{}'.format(i)
                                         ]).agg(['count', 'mean'])
            pivot_tmp['proportion'] = pivot_tmp[y]['count'] / len(df)
            pivot_tmp['lift'] = pivot_tmp[y]['mean'] / df[y].mean()

            pivot_tmp.columns = pd.MultiIndex(
                levels=[[y], ['count', 'mean', 'proportion', 'lift']],
                codes=[[0] * 4, list(range(0, 4))])
            desc_list = pivot_tmp.index.astype(str).tolist()
            rectified_index = pd.MultiIndex(levels=[[i], desc_list],
                                            codes=[[0] * len(desc_list),
                                                   range(0, len(desc_list))],
                                            names=['value', 'level'])
            pivot_tmp.index = rectified_index
            pivot = pd.concat([pivot, pivot_tmp])

            if not pivot.index.is_lexsorted() and sort_index:
                pivot.sort_index(level=pivot.index.names, inplace=True)

    return pivot


class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval(
                'lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(
                int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({
            'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + \
            (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(
            prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(
                variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new

    
