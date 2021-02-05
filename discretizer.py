import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from pandas.core.reshape.tile import _format_labels


def cardinality_encoder(x, splitter='qcut', q=10, winsor=True, append=False):
    x = pd.Series(x)
    # bins = pd.qcut(x=x, q=q, retbins=True, duplicates='drop', precision=4)[1]

    if splitter == 'qcut':
        quantiles = np.linspace(start=0, stop=1, num=q + 1)
        bins = algos.quantile(x, quantiles)
    elif splitter == 'cut':
        bins = np.linspace(start=x.min(), stop=x.max(), num=q + 1)
    else:
        raise ValueError('splitter is not defined.')

    if winsor:
        bins[-1] = np.inf
        bins[0] = -np.inf
    if append:
        bins = np.append(bins, append)
        bins = np.unique(bins)
        bins.sort()

    indices = bins.searchsorted(x, side='left').astype(np.float64)
    indices[x == bins[0]] = 1
    na_mask = np.isnan(x)
    if na_mask.any():
        np.putmask(indices, na_mask, 0)
    indices -= 1

    bins_formatted = _format_labels(bins=bins,
                                    precision=4,
                                    right=True,
                                    include_lowest=True)
    labels = algos.take_nd(arr=bins_formatted, indexer=indices)

    return bins, indices, labels


def describer(df, y=None, columns=None):
    pivot = pd.DataFrame()

    if not y:
        y = '__mark'
        df[y] = 0
    column_limit = columns if columns else set(df.columns) - set([y])

    for i in column_limit:
        _, _, df['interval_{}'.format(i)] = cardinality_encoder(
            x=df[i],
            splitter='qcut',
            q=10,
            winsor=False,
            append=[-np.inf, -1, 0, np.inf])
        pivot_tmp = df.groupby(by=['interval_{}'.format(i)],
                               dropna=False).agg({y: ['count', 'mean']})
        pivot_tmp['proportion'] = pivot_tmp[y]['count'] / len(df)
        pivot_tmp['lift'] = pivot_tmp[y]['mean'] / df[y].mean()
        pivot_tmp.columns = pd.MultiIndex.from_tuples(
            tuples=[(y, j) for j in ['count', 'mean', 'proportion', 'lift']])
        pivot_tmp.index = pd.MultiIndex.from_tuples(
            tuples=[(i, j) for j in pivot_tmp.index],
            names=['variable', 'range'])
        pivot = pd.concat([pivot, pivot_tmp])

    if y == '__mark':
        pivot = pivot[y].drop(columns=['mean', 'lift'])

    return pivot




from sklearn.model_selection import StratifiedKFold
from itertools import product


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

    
