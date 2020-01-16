import numpy as np
import pandas as pd
from discretizer import equifrequency_cutpoints, x_encoder
from sklearn.cluster import DBSCAN
from math import radians, cos, sin, asin, sqrt


class geographic_agglomerator(object):
    def __init__(self, capacity_floor=30, risk_ceiling=0.4, q=1000, bounds=2, **kwargs):
        self.capacity_floor = capacity_floor
        self.risk_ceiling = risk_ceiling
        self.q = q
        self.bounds = bounds
        self.kwargs = kwargs

    def fit(self, df, y, X=['longitude', 'latitude'], radians=False):
        # dependents
        self.longitude = X[0]
        self.latitude = X[1]
        self.X = X
        self.y = y

        # training
        if radians:
            self.db = DBSCAN(**self.kwargs).fit(np.radians(df[X]))
        else:
            self.db = DBSCAN(**self.kwargs).fit(df[X])
        # prediction
        df['labels'] = self.db.labels_

        # seeds
        blacklist = pd.pivot_table(
            df, index='labels', values=y, aggfunc={len, np.mean})
#         blacklist = blacklist.sort_values(by='mean', ascending=False)
        blacklist = blacklist[blacklist['len'] > self.capacity_floor]
        blacklist = blacklist[blacklist['mean']
                              > self.risk_ceiling].index.tolist()
        blacklist = df[np.isin(df['labels'], blacklist)
                       ].groupby('labels')[X].mean()
        self.blacklist = blacklist

        # measure
        self.measure_kwargs = {'blacklist': self.blacklist,
                               'X': [self.longitude, self.latitude]}
        df['distance_min'] = df.apply(
            shortest_distance, **self.measure_kwargs, axis=1)

        # threshold
        df_index = (~np.isinf(df['distance_min'])) & (
            df['distance_min'].notnull())
        self.cut_points, _ = equifrequency_cutpoints(
            df['distance_min'][df_index], q=self.q, winsor=True, append=False, de_dup=True)
        self.threshold = self.cut_points[self.bounds]

        # performance
        df['distance_min_level'] = x_encoder(
            df['distance_min'], splitter='qcut', q=self.q, order=True, fillna=len(self.cut_points))
        self.performance = df.groupby('distance_min_level')[[y]].mean()

    def transform(self, df):
        df['distance_min'] = df.apply(
            shortest_distance, **self.measure_kwargs, axis=1)
        df['distance_min_level'] = x_encoder(
            df['distance_min'], splitter='qcut', q=self.q, order=True, fillna=len(self.cut_points))
        df['hit'] = df['distance_min'].map(
            lambda x: 1 if x < self.threshold else 0)
        return df

    @staticmethod
    def geodistance(lng1, lat1, lng2, lat2):
        lng1, lat1, lng2, lat2 = map(
            radians, [round(lng1, 6), round(lat1, 6), round(lng2, 6), round(lat2, 6)])
        dlon = lng2-lng1
        dlat = lat2-lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        distance = 2*asin(sqrt(a))*6371*1000
        distance = round(distance/1000, 3)
        return distance

    @classmethod
    def shortest_distance(x, blacklist, X=['longitude', 'latitude']):
        lng1 = x[X[0]]
        lat1 = x[X[1]]
        if pd.isnull(lng1):
            result = np.inf
        else:
            l = []
            for i in set(blacklist.index):
                lng2 = blacklist.loc[i][X[0]]
                lat2 = blacklist.loc[i][X[1]]
                l = l+[geodistance(lng1, lat1, lng2, lat2)]
            result = min(l)
        return result

    if __name__ == "__main__":
        df = pd.read_pickle('train_20200115.pkl')

        kms_per_radian = 6371
        eps = 0.5 / kms_per_radian
        kwargs = {'eps': eps, 'min_samples': 5, 'metric': 'haversine',
                  'algorithm': 'ball_tree', 'n_jobs': -10}

        g = globals()
        test_collect = pd.DataFrame()
        task_index = set(df['apply_week'][df['apply_month'] > '2019-03'])

        for i in task_index:
            train = df[df['apply_week'] < i]
            test = df[(df['apply_week'] == i) & (df.user_type == 1)]
            g[f'ga_{i}'] = geographic_agglomerator(**kwargs)
            g[f'ga_{i}'].fit(train, outcome, predictors, radians=True)
            g[f'cut_points_{i}'] = g[f'ga_{i}'].cut_points
            g[f'blacklist_{i}'] = g[f'ga_{i}'].blacklist
            g[f'threshold_{i}'] = g[f'ga_{i}'].threshold
            g[f'performance_{i}'] = g[f'ga_{i}'].performance
            g[f'test_{i}'] = g[f'ga_{i}'].transform(test)
            test_collect = pd.concat(
                [test_collect, g[f'test_{i}']], ignore_index=True)
