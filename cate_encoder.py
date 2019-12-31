import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class AppCateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cate_dict, delimiter=',', prefix='app_', unknown='unknow'):
        self.delimiter = delimiter
        self.prefix = prefix
        self.unknown = unknown

        if type(cate_dict) == str:
            cate_dict = pickle.load(open(cate_dict, 'rb'))
        self.cate_list = [self.prefix + i for i in set(cate_dict.values())|set([self.unknown])]
        self.cate_dict = cate_dict

    def fit(self, dfs):
        return self

    def transform(self, dfs):
        dfs = dfs.map(lambda s:self.app2cate(s, self.cate_dict, self.delimiter, self.prefix, self.unknown))
        df = dfs.apply(pd.Series)

        for c in self.cate_list:
            if c not in df.columns:
                df[c] = np.nan
        df[df.notnull().any(axis=1)] = df[df.notnull().any(axis=1)].replace(np.nan, 0)

        tmp = df.divide(df.sum(axis=1), axis=0)
        tmp.columns = [i+'_ratio' for i in df.columns]
        df = pd.concat([df, tmp], axis=1)
        return df

    @staticmethod
    def app2cate(x, cate_dict, delimiter=',', prefix='app_', unknown='unknow'):
        try:
            x = set([s.strip() for s in x.split(delimiter)])
            l = [prefix + cate_dict.get(i, unknown) for i in x]
            dic = dict(Counter(l))
        except:
            dic = {}

        return dic
    
    if __name__ == "__main__":
        enc = AppCateEncoder(cate_dict=os.getcwd()+'/cate_dict_20180806_v1.pkl')
        pred_sample = '主题商店,360° 照片编辑器,withTV,美团,字典,贷上钱,S换机助手下载,工银融e联,欧朋流量助手,Excel,游戏中心,天气,微信,虎牙直播,迅雷,红包助手,中国工商银行,平安普惠,铃声多多,开心消消乐®,腾讯视频,录制屏幕,链接分享,提醒,汽车之家,慢动作编辑器,Word,PowerPoint,Google通讯录同步,UC浏览器,订阅日历,抖音短视频,百度输入法,QQ,酷狗音乐,图片编辑器,Google 日历同步,英雄战魂2,喜马拉雅FM-听书电台,光大银行,计算器,三星健康,欢乐斗地主,三星云,领英,视频剪辑器,Bixby 视觉,来分期,三星拍立淘,汽车点评,三星盖乐世好友,快牙,用钱宝,邮储银行,手机淘宝,三星书城,掌上生活,惠花花,放大镜,三星畅联,支付宝,招联好期贷,马上金融,网商银行, PP助手,360借条,手机贷,多享金汇,现金借款,阳光惠生活,追书神器'
        pred_sample = pd.Series(pred_sample)
        app_cate = enc.transform(pred_sample)
        app_cate[['app_汽车_ratio', 'app_理财_ratio', 'app_通讯社交_ratio','app_借贷_ratio']]
