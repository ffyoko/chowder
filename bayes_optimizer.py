from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
import xgboost as xgb

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours


def get_data():
    data, targets = make_classification(
        n_samples=1000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=134985745
    )
    return data, targets


def svc_cv(C, gamma, data, targets):
    estimator = SVC(C=C, gamma=gamma, random_state=2)
    cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=4)
    return cval.mean()


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss', cv=4)
    return cval.mean()


def xgb_cv(max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, dtrain):

    global RMSEbest
    global ITERbest

    paramt = {
        'booster': 'gbtree',
        'max_depth': max_depth,
        'gamma': gamma,
        'eta': 0.01,
        'objective': 'reg:linear',
        'nthread': 8,
        'silent': True,
        'eval_metric': 'rmse',
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'max_delta_step': max_delta_step,
        'seed': 1001
    }

    xgbr = xgb.cv(
        paramt,
        dtrain,
        num_boost_round=1000,
#         stratified = True,
        nfold=3,
        verbose_eval=False,
        early_stopping_rounds=50,
        metrics="rmse",
        show_stdv=True)

    cv_score = xgbr['test-rmse-mean'].iloc[-1]
    if (cv_score < RMSEbest):
        RMSEbest = cv_score
        ITERbest = len(xgbr)

    return (-1.0 * cv_score)


def optimize_svc(data, targets):
    def svc_crossval(expC, expGamma):
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)


def optimize_rfc(data, targets):
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999)
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)


def optimize_xgb(data, targets):

    def xgb_crossval(max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
        return xgb_cv(
            max_depth=max_depth.astype(int),
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step.astype(int),
            dtrain=xgb.DMatrix(data, label=targets)
        )

    optimizer = BayesianOptimization(
        f=xgb_crossval, 
        pbounds={
            'max_depth': (3, 10),
            'gamma': (0.00001, 1.0),
            'min_child_weight': (0, 5),
            'max_delta_step': (0, 5),
            'subsample': (0.5, 0.9),
            'colsample_bytree': (0.05, 0.4)
        })

    optimizer.maximize(init_points=10, n_iter=25, acq="ei", xi=0.01)

    print("Final result:", optimizer.max)


if __name__ == "__main__":
    data, targets = get_data()

    print(Colours.yellow("--- Optimizing SVM ---"))
    optimize_svc(data, targets)

    print(Colours.green("--- Optimizing Random Forest ---"))
    optimize_rfc(data, targets)

    RMSEbest = 10.
    ITERbest = 0
    print(Colours.blue("--- Optimizing XGBoost ---"))
    optimize_xgb(data, targets)
