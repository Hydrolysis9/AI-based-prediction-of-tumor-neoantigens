"""
超参数调优
"""
import optuna
from typing import Dict,Optional

import xgboost

from sklearn.metrics import roc_auc_score,recall_score,fbeta_score,average_precision_score
from sklearn.model_selection import cross_val_score,StratifiedKFold
import pandas as pd



class OptunaTuner:

    def __init__(self,
                 model_type: str= 'xgboost',
                 direction: str = 'maximize',
                 study_name: str = None,
                 storage: str = None,
                 random_state: int = 42):

        self.model_type = model_type
        self.direction = direction
        self.random_state = random_state
        self.best_params = None
        self.best_value = None
        self.best_model = None

        # 创建或加载study
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )

    def _get_xgboost_params(self,trial: optuna.Trial) -> Dict:

        params = {
            'n_estimators': trial.suggest_int('n_estimators',150,300,step=5),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': self.random_state,
            'use_label_encoder': False,
            'eval_metric': 'auc'
        }

        return params

    def _object_cv(self,trial: optuna.Trial,
                   X: pd.DataFrame,
                   y: pd.Series,
                   cv_folds: int = 5) -> float:

        params = self._get_xgboost_params(trial)
        model = xgboost.XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=self.random_state)
        scores = cross_val_score(model,X,y,cv=cv,scoring='roc_auc')

        return scores.mean()

    def _objective_val(self,trial: optuna.Trial,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame,
                       y_val: pd.Series,
                       tune_item: str='roc_auc'
                       ) -> float:

        params = self._get_xgboost_params(trial)
        model = xgboost.XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        if tune_item =='roc_auc':
            y_pred_proba = model.predict_proba(X_val)[:,1]
            val_auc = roc_auc_score(y_val,y_pred_proba)
            return val_auc

        elif tune_item =='recall':
            y_pred = model.predict(X_val)
            val_recall = recall_score(y_val, y_pred)
            return val_recall

        elif tune_item == 'f2':
            y_pred = model.predict(X_val)
            val_f2 = fbeta_score(y_val, y_pred, beta=2)
            return val_f2

        elif tune_item == 'pr_auc':
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            val_pr_auc = average_precision_score(y_val, y_pred_proba)
            return val_pr_auc
        else:
            print('无此类型输出')


    def tune(self,
             X: pd.DataFrame,
             y: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             tune_item = 'roc_auc',
             n_trials: int = 50,
             cv_folds: int = 5,
             timeout: Optional[int] = None,
             show_progress: bool = True
             ) -> Dict:

        if X_val is not None and y_val is not None:
            print("使用验证集进行调优...")
            objective = lambda trial: self._objective_val(
                trial,X,y,X_val,y_val,tune_item=tune_item
            )
        else:
            print(f'使用{cv_folds}折交叉验证进行调优...')
            objective = lambda trial : self._object_cv(
                trial,X,y,cv_folds
            )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        print(f'\n最佳AUC ：{self.best_value}')
        print('最佳参数')
        for k,v in self.best_params.items():
            print(f" {k}: {v}")

        return self.best_params


# ========== 便捷函数 ==========
def tune_with_validation(X_train, y_train, X_val, y_val,
                        model_type='xgboost', n_trials=30,tune_item='roc_auc'):
    tuner = OptunaTuner(model_type=model_type)
    best_params = tuner.tune(
        X_train,y_train,X_val,y_val,
        n_trials=n_trials,
        tune_item=tune_item
    )

    return best_params








