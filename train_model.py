"""
模型训练模块
功能：训练免疫原性预测模型
"""
from typing import Union,Optional,Any,List,Dict
import numpy as np
import pandas as pd
import joblib




class TreeModelTrainer:
    def __init__(self,model_type: str='random_forest',random_state: int =42,**params):
        self.model_type = model_type
        self.random_state = random_state
        self.params = params
        self.model = None
        self.base_model = None
        self.best_model = None
        self.feature_importance_ = None

    def _set_default_params(self):
        if not self.params:
            if self.model_type == 'random_forest':
                self.params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

            elif self.model_type == 'xgboost':
                self.params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': self.random_state,
                    'use_label_encoder': False,
                    'eval_metric': 'auc'
                }

    def _create_model(self):

        if self.model_type == 'random_forest':
            from sklearn.ensemble import  RandomForestClassifier
            return RandomForestClassifier(**self.params)
        elif self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(**self.params)

        else:
            raise ValueError(f"不支持的模型类型：{self.model_type}")


    def train(self,
              X_train: Union[pd.DataFrame,np.ndarray],
              y_train: Union[pd.DataFrame,np.ndarray],
              X_val: Optional[Union[pd.DataFrame,np.ndarray]] = None,
              y_val: Optional[Union[pd.DataFrame,np.ndarray]] = None) -> Any:

        self.model = self._create_model()

        # 训练
        if X_val is not None and self.model_type == 'xgboost':
            self.model.fit(
                X_train,y_train,
                eval_set=[(X_val,y_val)],
                verbose=False
            )

        else:
            self.model.fit(X_train,y_train)

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_

        print(f"✅ {self.model_type} 训练完成")
        return self.model


    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测类别"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        return self.model.predict_proba(X)

    def get_features_importance(self,feature_names: Optional[List[str]] = None) ->  Optional[pd.DataFrame]:

        if self.feature_importance_ is None:
            print("警告：该模型没有 feature_importances_ 属性")
            return None

        if feature_names is not None:
            # 确保长度匹配
            if len(feature_names) != len(self.feature_importance_):
                print(f"警告：特征名称数量 ({len(feature_names)}) 与重要性数量 ({len(self.feature_importance_)}) 不匹配")
                return pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(self.feature_importance_))],
                    'importance': self.feature_importance_
                }).sort_values('importance', ascending=False)

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance_
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            return pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(self.feature_importance_))],
                'importance': self.feature_importance_
            }).sort_values('importance', ascending=False)


    def save_model(self,path: str,metadata: Optional[Dict] = None):
        save_dict = {
            'model' : self.model,
            'model_type' : self.model_type,
            'model_params' : self.params
        }

        if metadata:
            save_dict['metadata'] = metadata
        import os
        os.makedirs(os.path.dirname(path),exist_ok=True)
        joblib.dump(save_dict,path)
        print(f"模型已保存到：{path}")

    @classmethod
    def load_model(cls,path: str):
        save_dict = joblib.load(path)

        trainer = cls(
            model_type=save_dict['model_type'],
            **save_dict.get('model_params',{})
        )
        trainer.model = save_dict['model']

        print(f"模型已加载：{path}")
        return trainer



# ========== 便捷函数 ==========

def creat_trainer(model_type: str = 'random_forest', **kwargs) -> TreeModelTrainer:

    return TreeModelTrainer(model_type=model_type,**kwargs)



# ========== 测试函数 ==========

if __name__=="__main__":
    print("=" * 50)
    print("测试模型训练模块")
    print("=" * 50)

    # 生成测试数据
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 转换为DataFrame（模拟真实场景）
    feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    print(f"\n训练集大小：{X_train_df.shape}")
    print(f"测试集大小：{X_test_df.shape}")

    # 测试不同模型
    for model_type in ['random_forest', 'xgboost']:
        print(f"\n--- 测试 {model_type} ---")

        # 创建训练器
        trainer = TreeModelTrainer(model_type=model_type)

        # 训练
        trainer.train(X_train_df, y_train)

        # 预测
        y_pred = trainer.predict(X_test_df)
        y_prob = trainer.predict_proba(X_test_df)

        # 计算准确率
        accuracy = (y_pred == y_test).mean()
        print(f"准确率：{accuracy:.4f}")

        # 特征重要性
        importance_df = trainer.get_features_importance(feature_names)
        if importance_df is not None:
            print("特征重要性（前5）：")
            print(importance_df.head())

        # 测试保存和加载
        trainer.save_model('temp_model.pkl')

        new_trainer = TreeModelTrainer()
        new_trainer.load_model('temp_model.pkl')

        # 清理
        import os

        os.remove('temp_model.pkl')

