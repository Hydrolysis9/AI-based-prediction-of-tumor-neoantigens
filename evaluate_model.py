"""
模型评估模块
"""

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

class ModelEvaluator:
    def __init__(self):
        self.matrics = {}

    def evaluate(self,model,X_test,y_test):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob)
        }

        return metrics

    def cross_validate(self, model, X, y, cv=5):
        """交叉验证"""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }