"""
特征数据处理
"""
from sklearn.decomposition import PCA,TruncatedSVD
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

class Decompostioner():

    def __init__(self,
                 n_components: int=20,
                 method: str='pca'):
        self.method = method
        self.n_components =n_components

    def reduce_dimensions_pca(self,features):
        """
        使用PCA对特征进行降维

        参数:
        features: 原始特征矩阵，形状 (n_samples, n_features)
        n_components: 要保留的主成分数量

        返回:
        reduced_features: 降维后的特征矩阵
        pca_model: 训练好的PCA模型，可用于后续转换
        """
        print(f"原始特征形状: {features.shape}")

        # 1. 标准化特征（对连续型PCA很重要）

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 2. 应用PCA
        pca = PCA(n_components=self.n_components, random_state=42)
        reduced_features = pca.fit_transform(features_scaled)
        reduced_features = pd.DataFrame(reduced_features)
        reduced_features.columns = [f'pca{i}' for i in range(1, len(reduced_features.columns) + 1)]

        print(f"降维后特征形状: {reduced_features.shape}")
        print(f"累计解释方差比例: {pca.explained_variance_ratio_.sum():.3f}")

        return reduced_features, pca


    def reduce_dimensions_truncatedsvd(self,feature):

        svd = TruncatedSVD(n_components=self.n_components,random_state=42)
        reduced_features = svd.fit_transform(feature)
        reduced_features = pd.DataFrame(reduced_features)
        reduced_features.columns = [f'svd{i}' for i in range(1, len(reduced_features.columns) + 1)]

        explained_variance = svd.explained_variance_ratio_.sum()
        print(f'降维后特征形状: {reduced_features.shape}')
        print(f"保留{self.n_components}维解释{explained_variance:.2%}的方差")

        return reduced_features,svd

    def reduce_dimensions(self,features):


        if self.method == 'pca':
            print(f"降维：PCA")
            reduced_features = self.reduce_dimensions_pca(features)
            return reduced_features

        elif self.method == 'svd':
            print(f"降维：TruncatedSVD")
            reduced_features = self.reduce_dimensions_truncatedsvd(features)
            return reduced_features

        else: print('no such method')




# 连续数值归一化
def zscore_normalization(df,normalization_columns):
    """
    Z-score标准化

    参数：
    df: 要标准化的Dataframe
    normalization_columns: 需要标准化的列名列表
    """
    df_normalized = df.copy()

    # 初始化标准化器
    scaler = StandardScaler()

    # 标准化
    df_normalized[normalization_columns] = scaler.fit_transform(df[normalization_columns])

    # 保存标准化参数
    normalization_info = {
        'method' : 'zscore',
        'column' : normalization_columns,
        'mean' : scaler.mean_,
        'std' : scaler.scale_,
    }

    print(f"✅ Z-score标准化完成")
    print(f"   均值: {scaler.mean_}...")  # 显示特征的均值
    print(f"   标准差: {scaler.scale_}...") # 显示特征的标准差
    print(df_normalized.head(3))

    return df_normalized, normalization_info

def target_correct(X,y,top:int = 10) -> List:
    # 检查特征与标签的相关性
    target_corr = []
    for col in X.columns:
        corr = abs(X[col].corr(y))
        target_corr.append({'feature': col, 'corr_with_target': corr})

    target_corr = pd.DataFrame(target_corr).sort_values('corr_with_target', ascending=False)
    print("\n与标签相关性最高的特征：")
    print(target_corr.head(top))

    return target_corr['feature'].iloc[:top].tolist()