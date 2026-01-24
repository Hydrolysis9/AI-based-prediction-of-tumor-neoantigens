import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from sklearn.decomposition import PCA

train_data = pd.read_csv(r"C:\Users\Hydrolysis\Desktop\TCGA_COAD_READ\tcell_table_export_1764489327.tsv\primary_feature_matrix.csv",
                        sep = ",",
                        )
print(train_data.head(3))
print(len(train_data))

# 序列特征稀疏矩阵降维
def reduce_dimensions_pca(features, n_components=20):
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
    # 注意：如果你的K-mer特征是计数或二元值，且已比较规整，可以跳过这步
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 2. 应用PCA
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(features_scaled)
    reduced_features = pd.DataFrame(reduced_features)
    reduced_features.columns = [f'pca{i}' for i in range(1,len(reduced_features.columns)+1)]

    print(f"降维后特征形状: {reduced_features.shape}")
    print(f"累计解释方差比例: {pca.explained_variance_ratio_.sum():.3f}")


    return reduced_features, pca

# 从原始矩阵中取出序列特征稀疏矩阵
k_mer_matrix = train_data.iloc[:,11:411]

# PCA降维
k_mer_matrix,pca = reduce_dimensions_pca(k_mer_matrix,n_components=50)

# 降维后矩阵与其他特征合并
train_data = pd.concat([train_data.iloc[:,:11],train_data.iloc[:,411:],k_mer_matrix],axis=1)
print(train_data.shape)
pass

### 数据缩放 ###

# IC50
train_data['IC50'] = np.log1p(train_data['IC50'])

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

# 创建特征交叉
train_data['ic50_x_presentation'] = train_data['IC50'] * train_data['presentation_score']
train_data['affinity_x_PCP'] = train_data['affinity'] * train_data['PCP']

# 数据标准化
train_data,norm_info = zscore_normalization(train_data,
                                            normalization_columns = ['length',
                                                                     'molecular_weight',
                                                                     'aromaticity',
                                                                     'instability_index',
                                                                     'isoelectric_point',
                                                                     'gravy',
                                                                     'IC50',
                                                                     'affinity',
                                                                     'processing_score',
                                                                     'presentation_score',
                                                                     'ic50_x_presentation',
                                                                     'PCP',
                                                                     'affinity_x_PCP'

                                                                    ])


# 分离特征和标签
feature_cols = [col for col in train_data.columns
                if col not in ['peptide','MHC_allele','label','immunogenicity',
                               'peptide_num','sample_name','best_allele'
                               ]]
x = train_data[feature_cols].copy()
y = train_data['label']
pass


# 检查标签分布
print('='*60)
print('数据平衡分析')
print(f"原始数据分布: {Counter[y]}")
print(f"阳性比例: {sum(y)/len(y):.2%}")
print(f"阴性比例: {(len(y)-sum(y))/len(y):.2%}")

# 分层分割
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n数据分割结果:")
print(f"  训练集: {x_train.shape[0]} 个样本")
print(f"  测试集: {x_test.shape[0]} 个样本")
print(f"  总样本: {len(x)} 个样本")

print(f"\n训练集标签分布:")
train_pos = sum(y_train == 1)
train_neg = sum(y_train == 0)
print(f"  阳性: {train_pos} ({train_pos/len(y_train):.1%})")
print(f"  阴性: {train_neg} ({train_neg/len(y_train):.1%})")
print(f"  比例: 1:{train_neg/train_pos:.2f}")

print(f"\n测试集标签分布:")
test_pos = sum(y_test == 1)
test_neg = sum(y_test == 0)
print(f"  阳性: {test_pos} ({test_pos/len(y_test):.1%})")
print(f"  阴性: {test_neg} ({test_neg/len(y_test):.1%})")
print(f"  比例: 1:{test_neg/test_pos:.2f}")


# 计算权重
classes = np.array([0, 1])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

print(f"\nSklearn 'balanced' 权重:")
print(f"  类别 0 (阴性) 权重: {class_weights[0]:.2f}")
print(f"  类别 1 (阳性) 权重: {class_weights[1]:.2f}")

# 创建权重字典
weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("\n" + "="*60)
print("模型训练与评估")
print("="*60)


### 使用随机森林建立基线 ###

rf_model =RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    verbose=0
)

# 模型训练
rf_model.fit(x_train, y_train)

y_pred_proba = rf_model.predict_proba(x_test)[:, 1]
y_pred = rf_model.predict(x_test)

# 计算性能指标
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
print(f"随机森林性能:")
print(f"AUC: {auc_score:.3f}")
print(f"准确率: {accuracy:.3f}")

# 训练XGboost模型

# 基础模型
xgb_base = xgb.XGBClassifier(
    n_estimators=210,
    gamma=0,
    max_depth=4,
    max_delta_step=0,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    colsample_bynode=1,
    colsample_bylevel=1,
    random_state=42,
    eval_metric='auc',
    scale_pos_weight=test_neg/test_pos

)

xgb_base.fit(
    X=x_train,
    y=y_train,
    eval_set=[(x_test, y_test)],
    verbose=False
)

# 预测
y_pred = xgb_base.predict(x_test)
y_pred_proba = xgb_base.predict_proba(x_test)[:, 1]


# 显示各种评估指标
print("=" * 50)
print("XGBoost 模型评估结果")
print("=" * 50)

print(f"\n1. 基础指标:")
print(f"   准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"   AUC分数: {roc_auc_score(y_test, y_pred_proba):.4f}")

train_auc = roc_auc_score(y_train,xgb_base.predict_proba(x_train)[:,1])
val_auc = roc_auc_score(y_test,xgb_base.predict_proba(x_test)[:,1])
print(f'训练集AUC:{train_auc:.3f},验证集AUC:{val_auc:.3f},差距：{train_auc - val_auc:.3f}')

feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': xgb_base.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 重要特征:")
print(feature_importance.head(15))



