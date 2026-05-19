import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from models.train_model import TreeModelTrainer
from utils.data_utils import export_data
from pipelines.prediction_pipline import PredictionPipeline
import matplotlib.pyplot as plt


# 读取结果
model = TreeModelTrainer.load_model('../models/version/model_20260319_180919_0.679853859485634.pkl')

# ====== IEDB ======
res_deepimmuno = pd.read_csv('IEDB_data/Deepimmuno_res.txt', sep='\t')
X_test = pd.read_csv('../data/external/X_test.csv')
res = pd.read_csv('IEDB_data/tool_features.csv')
y = pd.read_csv('IEDB_data/y_test.csv')


model_res = pd.DataFrame(model.predict_proba(X_test)[:,1],columns=['prob'])
model_res = pd.concat((model_res,res['peptide'],y),axis=1).sort_values('prob',ascending=False)

res_mhcflurry = pd.concat((res['peptide'],res['presentation_score'],y),axis=1).sort_values('presentation_score',ascending=False)
res_netmhc = pd.concat((res['peptide'],res['netMHCpan_%Rank'],y),axis=1).sort_values('netMHCpan_%Rank',ascending=True)

model_res = model_res[model_res['peptide'].str.len().isin([9,10])]
res_mhcflurry = res_mhcflurry[res_netmhc['peptide'].str.len().isin([9,10])]
res_netmhc = res_netmhc[res['peptide'].str.len().isin([9,10])]

res_deepimmuno = res_deepimmuno.merge(
    model_res[['peptide', 'label']],  # 只取需要的列
    on='peptide',              # 按peptide列对齐
    how='left'                 # 保留res_deepimmuno所有行
).sort_values('immunogenicity',ascending=False)

model_res = model_res.drop_duplicates(subset='peptide',keep='first')
res_deepimmuno = res_deepimmuno.drop_duplicates(subset='peptide',keep='first')
res_mhcflurry = res_mhcflurry.drop_duplicates(subset='peptide',keep='first')
res_netmhc = res_netmhc.drop_duplicates(subset='peptide',keep='first')


export_data(model_res,'../notebooks/terminal/res_model.csv')
export_data(res_netmhc,'../notebooks/terminal/res_netmhc.csv')
export_data(res_mhcflurry,'../notebooks/terminal/res_mhcflurry.csv')
export_data(res_deepimmuno,'../notebooks/terminal/res_deepimmuno.csv')


# 计算auc
auc_model = roc_auc_score(model_res['label'],model_res['prob'])
auc_mhcflurry = roc_auc_score(res_mhcflurry['label'],res_mhcflurry['presentation_score'])
auc_netmhc = roc_auc_score(res_netmhc['label'], -res_netmhc['netMHCpan_%Rank'])
auc_deepimmuno = roc_auc_score(res_deepimmuno['label'], res_deepimmuno['immunogenicity'])

# 计算top_K
def top_k(df:pd.DataFrame,k: int=20):
    df = df.head(k)
    counts = len(df[df['label'] == 1])
    res = counts / k
    return counts,res

top_20_mhcflurry,topk_mhcflurry = top_k(res_mhcflurry)
top_20_netmhc,topk_netmhc = top_k(res_netmhc)
top_20_deepimmuno,topk_deepimmuno = top_k(res_deepimmuno)
top_20_model,topk_model = top_k(model_res)

# 计算富集倍数
positive_rate_iedb = len(y[y['label'] == 1]) / len(y)
enrichment_factor_iedb = topk_model/positive_rate_iedb


print(f'\nauc_model:{auc_model}')
print(f'auc_netmhc:{auc_netmhc}')
print(f'auc_mhcflurry:{auc_mhcflurry}')
print(f'auc_deepimmuno:{auc_deepimmuno}')

print(f'\ntopk_model:{topk_model}')
print(f'topk_mhcflurry:{topk_mhcflurry}')
print(f'topk_netmhc:{topk_netmhc}')
print(f'topk_deepimmuno:{topk_deepimmuno}')

print(f'\nenrichment_factor:{enrichment_factor_iedb}')

# 可视化
total_df = pd.read_csv('../notebooks/terminal/total_df.csv')

# ROC曲线
fig, ax = plt.subplots(figsize=(6, 6))
plt.style.use('seaborn-v0_8-whitegrid')
models_auc = {'My_model':total_df['my_model_prob'],
          'DeepImmuno': total_df['Deepimmuno'],
          'MHCflurry': total_df['Deepimmuno'],
          'NetMHCpan': 1-total_df['netMHCpan_%Rank']
          }

auc_scores = {'My_model': auc_model,
          'DeepImmuno': auc_deepimmuno,
          'MHCflurry': auc_mhcflurry,
          'NetMHCpan': auc_netmhc
          }


colors_auc = {'My_model': '#2E86AB', 'DeepImmuno': '#A23B72',
          'MHCflurry': '#F18F01', 'NetMHCpan': '#C73E1D'}

for name, scores in models_auc.items():
    fpr, tpr, _ = roc_curve(total_df['label'], scores)
    plt.plot(fpr,tpr,
             label=f'{name}:(AUC = {auc_scores[name]:.3f})',
             color=colors_auc[name])

# 添加对角线（随机猜测）
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)', alpha=0.5)

# 设置图表属性
plt.xlabel('FPR', fontsize=14, fontweight='bold')
plt.ylabel('TPR', fontsize=14, fontweight='bold')
plt.title('ROC(IEDB)', fontsize=16, fontweight='bold', pad=20)

# 添加图例
plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)

# 设置坐标轴范围
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')



# Precision@top20
models_top = ['My Model', 'MHCflurry', 'NetMHCpan', 'DeepImmuno']
colors_top = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72']
precision_at_20 = [topk_model, topk_mhcflurry, topk_netmhc, topk_deepimmuno]

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.1

bars = plt.bar(models_top, precision_at_20, color=colors_top,width=0.5 ,edgecolor='black', linewidth=0.5, alpha=0.85,label=models_top)
# 添加数值标签
for bar, val in zip(bars, precision_at_20):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置坐标轴
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision@Top20', fontsize=14, fontweight='bold')
ax.set_title('Precision@Top20',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.05)

# 添加图例
ax.legend(loc='upper right', fontsize=11)

# 添加网格
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('precision_at_20_basic.png', dpi=300, bbox_inches='tight')



# Precision@top50
top_50_mhcflurry,topk_mhcflurry = top_k(res_mhcflurry,50)
top_50_netmhc,topk_netmhc = top_k(res_netmhc,50)
top_50_deepimmuno,topk_deepimmuno = top_k(res_deepimmuno,50)
top_50_model,topk_model = top_k(model_res,50)

precision_at_50 = [topk_model, topk_mhcflurry, topk_netmhc, topk_deepimmuno]

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.1

bars = plt.bar(models_top, precision_at_50, color=colors_top,width=0.5 ,edgecolor='black', linewidth=0.5, alpha=0.85,label=models_top)
# 添加数值标签
for bar, val in zip(bars, precision_at_50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置坐标轴
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision@Top50', fontsize=14, fontweight='bold')
ax.set_title('Precision@Top50',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.05)

# 添加图例
ax.legend(loc='upper right', fontsize=11)

# 添加网格
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('precision_at_50_basic.png', dpi=300, bbox_inches='tight')


# ========== ITSNdb ==========

# 导入数据
itsndb_res = pd.read_csv('../notebooks/ITSNdb_data/itsndb_res.csv')

# 计算auc
auc_model = roc_auc_score(itsndb_res['label'],itsndb_res['My_model'])
auc_mhcflurry = roc_auc_score(itsndb_res['label'],itsndb_res['mhcflurry'])
auc_netmhc = roc_auc_score(itsndb_res['label'],-itsndb_res['netMHCpan'])
auc_deepimmuno = roc_auc_score(itsndb_res['label'],itsndb_res['Deepimmuno'])



# ROC曲线
fig, ax = plt.subplots(figsize=(6, 6))
plt.style.use('seaborn-v0_8-whitegrid')
models_auc = {'My_model':itsndb_res['My_model'],
          'DeepImmuno': itsndb_res['Deepimmuno'],
          'MHCflurry': itsndb_res['mhcflurry'],
          'NetMHCpan': 1-itsndb_res['netMHCpan']
          }

auc_scores = {'My_model': auc_model,
          'DeepImmuno': auc_deepimmuno,
          'MHCflurry': auc_mhcflurry,
          'NetMHCpan': auc_netmhc
          }


colors_auc = {'My_model': '#2E86AB', 'DeepImmuno': '#A23B72',
          'MHCflurry': '#F18F01', 'NetMHCpan': '#C73E1D'}

for name, scores in models_auc.items():
    fpr, tpr, _ = roc_curve(itsndb_res['label'], scores)
    plt.plot(fpr,tpr,
             label=f'{name}:(AUC = {auc_scores[name]:.3f})',
             color=colors_auc[name])

# 添加对角线（随机猜测）
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)', alpha=0.5)

# 设置图表属性
plt.xlabel('FPR', fontsize=14, fontweight='bold')
plt.ylabel('TPR', fontsize=14, fontweight='bold')
plt.title('ROC(ITSNdb)', fontsize=16, fontweight='bold', pad=20)

# 添加图例
plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)

# 设置坐标轴范围
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('roc_comparison_ITSNdb.png', dpi=300, bbox_inches='tight')




# Precision@top20
top_20_mhcflurry,topk_mhcflurry = top_k(itsndb_res[['mhcflurry','label']].sort_values('mhcflurry',ascending=False))
top_20_netmhc,topk_netmhc = top_k(itsndb_res[['netMHCpan','label']].sort_values('netMHCpan',ascending=True))
top_20_deepimmuno,topk_deepimmuno = top_k(itsndb_res[['Deepimmuno','label']].sort_values('Deepimmuno',ascending=False))
top_20_model,topk_model = top_k(itsndb_res[['My_model','label']].sort_values('My_model',ascending=False))


models_top = ['My Model', 'MHCflurry', 'NetMHCpan', 'DeepImmuno']
colors_top = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72']
precision_at_20 = [topk_model, topk_mhcflurry, topk_netmhc, topk_deepimmuno]

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.1

bars = plt.bar(models_top, precision_at_20, color=colors_top,width=0.5 ,edgecolor='black', linewidth=0.5, alpha=0.85,label=models_top)
# 添加数值标签
for bar, val in zip(bars, precision_at_20):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置坐标轴
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision@Top20', fontsize=14, fontweight='bold')
ax.set_title('Precision@Top20',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.05)

# 添加图例
ax.legend(loc='upper right', fontsize=11)

# 添加网格
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('precision_at_20_basic_ITSNdb.png', dpi=300, bbox_inches='tight')





# Precision@top50
top_50_mhcflurry,topk_mhcflurry = top_k(itsndb_res[['mhcflurry','label']].sort_values('mhcflurry',ascending=False),50)
top_50_netmhc,topk_netmhc = top_k(itsndb_res[['netMHCpan','label']].sort_values('netMHCpan',ascending=True),50)
top_50_deepimmuno,topk_deepimmuno = top_k(itsndb_res[['Deepimmuno','label']].sort_values('Deepimmuno',ascending=False),50)
top_50_model,topk_model = top_k(itsndb_res[['My_model','label']].sort_values('My_model',ascending=False),50)


models_top = ['My Model', 'MHCflurry', 'NetMHCpan', 'DeepImmuno']
colors_top = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72']
precision_at_50 = [topk_model, topk_mhcflurry, topk_netmhc, topk_deepimmuno]

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.1

bars = plt.bar(models_top, precision_at_50, color=colors_top,width=0.5 ,edgecolor='black', linewidth=0.5, alpha=0.85,label=models_top)
# 添加数值标签
for bar, val in zip(bars, precision_at_50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置坐标轴
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision@Top50', fontsize=14, fontweight='bold')
ax.set_title('Precision@Top50',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.05)

# 添加图例
ax.legend(loc='upper right', fontsize=11)

# 添加网格
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('precision_at_50_basic_ITSNdb.png', dpi=300, bbox_inches='tight')

# 富集倍数
positive_rate_itsndb = len(itsndb_res[itsndb_res['label'] == 1]) / len(itsndb_res)
enrichment_factor_itsndb = topk_model/positive_rate_iedb

