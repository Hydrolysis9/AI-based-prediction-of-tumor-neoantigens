import numpy as np
from prediction_pipline import PredictionPipeline
import pandas as pd
from utils.data_utils import export_data

# 预加载预测流水线
pipeline = PredictionPipeline('../models/version/model_20260319_180919_0.679853859485634.pkl')

# 加载数据
mutation_seqs= pd.read_csv(r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\final_seq\processed\mutation_seq.csv',
                         sep=',')

prediction_df = pd.DataFrame(mutation_seqs['peptide'],columns=['peptide'])

allele_list = ['HLA-A*02:01','HLA-A*11:01','HLA-A*24:02','HLA-B*46:01','HLA-B*13:01','HLA-B*15:01',
               'HLA-B*58:01','HLA-C*01:02','HLA-C*03:03']

for i in allele_list:
    prediction_df['MHC_allele'] = len(prediction_df)*[i]


    prediction_df.to_csv(r'C:\Users\Hydrolysis\PycharmProjects\Beginning\data\external\mut_data.csv',index=False)
    try:
        res = pipeline.predict(r'C:\Users\Hydrolysis\PycharmProjects\Beginning\data\external\mut_data.csv',output_path=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\data\external\res.csv')
        res['gene'] = mutation_seqs['gene']
        export_data(res,f'../data/predicted/res_{i.replace("*", "").replace(":", "")}.csv')
    except:
        raise KeyError(f'预测失败:{i}')






