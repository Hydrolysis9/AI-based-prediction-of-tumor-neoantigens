import pandas as pd
import yaml
import os
from typing import Optional
import itertools


class PredictionPipeline:

    def __init__(self,model_path: str,config_path: str = '../config/config.yaml'):

        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model = None

    def _load_config(self,config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self):
        from models.train_model import TreeModelTrainer

        print(f'加载模型{self.model_path}')
        self.model = TreeModelTrainer.load_model(self.model_path)

        print('模型加载成功')
        return self

    def load_data(self,data_path: str) -> pd.DataFrame:

        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.txt'):
            df = pd.read_csv(data_path, sep='\t')
        else:
            raise ValueError(f"不支持的文件格式：{data_path}")

        print(f'加载:{len(df)}条记录')
        print(df)
        return df

    def extract_features(self,df: pd.DataFrame) -> pd.DataFrame:

        from features.sequence_features import extract_kmer_features_frequency
        from features.physicochemical_features import extract_physicochemical_features
        from features.tool_features import extract_tool_features

        print('\n提取序列特征')
        seq_df = extract_kmer_features_frequency(df['peptide'], k=2)
        aa_set = 'ACDEFGHIKLMNPQRSTVWY'
        dipeptides = [a + b for a, b in itertools.product(aa_set, repeat=2)]
        seq_features = pd.DataFrame(0,columns=dipeptides,index=seq_df.index)
        seq_features.update(df)

        print('\n提取理化特征')
        physico_features = extract_physicochemical_features(df['peptide'])

        print('\n提取工具预测特征')
        tool_features = extract_tool_features(df)

        # 合并所有特征
        all_features = pd.concat([seq_features, physico_features, tool_features], axis=1)
        from utils.data_utils import export_data
        export_data(all_features,'../data/processed/test.csv')
        all_features = all_features.drop('Affinity', axis=1)
        all_features = all_features.drop(['peptide_num', 'sample_name', 'peptide', 'best_allele'], axis=1)

        print(f"特征提取完成：{all_features.shape[1]}维")
        return all_features

    def predict(self,
                data_path: str,
                output_path: str,
                threshold: Optional[float] = None,
                save_proba: bool = True) -> pd.DataFrame:

        self.load_model()

        df = self.load_data(data_path)
        features = self.extract_features(df)

        print('\n进行预测')
        proba = self.model.predict_proba(features)[:,1]
        if threshold is None:
            threshold = 0.5

        predictions = (proba >= threshold).astype(int)

        result_df = df.copy()
        if save_proba:
            result_df['probability'] = proba
        result_df['confidence'] = pd.cut(
            proba,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['low', 'medium', 'high', 'very high']
        )

        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"\n结果已保存到：{output_path}")

        print(f"\n预测完成：{len(result_df)}条记录")
        print(f"正样本预测数：{sum(predictions)} ({sum(predictions) / len(predictions):.1%})")
        print(f"平均概率：{proba.mean():.3f}")

        return result_df


if __name__ == "__main__":
    # 测试
    pipeline = PredictionPipeline('../models/model.pkl')

    # 假设有一个测试文件
    test_file = '../data/raw/test_peptides.xlsx'
    if os.path.exists(test_file):
        results = pipeline.predict(test_file, '../data/predicted/predictions.csv')
        print(results.head())
    else:
        print("请准备测试文件")




