import os.path
import pandas as pd
import numpy as np
from utils.data_utils import load_and_clean_iedb
from sklearn.model_selection import train_test_split
from utils.data_utils import export_data
from datetime import datetime

class TrainingPipeline:

    def __init__(self):
        self.result = {}

    def run(self,data_path: str):

        if not os.path.exists('../data/processed/processed_data.csv'):
            # 加载并清洗数据

            df = load_and_clean_iedb(data_path)

            # 提取特征
            from features.sequence_features import extract_kmer_features_frequency
            from features.physicochemical_features import extract_physicochemical_features
            from features.tool_features import extract_tool_features


            print('\n提取序列特征')
            seq_features = extract_kmer_features_frequency(df['peptide'],k=2)

            print('\n提取理化特征')
            physico_features = extract_physicochemical_features(df['peptide'])

            print('\n提取工具预测特征')
            tool_features = extract_tool_features(df)

            # 合并所有特征
            all_features = pd.concat([seq_features,physico_features, tool_features], axis=1)
            all_features = all_features.drop('Affinity',axis=1)
            all_features = all_features.drop(['peptide_num','sample_name','peptide','best_allele'],axis=1)

            export_data(all_features,'../data/processed/processed_data.csv')

            # 处理缺失值
            if not all_features.isna().sum().sum() == 0:
                print(f"\n缺失值数量：{all_features.isna().sum().sum()}")
                all_features = all_features.fillna(all_features.median())

        else:
            all_features = pd.read_csv('../data/processed/processed_data.csv')
            df = load_and_clean_iedb(data_path)


        # 检查特征与标签的相关性
        from utils.feature_utils import target_correct
        tool_features = target_correct(all_features, df['label'])

        # 划分数据集
        # 1.划分出训练集
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_features, df['label'],
            test_size=0.2, random_state=42
        )

        # 2.划分验证集和测试集
        X_val,X_test,y_val,y_test = train_test_split(
            X_temp,y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )

        export_data(X_test,'../data/external/X_test.csv')
        export_data(y_test, '../notebooks/IEDB_data/y_test.csv')

        print(f"\n训练集标签分布:")
        train_pos = sum(y_train == 1)
        train_neg = sum(y_train == 0)
        print(f"  阳性: {train_pos} ({train_pos / len(y_train):.1%})")
        print(f"  阴性: {train_neg} ({train_neg / len(y_train):.1%})")
        print(f"  比例: 1:{train_neg / train_pos:.2f}")

        print(f"\n验证集标签分布:")
        val_pos = sum(y_val == 1)
        val_neg = sum(y_val == 0)
        print(f"  阳性: {val_pos} ({val_pos / len(y_val):.1%})")
        print(f"  阴性: {val_neg} ({val_neg / len(y_val):.1%})")
        print(f"  比例: 1:{val_neg / val_neg:.2f}")

        print(f"\n测试集标签分布:")
        test_pos = sum(y_test == 1)
        test_neg = sum(y_test == 0)
        print(f"  阳性: {test_pos} ({test_pos / len(y_test):.1%})")
        print(f"  阴性: {test_neg} ({test_neg / len(y_test):.1%})")
        print(f"  比例: 1:{test_neg / test_pos:.2f}")


        # 使用验证集调参
        from models.hyperparameter_tuning import tune_with_validation
        best_params = tune_with_validation(
            X_train,y_train,
            X_val,y_val,
            n_trials=200,
            tune_item='roc_auc'
        )


        # 训练模型
        from models.train_model import TreeModelTrainer
        trainer = TreeModelTrainer(model_type='xgboost',**best_params,scale_pos_weight=train_neg/train_pos)
        model = trainer.train(X_train,y_train)
        importance = trainer.get_features_importance(feature_names=X_train.columns)
        print(importance)


        # 评估
        from models.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model,X_test, y_test)
        print("\n=== 评估结果 ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 主模型
        main_model_path = '../models/model.pkl'
        trainer.save_model(main_model_path,metadata={
            'timestamp': timestamp,
            'metrics' : metrics
        })
        # 副本
        if metrics.get('auc') > 0.675:
            versioned_path = f'../models/version/model_{timestamp}_{(metrics.get('auc'))}.pkl'
            trainer.save_model(versioned_path)

        print(f"\n📁 模型保存位置：")
        print(f"   - 最新模型：{main_model_path}")


        return model,metrics


