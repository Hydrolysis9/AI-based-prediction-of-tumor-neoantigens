from typing import Union, List
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PhysicochemicalFeatureExtractor:
    """理化特征提取器"""

    def __init__(self):
        # 标准氨基酸集合
        self.VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

        # 特征名称列表
        self.FEATURE_NAMES = [
            'length',
            'molecular_weight',
            'aromaticity',
            'instability_index',
            'isoelectric_point',
            'gravy'
        ]

    def is_valid_peptide(self,peptide: str) -> bool:
        """检查肽段是否只包含有效氨基酸字母"""
        return all(char in self.VALID_AA for char in peptide)

    def _compute_peptide_features(self,peptide: str) -> dict:
        """计算单个肽段的理化特征"""
        # 无效肽段返回空值
        if not self.is_valid_peptide(peptide):
            return {name: np.nan for name in self.FEATURE_NAMES}

        try:
            analyzer = ProteinAnalysis(peptide)
            return {
                'length': len(peptide),
                'molecular_weight': analyzer.molecular_weight(),
                'aromaticity': analyzer.aromaticity(),
                'instability_index': analyzer.instability_index(),
                'isoelectric_point': analyzer.isoelectric_point(),
                'gravy': analyzer.gravy()
            }
        except Exception as e:
            print(f"警告：肽段 '{peptide}' 计算失败 - {e}")
            return {name: np.nan for name in self.FEATURE_NAMES}

    def extract_all_features(self,
                             peptides: Union[List[str],pd.Series,str],
                             prefix: str = 'physico') -> pd.DataFrame:
        """
         提取理化特征（支持单个或批量）

         Parameters:
         -----------
         peptides : List[str] or pd.Series or str
             肽段序列（单个字符串或序列列表）
         prefix : str
             特征名前缀

         Returns:
         --------
         pd.DataFrame
             理化特征数据框
         """
        # 处理单个肽段输入
        if isinstance(peptides,str):
            peptide = [peptides]

        # 处理Series输入
        if isinstance(peptides,pd.Series):
            index = peptides.index
            peptide = peptides.tolist()
        else:
            index = range(len(peptides))

        # 批量计算
        features_list = [self._compute_peptide_features(pep) for pep in peptides]

        # 转换为DataFrame
        features_df = pd.DataFrame(features_list, index=index)

        # 添加前缀
        features_df.columns = [f"{prefix}{col}" for col in features_df.columns]

        return features_df


# ========== 便携函数 ==========
def  extract_physicochemical_features(peptides: Union[str, List[str], pd.Series],
                                   prefix: str = 'physico_') -> pd.DataFrame:
    """
        便捷函数：提取理化特征（支持单个肽段或批量）
    """
    extractor = PhysicochemicalFeatureExtractor()
    return extractor.extract_all_features(peptides,prefix)


# ============ 测试代码 ============

if __name__ == "__main__":
    # 测试单个肽段
    print("测试单个肽段：")
    single = extract_physicochemical_features("AAAAA")
    print(single)
    print()

    # 测试批量肽段
    print("测试批量肽段：")
    batch = extract_physicochemical_features(["AAAAA", "ACDEFGHIK", "INVALID"],prefix='')
    print(batch)





