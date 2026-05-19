import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
class KMerFeatureExtractor:

    def __init__(self,k=2,use_frequency=True):
        """
        :param k:int
            kmer大小
        :param use_frequency:bool
            True: 使用频率（除以总长度）
            False: 使用原始计数
        """

        self.k = k
        self.use_frequency = use_frequency
        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(k,k),
            lowercase=False
        )
        self.feature_names = None

    def fit(self,peptides: pd.Series):
        """拟合vectorizer"""
        self.vectorizer.fit(peptides)
        self.feature_names = self.vectorizer.get_feature_names_out()

        return self

    def transform(self,peptides: pd.Series) -> pd.DataFrame:
        # 获取原始计数
        kmer_matrix = self.vectorizer.transform(peptides)
        kmer_df = pd.DataFrame(
            kmer_matrix.toarray(),
            columns=self.feature_names,
            index=peptides.index
        )

        if self.use_frequency:
            # 转化为频率（除以肽段长度）
            # 对于k-mer，每个肽段有 (len(pep) - k + 1) 个k-mer
            peptide_lengths = peptides.str.len()
            n_kmers = peptide_lengths - self.k + 1

            # 除以Kmer总数得到频率
            kmer_df = kmer_df.div(n_kmers,axis=0)

        return kmer_df

    def fit_transform(self,peptide: pd.Series) -> pd.DataFrame:
        """拟合并转化"""
        self.fit(peptide)
        return self.transform(peptide)

def get_feature_summary(kmer_df: pd.DataFrame) -> dict:
    """获取特征摘要"""
    summary = {
        'n_features': kmer_df.shape[1],
        'sparsity': 1 - kmer_df[(kmer_df > 0)].sum().sum() / (kmer_df.shape[0] * kmer_df.shape[1]),
        'non_zero_per_row': kmer_df[(kmer_df > 0)].sum(axis=1).describe().to_dict(),
        'most_common_kmers': kmer_df.mean().sort_values(ascending=False).head(10).to_dict()
    }
    return summary


def extract_kmer_features_frequency(peptides: pd.Series, k=2) -> pd.DataFrame:
    """
    提取k-mer频率特征
    """
    extractor = KMerFeatureExtractor(k=k, use_frequency=True)
    return extractor.fit_transform(peptides)


def extract_kmer_features_with_length(peptides: pd.Series, k=2) -> pd.DataFrame:
    """
    提取k-mer计数
    """
    extractor = KMerFeatureExtractor(k=k, use_frequency=False)
    kmer_df = extractor.fit_transform(peptides)


    return kmer_df

def extract_kmer_features_normalized(peptides: pd.Series,k=2) -> pd.DataFrame:
    """
    提取标准化后的K-mer特征
    """
    from sklearn.preprocessing import StandardScaler

    # 提取频率特征
    extractor = KMerFeatureExtractor(k=k,use_frequency=True)
    kmer_df = extractor.fit_transform(peptides)

    # 标准化
    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_df)
    kmer_scaled_df = pd.DataFrame(
        kmer_scaled,columns=kmer_df.columns,index=kmer_df.index
    )

    return kmer_scaled_df


# ============ 测试和比较 ============

def compare_kmer_methods(peptides: pd.Series):
    """
    比较不同k-mer提取方法
    """
    results = {}

    # 原始计数
    kmer_count = extract_kmer_features_with_length(peptides.copy())
    results['count'] = get_feature_summary(kmer_count)

    # 频率
    kmer_freq = extract_kmer_features_frequency(peptides.copy())
    results['frequency'] = get_feature_summary(kmer_freq)

    # 标准化
    kmer_norm = extract_kmer_features_normalized(peptides.copy())
    results['normalized'] = get_feature_summary(kmer_norm)

    return pd.DataFrame(results).T

# ==========主函数==========
def extract_sequence_feature(peptide: pd.Series,method: str='frequency',k: int=2) -> pd.DataFrame:
    """
    统一的序列特征提取接口

    Parameters:
    -----------
    peptides : pd.Series
        肽段序列
    method : str
        'frequency': 频率特征（推荐）
        'count': 计数 + 长度
        'normalized': 标准化后的频率
    k : int
        k-mer大小
    """
    if method == 'frequency':
        return extract_kmer_features_frequency(peptide,k)
    elif method == 'count':
        return extract_kmer_features_with_length(peptide,k)
    elif method == 'normalized':
        return  extract_kmer_features_normalized(peptide,k)
    else:
        raise ValueError(f'unknown method:{method},available method:frequency,count,normalized')

if __name__ == "__main__":
    # 测试代码
    test_peptides = pd.Series([
        "AAAAA",  # 长度5
        "ACDEFGHIK",  # 长度9
        "LMWVIP",  # 长度6
        "Y" * 15,  # 长度15
    ])

    res = extract_sequence_feature(test_peptides)
    print(res)