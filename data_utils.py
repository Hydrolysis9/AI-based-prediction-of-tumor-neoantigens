import re
from typing import Union,List

import pandas as pd

class IEDBDataloader:

    def __init__(self):
        # 标准氨基酸集合
        self.STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
        self.STANDARD_AA_PATTERN = r'^[ACDEFGHIKLMNPQRSTVWY]+$'

        # MHC血清型模式
        self.SEROTYPE_PATTERN = r'\bHLA-[ABC]w\d+\b'

        # 必需的列
        self.REQUIRED_COLUMNS = ['Epitope', 'MHC Restriction', 'Assay Description']

        # HLA标准化映射
        self.HLA_OFFICIAL_MAPPINGS = {
            # HLA-A 位点
            'A2': 'HLA-A*02:01', 'A0201': 'HLA-A*02:01', 'HLA-A2': 'HLA-A*02:01',
            'A1': 'HLA-A*01:01', 'A0101': 'HLA-A*01:01', 'HLA-A1': 'HLA-A*01:01',
            'A3': 'HLA-A*03:01', 'A0301': 'HLA-A*03:01', 'HLA-A3': 'HLA-A*03:01',
            'A11': 'HLA-A*11:01', 'A1101': 'HLA-A*11:01', 'HLA-A11': 'HLA-A*11:01',
            'A24': 'HLA-A*24:02', 'A2402': 'HLA-A*24:02', 'HLA-A24': 'HLA-A*24:02',

            # HLA-B 位点
            'B7': 'HLA-B*07:02', 'B0702': 'HLA-B*07:02', 'HLA-B7': 'HLA-B*07:02',
            'B8': 'HLA-B*08:01', 'B0801': 'HLA-B*08:01', 'HLA-B8': 'HLA-B*08:01',
            'B27': 'HLA-B*27:05', 'B2705': 'HLA-B*27:05', 'HLA-B27': 'HLA-B*27:05',
            'B35': 'HLA-B*35:01', 'B3501': 'HLA-B*35:01', 'HLA-B35': 'HLA-B*35:01',
            'B44': 'HLA-B*44:03', 'B4403': 'HLA-B*44:03', 'HLA-B44': 'HLA-B*44:03',
            'B51': 'HLA-B*51:01', 'B5101': 'HLA-B*51:01', 'HLA-B51': 'HLA-B*51:01',
            'B57': 'HLA-B*57:01', 'B5701': 'HLA-B*57:01', 'HLA-B57': 'HLA-B*57:01',

            # HLA-C 位点
            'C1': 'HLA-C*01:02', 'C0102': 'HLA-C*01:02', 'HLA-C1': 'HLA-C*01:02',
            'C2': 'HLA-C*02:02', 'C0202': 'HLA-C*02:02', 'HLA-C2': 'HLA-C*02:02',
            'C3': 'HLA-C*03:03', 'C0303': 'HLA-C*03:03', 'HLA-C3': 'HLA-C*03:03',
            'C4': 'HLA-C*04:01', 'C0401': 'HLA-C*04:01', 'HLA-C4': 'HLA-C*04:01',
            'C5': 'HLA-C*05:01', 'C0501': 'HLA-C*05:01', 'HLA-C5': 'HLA-C*05:01',
            'C6': 'HLA-C*06:02', 'C0602': 'HLA-C*06:02', 'HLA-C6': 'HLA-C*06:02',
            'C7': 'HLA-C*07:01', 'C0701': 'HLA-C*07:01', 'HLA-C7': 'HLA-C*07:01',
        }


        # 列名映射
        self.COLUMN_MAPPING = {
            'Epitope': 'peptide',
            'MHC Restriction': 'MHC_allele',
            'Assay Description': 'immunogenicity'
        }

    def load_iedb_data(self,filepath:str) -> pd.DataFrame:

        # 1.读取数据
        iedb_data = self._read_iedb_file(filepath)

        # 2.过滤标准氨基酸序列
        iedb_data = self._filter_standard_peptides(iedb_data,'Epitope')

        # 3.提取免疫原性结果
        iedb_data = self._extract_assay_results(iedb_data,'Assay Description')

        # 4.处理MHC分型
        iedb_data = self._process_mhc_restriction(iedb_data,'MHC Restriction')

        # 5.移除血清型数据
        iedb_data = self._remove_serotype(iedb_data,'MHC Restriction')

        # 6.标准化MHC格式
        iedb_data = self._standardize_mhc_format(iedb_data,'MHC Restriction')

        # 7.重命名列
        iedb_data = self._rename_columns(iedb_data)

        # 8.去重
        iedb_data = self._remove_duplicates(iedb_data)

        # 9.创建标签
        iedb_data = self._create_labels(iedb_data)

        # 10.显示导入结果
        self._display_final_summary(iedb_data)

        return iedb_data



    def _read_iedb_file(self,file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(
                file_path,
                sep='\t',
                skiprows=1,
                usecols=self.REQUIRED_COLUMNS
            )
            print(f'success:{file_path}')
            return data
        except Exception as e:
            print(f'fail:{e}')
            raise


    def _filter_standard_peptides(self,
                                  df: pd.DataFrame,
                                  column: str) -> pd.DataFrame:

        """
        filter peptides that contain standard amino acid

        :param df: input dataframe
        :param column: columns that needed processing
        :return: solved dataframe
        """


        #
        df[column] = df[column].str.extract(r'^([A-Z]+)')

        # filter standard aa
        mask = df[column].str.contains(
            self.STANDARD_AA_PATTERN,
            case=False,
            na=False
        )
        filter_df = df[mask].copy()
        print(f'标准氨基酸过滤:{len(df)} -> {len(filter_df)},已移除{len(df)-len(filter_df)}个非标准氨基酸肽段')

        return filter_df

    def _extract_assay_results(self,
                            df: pd.DataFrame,
                            column: str) -> pd.DataFrame:

        df[column] = df[column].str.extract(r'(Positive|Negative)')

        return df

    def _process_mhc_restriction(self,
                                 df: pd.DataFrame,
                                 column: str) -> pd.DataFrame:

        """处理MHC分型，提取可处理的MHC格式"""
        df[column] = df[column].str.extract(r'(H[^\s]*)')

        return df

    def _remove_serotype(self,
                         df: pd.DataFrame,
                         column: str) -> pd.DataFrame:

        """移除包含血清学命名的数据行"""
        mask  = ~df[column].str.contains(
            self.SEROTYPE_PATTERN,
            case=False,
            na=False
        )
        filtered_df = df[mask].copy()

        print(f"血清型过滤：{len(df)} -> {len(filtered_df)} (移除了{len(df) - len(filtered_df)}条血清型数据)")

        return filtered_df

    def _standardize_mhc_format(self,
                                df: pd.DataFrame,
                                column: str) -> pd.DataFrame:
        """
        标准化HLA等位基因格式

        处理三种情况：
        1. 高分辨率分型: HLA-A*02:01
        2. 低分辨率/通用型: HLA-A2, A0201
        3. 通用类别: HLA class I
        """

        def _standardize_single(allele):
            if pd.isna(allele):
                return allele

            allele_str = str(allele).strip()


            # 已经是高分辨率型
            if re.match(r'^HLA-([ABC])\*(\d{2,3}):(\d{2,3})$', allele_str):
                return allele_str

            # 低分辨率格式

            elif re.match(r"-(.*)$",allele_str):
                allele_str = re.search(r"-(.*)$", allele_str)
                allele_str = self.HLA_OFFICIAL_MAPPINGS.get(allele_str.group(1),allele_str)
                return allele_str

            # 通用类别
            if re.match(r'\bHLA\b', allele_str):
                return 'HLA-A*02:01'

            # 如果都无法匹配，返回原值并警告
            print(f"警告：无法标准化的MHC格式：{allele_str}")
            return allele_str

        df[column] = df[column].apply(_standardize_single)

        return df

    def _rename_columns(self,df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self.COLUMN_MAPPING)
        return df

    def _remove_duplicates(self,df: pd.DataFrame):
        """去除重复数据"""
        initial_len = len(df)
        df = df.drop_duplicates(subset=['peptide','MHC_allele'],keep="first")
        df = df.reset_index(drop=True) # 重设索引防止混乱

        print(f"去重处理：{initial_len} -> {len(df)} (移除了{initial_len-len(df)}条重复)")
        return df

    def _check_missing_values(self,df: pd.DataFrame):
        """检查缺失值"""
        total_na = df.isna().sum().sum()

        if total_na == 0:
            print("\n✅ 不存在缺失值")
        else:
            print(f"\n⚠️ 存在缺失值，总数：{total_na}")
            missing_rows = df[df.isna().any(axis=1)]
            print(f"包含缺失的行索引：{missing_rows.index.tolist()}")
            print("\n各列缺失情况：")
            print(df.isna().sum())


    def _create_labels(self,df: pd.DataFrame) -> pd.DataFrame:
        """创建二分类标签"""
        if 'immunogenicity' in df.columns:
            # 提取Positive/Negative标签
            df['label'] = df['immunogenicity'].str.contains('Positive', na=False).astype(int)

        return df

    def _display_final_summary(self, df: pd.DataFrame):
        """显示最终数据摘要"""
        print("\n" + "=" * 50)
        print("IEDB数据处理完成")
        print("=" * 50)
        print(f"最终数据量：{len(df)}")
        print(f"数据形状：{df.shape}")

        print("\n=== 列信息 ===")
        print(df.columns.tolist())

        print("\n=== 前4行数据 ===")
        print(df.head(4))

        print("\n=== 数据统计 ===")
        print(f"唯一肽段数：{df['peptide'].nunique()}")
        print(f"肽段长度范围：{df['peptide'].str.len().min()}-{df['peptide'].str.len().max()}")
        print(f"肽段平均长度：{df['peptide'].str.len().mean():.2f}")

        print(f"\n唯一MHC类型数：{df['MHC_allele'].nunique()}")
        print("MHC类型分布（前10）：")
        print(df['MHC_allele'].value_counts().head(10))

        # 检查缺失值
        self._check_missing_values(df)



# ========== 便携函数 ==========
def load_and_clean_iedb(filepath:str) -> pd.DataFrame:
    """
    一站式加载与清理IEDB数据
    """
    loader = IEDBDataloader()
    return loader.load_iedb_data(filepath)

def get_data_statistics(df: pd.DataFrame):
    """
    获取数据统计信息
    """
    stats = {'total_samples': len(df), 'unique_peptides': df['peptide'].nunique(),
             'unique_mhc': df['MHC_allele'].nunique(), 'peptide_length': {
            'min': int(df['peptide'].str.len().min()),
            'max': int(df['peptide'].str.len().max()),
            'mean': float(df['peptide'].str.len().mean()),
            'median': float(df['peptide'].str.len().median())
        }, 'missing_values': df.isna().sum().to_dict(), 'top_mhc': df['MHC_allele'].value_counts().head(10).to_dict()}

    # MHC top 10

    return stats


def export_data(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """
    导出数据
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'pickle':
        df.to_pickle(output_path)
    else:
        raise ValueError(f"不支持的导出格式：{format}")

    print(f"数据已导出到：{output_path}")





if __name__ == "__main__":
    # 测试完整流程
    print("测试IEDB数据完整处理流程...")

    # 创建测试数据
    test_data = """
Epitope	MHC Restriction	Assay Description
AAAAA (test)	HLA-A*02:01	Positive
BBBBB	HLA-B7	Positive-High
CCCCC	HLA-Cw4	Negative
DDDDD	HLA class I	Positive-Confirmed
EEEEE	HLA-A0201	Negative-Low
FFFFF	HLA-B*07:02 K66A	Positive
GGGGG	HLA-A*01:01	Negative"""

    # 保存测试文件
    test_file = "test_iedb.txt"
    with open(test_file, 'w') as f:
        f.write(test_data)

    # 测试加载和清洗
    try:
        df = load_and_clean_iedb(test_file)

        print("\n" + "=" * 50)
        print("数据统计信息：")
        stats = get_data_statistics(df)
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"测试失败：{e}")
    finally:
        import os

        os.remove(test_file)


