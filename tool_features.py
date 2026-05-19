"""
工具预测特征模块
功能：集成MHCflurry、NetChop、NetMHCpan等工具的预测结果
"""
from typing import Union,List
import numpy as np
from mhcflurry import Class1AffinityPredictor,Class1ProcessingPredictor,Class1PresentationPredictor
import pandas as pd
import subprocess
import os
import re
from tqdm import tqdm

# ========== mhcflurry 预测器 ==========

class MHCflurryPredictor:
    """mhcflurry预测器封装"""

    def __init__(self):
        self.affinity_predictor = None
        self.presentation_predictor = None
        self.processing_predictor = None
        self._load_predictors()

    def _load_predictors(self):
        try:
            print('\n=== 正在加载mhcflurry模型 ===')
            self.affinity_predictor = Class1AffinityPredictor.load()
            self.presentation_predictor = Class1PresentationPredictor.load()
            self.processing_predictor = Class1ProcessingPredictor.load()
            print('=== 模型加载完毕 ===\n')
        except ImportError:
            print("警告：mhcflurry未安装，请先安装：pip install mhcflurry")
        except Exception as e:
            print(f"警告：mhcflurry加载失败 - {e}")

    def predict_processing(self,peptides: Union[List[str],str]) -> pd.DataFrame:
        """预测抗原加工得分"""
        if self.processing_predictor is None:
            return pd.DataFrame({'PCP': [np.nan] * len(peptides) if isinstance(peptides, list) else [np.nan]})

        if isinstance(peptides,str):
            peptides = [peptides]

        try:
            scores = self.processing_predictor.predict(peptides)
            return pd.DataFrame({"peptides_processing_score":scores})
        except Exception as e:
            print(f"加工预测失败: {e}")
            return pd.DataFrame({'peptides_processing_score': [np.nan] * len(peptides)})

    def predict_affinity(self,peptides: Union[List[str],str],alleles: Union[List[str],str]) -> pd.DataFrame:
        """预测结合亲和力"""
        if self.affinity_predictor is None:
            return pd.DataFrame({'Affinity': [np.nan] * len(peptides)})

        # 统一为列表格式
        if isinstance(peptides, str):
            peptides = [peptides]
        if isinstance(alleles, str):
            alleles = [alleles]

        try:
            predictions = self.affinity_predictor.predict(
                peptides=peptides,
                alleles=alleles
            )

            # 转化为Dataframe
            result_df = pd.DataFrame({
                'peptide': peptides,
                'allele': alleles,
                'Affinity': predictions
        })
            return result_df['Affinity']
        except Exception as e:
            print(f"MHC亲和力预测错误: {e}")
            return pd.DataFrame({'Affinity': [np.nan] * len(peptides)})

    def predict_presentation(self,
                             df: pd.DataFrame,
                             peptide_col: str = 'peptide',
                             allele_col: str = 'MHC_allele') -> pd.DataFrame:
        """预测MHC呈递分数（按allele分组处理）"""
        if self.presentation_predictor is None:
            return pd.DataFrame(index=df.index).assign(presentation_score=np.nan)

        res = []

        for allele,group_df in df.groupby(allele_col):
            peptides = group_df[peptide_col].tolist()
            original_indices = group_df.index.tolist()
            print(f'计算呈递分数：{allele} - {len(peptides)}条肽段')

            try:
                predictions = self.presentation_predictor.predict(
                    peptides=peptides,
                    alleles=[allele]
                )
                predictions.index = original_indices
                res.append(predictions)


            except Exception as e:
                print(f"预测失败: {e}")
                # 失败时添加空值
                return pd.DataFrame({'presentation_score': [np.nan] * len(peptides)})

        res = pd.concat(res)
        res = res.sort_index()

        return pd.DataFrame(res)



# ========== NetChop 预测器 ==========
class NetChopPredictor:
    """NetChop预测器封装（基于WSL）"""
    def __init__(self,wsl_netchop_path: str = "/root/netchop-3.1/netchop"):
        self.netchop_path = wsl_netchop_path
        self.temp_files = []

    def create_fasta(self,
                    peptides: Union[List[str],pd.Series],
                    fasta_file: str = 'peptide.fasta') -> str:
        """将肽段列表转换为FASTA文件"""
        if isinstance(peptides, pd.Series):
            peptides = peptides.tolist()

        with open(fasta_file, 'w') as f:
            for i, peptide in enumerate(peptides, 1):
                if pd.notna(peptide):
                    f.write(f">peptide_{i}\n{peptide}\n")

        print(f"FASTA文件创建完成：{fasta_file} ({len(peptides)}条肽段)")
        self.temp_files.append(fasta_file)
        return fasta_file

    def run_netchop(self,
                    fasta_file: str,
                    output_file: str = 'netchop_output.txt',
                    clean_up: bool = True) -> bool:
        """运行NetChop预测蛋白酶体切割"""

        wsl_command = f"{self.netchop_path} {fasta_file} > {output_file}"

        try:
            result = subprocess.run(['wsl', 'bash', '-c', wsl_command],
                                    capture_output=True, text=True)

            if result.returncode == 0:
                print(f"NetChop执行成功！输出保存到: {output_file}")
                self.temp_files.append(output_file)

                # 预览结果
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        preview = f.readlines()[:5]
                    print("输出预览（前5行）:")
                    for line in preview:
                        print(f"  {line.strip()}")
                return True
            else:
                print(f"NetChop执行失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"NetChop调用失败: {e}")
            return False

    def parse_output(self,output_file: str) -> pd.DataFrame:
        """解析NetChop输出文件"""
        try:
            with open(output_file,'r') as f:
                content = f.read()

            # 匹配NetChop输出格式
            pattern = r'pos\s+AA\s+C\s+score\s+Ident\n--------------------------------------\n(.*?)\n--------------------------------------'
            blocks = re.findall(pattern, content, re.DOTALL)

            data = []
            for block in blocks:
                lines = block.strip().split('\n')
                peptide_name = None
                scores = []

                for line in lines:
                    match = re.match(r'\s*\d+\s+[A-Z]\s+[A-Z\.]\s+(\d+\.\d+)\s+(\w+)', line.strip())
                    if match:
                        score = float(match.group(1))
                        peptide = match.group(2)
                        scores.append(score)
                        if peptide_name is None:
                            peptide_name = peptide

                if peptide_name and scores:
                    data.append({'peptide': peptide_name, 'scores': scores})

            return pd.DataFrame(data)

        except Exception as e:
            print(f"解析NetChop输出失败: {e}")
            return pd.DataFrame()


    def extract_chop_feature(self,
                             df: pd.DataFrame,
                             scores_col: str = "scores",
                             threshold: float = 0.5) -> pd.DataFrame:

        """
        从切割分数提取特征

        参数：
            df: 包含scores列（数组）的DataFrame
            scores_col: 数组列的名称
            threshold: 切割阈值
        """

        def calculate_max_chop_length(scores):
            """计算连续最大切割长度"""
            if not isinstance(scores,(list,np.ndarray)):
                return {'max_chop_length': np.nan}

            max_length = 0
            current_length = 0
            for score in scores:
                if score > threshold:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 0

            return {'max_chop_length': max_length}

        features = df[scores_col].apply(calculate_max_chop_length)
        return pd.DataFrame(features.tolist())

    def predict_and_extract(self,
                            peptides: Union[List[str],pd.Series],
                            threshold: float = 0.5,
                            clean_up: bool = True) -> pd.DataFrame:
        # 创建临时文件
        fasta_file = 'temp_netchop.fasta'
        output_file = 'temp_netchop_out.txt'

        # 运行预测
        self.create_fasta(peptides,fasta_file)
        success = self.run_netchop(fasta_file,output_file)

        if success:
            # 解析结果
            res = self.parse_output(output_file)

            if not res.empty:
                # 提取特征
                features = self.extract_chop_feature(res)

        # 清理临时文件
        if clean_up:
            for f in [fasta_file, output_file]:
                if os.path.exists(f):
                    os.remove(f)

        return features if success else pd.DataFrame()


# ========== NetMHCpan 预测器 ==========

class NetMHCpanPredictor:
    """NetMHCpan预测器封装（基于WSL）"""

    def __init__(self,netmhcpan_path: str = "/root/netMHCpan-4.2/netMHCpan"):
        self.netmhcpan_path = netmhcpan_path
        self.temp_files = []

    def run_netmhcpan(self,df: pd.DataFrame,peptide_col: str = 'peptide',allele_col: str = 'MHC_allele') -> pd.DataFrame:

        # 准备结果Series
        netmhcpan_rank = pd.Series(index=df.index, dtype=float)

        # 准备数据
        df_copy = df[[peptide_col,allele_col]].copy()

        # 删除allele中的*
        df_copy['allele'] = df_copy[allele_col].str.replace('*', '', regex=False)

        # 按allele分组处理
        for allele, group in df_copy.groupby('allele'):
            print(f"处理NetMHCpan: {allele} ({len(group)}条肽段)")

            # 保存原始索引
            indices = group.index

            # 创建临时文件
            temp_peptide = 'tmp_peptide.txt'
            temp_output = 'NetMHCpan_out.xls'

            try:
                # 写入肽段
                group[peptide_col].to_csv(temp_peptide, index=False, header=False)

                # 运行命令
                cmd = f"{self.netmhcpan_path} -a {allele} -p {temp_peptide} -xls"
                result = subprocess.run(['wsl', 'bash', '-c', cmd],
                                        capture_output=True, text=True)

                if result.returncode == 0 and os.path.exists(temp_output):
                    # 读取结果
                    required_columns = ['Peptide', 'Rank']
                    res = pd.read_csv(temp_output, sep='\t', skiprows=1,
                                      usecols=required_columns)

                    # 创建肽段到分数的映射
                    rank_map = dict(zip(res['Peptide'], res['Rank']))

                    # 按原始顺序赋值
                    for idx, pep in zip(indices, group[peptide_col]):
                        if pep in rank_map:
                            netmhcpan_rank.loc[idx] = rank_map[pep]
                        else:
                            netmhcpan_rank.loc[idx] = np.nan

                    print(f"  ✅ 成功")
                else:
                    print(f"  ❌ 失败")
                    netmhcpan_rank.loc[indices] = np.nan

            except Exception as e:
                print(f"  ❌ 错误: {e}")
                netmhcpan_rank.loc[indices] = np.nan

            finally:
                # 清理临时文件
                for f in [temp_peptide, temp_output]:
                    if os.path.exists(f):
                        os.remove(f)

        return pd.DataFrame({'netMHCpan_%Rank': netmhcpan_rank})

# ========== NetMHCstabpan 预测器 ==========

class NetMHCstabpanPredictor:

    def __init__(self,netmhcstabpan_path: str = "/root/netMHCstabpan-1.0/netMHCstabpan"):
        self.netmhcstabpan_path = netmhcstabpan_path
        self.temp_files = []

    def run_netmhcstabpan(self,df: pd.DataFrame,peptide_col: str = 'peptide',allele_col: str = 'MHC_allele') -> pd.DataFrame:

        # 准备结果Series
        netmhcstabpan_rank = []

        # 准备数据
        df_copy = df[[peptide_col,allele_col]].copy()

        # 删除allele中的*
        df_copy['allele'] = df_copy[allele_col].str.replace('*', '', regex=False)
        peptides = df_copy[peptide_col].tolist()
        alleles = df_copy['allele'].tolist()

        # 按allele分组处理
        for peptide, allele in zip(peptides, alleles):


            # 创建临时文件
            temp_peptide = 'tmp_peptide.txt'
            temp_output = 'NetMHCstabpan.xls'

            try:
                # 写入肽段
                peptide_txt = pd.DataFrame([peptide])
                peptide_txt.to_csv(temp_peptide, index=False, header=False)

                # 运行命令
                cmd = f"{self.netmhcstabpan_path} -a {allele} -p {temp_peptide} -s 0 -xls"
                result = subprocess.run(['wsl', 'bash', '-c', cmd],
                                        capture_output=True, text=True)

                if result.returncode == 0 and os.path.exists(temp_output):
                    # 读取结果

                    res = pd.read_csv(temp_output, sep='\t', skiprows=1,
                                      usecols=['Rank'])
                    netmhcstabpan_rank.append(res['Rank'].values[0])

                else:
                    print(f" ❌ 失败: {peptide} - {allele}")
                    netmhcstabpan_rank.append(np.nan)

            except Exception as e:
                print(f"  ❌ 错误: {e}")
                netmhcstabpan_rank.append(np.nan)

            finally:
                # 清理临时文件
                for f in [temp_peptide, temp_output]:
                    if os.path.exists(f):
                        os.remove(f)

        return pd.DataFrame({'netMHCstabpan_%Rank': netmhcstabpan_rank})



# ========== 统一的提取器 ==========
class ToolFeatureExtractor:

    def __init__(self,use_mhcflurry=True, use_netchop=True, use_netmhcpan=True,use_netmhcstabpan=True):
        self.use_mhcflurry = use_mhcflurry
        self.use_netchop = use_netchop
        self.use_netmhcpan = use_netmhcpan
        self.use_netmhcstabpan = use_netmhcstabpan

        # 初始化各预测器
        self.mhcflurry = MHCflurryPredictor() if use_mhcflurry else None
        self.netchop = NetChopPredictor() if use_netchop else None
        self.netmhcpan = NetMHCpanPredictor() if use_netmhcpan else None
        self.netmhcstabpan = NetMHCstabpanPredictor() if use_netmhcstabpan else None

    def extract_all_features(self,
                             df: pd.DataFrame,
                             peptide_col: str = 'peptide',
                             allele_col: str = 'MHC_allele') -> pd.DataFrame:
        """
        提取所有工具预测特征 - 所有特征都保持原始索引
        """
        # 初始化特征DataFrame，使用原始索引
        all_features = pd.DataFrame(index=df.index)

        # 1.MHCflurry 加工特征
        if self.mhcflurry:
            processing_features = self.mhcflurry.predict_processing(
                df[peptide_col].tolist(),
            )
            all_features = pd.concat([all_features, processing_features], axis=1)

        # 2. MHCflurry 呈递特征
        if self.mhcflurry:
            presentation_features = self.mhcflurry.predict_presentation(
                df,
                peptide_col=peptide_col,
                allele_col=allele_col
            )
            all_features = pd.concat([all_features, presentation_features], axis=1)

        # 3. MHCflurry 结合特征
        if self.mhcflurry:
            affinity_features = self.mhcflurry.predict_affinity(
                df[peptide_col].tolist(),
                df[allele_col].tolist()
            )
            all_features = pd.concat([all_features, affinity_features], axis=1)

        # 4. NetChop 特征
        if self.netchop:
            print("\n--- 提取NetChop切割特征 ---")
            netchop_features = self.netchop.predict_and_extract(
                df[peptide_col],
            )
            if not netchop_features.empty:
                all_features = pd.concat([all_features, netchop_features], axis=1)

        # 5. NetMHCpan 特征
        if self.netmhcpan and allele_col in df.columns:
            print("\n--- 提取NetMHCpan结合特征 ---")
            netmhcpan_features = self.netmhcpan.run_netmhcpan(
                df,
                peptide_col=peptide_col,
                allele_col=allele_col
            )
            all_features = pd.concat([all_features, netmhcpan_features], axis=1)

        if self.netmhcstabpan:
        # 6. NetMHCstabpan 特征
            print("\n--- 提取NetMHCstabpan稳定性特征")
            netmhcstabpan_features = self.netmhcstabpan.run_netmhcstabpan(
                df,
                peptide_col=peptide_col,
                allele_col=allele_col
            )
            all_features = pd.concat([all_features,netmhcstabpan_features],axis=1)

        return all_features


# ============ 便捷函数 ============

def extract_tool_features(df: pd.DataFrame,
                          peptide_col: str = 'peptide',
                          allele_col: str = 'MHC_allele',
                          features: List[str] = ['processing','presentation', 'netchop', 'netmhcpan','netmhcstabpan']) -> pd.DataFrame:
    """
    便捷函数：提取工具预测特征 - 返回带原始索引的特征DataFrame

    使用示例：
    ```python
    # 原始数据
    cleaned_df = load_and_clean_iedb("data/raw/iedb_data.tsv")

    # 提取工具特征（保持索引）
    tool_features = extract_tool_features(cleaned_df)

    # 最后统一合并所有特征
    final_df = pd.concat([cleaned_df, tool_features], axis=1)
    ```
    """
    use_mhcflurry = any(f in features for f in ['processing', 'presentation'])
    use_netchop = 'netchop' in features
    use_netmhcpan = 'netmhcpan' in features
    use_netmhcstabpan = 'netmhcstabpan' in features

    extractor = ToolFeatureExtractor(
        use_mhcflurry=use_mhcflurry,
        use_netchop=use_netchop,
        use_netmhcpan=use_netmhcpan,
        use_netmhcstabpan=use_netmhcstabpan
    )

    return extractor.extract_all_features(df, peptide_col, allele_col)


# ============ 测试代码 ============

if __name__ == "__main__":
    # 创建测试数据（带自定义索引）
    test_df = pd.DataFrame({
        'peptide': ['APIWPYELLY', 'LIVDSSLCDL', 'SIINFEKL', 'CMTWNAMNL'],
        'MHC_allele': ['HLA-A*02:01', 'HLA-A*24:02','HLA-B*35:01', 'HLA-A*02:01']
    })  # 自定义索引

    print("=" * 50)
    print("测试工具特征提取（保持索引）")
    print("=" * 50)
    print("原始数据：")
    print(test_df)

    # 测试提取特征
    tool_features = extract_tool_features(
        test_df,
        features=['processing','presentation','netchop','netmhcpan','netmhcstabpan']
    )

    print("\n提取的特征（带原始索引）：")
    print(tool_features)

    print("\n索引是否一致：", test_df.index.equals(tool_features.index))

    # 最后统一合并
    final_df = pd.concat([test_df, tool_features], axis=1)
    print("\n最终合并结果：")
    print(final_df)








