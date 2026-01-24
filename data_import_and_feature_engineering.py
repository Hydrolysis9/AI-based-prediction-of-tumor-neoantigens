"""训练集数据处理 """
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from Bio.SeqUtils import ProtParam
from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor,Class1ProcessingPredictor
import logging
import re
import os


#### IEDB数据导入 ###
def iedb_data_load(filepath):

    # 定义需要的列
    REQUIRE_COLUMN = ['Epitope','MHC Restriction','Assay Description']

    # 读取数据：跳过第一行，只读取有效列的数据
    iedb_data = pd.read_csv(filepath,sep = "\t",skiprows= 1,usecols=REQUIRE_COLUMN)

    # 提取序列名称
    """部分肽段数据含非标准氨基酸，在数据导入时过滤"""
    iedb_data['Epitope'] = iedb_data['Epitope'].str.extract(r'^([A-Z]+)')
    stanard_amino = r'^[ACDEFGHIKLMNPQRSTVWY]+$'
    mask = iedb_data['Epitope'].str.contains(stanard_amino,case=False, na=False)
    iedb_data = iedb_data[mask]

    # 提取免疫原性结果
    iedb_data['Assay Description'] = iedb_data['Assay Description'].str.extract(r'(Positive|Negative)')

    # 提取MHC结合序列
    """
    从IEDB得到的MHC数据格式不统一，先提取主要内容，方便后续格式的标准化
        目前发现的存在的格式：
        高分辨率分型: HLA-[A-C]*XX:XX
        低分辨率或通用型: HLA-[A-C]X
        通用类别: HLA class I
        带注释的突变型: 如HLA-A*02:01 K66A mutant
        血清学分型：如HLA-Cw4，血清学分型对应多个高分辨率等位基因，无法直接替换为单一高分辨率分型，无法精确处理，故删去包含血清学命名的数据行
        
    此处提取第一个空格前的所有内容
    """
    iedb_data['MHC Restriction'] = iedb_data['MHC Restriction'].str.extract(r'(H[^\s]*)')
     # 移除包含血清学命名的数据行
    serotype = r'\bHLA-[ABC]w\d+\b'
    mask1 = ~iedb_data['MHC Restriction'].str.contains(serotype, case=False, na=False) # 选取非血清名的数据行
    iedb_data = iedb_data[mask1]

    # 展示导入结果
    print(f'加载数据量：{len(iedb_data)}')
    print(f'列名：{iedb_data.columns.tolist()}')

    print("\n=== 前3行数据 ===")
    pd.set_option('display.max_columns', None)
    print(iedb_data.head(3))

    return iedb_data

# 数据导入
iedb_df = iedb_data_load(
    r'C:\Users\Hydrolysis\Desktop\TCGA_COAD_READ\tcell_table_export_1764489327.tsv\IEDB20251103.tsv')
pass


### 数据清洗与标签提取 ###

# 标准化HLA等位基因
def standard_mhc_format(allele):
    """规范化HLA数据"""
    HLA_OFFICIAL_MAPPINGS = {
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

    standard_allele = []

    """基于数据导入时提取的MHC格式进行标准化"""
    for alle in allele:
        if re.match(r'^HLA-([ABC])\*(\d{2,3}):(\d{2,3})$', str(alle)):  # 高分辨率分型: HLA-[A-c]*XX:XX
            standard_allele.append(alle)

        elif re.match(r"-(.*)$", str(alle)):  # 低分辨率或通用型: HLA-[A-c]X
            alle = re.search(r"-(.*)$", str(alle))
            alle = HLA_OFFICIAL_MAPPINGS.get(alle.group(1), alle)
            standard_allele.append(alle)

        elif re.match(r'\bHLA\b', str(alle)):  # 通用类别:HLA Class I，替换为最通用等位基因HLA-A*02:01
            standard_allele.append('HLA-A*02:01')

    return standard_allele

std_allelle = standard_mhc_format(iedb_df['MHC Restriction'].tolist())
iedb_df['MHC Restriction'] = std_allelle

def iedb_data_clean(iedb_df):

    # 列名映射字典
    column_mapping = {
        'Epitope':'peptide',  # 肽段序列列
        'MHC Restriction':'MHC_allele',  # MHC等位基因列
        'Assay Description':'immunogenicity' # 免疫原性结果列
    }
    # 重命名列
    iedb_df = iedb_df.rename(columns=column_mapping)

    # 去重
    iedb_df = iedb_df.drop_duplicates(subset=['peptide','MHC_allele'],keep="first")

    # 检查是否有缺失值
    if iedb_df.isna().sum().sum() == 0:
        print('\n不存在缺失值')
    else:
        missing_rows = iedb_df[iedb_df.isna().any(axis = 1)]
        print('\n存在缺失值')
        print('包含缺失的行索引：')
        print(missing_rows.index.tolist())

    # 免疫标签提取
    if 'immunogenicity' in iedb_df.columns:
        # 二分类标签
       iedb_df['label'] = iedb_df['immunogenicity'].str.contains('Positive').astype(int)

    print('\n=== 显示前三列 ===')
    print(iedb_df.shape)
    print(iedb_df.head(3))

    return iedb_df


# 清洗数据
cleaned_iedb =iedb_data_clean(iedb_df)



### 特征工程 ###

# 计算肽段的序列特征
def sequence_features(peptide,k=2):
    """k-mer编码会因为肽段长度不同而产生特征维度不一致的特征，由于技术有限，这里利用固定长度k-mer（k=2）进行频次统计"""

    # CountVectorizer方法生成k-mers
    vectorizer = CountVectorizer(analyzer='char',
                                 ngram_range=(k, k),
                                 lowercase=False,
                                 )

    kmer_matrix = vectorizer.fit_transform(peptide)
    # 与特征名合并
    kmer = pd.DataFrame(kmer_matrix.toarray(),
                        columns=vectorizer.get_feature_names_out(),
                        index=peptide.index)


    return kmer

kmer = sequence_features(cleaned_iedb['peptide'])
print(f'得到{len(kmer)}个序列特征结果')



# 计算肽段的理化特征
def is_valid_peptide(peptide):
    """检查肽段是否只包含有效氨基酸字母"""
    valid_char = set('ACDEFGHIKLMNPQRSTVWY')
    return all(char in valid_char for char in peptide)

def biochemical_features(peptide):
    """跳过无效肽段"""
    if not is_valid_peptide(peptide):
        return {
                'length':len(peptide),
                'molecular_weight':None, # 分子量
                'aromaticity':None,   #芳香性
                'instability_index':None,   # 不稳定指数
                'isoelectric_point': None,  # 等电点
                'gravy': None     # 疏水性
                }
    else:
    # 创建分析对象
        analyzer = ProtParam.ProteinAnalysis(peptide)

        features = {
                    'length':len(peptide),
                    'molecular_weight':analyzer.molecular_weight(), # 分子量
                    'aromaticity':analyzer.aromaticity(),   #芳香性
                    'instability_index':analyzer.instability_index(),   # 不稳定指数
                    'isoelectric_point': analyzer.isoelectric_point(),  # 等电点
                    'gravy': analyzer.gravy() #疏水性
                    }
        return features

bio_features = cleaned_iedb['peptide'].apply(biochemical_features).apply(pd.Series)
print(f'得到{len(bio_features)}个理化预测结果')




# 计算免疫特征

# 设置日志级别减少输出
logging.getLogger('mhcflurry').setLevel(logging.WARNING)

def setup_mhcflurry_predictors():
    """设置MHCflurry预测器"""
    print('\n=== 正在加载MHCflurry模型 ===')
    affinity_predictor = Class1AffinityPredictor.load()
    presentation_predictor = Class1PresentationPredictor.load()
    mhc_processing_predictor = Class1ProcessingPredictor.load()
    print('\n=== 模型加载完毕 ===')

    return affinity_predictor,presentation_predictor,mhc_processing_predictor

def mhc_processing_predictor(peptide,processing_predictor):
    """计算肽段加工特征:抗原加工预测得分"""
    res = processing_predictor.predict(peptide)
    res = pd.DataFrame(res,columns=['PCP'])
    return res


def mhc_affinity_predictor(peptide,allele,affinity_predictor):
    """计算MHC结合亲和力特征：IC50"""

    # 提取值为列表
    peptide = peptide.tolist()
    allele = allele.tolist()

    # 将HLA数据规范化
    allele = standard_mhc_format(allele)

    try:
        # 预测结合亲和力
        predictions = affinity_predictor.predict(peptides = peptide,alleles = allele)

        if len(peptide) == 0:
            return None

        return predictions

    except Exception as e:
        print(f"MHC亲和力预测错误 {peptide}-{allele}: {e}")
        return None

def mhc_presentation_predictor(df,presentation_predictor,peptide='peptide',allele='mhc_allele'):
    """计算MHC呈递分数"""
    allele_group = df.groupby(allele)

    res = []

    for alle,group_df in allele_group:

        alle = standard_mhc_format([alle])
        peptides = group_df[peptide].tolist()

        # 计算呈递分数
        print(f'进行呈递分数预测：{alle}-{peptides}')

        predictions = presentation_predictor.predict(peptides = peptides,alleles = alle )
        res.append(predictions)

    res = pd.concat(res,ignore_index=True)
    return res


# 运行免疫特征预测模型
affinity_predictor, presentation_predictor,processing_predictor = setup_mhcflurry_predictors()
mhc = mhc_affinity_predictor(cleaned_iedb['peptide'],cleaned_iedb['MHC_allele'],affinity_predictor)     # IC50
mhc_pres = mhc_presentation_predictor(cleaned_iedb,
                                      presentation_predictor=presentation_predictor,
                                      peptide='peptide',
                                      allele='MHC_allele')                                              # 呈递预测
mhc_proc = mhc_processing_predictor(cleaned_iedb['peptide'],processing_predictor)
mhc = pd.DataFrame(mhc,index=cleaned_iedb.index,columns=['IC50'])

print(f'得到{len(mhc)}个结合亲和力结果')
print(f'得到{len(mhc_pres)}个呈递预测结果')

# 合并各特征结果
merge_iedb = pd.concat([cleaned_iedb,bio_features,mhc,kmer,mhc_proc],axis=1)
merge_iedb = pd.merge(merge_iedb,mhc_pres,on='peptide')
merge_iedb = merge_iedb.query('MHC_allele == best_allele')

print(merge_iedb.shape)
print(f'含有的缺失值：{merge_iedb.isna().sum().sum()}')


# 确保你的矩阵是DataFrame
# 如果matrix是numpy数组或稀疏矩阵：
# df = pd.DataFrame(matrix, columns=feature_names, index=row_names)

def export_to_csv(df, folder_path, filename, index=False, encoding='utf-8'):
    """
    将DataFrame导出为CSV文件

    参数:
        df: 要导出的DataFrame
        folder_path: 文件夹路径
        filename: 文件名（不含.csv）
        index: 是否保留索引
        encoding: 文件编码
    """

    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)

    # 构建完整路径
    filepath = os.path.join(folder_path, f"{filename}.csv")

    # 导出到CSV
    df.to_csv(filepath, index=index, encoding=encoding)

    print(f"✅ 文件已保存到: {filepath}")
    print(f"📊 数据形状: {df.shape}")
    print(f"💾 文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

    return filepath


# 使用示例
output_folder = r"C:\Users\Hydrolysis\Desktop\TCGA_COAD_READ\tcell_table_export_1764489327.tsv"
export_to_csv(merge_iedb, output_folder, "primary_feature_matrix")
export_to_csv(cleaned_iedb,output_folder,'cleaned')


