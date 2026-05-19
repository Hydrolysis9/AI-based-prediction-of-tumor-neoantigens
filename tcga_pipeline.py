import os
import shutil
import gzip
from pathlib import Path
import pandas as pd
import re
from utils.data_utils import export_data
from pyensembl import EnsemblRelease
import logging
import ast


def extract_tcga_files(source_folder,target_folder,end_with):
    """
    :param source_folder: TCGA下载的数据路径
    :param target_folder: 提取到文件的存放位置路径
    :param end_with: 文件形式（.tsv : RNA_seq .gz: maf压缩文件）
    :return: 将下载数据提取到预处理文件夹
    暂时仅支持绝对路径
    """

    os.makedirs(target_folder,exist_ok=True)

    total_files = 0

    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder,item)

        if os.path.isdir(item_path):
            print(f"处理文件{item}")

            for file in os.listdir(item_path):
                if file.endswith(end_with):
                    file_path = os.path.join(item_path,file)
                    new_filename = f"{file}"
                    target_path = os.path.join(target_folder,new_filename)

                    shutil.copy2(file_path,target_path)
                    total_files += 1
                    print(f"  已复制: {file} -> {target_path}")

    if end_with == '.tsv':
        print(f"\n✅ 完成！共提取 {total_files} 个文件到 {target_folder}")

    elif end_with == '.gz':
        gz_folder = Path(target_folder)
        for gz_file in gz_folder.glob('*.gz'):
            outputfile = Path(target_folder) / gz_file.stem
            with gzip.open(gz_file,'rb') as gz:
                outputfile.write_bytes(gz.read())

            gz_file.unlink()

        print(f"\n✅ 完成！共提取 {total_files} 个解压文件到 {target_folder}")
    else:
        print('无效文件格式')


def extract_tpm(input_folder: str,output_folder: str) -> pd.DataFrame:
    """
    :param input_folder: 提取的tsv文件的存放路径
    :param output_folder: 提取出的matrix的存放路径
    :return: 一个matrix，存放所有样本的表达数据
    暂时仅支持绝对路径
    """
    tsv_files = list(Path(input_folder).glob("*.tsv"))
    print(f'找到 {len(tsv_files)} 个TSV文件')

    res = []
    col_names = ['gene_id', 'gene_name', 'gene_type', 'unstranded',
                 'stranded_first', 'stranded_second', 'tpm_unstranded',
                 'fpkm_unstranded', 'fpkm_uq_unstranded']

    for file in tsv_files:
        sample = file.name

        df = (pd.read_csv(file, sep='\t', skiprows=5, names=col_names)
              .query("gene_type == 'protein_coding'")
              .filter(['gene_name', 'tpm_unstranded'])
              .rename(columns={'tpm_unstranded': sample}))

        res.append(df)

    res_df = pd.concat(res, axis=1).loc[:, ~pd.concat(res, axis=1).columns.duplicated()]

    export_data(res_df,output_folder)
    return res_df


def map_mutation_expression(input_folder: str, sample_sheet: str):
    """
    :param input_folder: 提取的突变数据的存放路径
    :param sample_sheet: TCGA上下载的 gdc_sample_sheet.tsv 的路径
    :return: 一个mapping_sheet 为对应的突变
    """
    res = []
    mut_files = list(Path(input_folder).glob('*.maf'))

    sample_sheet = pd.read_csv(sample_sheet, sep='\t')

    for file in mut_files:
        mut = pd.read_csv(file, sep='\t', comment='#',low_memory=False)['Tumor_Sample_Barcode']
        if not len(mut) == 0:

            mut = mut.iloc[0]
            mut_sample = re.search(r"^([^-]+-[^-]+-[^-]+)", mut).group(1)

            expr = sample_sheet[sample_sheet['Case ID'] == mut_sample]['File Name']
            if not expr.empty:
                res.append([file.name,expr.iloc[0]])

    res = pd.DataFrame(res, columns=['mut_file', 'expr_file'])
    export_data(res,'../tcga_process/data/mapping.csv')

    return res


def select_data(mutation_files, expression_files, mapping):
    """
    借助mapping_sheet过滤不匹配的突变及表达数据
    :param mutation_files:
    :param expression_files:
    :param mapping:
    :return: 保留匹配的突变-表达数据
    """
    mut_files = list(Path(mutation_files).glob('*.maf'))
    expr_files = list(Path(expression_files).glob('*.tsv'))
    mapping_sheet = pd.read_csv(mapping, sep=',')

    for mut_file in mut_files:
        if not mut_file.name in list(mapping_sheet['mut_file']):
            mut_file.unlink()

    for expr_file in expr_files:
        if not expr_file.name in list(mapping_sheet['expr_file']):
            expr_file.unlink()




def extract_mutation(input_folder: str,output_folder: str) :
    """
    :param input_folder: 已提取出的完整maf文件的路径
    :param output_folder: 输出路径
    :return: 提取maf文件中的核心字段
    """
    maf_files = list(Path(input_folder).glob('*.maf'))
    print(f'找到{len(maf_files)}个MAF文件')
    required_columns = ['Hugo_Symbol','Variant_Classification','Protein_position','t_ref_count','t_alt_count','Amino_acids']
    for file in maf_files:
        df = pd.read_csv(file,comment='#',skiprows=7,sep='\t',usecols=required_columns)

        df_filter = df[
            (df['Variant_Classification'] == 'Missense_Mutation') &
            (df['Amino_acids'].str.contains(r'^[A-Z]/[A-Z]$', na=False))
            ].copy()
        df_filter['VAF'] = df_filter['t_alt_count'] / (df_filter['t_alt_count'] + df_filter['t_ref_count'])

        if not os.path.exists(output_folder):
            os.makedirs(output_folder,exist_ok=True)

        df_filter.to_csv(Path(output_folder) / file.name,sep='\t',index=False)

    print(f'已提取{len(maf_files)}个MAF文件')


def generate_mutation_seq(maf):
    """
    :param maf: 已提取核心字段的maf文件
    :return: 突变肽段列表
    """

    logging.getLogger('pyensembl').setLevel(logging.WARNING)
    ensembl = EnsemblRelease(104, species='human')
    df = pd.read_csv(maf,sep='\t')
    gene_names = ensembl.gene_names()
    res = []
    for gene_name,total_length,mutation_aa,vaf in zip(df['Hugo_Symbol'],df['Protein_position'],df['Amino_acids'],df['VAF']):
        if gene_name in gene_names:
            transcript_ids = ensembl.transcript_ids_of_gene_name(gene_name)
            protein_seq = [ensembl.transcript_by_id(seq).protein_sequence for seq in transcript_ids]
            target_seq = [seq for seq in protein_seq if
                          seq is not None and len(seq) == int(total_length.split('/')[-1])]
            if len(target_seq) > 0:
                target_seq = str(target_seq[0])
                mutation_pos = int(total_length.split('/')[0]) - 1
                mutation_total_seq = target_seq[:mutation_pos] + mutation_aa.split('/')[1] + target_seq[
                                                                                             mutation_pos + 1:]

                mutation_seq = [mutation_total_seq[mutation_pos - 4:mutation_pos + 5],  # P5
                                mutation_total_seq[mutation_pos - 5:mutation_pos + 4],  # P6
                                mutation_total_seq[mutation_pos - 3:mutation_pos + 6]]  # P4
                res.append([gene_name, mutation_seq, vaf])
            else:
                continue
        else:
            continue

    return pd.DataFrame(res,columns=['gene_name','mutation_seq','VAF'])


def get_expression_mutation(maf_files: str, mapping_sheet: str, expression_matrix: str,output_folder: str):
    """
    :param maf_files: 已提取核心肽段的maf文件夹路径
    :param mapping_sheet: 生成的突变-表达对应表
    :param expression_matrix: 已提取的表达数据矩阵
    :param output_folder: 输出文件夹
    :return: 合并突变序列-VAF-TPM
    """
    print('正在提取 VAF-TPM-seq')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True)

    mapping = pd.read_csv(mapping_sheet, sep=',')
    tpm_df = pd.read_csv(expression_matrix, sep=',')
    i = 0
    for file in Path(maf_files).glob('*.maf'):
        i += 1
        maf = generate_mutation_seq(file)
        mapping_expr = mapping.loc[mapping['mut_file'] == file.name, 'expr_file'].iloc[0]

        expr = tpm_df[['gene_name', mapping_expr]] \
            .rename(columns={mapping_expr: 'TPM'}) \
            .drop_duplicates(subset=['gene_name'])

        expr = expr[expr['gene_name'].isin(maf['gene_name'])]

        res = pd.merge(maf, expr, on='gene_name')

        output = Path(output_folder) / f'Sample{i}'
        res.to_csv(output)


def get_mutation_seq(input_folder: str,
                     output_folder: str,
                     vaf_threshold: float = 0.1,
                     tpm_threshold: float = 5.0
                     ):
    """
    :param input_folder: 已提取vaf和tpm的样本文件夹
    :param output_folder: 最终输出文件夹
    :param vaf_threshold: 用于筛选的vaf阈值
    :param tpm_threshold: 用于筛选的tpm阈值
    :return: 提取含tpm和vaf的突变肽段表
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True)

    res = []

    seq_files = list(Path(input_folder).rglob('*'))
    for seq_file in seq_files:
        df = pd.read_csv(seq_file, sep=',')
        df_filtered = df.query('VAF > @vaf_threshold and TPM > @tpm_threshold').drop('Unnamed: 0',axis=1).reset_index(drop=True)
        for seqs,vaf,tpm,gene in zip(df_filtered['mutation_seq'].apply(ast.literal_eval),df_filtered['VAF'],df_filtered['TPM'],df_filtered['gene_name']):
            for seq in seqs:

                res.append([seq,vaf,tpm,gene])

    res = pd.DataFrame(res,columns=['peptide','VAF','TPM','gene'])
    res = res[res['peptide'].str.len().between(8, 15)]
    res.to_csv(Path(output_folder) / 'mutation_seq.csv')
    print(res)
    return res


# ========== TCGA数据处理流水线 ==========
if __name__ == '__main__':
    extract_tcga_files(source_folder=r'C:\Users\Hydrolysis\Desktop\TCGA_COAD_READ\PAAD_expression',
                       target_folder='../tcga_process/data/tcga_expression/raw',
                       end_with='.tsv')

    extract_tcga_files(source_folder=r'C:\Users\Hydrolysis\Desktop\TCGA_COAD_READ\PAAD_mutation',
                       target_folder='../tcga_process/data/tcga_mutation/raw',
                       end_with='.gz')

    select_data(mutation_files=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_mutation\raw',
                expression_files=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_expression\raw',
                mapping=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\mapping.csv')

    extract_tpm(input_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_expression\raw',
                output_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_expression\processed\tpm_matrix.csv')

    map_mutation_expression(input_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_mutation\raw',
                            sample_sheet=r"C:\Users\Hydrolysis\Downloads\gdc_sample_sheet.2026-03-12.tsv")


    extract_mutation(input_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_mutation\raw',
                     output_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_mutation\processed')

    get_expression_mutation(
        maf_files=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_mutation\processed',
        mapping_sheet=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\mapping.csv',
        expression_matrix=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\tcga_expression\processed\tpm_matrix.csv',
        output_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\final_seq\raw')

    get_mutation_seq(input_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\final_seq\raw',
                      output_folder=r'C:\Users\Hydrolysis\PycharmProjects\Beginning\tcga_process\data\final_seq\processed')