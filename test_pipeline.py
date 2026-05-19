import os
from pipelines.training_pipeline import TrainingPipeline

def main():
    """运行测试"""
    print("="*60)
    print("开始测试新抗原预测训练流水线")
    print("="*60)

    data_path = "../data/raw/IEDB20251103.tsv"

    if not os.path.exists(data_path):
        print(f'错误：数据文件不存在 - {data_path}')
        return

    pipeline = TrainingPipeline()

    try:
        model,metrics = pipeline.run(data_path)

        print("\n" + "=" * 60)
        print("流水线运行成功！")
        print("=" * 60)

    except Exception as e:
        print(f"\n运行失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()