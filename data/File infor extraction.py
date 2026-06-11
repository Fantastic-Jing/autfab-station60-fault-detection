import os

DATASET_DIR = r"C:\Users\DemoBox\Desktop\H_da\TeamProject\DevelopmentField\DescDatasetsFaults\TimeSeriesDataLabelled - Raw"

def extract_dataset_metadata(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：路径不存在，请检查配置：{folder_path}")
        return

    # 筛选出文件夹中所有的 csv 文件
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    total_files = len(csv_files)
    
    print("-" * 60)
    print("数据集宏观统计结果:")
    print(f"总共检测到的 CSV 文件数量: {total_files}")
    print("-" * 60)
    print(f"{'CSV 文件名':<40} | {'包含的行数 (时间步长)':<15}")
    print("-" * 60)
    
    total_rows_all_files = 0
    
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # 仅读取索引或执行轻量级计数，不将全部内容加载进内存
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 减去 1 是为了扣除表头行（行表头）
                row_count = sum(1 for line in f) - 1
            
            print(f"{file_name:<40} | {row_count:<15}")
            total_rows_all_files += max(0, row_count)
            
        except Exception as e:
            print(f"{file_name:<40} | 错误: 无法读取 ({str(e)})")
            
    print("-" * 60)
    print(f"全量文件累加总行数: {total_rows_all_files}")
    print("-" * 60)

if __name__ == "__main__":
    extract_dataset_metadata(DATASET_DIR)