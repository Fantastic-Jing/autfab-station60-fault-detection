import os

# =====================================================================
# 配置区域：请将下面的路径替换为你存放 CSV 文件的本地文件夹绝对路径
# =====================================================================
DATASET_DIR = r"C:\Users\DemoBox\Desktop\H_da\TeamProject\DevelopmentField\DescDatasetsFaults\TimeSeriesDataLabelled - Raw"

def extract_label_segments(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：路径不存在：{folder_path}")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    
    print("-" * 80)
    print("每个文件的 Label 变化区间及顺序统计结果:")
    print("-" * 80)
    
    delimiters = [';', ',', '\t']
    
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header_line = f.readline()
                chosen_delim = ','
                for d in delimiters:
                    if d in header_line:
                        chosen_delim = d
                        break
                
                current_label = None
                start_row = 1  # 数据起始行（扣除表头后，第一条数据定义为第1行）
                current_row_idx = 0
                segments = []
                
                for line in f:
                    parts = line.strip().split(chosen_delim)
                    if not parts or parts == ['']:
                        continue
                    
                    current_row_idx += 1
                    # 提取最后一列的 Label
                    label_val = parts[-1].strip()
                    
                    # 初始化第一个 Label
                    if current_label is None:
                        current_label = label_val
                        start_row = current_row_idx
                        continue
                    
                    # 如果 Label 发生变化，记录上一个区间
                    if label_val != current_label:
                        segments.append(f"{start_row}-{current_row_idx-1}行: {current_label}")
                        current_label = label_val
                        start_row = current_row_idx
                
                # 记录文件末尾的最后一个区间
                if current_label is not None:
                    segments.append(f"{start_row}-{current_row_idx}行: {current_label}")
                
                # 输出当前文件的统计结果
                if segments:
                    segments_str = " | ".join(segments)
                    print(f"{file_name:<50} -> {segments_str}")
                else:
                    print(f"{file_name:<50} -> 错误: 未提取到有效数据")
                    
        except Exception as e:
            print(f"{file_name:<50} -> 错误: {str(e)}")
            
    print("-" * 80)

if __name__ == "__main__":
    extract_label_segments(DATASET_DIR)