import os

# =====================================================================
# 配置区域
# =====================================================================
DATASET_DIR = r"C:\Users\DemoBox\Desktop\H_da\TeamProject\DevelopmentField\DescDatasetsFaults\TimeSeriesDataLabelled - Raw"

def analyze_absolute_global_rate_final(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：路径不存在：{folder_path}")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    
    print("-" * 80)
    print("PLC 全局时间跨度精确递归统计 (已修正时间回绕):")
    print("-" * 80)
    print(f"{'CSV 文件名':<50} | {'全局平均间隔':<12} | {'全局稳定频率':<10}")
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
                
                total_diff_ms = 0.0
                valid_gaps_count = 0
                
                # 寻找第一行有效的初始数据
                prev_line = f.readline()
                while prev_line:
                    prev_parts = prev_line.strip().split(chosen_delim)
                    if len(prev_parts) >= 7:
                        try:
                            prev_sec = float(prev_parts[5])
                            prev_nano = float(prev_parts[6])
                            break
                        except ValueError:
                            pass
                    prev_line = f.readline()
                
                # 开始流式递归全量数据
                for line in f:
                    parts = line.strip().split(chosen_delim)
                    if len(parts) >= 7:
                        try:
                            curr_sec = float(parts[5])
                            curr_nano = float(parts[6])
                            
                            # 计算秒数差值
                            sec_diff = curr_sec - prev_sec
                            
                            # 核心：修正工控 PLC 常见的分钟/小时时间戳回绕 (例：从59秒到0秒)
                            if sec_diff < -30: 
                                sec_diff += 60
                            elif sec_diff > 30: # 逆向边缘情况处理
                                sec_diff -= 60
                                
                            nano_diff = curr_nano - prev_nano
                            
                            # 转换为毫秒并累加
                            gap_ms = (sec_diff * 1000.0) + (nano_diff / 1e6)
                            
                            # 过滤掉由于网络极度卡顿导致的瞬时异常超大延迟，只统计正常硬件步进
                            if 0 <= gap_ms < 1000.0:
                                total_diff_ms += gap_ms
                                valid_gaps_count += 1
                            
                            prev_sec = curr_sec
                            prev_nano = curr_nano
                        except ValueError:
                            continue
                
                if valid_gaps_count > 0 and total_diff_ms > 0:
                    avg_interval = total_diff_ms / valid_gaps_count
                    frequency_hz = 1000.0 / avg_interval
                    print(f"{file_name:<50} | {avg_interval:>9.2f} ms | {frequency_hz:>8.1f} Hz")
                else:
                    print(f"{file_name:<50} | 错误: 无法提取有效时间步进")
                    
        except Exception as e:
            print(f"{file_name:<50} | 错误: {str(e)}")
            
    print("-" * 80)

if __name__ == "__main__":
    analyze_absolute_global_rate_final(DATASET_DIR)