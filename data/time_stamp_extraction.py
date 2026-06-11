import os

# Config area
DATASET_DIR = r".\Raw"

def analyze_absolute_global_rate_final(folder_path):
    """Compute average time interval and frequency per CSV.

    This function corrects common PLC timestamp wraparound.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Path does not exist: {folder_path}")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

    print("-" * 80)
    print("PLC global time-span stats (wraparound corrected):")
    print("-" * 80)
    print(f"{'CSV filename':<50} | {'Avg interval':<12} | {'Stable frequency':<10}")
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

                # Find first valid data row
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

                # Stream through data rows
                for line in f:
                    parts = line.strip().split(chosen_delim)
                    if len(parts) >= 7:
                        try:
                            curr_sec = float(parts[5])
                            curr_nano = float(parts[6])

                            # Compute seconds difference
                            sec_diff = curr_sec - prev_sec

                            # Fix common PLC timestamp wraparound (e.g., 59 -> 0 seconds)
                            if sec_diff < -30:
                                sec_diff += 60
                            elif sec_diff > 30:  # reverse edge-case handling
                                sec_diff -= 60

                            nano_diff = curr_nano - prev_nano

                            # Convert to milliseconds and accumulate
                            gap_ms = (sec_diff * 1000.0) + (nano_diff / 1e6)

                            # Filter out extreme delays; count only normal hardware steps
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
                    print(f"{file_name:<50} | Error: Cannot extract valid time step")

        except Exception as e:
            print(f"{file_name:<50} | Error: {str(e)}")

    print("-" * 80)


if __name__ == "__main__":
    analyze_absolute_global_rate_final(DATASET_DIR)