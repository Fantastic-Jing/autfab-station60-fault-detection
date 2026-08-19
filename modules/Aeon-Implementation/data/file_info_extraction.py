import os

# Simple utility to list CSV files and count their data rows.
DATASET_DIR = r".\Raw"

def extract_dataset_metadata(folder_path):
    """Print CSV filenames and their row counts (excluding header).

    This counts lines without loading the full file into memory.
    """
    if not os.path.exists(folder_path):
        print(f"Error: path does not exist, check configuration: {folder_path}")
        return

    # Filter all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    total_files = len(csv_files)

    print("-" * 60)
    print("Dataset summary:")
    print(f"Total CSV files detected: {total_files}")
    print("-" * 60)
    print(f"{'CSV filename':<40} | {'Row count (time steps)':<15}")
    print("-" * 60)

    total_rows_all_files = 0

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Read index / count lines without loading whole file into memory
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Subtract 1 to exclude header row
                row_count = sum(1 for line in f) - 1

            print(f"{file_name:<40} | {row_count:<15}")
            total_rows_all_files += max(0, row_count)

        except Exception as e:
            print(f"{file_name:<40} | Error: cannot read ({str(e)})")

    print("-" * 60)
    print(f"Total accumulated rows across all files: {total_rows_all_files}")
    print("-" * 60)


if __name__ == "__main__":
    extract_dataset_metadata(DATASET_DIR)