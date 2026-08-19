import os

# Config area: replace DATASET_DIR with your CSV folder path if needed
DATASET_DIR = r".\Raw"

def extract_label_segments(folder_path):
    """Extract consecutive label segments per CSV and print ranges.

    Each segment shows start-end rows and the label value.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Path does not exist: {folder_path}")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

    print("-" * 80)
    print("Label change intervals and order per file:")
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
                start_row = 1  # Data start row (after header, first data row is 1)
                current_row_idx = 0
                segments = []

                for line in f:
                    parts = line.strip().split(chosen_delim)
                    if not parts or parts == ['']:
                        continue

                    current_row_idx += 1
                    # Extract label from the last column
                    label_val = parts[-1].strip()

                    # Initialize first label
                    if current_label is None:
                        current_label = label_val
                        start_row = current_row_idx
                        continue

                    # If label changes, record previous segment
                    if label_val != current_label:
                        segments.append(f"{start_row}-{current_row_idx-1} rows: {current_label}")
                        current_label = label_val
                        start_row = current_row_idx

                # Record the last segment at file end
                if current_label is not None:
                    segments.append(f"{start_row}-{current_row_idx} rows: {current_label}")

                # Print results for the current file
                if segments:
                    segments_str = " | ".join(segments)
                    print(f"{file_name:<50} -> {segments_str}")
                else:
                    print(f"{file_name:<50} -> Error: No valid data extracted")

        except Exception as e:
            print(f"{file_name:<50} -> Error: {str(e)}")

    print("-" * 80)


if __name__ == "__main__":
    extract_label_segments(DATASET_DIR)