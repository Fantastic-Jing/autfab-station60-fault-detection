"""
================================================================================
Convert and Interleave Multivariate Time Series Data

PURPOSE:
    This script reads multivariate time series data from CSV or text files,
    parses individual values from each row, and interleaves all values
    into a single output line in sktime .ts format.
    Multiple input files can be processed sequentially, each producing one data line
    that is appended to the same output file.

    If the output .ts file does not exist, the script writes the full metadata header
    first and then writes the first data line.
    If the output .ts file already exists, the script validates the existing header
    and appends only the new data line.

INPUT FORMAT:
    - Each row represents one time series source row used in the interleaving process.
    - Multiple dimensions are separated by a configurable delimiter (default: ";").
    - Within each dimension, values are separated by a configurable delimiter
      (default: ",").
    - Values from all rows are interleaved in the output.
    - The first non-empty line is skipped and treated as a header line.

OUTPUT FORMAT:
    - sktime .ts format with metadata header and @data section.
    - One output data line is created per input file.
    - Each data line contains interleaved values separated into timestamp groups by ':'.
    - The class label is appended at the end of the line.
    - Metadata such as @problemName, @dimensions, @equalLength, and @classLabel
      can be provided through CLI options.

EXAMPLE:
    Call syntax:
        python convert_to_tsV3.py <input_file> <label> <ignored_columns> <output_file>
            --problem-name <dataset_name>
            --class-labels "E01 E02 E03"
            --n-dims <number_of_dimensions>

        ignored_columns = 0 means no ignored columns,
        ignored_columns = 1 means ignore the first column,
        ignored_columns = 2 means ignore the first two columns, and so on.

    First call (creates output file with header):
        python convert_to_tsV3.py test1.csv label1 1 output.ts
            --problem-name CustomDataset
            --class-labels "label1 label2"
            --n-dims 8

    Second call (appends only a new data line):
        python convert_to_tsV3.py test2.csv label2 2 output.ts
            --problem-name CustomDataset
            --class-labels "label1 label2"
            --n-dims 8

    Input CSV file 1:
        1;2;3;4;5;6;7;8;9
        10;11;12;13,14;15;16;17;18

    Input CSV file 2:
        31;32;33;34;35;36;37;38;39
        410;411;412;413,414;415;416;417;418

    Output .ts:
        @problemName CustomDataset
        @timeStamps false
        @missing false
        @univariate false
        @dimensions 8
        @equalLength false
        @classLabel true label1 label2
        @data
        2,11:3,12:4,13:5,14:6,15:7,16:8,17:9,18:label1
        33,412:34,413:35,414:36,415:37,416:38,417:39,418:label2
================================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence


def split_sample(line: str, dim_sep: str) -> List[str]:
    parts = [p.strip() for p in line.strip().split(dim_sep)]
    return [p for p in parts if p != ""]


def normalize_dim_values(dim_text: str, value_sep: str) -> str:
    raw = dim_text.strip()
    if value_sep == "whitespace":
        vals = [v for v in raw.replace(",", " ").split() if v]
    else:
        vals = [v.strip() for v in raw.split(value_sep) if v.strip() != ""]
    return ",".join(vals)


def parse_class_labels(class_labels_text: str | None) -> List[str] | None:
    if class_labels_text is None:
        return None
    labels = [item.strip() for item in class_labels_text.replace(",", " ").split() if item.strip()]
    return labels or None


def build_header(
    problem_name: str,
    n_dims: int,
    class_labels: Sequence[str] | None,
    equal_length: bool,
) -> List[str]:
    header = [
        f"@problemName {problem_name}",
        "@timeStamps false",
        "@missing false",
        "@univariate false" if n_dims > 1 else "@univariate true",
        f"@dimensions {n_dims}",
        f"@equalLength {'true' if equal_length else 'false'}",
    ]
    if class_labels:
        header.append(f"@classLabel true {' '.join(class_labels)}")
    else:
        header.append("@classLabel false")
    header.append("@data")
    return header


def build_data_line(
    input_path: Path,
    label: str,
    dim_sep: str,
    value_sep: str,
    ignored_columns: int = 0,
) -> str:
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]

    if lines:
        lines = lines[1:]

    all_rows_values: list[list[str]] = []
    for line in lines:
        dims = split_sample(line, dim_sep)
        dims = dims[ignored_columns:]

        row_values: list[str] = []
        for dim in dims:
            normalized = normalize_dim_values(dim, value_sep)
            values = [v for v in normalized.split(",") if v != ""]
            row_values.extend(values)

        if row_values:
            all_rows_values.append(row_values)

    if not all_rows_values:
        raise ValueError(f"No valid samples found in input file: {input_path}")

    grouped_values: list[str] = []
    max_values = max(len(row) for row in all_rows_values)
    for idx in range(max_values):
        group = [row[idx] for row in all_rows_values if idx < len(row)]
        if group:
            grouped_values.append(",".join(group))

    return ":".join(grouped_values) + ":" + label


def validate_existing_header(output_path: Path, expected_header: Sequence[str]) -> None:
    existing_lines = output_path.read_text(encoding="utf-8").splitlines()
    existing_header: list[str] = []
    for line in existing_lines:
        existing_header.append(line.strip())
        if line.strip().lower() == "@data":
            break

    if existing_header != list(expected_header):
        raise ValueError(
            "Existing output header does not match the requested metadata. "
            "Use a new output file or remove the old one before rerunning."
        )


def convert_file(
    input_path: Path,
    output_path: Path,
    label: str,
    problem_name: str,
    dim_sep: str,
    value_sep: str,
    ignored_columns: int,
    class_labels: Sequence[str] | None,
    n_dims: int,
    equal_length: bool,
) -> None:
    data_line = build_data_line(
        input_path=input_path,
        label=label,
        dim_sep=dim_sep,
        value_sep=value_sep,
        ignored_columns=ignored_columns,
    )

    header = build_header(
        problem_name=problem_name,
        n_dims=n_dims,
        class_labels=class_labels,
        equal_length=equal_length,
    )

    if output_path.exists():
        validate_existing_header(output_path, header)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(data_line + "\n")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(header + [data_line]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert row-based multivariate data and append to sktime .ts format. "
            "If output does not exist, write metadata header and @data first; "
            "otherwise only append the data line."
        )
    )

    parser.add_argument("input", type=Path, help="Input text/csv file")
    parser.add_argument("label", type=str, help="Class label for this input file")
    parser.add_argument("ignored_columns", type=int, help="Number of leading columns to ignore")
    parser.add_argument("output", type=Path, help="Output .ts file")

    parser.add_argument("--problem-name", default="CustomDataset", help="Dataset name for @problemName")
    parser.add_argument("--class-labels", default=None, help="Class labels for @classLabel, e.g. 'E01 E02 E03' or 'E01,E02,E03'")
    parser.add_argument("--n-dims", type=int, required=True, help="Number of dimensions for @dimensions")
    parser.add_argument("--equal-length", action="store_true", help="Set @equalLength true")
    parser.add_argument("--dim-sep", default=";", help="Separator between dimensions in one row")
    parser.add_argument("--value-sep", default=",", help="Separator between values inside one dimension")

    args = parser.parse_args()
    class_labels = parse_class_labels(args.class_labels)

    convert_file(
        input_path=args.input,
        output_path=args.output,
        label=args.label,
        problem_name=args.problem_name,
        dim_sep=args.dim_sep,
        value_sep=args.value_sep,
        ignored_columns=args.ignored_columns,
        class_labels=class_labels,
        n_dims=args.n_dims,
        equal_length=args.equal_length,
    )


if __name__ == "__main__":
    main()
