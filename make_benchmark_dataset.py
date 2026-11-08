"""Create a benchmark subset without changing the existing prepare_data.py."""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chọn các articleType gần nhau để tạo bộ dữ liệu benchmark."
    )
    parser.add_argument("--csv", type=Path, default=Path("styles.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_dataset.csv"))
    parser.add_argument("--output-image-dir", type=Path, default=Path("benchmark_images"))
    parser.add_argument(
        "--categories",
        help="articleType cần lấy, phân cách dấu phẩy; phải khớp tên trong CSV.",
    )
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV nguồn: {args.csv.resolve()}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {args.image_dir.resolve()}")

    dataframe = pd.read_csv(args.csv, on_bad_lines="skip")
    required = {"id", "articleType"}
    missing_columns = required.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing_columns))}")

    if not args.categories:
        print(dataframe["articleType"].value_counts().to_string())
        print("\nChạy lại với --categories \"Tên lớp 1,Tên lớp 2,...\" để tạo dữ liệu.")
        return
    if args.samples_per_category < 1:
        raise ValueError("samples-per-category phải lớn hơn 0.")

    categories = [name.strip() for name in args.categories.split(",") if name.strip()]
    if len(categories) < 2:
        raise ValueError("Benchmark cần ít nhất hai category.")
    available = set(dataframe["articleType"].dropna().astype(str))
    unknown = set(categories).difference(available)
    if unknown:
        raise ValueError("Không có category trong CSV: " + ", ".join(sorted(unknown)))

    output_rows = []
    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    selected = dataframe[dataframe["articleType"].isin(categories)]
    for category in categories:
        category_rows = selected[selected["articleType"] == category]
        category_rows = category_rows.drop_duplicates(subset="id").sample(
            frac=1, random_state=args.seed
        )
        copied = 0
        for _, row in category_rows.iterrows():
            if copied >= args.samples_per_category:
                break
            source = args.image_dir / f"{row['id']}.jpg"
            if source.is_file():
                shutil.copy2(source, args.output_image_dir / source.name)
                output_rows.append(row)
                copied += 1
        print(f"{category}: {copied} ảnh")
        if copied < 3:
            raise ValueError(
                f"Category {category!r} chỉ có {copied} ảnh hợp lệ; cần ít nhất 3."
            )

    output = pd.DataFrame(output_rows, columns=dataframe.columns)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Đã lưu {len(output)} dòng: {args.output_csv.resolve()}")
    print(f"Ảnh đã lưu tại: {args.output_image_dir.resolve()}")


if __name__ == "__main__":
    main()
