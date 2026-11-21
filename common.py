from io import BytesIO
from pathlib import Path

import pandas as pd


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read the first worksheet or a comma-delimited UTF-8/GB18030 file."""
    extension = Path(filename).suffix.lower()
    if extension == ".xlsx":
        return pd.read_excel(BytesIO(data), engine="openpyxl", dtype=object)
    if extension == ".csv":
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                return pd.read_csv(
                    BytesIO(data), encoding=encoding, dtype=object, skip_blank_lines=False
                )
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 编码无法识别，请另存为 UTF-8 后重试。")
    raise ValueError("请上传 .xlsx 或 .csv 格式的文件。")


def excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.sheets[name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            # Preserve user-supplied text that starts with = as literal Excel text.
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
    return output.getvalue()

