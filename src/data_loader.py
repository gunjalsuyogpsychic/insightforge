import pandas as pd
from pathlib import Path

def load_sales_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sales CSV not found: {path}")
    df = pd.read_csv(p)

    # Try to parse common date columns if present
    for col in ["date", "Date", "order_date", "Order Date", "timestamp", "Order_Date", "OrderDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            break

    return df

def load_records_xlsx(path: str) -> dict:
    """
    Returns dict of {sheet_name: DataFrame}
    """
    p = Path(path)
    if not p.exists():
        # Some projects may not use the xlsx file; keep graceful.
        return {}
    xls = pd.ExcelFile(p)
    sheets = {}
    for name in xls.sheet_names:
        sheets[name] = xls.parse(name)
    return sheets
