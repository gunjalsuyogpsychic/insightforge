import pandas as pd
import numpy as np

def guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def compute_summary_tables(df: pd.DataFrame) -> dict:
    date_col = guess_col(df, ["date", "order_date", "Order Date", "Date", "Order_Date", "OrderDate"])
    sales_col = guess_col(df, ["sales", "Sales", "revenue", "Revenue", "amount", "Amount", "total", "Total"])
    product_col = guess_col(df, ["product", "Product", "item", "Item", "sku", "SKU", "Category", "category"])
    region_col = guess_col(df, ["region", "Region", "state", "State", "country", "Country", "City", "city"])
    customer_col = guess_col(df, ["customer", "Customer", "customer_id", "Customer ID", "CustomerID"])
    age_col = guess_col(df, ["age", "Age"])
    gender_col = guess_col(df, ["gender", "Gender", "sex", "Sex"])

    out = {
        "meta": {
            "date_col": date_col,
            "sales_col": sales_col,
            "product_col": product_col,
            "region_col": region_col,
            "customer_col": customer_col,
            "age_col": age_col,
            "gender_col": gender_col,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns),
        },
        "tables": {}
    }

    if sales_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        out["tables"]["numeric_stats"] = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()
        return out

    work = df.copy()
    work[sales_col] = pd.to_numeric(work[sales_col], errors="coerce").fillna(0)

    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    out["tables"]["kpis"] = pd.DataFrame([{
        "total_sales": float(work[sales_col].sum()),
        "avg_sales": float(work[sales_col].mean()),
        "median_sales": float(work[sales_col].median()),
        "std_sales": float(work[sales_col].std(ddof=1)) if len(work) > 1 else 0.0,
        "num_transactions": int(len(work)),
        "num_customers": int(work[customer_col].nunique()) if customer_col else None
    }])

    if date_col is not None:
        t = work.dropna(subset=[date_col]).copy()
        if len(t) > 0:
            t["month"] = t[date_col].dt.to_period("M").astype(str)
            monthly = t.groupby("month", as_index=False)[sales_col].sum().sort_values("month")
            out["tables"]["sales_monthly"] = monthly.rename(columns={sales_col: "total_sales"})

            t["quarter"] = t[date_col].dt.to_period("Q").astype(str)
            quarterly = t.groupby("quarter", as_index=False)[sales_col].sum().sort_values("quarter")
            out["tables"]["sales_quarterly"] = quarterly.rename(columns={sales_col: "total_sales"})

    if product_col is not None:
        prod = work.groupby(product_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
        out["tables"]["sales_by_product"] = prod.rename(columns={sales_col: "total_sales"}).head(30)

    if region_col is not None:
        reg = work.groupby(region_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
        out["tables"]["sales_by_region"] = reg.rename(columns={sales_col: "total_sales"}).head(30)

    seg_tables = {}
    if gender_col is not None:
        g = work.groupby(gender_col, as_index=False)[sales_col].agg(["count","sum","mean"])
        g.columns = ["count", "total_sales", "avg_sales"]
        seg_tables["by_gender"] = g.reset_index().rename(columns={gender_col: "gender"})
    if age_col is not None:
        ages = pd.to_numeric(work[age_col], errors="coerce")
        bins = [0, 18, 25, 35, 45, 55, 65, 200]
        labels = ["<18","18-24","25-34","35-44","45-54","55-64","65+"]
        work["age_bucket"] = pd.cut(ages, bins=bins, labels=labels, right=False)
        a = work.groupby("age_bucket", as_index=False)[sales_col].agg(["count","sum","mean"])
        a.columns = ["count", "total_sales", "avg_sales"]
        seg_tables["by_age_bucket"] = a.reset_index()
    if customer_col is not None:
        cust = work.groupby(customer_col, as_index=False)[sales_col].agg(["count","sum","mean"])
        cust.columns = ["frequency", "total_sales", "avg_sales"]
        seg_tables["by_customer"] = cust.reset_index().head(50)

    if seg_tables:
        out["tables"]["customer_segmentation"] = seg_tables

    return out
