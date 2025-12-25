import pandas as pd
import matplotlib.pyplot as plt

def plot_sales_over_time(monthly_df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(monthly_df["month"], monthly_df["total_sales"])
    ax.set_title("Sales Trend (Monthly)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_top_categories(df: pd.DataFrame, category_col: str, value_col: str, title: str, top_n: int = 10):
    fig, ax = plt.subplots()
    top = df.head(top_n).copy()
    ax.bar(top[category_col].astype(str), top[value_col])
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
