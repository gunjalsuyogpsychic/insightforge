import streamlit as st
import pandas as pd

from config import settings
from src.data_loader import load_sales_csv, load_records_xlsx
from src.analytics import compute_summary_tables
from src.retriever import tables_to_kbitems, build_or_load_faiss, retrieve
from src.rag_chain import get_llm, build_prompt, run_rag, JSONChatMemory
from src.viz import plot_sales_over_time, plot_top_categories
from src.eval import run_qa_eval

st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

@st.cache_data
def load_all_data():
    sales = load_sales_csv(settings.DATA_SALES_CSV)
    records = load_records_xlsx(settings.DATA_RECORDS_XLSX)
    return sales, records

@st.cache_data
def build_kb(sales_df: pd.DataFrame):
    summary = compute_summary_tables(sales_df)
    kbitems = tables_to_kbitems(summary)
    return summary, kbitems

@st.cache_resource
def init_vector_db(kbitems):
    db = build_or_load_faiss(kbitems, settings.FAISS_DIR, settings.EMBEDDING_MODEL)
    return db

def main():
    st.title("InsightForge — AI-Powered Business Intelligence Assistant")

    # Sidebar controls
    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox(
        "LLM Provider",
        ["groq", "ollama"],
        index=0 if settings.LLM_PROVIDER=="groq" else 1
    )
    st.sidebar.caption("Groq requires GROQ_API_KEY env var. Ollama requires local Ollama model.")
    st.sidebar.divider()
    st.sidebar.markdown("**Data files**")
    st.sidebar.code(f"{settings.DATA_SALES_CSV}\n{settings.DATA_RECORDS_XLSX}")

    # Load data
    try:
        sales_df, records = load_all_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    summary, kbitems = build_kb(sales_df)
    db = init_vector_db(kbitems)

    # LLM + prompt + memory
    llm = get_llm(provider, settings.GROQ_API_KEY, settings.GROQ_MODEL, settings.OLLAMA_MODEL)
    prompt = build_prompt()
    memory = JSONChatMemory(settings.MEMORY_FILE, max_turns=8)

    tab1, tab2, tab3 = st.tabs(["Chat Assistant", "Dashboards", "Evaluation"])

    with tab1:
        st.subheader("Ask questions about your business data (RAG + Memory)")

        q = st.text_input("Enter a question", placeholder="e.g., Which region has the highest sales and why?")

        if st.button("Ask", type="primary") and q.strip():
            docs = retrieve(db, q, k=4)
            ans = run_rag(llm, prompt, docs, q, memory)
            st.markdown(ans)

        with st.expander("Dataset overview"):
            st.write("Columns:", summary["meta"]["columns"])
            st.write("Rows:", summary["meta"]["n_rows"])
            st.dataframe(sales_df.head(20))

        with st.expander("Retrieved context preview (debug)"):
            if q.strip():
                docs = retrieve(db, q, k=4)
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Doc {i} — {d.metadata}**")
                    st.code(d.page_content[:2000])

    with tab2:
        st.subheader("Visual Insights")
        tables = summary.get("tables", {})

        # Monthly sales plot if available
        if "sales_monthly" in tables and tables["sales_monthly"] is not None and len(tables["sales_monthly"]) > 0:
            monthly = tables["sales_monthly"]
            fig = plot_sales_over_time(monthly)
            st.pyplot(fig)
        else:
            st.info("No monthly sales table found (date column might be missing).")

        colA, colB = st.columns(2)

        # Top products
        with colA:
            if "sales_by_product" in tables and tables["sales_by_product"] is not None and len(tables["sales_by_product"]) > 0:
                prod = tables["sales_by_product"]
                st.pyplot(plot_top_categories(prod, prod.columns[0], "total_sales", "Top Products by Sales", top_n=10))
            else:
                st.info("No product breakdown detected.")

        # Top regions
        with colB:
            if "sales_by_region" in tables and tables["sales_by_region"] is not None and len(tables["sales_by_region"]) > 0:
                reg = tables["sales_by_region"]
                st.pyplot(plot_top_categories(reg, reg.columns[0], "total_sales", "Top Regions by Sales", top_n=10))
            else:
                st.info("No region breakdown detected.")

        st.markdown("### KPI Table")
        if "kpis" in tables and tables["kpis"] is not None:
            st.dataframe(tables["kpis"])

    with tab3:
        st.subheader("Model Evaluation (QAEvalChain)")

        st.write("Provide a few reference Q/A pairs to evaluate response quality.")
        default_examples = [
            {"query": "What is the total sales?", "answer": "Should return total_sales from KPIs."},
            {"query": "Which product has the highest sales?", "answer": "Should mention top product and its sales."},
            {"query": "Show sales trend month over month.", "answer": "Should summarize monthly trend using sales_monthly table."},
        ]

        examples = st.text_area(
            "Examples (JSON list)",
            value=json.dumps(default_examples, indent=2),
            height=180
        )

        if st.button("Run Evaluation"):
            try:
                ex = json.loads(examples)
                preds = []
                for e in ex:
                    docs = retrieve(db, e["query"], k=4)
                    result = run_rag(llm, prompt, docs, e["query"], memory)
                    preds.append({"query": e["query"], "result": result})

                graded = run_qa_eval(llm, ex, preds)
                st.json(graded)
            except Exception as err:
                st.error(f"Evaluation failed: {err}")

if __name__ == "__main__":
    main()
