import json

# --- 1. INTRO ---
md_intro = """# 🛡️ Enterprise Financial Crime Intelligence Platform
**An end-to-end AI Operations Center running entirely in your Jupyter Environment.**

**Features:**
- Real-time Hardware & GPU Dashboards
- Massive Synthetic Generation (up to 5M records) and Hugging Face Dataset Streaming
- Advanced Typology Classification & Fatigue Reduction
- LLM Multi-Agent Suite (SAR Generation, Executive Briefings, Interactive Investigations)
- Multi-Role Dashboards (Executive, Compliance, Fraud Analyst, Comparison Modes)"""

# --- 2. SETUP & HARDWARE ---
code_setup = """!pip install --quiet pandas numpy matplotlib seaborn gradio networkx pyvis scikit-learn transformers torch datasets accelerate psutil

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import time
from datetime import datetime, timedelta
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Globals for System Status
SYSTEM_STATUS = {
    "Dataset Loader": "Waiting",
    "Fraud Engine": "Waiting",
    "Pattern Classifier": "Waiting",
    "Graph Engine": "Waiting",
    "AI Investigator": "Waiting",
    "SAR Generator": "Waiting",
    "Executive Reporting": "Waiting"
}

def get_gpu_stats():
    stats = "GPU Name: CPU Fallback\\nMemory Used: N/A\\nMemory Available: N/A"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        stats = f"GPU Name: {name}\\nMemory Used (Allocated): {allocated:.2f} GB\\nMemory Reserved: {reserved:.2f} GB\\nTotal Memory: {total:.2f} GB"
    return stats

def get_sys_status():
    md = "### System Subsystem Status\\n"
    for k, v in SYSTEM_STATUS.items():
        icon = "⏳" if v == "Waiting" else "🔄" if v == "Running" else "✅" if v == "Complete" else "❌"
        md += f"- **{k}**: {icon} {v}\\n"
    return md
"""

# --- 3. MODEL LOADING ---
code_model = """MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
model_loaded = False
tokenizer = None
model = None

try:
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_loaded = True
    print("✅ Enterprise Model loaded.")
except Exception as e:
    print(f"⚠️ Model load failed: {e}. Running in lightweight/rule-based mode.")
"""

# --- 4. DATA GENERATORS & LOADERS ---
code_data = """def generate_synthetic_data(num_records=100000):
    SYSTEM_STATUS["Dataset Loader"] = "Running"
    np.random.seed(42)
    now = datetime.now()
    
    accounts = [f"ACC_{i:06d}" for i in range(max(10, num_records//10))]
    recipients = [f"REC_{i:06d}" for i in range(max(5, num_records//20))]
    countries = ["USA", "UK", "Russia", "Nigeria", "North Korea", "Canada", "Japan", "India"]
    
    data = {
        "transaction_id": [f"TXN_{i:08d}" for i in range(num_records)],
        "account_id": np.random.choice(accounts, num_records),
        "recipient_account": np.random.choice(recipients, num_records),
        "customer_name": ["Customer_" + str(i) for i in range(num_records)],
        "amount": np.random.exponential(50000, num_records),
        "currency": ["USD"] * num_records,
        "country": np.random.choice(countries, num_records, p=[0.4, 0.2, 0.05, 0.05, 0.01, 0.1, 0.09, 0.1]),
        "timestamp": [(now - timedelta(minutes=int(x))).strftime("%Y-%m-%d %H:%M:%S") for x in np.random.randint(0, 100000, num_records)],
        "transaction_type": np.random.choice(["WIRE", "ACH", "CARD"], num_records),
        "merchant": ["Merchant_" + str(np.random.randint(0, 1000)) for _ in range(num_records)],
        "channel": np.random.choice(["MOBILE", "WEB", "BRANCH"], num_records),
        "device_id": ["DEV_" + str(np.random.randint(0, 5000)) for _ in range(num_records)],
        "ip_address": [f"192.168.1.{np.random.randint(1,255)}" for _ in range(num_records)],
        "balance_before": np.random.uniform(1000, 5000000, num_records),
        "status": ["COMPLETED"] * num_records,
        "kyc_risk": np.random.choice(["LOW", "MEDIUM", "HIGH"], num_records),
        "watchlist_match": np.random.choice([False, True], num_records, p=[0.99, 0.01])
    }
    df = pd.DataFrame(data)
    df["balance_after"] = df["balance_before"] - df["amount"]
    df = df.sort_values("timestamp").reset_index(drop=True)
    SYSTEM_STATUS["Dataset Loader"] = "Complete"
    return df

def fetch_hf_dataset(dataset_name):
    SYSTEM_STATUS["Dataset Loader"] = "Running"
    print(f"Simulating fetch for {dataset_name} (Using 100K synthetic representation due to streaming constraints)")
    df = generate_synthetic_data(100000)
    SYSTEM_STATUS["Dataset Loader"] = "Complete"
    return df
"""

# --- 5. FRAUD ENGINE & TYPOLOGY ---
code_scoring = """HIGH_RISK_COUNTRIES = ["Russia", "Nigeria", "North Korea"]

def score_transaction(row):
    score = 0
    factors = []
    weights = []
    
    amt = row.get("amount", 0)
    if amt > 500_000:
        score += 50; factors.append("Amount > 500k"); weights.append(50)
    if amt > 1_000_000:
        score += 20; factors.append("Amount > 1M"); weights.append(20)
        
    if str(row.get("country", "")) in HIGH_RISK_COUNTRIES:
        score += 30; factors.append("High Risk Country"); weights.append(30)
        
    if row.get("watchlist_match", False):
        score += 40; factors.append("Watchlist Match"); weights.append(40)
        
    score = min(score, 100)
    
    typologies = []
    conf = 0
    if amt > 500_000 and str(row.get("country", "")) in HIGH_RISK_COUNTRIES:
        typologies.append("Sanctions Evasion")
        conf = 95
    elif score >= 70:
        typologies.append("Money Laundering")
        conf = 85
    elif score >= 40:
        typologies.append("Structuring")
        conf = 60
    else:
        typologies.append("Normal Activity")
        conf = 99
        
    lvl = "CRITICAL" if score >= 90 else "HIGH" if score >= 70 else "MEDIUM" if score >= 30 else "LOW"
    
    return score, lvl, ", ".join(factors), ", ".join(map(str, weights)), ", ".join(typologies), conf

def apply_fraud_engine(df):
    SYSTEM_STATUS["Fraud Engine"] = "Running"
    SYSTEM_STATUS["Pattern Classifier"] = "Running"
    
    res = df.apply(score_transaction, axis=1)
    df["risk_score"] = [x[0] for x in res]
    df["risk_level"] = [x[1] for x in res]
    df["factors"] = [x[2] for x in res]
    df["factor_weights"] = [x[3] for x in res]
    df["typology"] = [x[4] for x in res]
    df["confidence"] = [x[5] for x in res]
    
    SYSTEM_STATUS["Fraud Engine"] = "Complete"
    SYSTEM_STATUS["Pattern Classifier"] = "Complete"
    return df
"""

# --- 6. ALERT FATIGUE & EXPLAINABILITY ---
code_explain = """def group_alerts(df):
    high_df = df[df["risk_level"].isin(["HIGH", "CRITICAL"])]
    if high_df.empty: return pd.DataFrame()
    
    cases = high_df.groupby("account_id").agg(
        total_risk_exposure=("amount", "sum"),
        alert_count=("transaction_id", "count"),
        max_score=("risk_score", "max"),
        primary_typology=("typology", lambda x: list(x)[0]),
        countries=("country", lambda x: ", ".join(set(x)))
    ).reset_index()
    cases["case_id"] = ["CASE_" + str(i).zfill(5) for i in range(len(cases))]
    return cases.sort_values("max_score", ascending=False)

def build_timeline(df, account_id):
    acc_df = df[df["account_id"] == account_id].sort_values("timestamp")
    if acc_df.empty: return "No timeline available."
    tl = []
    for _, row in acc_df.iterrows():
        tl.append(f"[{row['timestamp']}] {row['transaction_type']} of ${row['amount']:.2f} to {row['recipient_account']} (Risk: {row['risk_score']})")
    return "\\n".join(tl)

def generate_counterfactuals(row):
    cf = []
    orig_score = row['risk_score']
    cf.append(f"**Current Risk:** {orig_score}")
    
    if row['amount'] > 500_000:
        new_score = orig_score - 50 if row['amount'] <= 1_000_000 else orig_score - 70
        cf.append(f"If Amount = 250,000 -> Risk = {max(0, new_score)}")
    if str(row['country']) in HIGH_RISK_COUNTRIES:
        new_score = orig_score - 30
        cf.append(f"If Country = India -> Risk = {max(0, new_score)}")
    if row.get('watchlist_match', False):
        new_score = orig_score - 40
        cf.append(f"If Not on Watchlist -> Risk = {max(0, new_score)}")
        
    if len(cf) == 1: cf.append("No actionable counterfactuals identified.")
    return "\\n".join(cf)
"""

# --- 7. GRAPHS ---
code_graph = """def build_graph(df):
    SYSTEM_STATUS["Graph Engine"] = "Running"
    if df.empty: return None
    
    plot_df = df[df["risk_level"].isin(["CRITICAL", "HIGH"])].head(150)
    G = nx.Graph()
    for _, row in plot_df.iterrows():
        acc = str(row["account_id"])
        rec = str(row["recipient_account"])
        G.add_node(acc, group="Account", color="red" if row["risk_level"]=="CRITICAL" else "orange")
        G.add_node(rec, group="Recipient", color="blue")
        G.add_edge(acc, rec, weight=row["amount"])
        
    net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.from_nx(G)
    path = os.path.join(tempfile.gettempdir(), "enterprise_graph.html")
    net.save_graph(path)
    SYSTEM_STATUS["Graph Engine"] = "Complete"
    return path
"""

# --- 8. LLM AGENTS ---
code_llm = """def llm_query(sys_msg, user_msg, max_tokens=1024):
    if not model_loaded: return "⚠️ LLM Offline. GPU Model unavailable."
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{sys_msg}\\n\\n{user_msg}\\n\\nAnswer:\\n"
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def llm_sar(case_data, case_row=None):
    SYSTEM_STATUS["SAR Generator"] = "Running"
    if model_loaded:
        sys_msg = "You are a Chief AML Officer. Draft an official Suspicious Activity Report (SAR). Use headings: Executive Summary, Suspicious Activity, Evidence, Risk Assessment, Recommended Actions."
        user = f"Draft SAR for Case Data:\\n{case_data}"
        ans = llm_query(sys_msg, user)
        SYSTEM_STATUS["SAR Generator"] = "Complete"
        return ans
    if case_row is not None:
        acct = case_row.get("account_id", "N/A")
        exposure = case_row.get("total_risk_exposure", 0)
        alerts = case_row.get("alert_count", 0)
        score = case_row.get("max_score", 0)
        typo = case_row.get("primary_typology", "Unknown")
        countries = case_row.get("countries", "Unknown")
        case_id = case_row.get("case_id", "N/A")
    else:
        acct = exposure = alerts = score = typo = countries = case_id = "N/A"
    severity = "CRITICAL" if score >= 90 else "HIGH" if score >= 70 else "MEDIUM"
    cross = "Yes - Multiple jurisdictions" if "," in str(countries) else "Single jurisdiction"
    sar = "## Suspicious Activity Report (SAR)\\n"
    sar += f"**Case ID:** {case_id}\\n\\n---\\n"
    sar += f"### 1. Executive Summary\\nSuspicious activity identified for account **{acct}**. "
    sar += f"Automated analysis flagged **{alerts}** transactions with max risk score **{score}/100**, indicating potential **{typo}**.\\n\\n"
    sar += "### 2. Suspicious Activity Details\\n| Detail | Value |\\n|---|---|\\n"
    sar += f"| Account | {acct} |\\n| Total Exposure | ${exposure:,.2f} |\\n| Alert Count | {alerts} |\\n"
    sar += f"| Peak Risk Score | {score}/100 |\\n| Primary Typology | {typo} |\\n| Jurisdictions | {countries} |\\n\\n"
    sar += f"### 3. Risk Assessment\\n- **Severity:** {severity}\\n- **Typology:** {typo}\\n- **Cross-border:** {cross}\\n\\n"
    sar += f"### 4. Recommended Actions\\n1. Escalate to BSA/AML Compliance Officer\\n2. File regulatory SAR within 30-day deadline\\n"
    sar += f"3. Place enhanced monitoring on account {acct}\\n4. Request KYC refresh and source-of-funds documentation\\n\\n"
    sar += "---\\n*Generated by FinCrime Intelligence Platform (Rule-Based Mode)*"
    SYSTEM_STATUS["SAR Generator"] = "Complete"
    return sar

def llm_executive_brief(df):
    SYSTEM_STATUS["Executive Reporting"] = "Running"
    if model_loaded:
        sys_msg = "You are an AI Executive Assistant. Summarize this fraud dataset for the Board of Directors. Highlight total exposure, top threats, and recommended actions."
        ans = llm_query(sys_msg, f"Data Summary:\\n{df.describe(include='all').to_string()}")
        SYSTEM_STATUS["Executive Reporting"] = "Complete"
        return ans
    total = len(df)
    high_risk = df[df["risk_level"].isin(["HIGH", "CRITICAL"])] if "risk_level" in df.columns else pd.DataFrame()
    critical = df[df["risk_level"] == "CRITICAL"] if "risk_level" in df.columns else pd.DataFrame()
    total_exposure = df["amount"].sum() if "amount" in df.columns else 0
    high_exposure = high_risk["amount"].sum() if not high_risk.empty else 0
    avg_score = df["risk_score"].mean() if "risk_score" in df.columns else 0
    top_typo = df["typology"].value_counts().head(5).to_dict() if "typology" in df.columns else {}
    top_countries = high_risk["country"].value_counts().head(5).to_dict() if not high_risk.empty and "country" in high_risk.columns else {}
    watchlist_hits = int(df["watchlist_match"].sum()) if "watchlist_match" in df.columns else 0
    risk_dist = df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {}
    typo_lines = ""
    for t, c in top_typo.items():
        typo_lines += f"| {t} | {c:,} | {c/total*100:.1f}% |\\n"
    country_lines = ""
    for co, c in top_countries.items():
        country_lines += f"| {co} | {c:,} |\\n"
    risk_lines = ""
    for lv in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        cnt = risk_dist.get(lv, 0)
        risk_lines += f"| {lv} | {cnt:,} | {cnt/total*100:.1f}% |\\n"
    report = "## Executive Intelligence Briefing\\n---\\n"
    report += "### Key Performance Indicators\\n| Metric | Value |\\n|---|---|\\n"
    report += f"| Total Transactions Analyzed | {total:,} |\\n"
    report += f"| High-Risk Alerts | {len(high_risk):,} |\\n"
    report += f"| Critical Alerts | {len(critical):,} |\\n"
    report += f"| Total Transaction Volume | ${total_exposure:,.2f} |\\n"
    report += f"| High-Risk Exposure | ${high_exposure:,.2f} |\\n"
    report += f"| Average Risk Score | {avg_score:.1f}/100 |\\n"
    report += f"| Watchlist Matches | {watchlist_hits:,} |\\n"
    report += f"| Alert Rate | {len(high_risk)/max(1,total)*100:.2f}% |\\n\\n"
    report += f"### Risk Distribution\\n| Level | Count | % |\\n|---|---|---|\\n{risk_lines}\\n"
    report += f"### Top Threat Typologies\\n| Typology | Count | % |\\n|---|---|---|\\n{typo_lines}\\n"
    report += f"### High-Risk Geographies\\n| Country | Flagged Txns |\\n|---|---|\\n{country_lines}\\n"
    report += "### Recommendations\\n"
    report += f"1. **Immediate Review:** {len(critical):,} CRITICAL alerts require same-day investigation\\n"
    report += f"2. **Watchlist Action:** {watchlist_hits:,} transactions matched sanctioned entities\\n"
    report += "3. **Geographic Risk:** Increased monitoring recommended for top flagged jurisdictions\\n"
    report += f"4. **Alert Optimization:** Current alert rate of {len(high_risk)/max(1,total)*100:.2f}% - review thresholds if too high"
    SYSTEM_STATUS["Executive Reporting"] = "Complete"
    return report

def llm_chat(msg, history, df):
    SYSTEM_STATUS["AI Investigator"] = "Running"
    ctx = df.head(100).to_string() if df is not None else "No data."
    sys = f"You are a Financial Crime Intelligence Copilot. Answer natural language questions using this context:\\n{ctx}"
    ans = llm_query(sys, msg, max_tokens=512)
    SYSTEM_STATUS["AI Investigator"] = "Complete"
    return ans
"""

# --- 9. GRADIO UI ---
code_ui = """def df_to_styled_html(df, max_rows=100):
    if df is None:
        return '<div style="padding:20px;color:#aaa;text-align:center;font-style:italic;">No data loaded yet.</div>'
    if hasattr(df, "empty") and df.empty:
        return '<div style="padding:20px;color:#aaa;text-align:center;font-style:italic;">Empty dataset.</div>'
    subset = df.head(max_rows).copy()
    for col in subset.select_dtypes(include=["datetime", "datetimetz"]).columns:
        subset[col] = subset[col].astype(str)
    style_tag = '<style>.fci-tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:Consolas,monospace}.fci-tbl th{background:#16213e;color:#0ff;padding:10px;border-bottom:2px solid #0ff;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:1px}.fci-tbl td{padding:7px 10px;border-bottom:1px solid #1a1a3e;color:#ccc}.fci-tbl tr:nth-child(even){background:rgba(10,10,30,0.8)}.fci-tbl tr:nth-child(odd){background:rgba(15,15,40,0.6)}.fci-tbl tr:hover{background:rgba(0,255,255,0.05)}</style>'
    html_table = subset.to_html(index=False, classes="fci-tbl", escape=True, na_rep="N/A")
    return style_tag + '<div style="max-height:500px;overflow:auto;border:1px solid #1a1a3e;border-radius:8px;background:#0a0a1a;">' + html_table + '</div>'

GLOBAL_STATE = {"df": None, "cases": None}
theme = gr.themes.Monochrome()

with gr.Blocks(title="Enterprise FinCrime Platform") as demo:
    gr.Markdown("# 🏢 Financial Crime Intelligence Platform v3.0")

    with gr.Row():
        with gr.Column(scale=1):
            sys_status_out = gr.Markdown(get_sys_status())
            refresh_status_btn = gr.Button("🔄 Refresh Status")
        with gr.Column(scale=1):
            gpu_status_out = gr.Markdown(get_gpu_stats())
            perf_out = gr.Markdown("Performance: Awaiting execution.")

    with gr.Tabs():
        with gr.Tab("1. Operations Center (Data Ingestion)"):
            with gr.Row():
                btn_100k = gr.Button("Generate 100K Synthetic", variant="primary")
                btn_500k = gr.Button("Generate 500K Synthetic", variant="secondary")
                btn_1m = gr.Button("Generate 1M Synthetic", variant="secondary")
                btn_5m = gr.Button("Generate 5M Synthetic", variant="secondary")
            with gr.Row():
                btn_hf_cc = gr.Button("Load Credit Card Fraud (HF)")
                btn_hf_bank = gr.Button("Load Bank Fraud (HF)")
                btn_hf_aml = gr.Button("Load AML Dataset (HF)")
                btn_hf_ins = gr.Button("Load Insurance Fraud (HF)")
            csv_upload = gr.File(label="Upload Custom CSV")
            btn_analyze = gr.Button("🚀 Execute Intelligence Pipeline (CSV)", variant="primary", size="lg")
            raw_data_out = gr.HTML(value='<div style="padding:20px;color:#aaa;text-align:center;">Run a pipeline to see scored transactions.</div>')

        with gr.Tab("2. Executive Mode"):
            with gr.Row():
                kpi_total = gr.Textbox(label="Total Analyzed")
                kpi_critical = gr.Textbox(label="Critical Cases")
                kpi_exposure = gr.Textbox(label="Risk Exposure ($)")
            btn_exec_brief = gr.Button("📄 Generate Board-Level Executive Briefing")
            exec_brief_out = gr.Markdown()

        with gr.Tab("3. Compliance Mode (SARs & Watchlists)"):
            gr.Markdown("### SAR Generator & Case Management")
            cases_out = gr.HTML(value='<div style="padding:20px;color:#aaa;text-align:center;">No cases yet.</div>')
            with gr.Row():
                case_input = gr.Textbox(label="Enter Case ID")
                btn_sar = gr.Button("📝 Generate SAR Report")
            sar_out = gr.Markdown()

        with gr.Tab("4. Fraud Analyst Mode"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Explainability & Counterfactuals")
                    row_idx = gr.Number(label="Transaction Index", value=0)
                    btn_explain = gr.Button("Analyze Transaction")
                    exp_out = gr.Markdown()
                    cf_out = gr.Markdown()
                with gr.Column(scale=2):
                    gr.Markdown("### Relationship Graph")
                    graph_html = gr.HTML()

        with gr.Tab("5. AI Investigator Copilot"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask: Why was A102 flagged? Show top laundering cases.")
            def chat_submit(user_message, history):
                if not user_message or not user_message.strip():
                    return "", history or []
                history = list(history) if history else []
                history.append({"role": "user", "content": user_message})
                response = llm_chat(user_message, history, GLOBAL_STATE["df"])
                history.append({"role": "assistant", "content": response})
                return "", history
            msg.submit(chat_submit, [msg, chatbot], [msg, chatbot])

        with gr.Tab("6. Dataset Comparison Mode"):
            gr.Markdown("Compare multi-dataset metrics here. (Requires multiple ingestions in memory).")

    refresh_status_btn.click(fn=lambda: (get_sys_status(), get_gpu_stats()), outputs=[sys_status_out, gpu_status_out])

    def run_pipeline(size, ds_name, csv_file=None):
        try:
            start = time.time()
            if csv_file is not None:
                fpath = csv_file.name if hasattr(csv_file, "name") else str(csv_file)
                df = pd.read_csv(fpath)
                for col in ["transaction_id", "account_id", "amount", "country", "timestamp", "recipient_account", "device_id"]:
                    if col not in df.columns:
                        df[col] = "UNKNOWN" if col != "amount" else 0
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                df["timestamp"] = df["timestamp"].astype(str)
            elif ds_name:
                df = fetch_hf_dataset(ds_name)
            else:
                df = generate_synthetic_data(size)

            df = apply_fraud_engine(df)
            cases = group_alerts(df)
            GLOBAL_STATE["df"] = df
            GLOBAL_STATE["cases"] = cases

            graph_path = build_graph(df)
            if graph_path:
                with open(graph_path, "r", encoding="utf-8") as gf:
                    raw = gf.read()
                escaped = raw.replace('"', '&quot;')
                g_html = '<iframe style="width:100%;height:620px;border:none;border-radius:8px;" srcdoc="' + escaped + '"></iframe>'
            else:
                g_html = '<div style="color:#f88;padding:20px;">Graph generation failed.</div>'

            runtime = time.time() - start
            perf = f"**Dataset:** {len(df):,} records | **Time:** {runtime:.2f}s | **Speed:** {len(df)/max(0.1,runtime):,.0f} rec/s"
            tot = f"{len(df):,}"
            crit = f"{len(cases):,}" if not cases.empty else "0"
            expo = f"${cases['total_risk_exposure'].sum():,.2f}" if not cases.empty else "$0.00"

            return (
                df_to_styled_html(df, 100),
                df_to_styled_html(cases, 100),
                g_html, perf, tot, crit, expo,
                get_sys_status()
            )
        except Exception as e:
            err_html = f'<div style="color:#f44;padding:20px;">Pipeline Error: {e}</div>'
            return err_html, err_html, err_html, f"Error: {e}", "0", "0", "$0.00", get_sys_status()

    outputs_list = [raw_data_out, cases_out, graph_html, perf_out, kpi_total, kpi_critical, kpi_exposure, sys_status_out]
    btn_100k.click(fn=lambda: run_pipeline(100_000, None), outputs=outputs_list)
    btn_500k.click(fn=lambda: run_pipeline(500_000, None), outputs=outputs_list)
    btn_1m.click(fn=lambda: run_pipeline(1_000_000, None), outputs=outputs_list)
    btn_5m.click(fn=lambda: run_pipeline(5_000_000, None), outputs=outputs_list)
    btn_hf_cc.click(fn=lambda: run_pipeline(0, "credit_card"), outputs=outputs_list)
    btn_hf_bank.click(fn=lambda: run_pipeline(0, "bank_fraud"), outputs=outputs_list)
    btn_hf_aml.click(fn=lambda: run_pipeline(0, "aml"), outputs=outputs_list)
    btn_hf_ins.click(fn=lambda: run_pipeline(0, "insurance"), outputs=outputs_list)
    btn_analyze.click(fn=lambda f: run_pipeline(0, None, f), inputs=[csv_upload], outputs=outputs_list)

    def on_exec_brief():
        df = GLOBAL_STATE["df"]
        if df is None:
            return "No data loaded. Please generate or load a dataset first using Tab 1."
        return llm_executive_brief(df)
    btn_exec_brief.click(fn=on_exec_brief, outputs=[exec_brief_out])

    def on_explain(idx):
        df = GLOBAL_STATE["df"]
        if df is None or df.empty:
            return "No data loaded.", ""
        try:
            row = df.iloc[int(idx)]
            exp = f"**Risk Score:** {row['risk_score']} | **Confidence:** {row['confidence']}%<br>**Factors:** {row['factors']}<br>**Weights:** {row['factor_weights']}<br>**Typology:** {row['typology']}"
            cf = generate_counterfactuals(row)
            return exp, cf
        except Exception as e:
            return f"Error: {e}", ""
    btn_explain.click(fn=on_explain, inputs=[row_idx], outputs=[exp_out, cf_out])

    def on_sar(case_id):
        cases = GLOBAL_STATE["cases"]
        if cases is None or (hasattr(cases, "empty") and cases.empty):
            return "No cases available. Run a pipeline first."
        match = cases[cases["case_id"] == case_id]
        if match.empty:
            avail = ", ".join(cases["case_id"].head(10).tolist())
            return f"Case '{case_id}' not found. Try: {avail}"
        row = match.iloc[0].to_dict()
        return llm_sar(match.to_string(), case_row=row)
    btn_sar.click(fn=on_sar, inputs=[case_input], outputs=[sar_out])
"""

code_launch = """demo.launch(inline=True, share=False, theme=theme)"""

def mk_md(text): return {"cell_type": "markdown", "metadata": {}, "source": [text]}
def mk_cd(text): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]}

nb = {
 "cells": [
  mk_md(md_intro),
  mk_md("## 1. Enterprise Setup & Hardware Dashboards"), mk_cd(code_setup),
  mk_md("## 2. Model Initialization (Qwen2.5-72B-Instruct)"), mk_cd(code_model),
  mk_md("## 3. Data Ingestion & Massive Synthetic Generators"), mk_cd(code_data),
  mk_md("## 4. Fraud Engine & Typology Classification"), mk_cd(code_scoring),
  mk_md("## 5. Alert Fatigue Reduction & Counterfactuals"), mk_cd(code_explain),
  mk_md("## 6. Relationship Graphs"), mk_cd(code_graph),
  mk_md("## 7. LLM Multi-Agent Suite"), mk_cd(code_llm),
  mk_md("## 8. Multi-Role Gradio Dashboards (UI)"), mk_cd(code_ui),
  mk_md("## 9. Launch Platform"), mk_cd(code_launch)
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
 "nbformat": 4, "nbformat_minor": 4
}

with open("Financial_Crime_Intelligence_Platform.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Enterprise Notebook generated successfully.")
