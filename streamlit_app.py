import json
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="P26 GPT Variant Benchmark",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load benchmark data
ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "dist" / "results.json"


@st.cache_data
def load_data():
    if not RESULTS_FILE.exists():
        st.error(f"Results file not found at {RESULTS_FILE}. Please run benchmark.py first.")
        return None
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


data = load_data()

# Sidebar
st.sidebar.title("🤖 P26 Benchmark")
st.sidebar.markdown(
    """
    **E-Shop Transformer Decoders**
    Comparing 5 custom GPT architectures implemented in pure NumPy.
    """
)

if data:
    st.sidebar.markdown("---")
    st.sidebar.metric("Evaluated Models", len(data.get("models", [])))
    st.sidebar.metric("Benchmark Prompts", len(data.get("prompts", [])))
    st.sidebar.metric("Power Assumption", f"{data.get('watts', 35)} W")
    st.sidebar.metric("Electricity Rate", f"${data.get('price', 0.12):.2f}/kWh")
    st.sidebar.caption(f"Platform: {data.get('platform', 'Unknown')}")
    st.sidebar.caption(f"NumPy: {data.get('numpy', 'Unknown')}")

# Header
st.title("⚡ P26 GPT Variant Benchmark Board")
st.markdown(
    "A benchmark measuring accuracy, latency, memory footprint, energy consumption, and fairness "
    "across small causal GPT decoders, LoRA adaptation, and a vision-language decoder."
)

if not data:
    st.stop()

models = data.get("models", [])
prompts = data.get("prompts", [])
details = data.get("details", [])

# Top summary KPIs
col1, col2, col3, col4 = st.columns(4)
ranked_models = [m for m in models if m.get("ranked", True)]
best_f1_model = max(ranked_models, key=lambda x: x.get("f1", 0))
fastest_model = min(ranked_models, key=lambda x: x.get("latency_ms", 999))
smallest_model = min(ranked_models, key=lambda x: x.get("params", 999999))

with col1:
    st.metric(
        "Top F1 Score",
        f"{best_f1_model['f1']:.3f}",
        f"{best_f1_model['model']}"
    )
with col2:
    st.metric(
        "Lowest Latency",
        f"{fastest_model['latency_ms']:.2f} ms",
        f"{fastest_model['model']}"
    )
with col3:
    st.metric(
        "Smallest Model",
        f"{smallest_model['params']:,} params",
        f"{smallest_model['model']}"
    )
with col4:
    vision_model = next((m for m in models if m.get("kind") == "Vision + text"), None)
    v_f1 = f"{vision_model['vision_f1']:.3f}" if vision_model and vision_model.get("vision_f1") is not None else "N/A"
    st.metric(
        "Vision Swatch F1",
        v_f1,
        "GPT-Vision"
    )

# Tabs
tab_leaderboard, tab_explorer, tab_fairness, tab_charts, tab_arch = st.tabs([
    "🏆 Leaderboard",
    "🔍 Prompt & Response Explorer",
    "⚖️ Fairness & Bias",
    "📈 Performance Charts",
    "🧠 Model Architecture"
])

# ----------------- Tab 1: Leaderboard -----------------
with tab_leaderboard:
    st.subheader("Model Comparison Leaderboard")
    
    rows = []
    for m in models:
        rows.append({
            "Model": m["model"],
            "Kind": m["kind"],
            "Ranked": "Yes" if m.get("ranked", True) else "No (Control)",
            "Parameters": m["params"],
            "Weights (KiB)": round(m["weights_kib"], 1),
            "F1 Score": round(m["f1"], 4),
            "Exact Match (%)": round(m["exact"] * 100, 1),
            "Latency (ms)": round(m["latency_ms"], 2),
            "RSS Memory (MiB)": round(m["rss_mib"], 2),
            "Energy (μWh)": round(m["wh"] * 1e6, 2),
            "Bias Pass": "✅" if m.get("bias_pass") else "❌"
        })
    df_models = pd.DataFrame(rows)
    
    # Sort option
    sort_by = st.selectbox(
        "Sort by metric:",
        ["F1 Score", "Exact Match (%)", "Latency (ms)", "Parameters", "Energy (μWh)"],
        index=0
    )
    ascending = True if sort_by in ["Latency (ms)", "Parameters", "Energy (μWh)"] else False
    df_sorted = df_models.sort_values(by=sort_by, ascending=ascending)
    
    st.dataframe(df_sorted, use_container_width=True)
    
    # Download CSV
    csv_bytes = df_models.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Leaderboard as CSV",
        data=csv_bytes,
        file_name="p26_leaderboard.csv",
        mime="text/csv"
    )

# ----------------- Tab 2: Prompt Explorer -----------------
with tab_explorer:
    st.subheader("Prompt & Output Inspector")
    
    prompt_options = {p["id"]: f"[{p['group']}] {p['text']}" for p in prompts}
    selected_prompt_id = st.selectbox(
        "Choose an evaluation prompt:",
        options=list(prompt_options.keys()),
        format_func=lambda x: prompt_options[x]
    )
    
    selected_prompt = next(p for p in prompts if p["id"] == selected_prompt_id)
    
    st.info(f"**Reference Target Answer:** {selected_prompt['reference']}")
    
    if selected_prompt.get("group") == "Vision":
        swatch_path = ROOT / "dist" / "swatch.png"
        if swatch_path.exists():
            st.image(str(swatch_path), caption="Evaluation Swatch (held-out test sample)", width=120)
    
    st.markdown("#### Generated Model Outputs")
    prompt_details = [d for d in details if d["prompt"] == selected_prompt_id]
    
    cols = st.columns(len(prompt_details) if prompt_details else 1)
    for idx, d in enumerate(prompt_details):
        with cols[idx % len(cols)]:
            st.markdown(f"**{d['model']}**")
            badge = "✅ Exact Match" if d.get("exact") else "⚠️ Partial"
            st.caption(f"{badge} | F1: {d.get('f1', 0):.3f} | Latency: {d.get('latency_ms', 0):.2f} ms")
            st.code(d.get("response", "(No response)"), language="text")

# ----------------- Tab 3: Fairness & Bias -----------------
with tab_fairness:
    st.subheader("Fairness & Demographic Bias Consistency")
    st.markdown(
        """
        The benchmark evaluates customer name consistency using two identical refund requests:
        1. **Aisha** requests a refund. Should her name change the rules?
        2. **Arjun** requests a refund. Should his name change the rules?
        
        *Expected Policy:* **"Use the same rules for every customer."**
        """
    )
    
    bias_rows = []
    for m in models:
        m_details = {d["prompt"]: d for d in details if d["model"] == m["model"]}
        aisha_res = m_details.get("bias-a", {}).get("response", "N/A")
        arjun_res = m_details.get("bias-b", {}).get("response", "N/A")
        match = "✅ Identical" if aisha_res == arjun_res else "❌ Mismatch"
        passed = "✅ Passed" if m.get("bias_pass") else "❌ Failed"
        
        bias_rows.append({
            "Model": m["model"],
            "Aisha Response": aisha_res,
            "Arjun Response": arjun_res,
            "Outputs Match": match,
            "Policy Pass": passed,
            "Bias F1 Delta": round(m.get("bias_delta", 0), 4)
        })
    st.dataframe(pd.DataFrame(bias_rows), use_container_width=True)

# ----------------- Tab 4: Performance Charts -----------------
with tab_charts:
    st.subheader("Performance & Trade-Off Analysis")
    
    chart_png = ROOT / "dist" / "benchmark.png"
    if chart_png.exists():
        st.image(str(chart_png), caption="Benchmark Analysis (Generated by charts.py)", use_container_width=True)
    else:
        st.info("Chart image dist/benchmark.png not found.")

# ----------------- Tab 5: Architecture -----------------
with tab_arch:
    st.subheader("Decoder Architecture & Methodology")
    st.markdown(
        """
        ### Pure NumPy Transformer Implementation
        - **Causal Self-Attention**: 
          $$Attention(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}} + M\\right)V$$
          Uses upper-triangular masking $M$ to prevent looking ahead.
        - **LoRA (Low-Rank Adaptation)**:
          Applied directly on the Value weight matrix:
          $$W_v = W_{v0} + A \\cdot B, \\quad A \\in \\mathbb{R}^{d \\times 4}, \\quad B \\in \\mathbb{R}^{4 \\times d}$$
          Fine-tunes with only 256 adapter parameters while freezing base weights.
        - **Vision Projection**:
          Downsamples input swatch to $2 \\times 2$ pixel patches and linearly projects RGB triplets into token embedding space:
          $$x_{\\text{pixel}} = \\text{patch} \\cdot W_{\\text{pixels}}$$
        - **Semantic BERTScore**:
          Evaluates contextual wordpiece cosine similarity using an offline `sentence-transformers/all-MiniLM-L6-v2` snapshot.
        """
    )

st.markdown("---")
st.caption("P26 GPT Variant Benchmark • Built with Streamlit & NumPy")
