# P26 GPT Variant Benchmark Board

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://8gdxxyermqk2b32altihlg.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/iGoTerror/P_26)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)

An end-to-end benchmarking and evaluation suite for custom Transformer decoders built from scratch in pure NumPy. The project compares four differently sized causal GPT-style text decoders and a multimodal vision-language decoder on a synthetic e-commerce customer support domain ("E-Shop").

---

## 🚀 Live Demo

- **Interactive Streamlit Web App**: **[https://8gdxxyermqk2b32altihlg.streamlit.app/](https://8gdxxyermqk2b32altihlg.streamlit.app/)**
- **Static Board (HTML/JS)**: Available in `dist/index.html` (deployable to GitHub Pages).

---

## 📌 Models Evaluated

All models are single-head causal self-attention decoders implemented with pure NumPy analytical gradients:

| Model | Kind | Parameters | Weights (FP32) | Latency (ms) | Mean F1 | Exact Match | Key Trait |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GPT-16** | Text | 7,040 | 27.5 KiB | 2.33 ms | **0.977** | **88.9%** | Best overall accuracy & latency balance |
| **GPT-24** | Text | 12,096 | 47.3 KiB | 2.13 ms | 0.704 | 55.6% | Medium-small baseline |
| **GPT-32 base** | Text (Control) | 18,176 | 71.0 KiB | 2.21 ms | 0.616 | 44.4% | Pre-adaptation baseline control |
| **GPT-32 + LoRA** | Text | 18,432 | 72.0 KiB | 2.25 ms | 0.638 | 55.6% | Rank-4 LoRA on Value projection ($V + A \cdot B$) |
| **GPT-48** | Text | 39,408 | 154.0 KiB | 2.76 ms | 0.643 | 55.6% | Largest parameter footprint |
| **GPT-Vision** | Vision + Text | 28,140 | 110.0 KiB | 3.51 ms | 0.732 | 44.4% | $2 \times 2$ RGB patch projection for product swatches |

---

## 💻 Running the Streamlit App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iGoTerror/P_26.git
   cd P_26
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   Or:
   ```bash
   python -m streamlit run streamlit_app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Static Web Dashboard

You can also run the bundled zero-dependency client dashboard:
```powershell
python -m http.server 8000 --directory dist
```
Then visit `http://localhost:8000` or double-click `dist/index.html`.

---

## 🔬 Reproduce the Benchmark Run

To re-train all models from seed 26, measure latency/memory/energy, and rebuild `dist/`:

1. **Install requirements & download MiniLM BERT encoder:**
   ```powershell
   python -m pip install -r requirements.txt huggingface_hub
   hf download sentence-transformers/all-MiniLM-L6-v2 model.safetensors tokenizer.json --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir encoder
   ```

2. **Run benchmark:**
   ```powershell
   python benchmark.py --encoder encoder
   ```

3. **Generate Submission PDF Report:**
   ```powershell
   python report.py --roll YOUR_ROLL --name "YOUR NAME" --live-url https://8gdxxyermqk2b32altihlg.streamlit.app/ --github-url https://github.com/iGoTerror/P_26 --output P26_Benchmark_Report.pdf
   ```

---

## 📊 Evaluation Metrics & Methodology

- **Accuracy & Semantics**:
  - **Exact Match**: Case- and punctuation-normalized strict equality.
  - **BERTScore (Precision, Recall, F1)**: Greedy cosine matching on contextual WordPiece embeddings from a frozen MiniLM encoder.
- **Hardware & Efficiency**:
  - **Latency**: Median latency across 5 warmed serial trials per prompt.
  - **Memory (RSS)**: Sampled process resident set size via `psutil`.
  - **Energy & Cost**: Wh estimated by $W \times \text{ms} / 3{,}600{,}000$ (default 35W) and USD cost at \$0.12/kWh.
- **Demographic Fairness**:
  - Counterfactual name-swap test (*Aisha* vs. *Arjun*) on identical refund queries to verify consistent, unbiased policy adherence.
- **Visual Grounding**:
  - Swatch color identification test on held-out images alongside three sanity-check controls (red, green, blue).

---

## 📂 Project Structure

```
├── streamlit_app.py           # Streamlit web dashboard
├── app.py                     # Entrypoint redirect to streamlit_app
├── benchmark.py               # Main training & evaluation benchmark runner
├── decoder.py                 # Pure NumPy causal self-attention decoder + LoRA
├── dataset.py                 # E-Shop policy dataset & evaluation prompts
├── semantic.py                # BERTScore contextual similarity calculator
├── charts.py                  # Matplotlib performance charts generator
├── report.py                  # Automated academic PDF report generator
├── requirements.txt           # Python dependencies
├── .github/workflows/pages.yml# GitHub Pages deployment workflow
└── dist/                      # Static web assets, charts, and recorded run data
    ├── index.html
    ├── results.json
    ├── leaderboard.csv
    ├── benchmark.png
    └── swatch.png
```
