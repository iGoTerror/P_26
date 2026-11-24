# P26 GPT Variant Benchmark Board

The static app visualizes measured outputs from a Python benchmark for a synthetic E-Shop. Four differently sized GPT-style text decoders share nine text cases with a fifth, vision-language decoder. The tenth prompt requires an image. The dashboard includes a sortable comparison table, Python-generated charts, the complete prompt and response explorer, and CSV export.

## View the board

```powershell
python -m http.server 8000 --directory dist
```

Open http://localhost:8000, or open `dist/index.html` directly. Static hosting should publish the contents of `dist`. For GitHub Pages, choose GitHub Actions as the Pages source in repository settings.

## Reproduce the run

```powershell
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
hf download sentence-transformers/all-MiniLM-L6-v2 model.safetensors tokenizer.json --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir encoder
python benchmark.py --encoder encoder
python report.py --roll YOUR_ROLL --name YOUR_NAME --output YOUR_ROLL_YOUR_NAME_P26.pdf
```

`benchmark.py` trains all models from seed 26, measures inference, and rebuilds the JSON, CSV, swatch and charts in `dist`. Weights are retrained on each run. The bundled snapshot is a recorded CPU run. Training and scoring run offline after acquiring the MiniLM encoder. Set `--watts` and `--rate` to change the electricity assumptions. The report script accepts `--live-url` and `--github-url` for verified links.

## Scoring and limitations

- A common text slice compares all entrants fairly. Vision is evaluated on a held-out swatch and three same-question color checks.
- A small single-head decoder generates answers autoregressively from a word vocabulary. GPT-32 uses rank-4 LoRA on the value projection, with the remaining weights frozen. Its source checkpoint is an unranked control.
- Raw BERTScore precision, recall and F1 use greedy cosine matching of final-layer, contextual MiniLM WordPiece embeddings. No IDF weights or baseline rescaling are used. Exact match normalizes case and punctuation spacing.
- Five warmed, serial trials per prompt produce a median latency per prompt. The summary averages those medians. Weight bytes and sampled process RSS use distinct units. Sampled RSS contains Python and previously allocated runtime memory.
- Energy estimates use W × ms ÷ 3,600,000 in Wh. Cost is Wh ÷ 1,000 × USD/kWh. The estimate uses runtime and a power assumption. It excludes training, idle usage and hosting overhead.
- One name swap tests consistency for a single question. Distinct names are represented in training vocabulary through unrelated cases. The probe also checks matching the expected equal-treatment answer. The benchmark uses synthetic, closed-domain data and a single seed.
