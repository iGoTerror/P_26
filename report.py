"""Build the P26 submission PDF from the measured snapshot."""
import argparse
import json
from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(24, 93, 91)
        self.cell(0, 7, "P26  /  GPT VARIANT BENCHMARK BOARD", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", size=9)
        self.set_text_color(54, 80, 83)
        self.cell(0, 6, f"P26  /  {self.page_no()}", align="R")

    def page_title(self, text):
        self.set_font("Helvetica", "B", 23)
        self.set_text_color(17, 45, 53)
        self.multi_cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def heading(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(24, 93, 91)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

    def para(self, text):
        self.set_font("Helvetica", size=11)
        self.set_text_color(25, 43, 52)
        self.multi_cell(0, 5.4, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def url(self, label, url):
        self.set_font("Helvetica", "U", 10)
        self.set_text_color(24, 93, 91)
        self.multi_cell(0, 5, f"{label}  {url}", link=url, new_x="LMARGIN", new_y="NEXT")


def build(args):
    data = json.loads((ROOT/"dist/results.json").read_text(encoding="utf-8"))
    rows = sorted((r for r in data["models"] if r["ranked"]), key=lambda r: -r["f1"])
    base = next(r for r in data["models"] if not r["ranked"])
    lora = next(r for r in rows if r["adapter"])
    vision = next(r for r in rows if r["vision_f1"] is not None)
    pdf = Report()
    pdf.set_margins(16, 15, 16)
    pdf.set_auto_page_break(True, 17)
    pdf.set_title("P26 GPT Variant Benchmark Board")
    pdf.set_author(args.name)
    pdf.add_page()
    pdf.page_title("GPT Variant Benchmark Board")
    pdf.para(f"Project P26 | {args.roll} | {args.name}")
    pdf.para("A controlled CPU comparison of four GPT-style text decoders and a small vision-language model for a synthetic E-Shop chatbot.")
    if args.live_url:
        pdf.url("Live app", args.live_url)
    if args.github_url:
        pdf.url("GitHub", args.github_url)
    pdf.heading("U1 / Evaluation design")
    pdf.para("The seven shop questions test policy retrieval. Two names form a counterfactual bias pair and the final prompt tests visual grounding. All five models take the same nine text prompts. The separately scored tenth prompt includes image pixels.")
    for i, p in enumerate(data["prompts"], 1):
        pdf.para(f"{i:02d}  {p['text']}\n      Reference: {p['reference']}")
    pdf.para("Primary metric: mean BERTScore F1. Secondary metrics: normalized exact match, generation latency, FP32 weight memory, sampled process RSS, and estimated inference energy and electricity cost.")

    pdf.add_page()
    pdf.page_title("U2 / Size and LoRA")
    pdf.para("All entrants generate answer tokens left to right with a causal attention mask, learned position and token embeddings, a residual feed-forward block and a vocabulary head. Width controls the parameter count. Base training uses synthetic shop templates. Evaluation questions are withheld. The shop policy answers are shared by design.")
    pdf.para(f"GPT-32 was adapted with rank-4 LoRA on its value projection. The effective matrix is W + AB. A and B represent {lora['adapter']} trainable parameters and add 1 KiB in FP32. A frozen-weight equality check was performed after 160 adapter updates on eight separate paraphrases.")
    pdf.para(f"The paired pre-adaptation checkpoint moved from {base['params']:,} to {lora['params']:,} parameters. Its F1 changed from {base['f1']:.3f} to {lora['f1']:.3f} and exact match from {base['exact']:.1%} to {lora['exact']:.1%}. This measures the combined effect of the adapter and the additional supervised examples.")
    pdf.heading("Measured leaderboard")
    pdf.set_font("Helvetica", size=8.5)
    with pdf.table(width=178, col_widths=(28, 15, 12, 14, 15, 16, 15, 15, 15), line_height=7) as table:
        table.row(("Model", "Params", "F1", "Exact %", "ms", "KiB", "RSS MiB", "uWh*", "nUSD*"))
        for r in rows:
            table.row((r["model"], f"{r['params']:,}", f"{r['f1']:.3f}", f"{r['exact']*100:.1f}",
                       f"{r['latency_ms']:.2f}", f"{r['weights_kib']:.1f}", f"{r['rss_mib']:.1f}",
                       f"{r['wh']*1e6:.2f}", f"{r['cost_usd']*1e9:.2f}"))
    pdf.ln(3)
    pdf.para(f"* Estimated at {data['watts']:g} W and USD {data['price']:g}/kWh. RSS is total process resident memory sampled after generation. Weight memory is the FP32 parameter footprint.")
    pdf.image(str(ROOT/"dist/benchmark.png"), w=178)

    pdf.add_page()
    pdf.page_title("U3 / GPT, BERT and multimodality")
    pdf.heading("GPT variants and the E-Shop chatbot")
    pdf.para("Decoder-only GPT uses causal context to predict the next token. GPT-2 popularized generative transfer through language modeling. GPT-3 demonstrated in-context learning through prompting. In this experiment, the controlled variants share architecture and training rules with different widths. The E-Shop chatbot takes a customer question and generates the shop policy or product description.")
    pdf.para("A production design could retrieve current policy, order and product data, restrict answers to verified facts, and hand off ambiguous or expensive actions to staff. The experiment uses a fixed fictional policy and avoids real transactions. The measured exact-match rates show how often each generated response reproduced the policy reference.")
    pdf.heading("BERT and contextual comparison")
    pdf.para("BERT encodes tokens using bidirectional context. The scorer uses a frozen six-layer MiniLM BERT encoder. Each candidate token matches its most similar reference token by cosine similarity to compute precision. Reversing the match yields recall. Their harmonic mean is raw F1. The implementation excludes special tokens and uses the final hidden layer with uniform token weights.")
    pdf.para("This process can reward paraphrases. It can also give a partial score to an answer with a wrong number, missing condition or irrelevant text. Exact policy match and raw outputs expose these cases for review.")
    pdf.heading("Cross-modal model and diffusion")
    pdf.para(f"The vision entrant pools user-supplied RGB pixels into four patches, linearly projects them into text embedding space and inserts them as prefix tokens. Causal attention combines those image tokens with the question. The image case F1 was {vision['vision_f1']:.3f}.")
    controls = [r for r in data["details"] if r["group"] == "Control"]
    pdf.para("Same question, changed pixels: " + "; ".join(f"{r['prompt'][8:]} yielded '{r['response']}'" for r in controls) + ". This is a small grounding check.")
    pdf.para("Diffusion learns to reverse a noising process. Latent diffusion performs denoising in compressed image space. Text conditioning can guide product-image generation through cross-attention. The benchmark's cross-modal task reads product images and outputs text, so it evaluates a vision-language decoder. Diffusion remains a syllabus comparison and was outside the measured run.")

    pdf.add_page()
    pdf.page_title("U4 / Python harness and figures")
    pdf.para("dataset.py defines disjoint prompt templates. decoder.py implements the decoder, backpropagation and adaptation in NumPy. benchmark.py trains entrants, evaluates fixed inputs, computes metrics and exports JSON and CSV. semantic.py provides contextual scores. charts.py builds the comparison figures with Matplotlib. The browser reads the resulting snapshot and supports sorting and prompt-level inspection.")
    pdf.para(f"Run metadata: seed {data['seed']}; {data['steps']} base updates; {data['repeats']} warmed batch-one trials per prompt; NumPy {data['numpy']}. Host: {data['platform']}. CPU: {data['processor'] or 'not reported'}. UTC run end: {data['timestamp'][:19]}.")
    pdf.para("Generation latency covers tokenization, model decoding and image preprocessing where relevant. It excludes training and the semantic scorer. The reported latency is the mean of prompt-level medians. RSS captures sampled process memory and includes the interpreter and allocator state. Minor timing variation is expected on a repeat run.")
    pdf.heading("U5 / Cost, energy and bias")
    pdf.para(f"With assumed average draw of {data['watts']:g} W, per-response energy in Wh is watts times latency in ms divided by 3,600,000. Electricity cost is Wh divided by 1,000 times USD {data['price']:g}/kWh. These estimates exclude training, host infrastructure and measured power telemetry.")
    pdf.para(f"The Aisha and Arjun prompts differ only by name. {sum(r['bias_match'] for r in rows)} of five entrants produced identical responses and {sum(r['bias_pass'] for r in rows)} reproduced the expected equal-treatment answer on both. The check captures one behavioral pair. It cannot support broader fairness claims.")
    pdf.heading("Conclusion")
    pdf.para(f"{rows[0]['model']} had the highest mean text F1 ({rows[0]['f1']:.3f}). The small, single-seed synthetic task limits generalization. Model size, parameter memory and latency describe this run. Raw responses show where stronger shop-policy grounding is needed.")
    pdf.heading("References")
    for label, url in [
        ("GPT-3", "https://arxiv.org/abs/2005.14165"),
        ("BERT", "https://arxiv.org/abs/1810.04805"),
        ("LoRA", "https://arxiv.org/abs/2106.09685"),
        ("BERTScore", "https://arxiv.org/abs/1904.09675"),
        ("Latent Diffusion", "https://arxiv.org/abs/2112.10752"),
        ("MiniLM encoder", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"),
    ]:
        pdf.url(label, url)
    pdf.output(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll", default="RollNo pending")
    parser.add_argument("--name", default="Name pending")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-url")
    parser.add_argument("--github-url")
    build(parser.parse_args())

