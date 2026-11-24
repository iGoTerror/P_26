"""Train and evaluate the five entrants. Run from any directory."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import csv
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import psutil
from dataset import evaluation, examples, POLICY
from decoder import Decoder
from semantic import SemanticScore

ROOT = Path(__file__).resolve().parent
SPECIAL = ("<pad>", "<unk>", "<q>", "<a>", "<eos>", "<pixel>")


def words(text):
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())


def patches(image):
    if image is None:
        return np.zeros((4, 3), dtype="f")
    rgb = np.asarray(image.resize((2, 2)), dtype="f") / 255
    return rgb.reshape(4, 3)


def prefix(question, vocab, vision):
    ids = [vocab["<pixel>"]]*4 if vision else []
    return ids + [vocab["<q>"]] + [vocab.get(w, 1) for w in words(question)] + [vocab["<a>"]]


def batch(rows, vocab, vision):
    prefixes = [prefix(q, vocab, vision) for q, _, _ in rows]
    sequences = [p + [vocab[w] for w in words(answer)] + [vocab["<eos>"]]
                 for p, (_, answer, _) in zip(prefixes, rows)]
    length = max(map(len, sequences))
    ids = np.asarray([s + [0]*(length-len(s)) for s in sequences])
    mask = np.asarray([[len(p)-1 <= i < len(s)-1 for i in range(length-1)]
                       for p, s in zip(prefixes, sequences)])
    rgb = np.asarray([patches(image) for _, _, image in rows]) if vision else None
    return ids[:, :-1], ids[:, 1:], mask, rgb


def generate(model, question, image, vocab):
    ids = prefix(question, vocab, model.vision)
    rgb = patches(image)[None] if model.vision else None
    tokens = list(vocab)
    answer = []
    for _ in range(20):
        logits = model.forward(np.asarray([ids]), rgb)[0, -1]
        logits[[0, 1, 2, 3, 5]] = -np.inf
        index = int(logits.argmax())
        if index == vocab["<eos>"]:
            break
        answer.append(tokens[index])
        ids.append(index)
    return re.sub(r" ([.,?!'])", r"\1", " ".join(answer))


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    truth = evaluation()
    train = examples(True) + examples(adaptation=True)
    vocab = {token: i for i, token in enumerate(SPECIAL + tuple(sorted(
        {w for q, a, _ in train for w in words(q) + words(a)})))}
    known_training = {q for q, _, _ in train}
    assert len(truth) == 10 and not any(q in known_training for _, _, q, _, _ in truth)
    prompts = [dict(id=key, group=group, text=q, reference=ref)
               for key, group, q, ref, _ in truth]
    truth[-1][-1].save(output / "swatch.png")
    details, models = [], []
    for name, width, vision in [("GPT-16", 16, False), ("GPT-24", 24, False),
                                ("GPT-32 + LoRA", 32, False), ("GPT-48", 48, False),
                                ("GPT-Vision", 40, True)]:
        model = Decoder(len(vocab), width, 26, vision)
        model.fit(*batch(examples(vision), vocab, vision), args.steps, .009)

        def measure(label, ranked=True):
            process = psutil.Process()
            generate(model, truth[0][2], None, vocab)
            base_rss = process.memory_info().rss
            for key, group, q, ref, image in truth:
                if image is not None and not vision:
                    continue
                timings, rss = [], []
                for _ in range(args.repeats):
                    start = perf_counter_ns()
                    answer = generate(model, q, image, vocab)
                    timings.append((perf_counter_ns()-start)/1e6)
                    rss.append(process.memory_info().rss)
                latency = float(np.median(timings))
                energy = args.watts * latency / 3_600_000
                details.append(dict(model=label, prompt=key, group=group, response=answer,
                                    reference=ref, exact=words(answer)==words(ref),
                                    latency_ms=latency, rss_mib=max(rss)/2**20,
                                    wh=energy, cost_usd=energy/1000*args.rate, timing_ms=timings))
            params = sum(p.size for p in model.params.values())
            adapter = sum(model.params[k].size for k in ("a", "b") if k in model.params)
            models.append(dict(model=label, kind="Vision + text" if vision else "Text",
                               params=params, adapter=adapter, weights_kib=4*params/1024,
                               baseline_rss_mib=base_rss/2**20, ranked=ranked))

        if name.endswith("LoRA"):
            measure("GPT-32 base", False)
            before = {k: v.copy() for k, v in model.params.items()}
            model.add_lora()
            model.fit(*batch(examples(adaptation=True), vocab, False), 160, .015)
            assert all(np.array_equal(v, model.params[k]) for k, v in before.items())
            assert np.any(model.params["b"] != 0)
        measure(name)
        if vision:
            for color in ("red", "green", "blue"):
                from dataset import swatch
                from random import Random
                details.append(dict(model=name, prompt=f"control-{color}", group="Control",
                                    response=generate(model, truth[-1][2], swatch(color, Random(275)), vocab),
                                    reference=f"The swatch is {color}."))
        print("Measured", name, flush=True)
    scorer = SemanticScore(args.encoder)
    for row in details:
        row["precision"], row["recall"], row["f1"] = scorer.score(row["response"], row["reference"])
        row.setdefault("exact", words(row["response"]) == words(row["reference"]))
    for model in models:
        rows = {r["prompt"]: r for r in details if r["model"] == model["model"]}
        common = [r for r in rows.values() if r["group"] in ("Shop", "Bias")]
        for key in ("f1", "exact", "latency_ms", "wh", "cost_usd"):
            model[key] = float(np.mean([r[key] for r in common]))
        model["rss_mib"] = max(r["rss_mib"] for r in common)
        model["bias_delta"] = abs(rows["bias-a"]["f1"] - rows["bias-b"]["f1"])
        model["bias_match"] = rows["bias-a"]["response"] == rows["bias-b"]["response"]
        model["bias_pass"] = rows["bias-a"]["exact"] and rows["bias-b"]["exact"]
        model["vision_f1"] = rows.get("vision", {}).get("f1")
    result = dict(models=models, prompts=prompts, details=details, watts=args.watts, price=args.rate,
                  steps=args.steps, repeats=args.repeats, seed=26, platform=platform.platform(),
                  processor=platform.processor(), numpy=np.__version__, encoder_revision=args.encoder.name,
                  timestamp=datetime.now(timezone.utc).isoformat(), policy=list(POLICY.values()))
    serialized = json.dumps(result, indent=2)
    (output/"results.json").write_text(serialized, encoding="utf-8")
    (output/"recorded-run.js").write_text("globalThis.benchmarkRun = " + serialized + ";\n", encoding="utf-8")
    with (output/"leaderboard.csv").open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=list(models[0]))
        csv_writer.writeheader()
        csv_writer.writerows(models)
    from charts import plot
    plot(result, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True, help="MiniLM snapshot with safetensors and tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT/"dist")
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--watts", type=float, default=35)
    parser.add_argument("--rate", type=float, default=.12, help="USD per kWh")
    args = parser.parse_args()
    if min(args.steps, args.repeats) < 1 or not all(np.isfinite(v) and v >= 0 for v in (args.watts, args.rate)):
        parser.error("Steps and repeats must be positive. Watts and rate must be finite and nonnegative.")
    run(args)
