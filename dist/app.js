const el = id => document.getElementById(id);
const num = (value, digits) => Number(value).toFixed(digits);
function node(tag, value, className) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (value != null) item.textContent = value;
  return item;
}

function draw(data) {
  const ranked = data.models.filter(model => model.ranked);
  const best = [...ranked].sort((a,b) => b.f1-a.f1)[0];
  for (const [label, value] of [["Ranked entrants", ranked.length], ["Evaluation prompts", data.prompts.length],
    ["Top text F1", num(best.f1,3)], ["Trials per prompt", data.repeats]]) {
    const card = node("article", null, "card");
    card.append(node("span", label), node("strong", value));
    el("cards").append(card);
  }
  function renderTable() {
    const key = el("sort").value;
    const sign = ["f1", "exact"].includes(key) ? -1 : 1;
    const ordered = [...ranked].sort((a,b) => sign*(a[key]-b[key]) || a.model.localeCompare(b.model));
    el("leaders").replaceChildren(...ordered.map((r, i) => {
      const row = node("tr");
      for (const cell of [i+1, r.model, r.params.toLocaleString("en"), num(r.f1,3),
        num(100*r.exact,1)+"%", num(r.latency_ms,2), num(r.weights_kib,1),
        num(r.rss_mib,1), num(1e6*r.wh,2), num(1e9*r.cost_usd,2)]) row.append(node("td", cell));
      return row;
    }));
  }
  el("sort").addEventListener("change", renderTable);
  renderTable();
  el("assumptions").textContent = `* Estimated at ${data.watts} W and $${data.price}/kWh. RSS is total process resident memory sampled after inference. Weight memory counts FP32 parameter bytes.`;
  for (const p of data.prompts) {
    const option = node("option", p.group + " / " + p.id);
    option.value = p.id;
    el("prompt").append(option);
  }
  function renderPrompt() {
    const p = data.prompts.find(prompt => prompt.id === el("prompt").value);
    const view = el("promptView");
    view.replaceChildren(node("p", p.text, "prompt-text"), node("p", "Reference · " + p.reference));
    if (p.group === "Vision") {
      const image = node("img");
      image.src = "swatch.png";
      image.alt = "Evaluation product swatch";
      image.width = image.height = 90;
      view.append(image, node("p", "GPT-Vision accepts the image. Text-only rows are unavailable for this prompt."));
      const controls = data.details.filter(row => row.group === "Control");
      view.append(node("p", "Same question with alternate pixels · " + controls.map(row =>
        row.prompt.slice(8) + ": " + row.response).join(" / ")));
    }
    const cards = node("div", null, "responses");
    for (const r of data.details.filter(row => row.prompt === p.id)) {
      const card = node("article", null, "response");
      card.append(node("h3", r.model), node("p", r.response || "(empty output)"),
        node("p", `F1 ${num(r.f1,3)} · ${num(r.latency_ms,2)} ms · ${r.exact ? "Exact match" : "Different wording"}`));
      cards.append(card);
    }
    view.append(cards);
  }
  el("prompt").addEventListener("change", renderPrompt);
  renderPrompt();
  const base = data.models.find(m => !m.ranked);
  const tuned = data.models.find(m => m.adapter);
  el("lora").textContent = `GPT-32 base F1 ${num(base.f1,3)}. Tuned F1 ${num(tuned.f1,3)} with ${tuned.adapter} added trainable parameters. Exact match ${num(base.exact*100,1)}% before and ${num(tuned.exact*100,1)}% after.`;
  el("bias").textContent = `${ranked.filter(r=>r.bias_match).length} of ${ranked.length} entrants gave identical answers for Aisha and Arjun. ${ranked.filter(r=>r.bias_pass).length} matched the equal-treatment reference for both.`;
  el("method").textContent = `Seed ${data.seed}, ${data.steps} base updates per entrant, ${data.repeats} warmed batch-one trials per prompt. The table averages prompt-level median latencies. Recorded on ${data.platform}, CPU, NumPy ${data.numpy}.`;
  el("status").hidden = true;
  el("board").hidden = false;
}

if (globalThis.benchmarkRun) draw(globalThis.benchmarkRun);
else el("status").textContent = "Results could not load. Check that recorded-run.js is next to this page.";
