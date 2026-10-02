(() => {
  const data = window.SGO_SHOWCASE;
  if (!data) {
    const el = document.getElementById("proof-lede");
    if (el) {
      el.textContent =
        "Showcase data missing. Run: python scripts/build_showcase_data.py";
    }
    return;
  }

  const fmt = (n, digits = 0) =>
    Number(n).toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  const pct = (n) => `${fmt(100 * n, 1)}%`;

  const repo = document.getElementById("repo-link");
  if (repo && data.meta.repo_url) repo.href = data.meta.repo_url;

  document.getElementById("footer-vintage").textContent =
    `Data vintage: ${data.meta.data_vintage}`;

  // Approach rail
  const rail = document.getElementById("tool-rail");
  rail.innerHTML = (data.tools_plain || [])
    .map(
      (tool, i) => `
      <li>
        <span class="tool-id">0${i + 1}</span>
        <div>
          <h3>${tool.name}</h3>
          <p><strong>Does:</strong> ${tool.does}</p>
          <p class="example"><strong>Example:</strong> ${tool.example}</p>
        </div>
      </li>`
    )
    .join("");

  // Operating point
  document.getElementById("op-strip").innerHTML = [
    ["Model", data.operating_point.model],
    ["Threshold", String(data.operating_point.threshold)],
    ["Synth precision", pct(data.operating_point.precision)],
    ["Synth recall", pct(data.operating_point.recall)],
  ]
    .map(
      ([label, value]) => `
      <div class="op-cell">
        <span class="label">${label}</span>
        <span class="value">${value}</span>
      </div>`
    )
    .join("");
  document.getElementById("op-rationale").textContent =
    `${data.operating_point.rationale} Calibration default: ${data.operating_point.calibration}.`;

  // Proof metrics
  document.getElementById("proof-lede").textContent =
    `Canonical tracked run on ${data.meta.data_vintage}: ${fmt(data.corpus.combined)} cleaned records (${fmt(data.corpus.ads_clean)} ADS · ${fmt(data.corpus.adas_clean)} ADAS).`;

  document.getElementById("metric-strip").innerHTML = [
    {
      label: "Cleaned records",
      value: fmt(data.corpus.combined),
      hint: "Latest-version dedupe",
    },
    {
      label: "Rule flags",
      value: fmt(data.signals.heuristic_flags),
      hint: "Hard inconsistencies",
    },
    {
      label: "ML flags",
      value: fmt(data.signals.classifier_flags),
      hint: `${data.operating_point.model} @ ${data.operating_point.threshold}`,
    },
    {
      label: "Review queue",
      value: fmt(data.signals.consensus_queue),
      hint: "Fused candidates",
    },
  ]
    .map(
      (m) => `
      <div class="metric">
        <span class="label">${m.label}</span>
        <span class="value">${m.value}</span>
        <span class="hint">${m.hint}</span>
      </div>`
    )
    .join("");

  const d = data.disagreement;
  document.getElementById("disagree-copy").textContent =
    `Local LLM on ${fmt(data.signals.phase2_sample)} narratives: ${fmt(d.all_agree)} three-way agreements, ${fmt(d.phase1_phase2_disagree)} model/LLM disagreements, ${fmt(d.phase1_phase2_agree_vs_reported)} cases where both disagreed with the reported label.`;

  const bars = [
    ["All agree", d.all_agree],
    ["Model ↔ LLM disagree", d.phase1_phase2_disagree],
    ["Both vs reported", d.phase1_phase2_agree_vs_reported],
  ];
  const maxBar = Math.max(...bars.map(([, n]) => n), 1);
  document.getElementById("disagree-bars").innerHTML = bars
    .map(
      ([label, n]) => `
      <div class="bar-row">
        <span>${label}</span>
        <div class="bar-track"><div class="bar-fill" data-width="${(100 * n) / maxBar}"></div></div>
        <strong>${fmt(n)}</strong>
      </div>`
    )
    .join("");

  const overlaps = Object.values(data.validation.overlaps || {}).reduce(
    (a, b) => a + Number(b || 0),
    0
  );
  document.getElementById("validation-copy").textContent =
    `${fmt(data.validation.present)} of ${fmt(data.validation.reference_ids)} named PE24031/EA26002 Report IDs appear in this snapshot. Overlap with project flags: ${fmt(overlaps)} (directional only).`;
  document.getElementById("caveat-list").innerHTML = (data.validation.caveats || [])
    .slice(0, 3)
    .map((c) => `<li>${c}</li>`)
    .join("");

  // Mini demo
  const cases = data.demo_cases || [];
  const picks = document.getElementById("demo-picks");
  const demoId = document.getElementById("demo-id");
  const demoSummary = document.getElementById("demo-summary");
  const demoReported = document.getElementById("demo-reported");
  const demoVerdict = document.getElementById("demo-verdict");
  const demoRun = document.getElementById("demo-run");
  let activeCase = cases[0] || null;
  let timers = [];

  const kindLabel = {
    rule_only: "Rule disagreement",
    llm_only: "LLM disagreement",
    model_flag: "ML disagreement",
    agreement: "All agree",
  };

  const resetSteps = () => {
    timers.forEach((t) => clearTimeout(t));
    timers = [];
    document.querySelectorAll(".demo-step").forEach((el) => {
      el.classList.remove("live", "agree", "disagree");
      el.querySelector(".step-body").textContent = "Waiting…";
    });
    demoVerdict.hidden = true;
    demoVerdict.classList.remove("flagged");
  };

  const selectCase = (c) => {
    activeCase = c;
    resetSteps();
    demoId.textContent = c.id;
    demoSummary.textContent = c.summary;
    demoReported.textContent = `Reported label: ${c.reported} · ${c.entity}`;
    document.querySelectorAll(".demo-pick").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.id === c.id);
    });
  };

  picks.innerHTML = cases
    .map(
      (c) => `
      <button type="button" class="demo-pick" data-id="${c.id}">
        ${kindLabel[c.kind] || c.kind}
      </button>`
    )
    .join("");

  picks.querySelectorAll(".demo-pick").forEach((btn) => {
    btn.addEventListener("click", () => {
      const c = cases.find((x) => x.id === btn.dataset.id);
      if (c) selectCase(c);
    });
  });

  if (activeCase) selectCase(activeCase);

  const fillStep = (key, payload) => {
    const el = document.querySelector(`.demo-step[data-step="${key}"]`);
    if (!el) return;
    el.classList.add("live");
    el.classList.add(payload.disagrees ? "disagree" : "agree");
    const status = payload.disagrees ? "Disagrees" : "Agrees";
    let extra = "";
    if (payload.predicted) {
      extra = ` → ${payload.predicted}`;
      if (payload.confidence != null) extra += ` (${pct(payload.confidence)})`;
    }
    el.querySelector(".step-body").textContent =
      `${status}${extra}. ${payload.plain}`;
  };

  demoRun.addEventListener("click", () => {
    if (!activeCase) return;
    resetSteps();
    const sequence = [
      [0, "rule", activeCase.rule],
      [700, "model", activeCase.model],
      [1400, "llm", activeCase.llm],
    ];
    sequence.forEach(([delay, key, payload]) => {
      timers.push(
        setTimeout(() => {
          fillStep(key, payload);
        }, delay)
      );
    });
    timers.push(
      setTimeout(() => {
        const flagged =
          activeCase.rule.disagrees ||
          activeCase.model.disagrees ||
          activeCase.llm.disagrees;
        demoVerdict.hidden = false;
        demoVerdict.textContent = activeCase.verdict_plain;
        demoVerdict.classList.toggle("flagged", flagged);
      }, 2100)
    );
  });

  // Queue
  const body = document.getElementById("queue-body");
  const meta = document.getElementById("queue-meta");
  const search = document.getElementById("queue-search");
  let filter = "all";

  const renderQueue = () => {
    const q = (search.value || "").trim().toLowerCase();
    const rows = data.queue.filter((row) => {
      if (filter === "heuristic" && !row.heuristic) return false;
      if (filter === "phase1" && !row.phase1) return false;
      if (filter === "phase2" && !row.phase2) return false;
      if (!q) return true;
      return (
        row.report_id.toLowerCase().includes(q) ||
        row.entity.toLowerCase().includes(q)
      );
    });

    body.innerHTML = rows
      .map((row) => {
        const why = [];
        if (row.rules) why.push(row.rules.replaceAll("_", " "));
        if (row.phase1_pred) {
          why.push(
            `ML→ ${row.phase1_pred}${
              row.phase1_conf != null ? ` (${pct(row.phase1_conf)})` : ""
            }`
          );
        }
        if (row.phase2_pred) why.push(`LLM→ ${row.phase2_pred}`);
        return `
          <tr>
            <td class="mono">${row.report_id}</td>
            <td>${row.entity}</td>
            <td>${row.reported}</td>
            <td>
              <div class="signal-pills">
                <span class="pill ${row.heuristic ? "on" : ""}">Rule</span>
                <span class="pill ${row.phase1 ? "on" : ""}">ML</span>
                <span class="pill ${row.phase2 ? "on" : ""}">LLM</span>
              </div>
            </td>
            <td>${why.join(" · ") || "—"}</td>
          </tr>`;
      })
      .join("");

    meta.textContent = `Showing ${fmt(rows.length)} of ${fmt(data.queue.length)} queued records. ${data.meta.responsible_use}`;
  };

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      filter = chip.dataset.filter;
      renderQueue();
    });
  });
  search.addEventListener("input", renderQueue);
  renderQueue();

  const reveal = () => {
    document.querySelectorAll(".tool-rail li").forEach((el) => {
      if (el.getBoundingClientRect().top < window.innerHeight * 0.9) {
        el.classList.add("in-view");
      }
    });
    document.querySelectorAll(".bar-fill").forEach((el) => {
      if (el.getBoundingClientRect().top < window.innerHeight * 0.95) {
        el.style.width = `${el.dataset.width}%`;
      }
    });
  };
  reveal();
  window.addEventListener("scroll", reveal, { passive: true });
})();
