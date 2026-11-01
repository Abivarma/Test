# Enterprise AI War Room Simulator

A local, deterministic simulator for rehearsing three high-stakes enterprise AI failures and the executive decisions that contain them. You can test **cost vs. latency vs. risk** trade-offs here before you change anything in production.

| Failure mode | Dataset | Executive lever |
|---|---|---|
| Unbounded agent retry loop → token-cost explosion and shared-quota blast radius | `data/telemetry_logs.json` | **Hard token circuit breaker** |
| RAG over a shared multi-tenant vector index with no ACL filter → HR salary PII in a junior analyst's context (OWASP LLM08:2025) | `data/vector_audit_log.csv` | **Vector metadata pre-filtering** |
| Overprovisioned frontier models plus unproven value → "Pilot Purgatory" | `data/pilot_metrics.csv` | **Dynamic model routing** plus a portfolio triage (SCALE / FIX / INSTRUMENT / HALT / KILL) |

Every metric is recomputed **before vs. after** whenever a lever moves.

---

## 1. Quick start (Apple Silicon)

```bash
git clone <this repo> Enterprise-AI-WarRoom-Simulator && cd Enterprise-AI-WarRoom-Simulator
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11–3.13
pip install -r requirements.txt                          # all arm64 wheels, no compiler, no GPU
python -m pytest -q                                      # 9 acceptance tests, ~3 s
streamlit run app.py                                     # http://localhost:8501
```

Other targets: `make data` regenerates the datasets deterministically. `make smoke` prints the before/after table in the terminal.

**Optional local LLM brief (LM Studio / MLX).** The *Decision log & brief* tab can draft a board-level SEV-1 brief with a local model. It uses any OpenAI-compatible server:

```bash
# LM Studio: load a model, start the server (default :1234). Nothing else to set.
# or MLX:
pip install mlx-lm && mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
export WARROOM_LLM_BASE_URL=http://localhost:8080/v1
```

The LLM only **narrates** numbers the engine already computed. It is never in the decision path, so a hallucination cannot change a metric. If no server is reachable, a deterministic template is used instead. You do not need IBM Bob or any cloud tokens to run this repo.

---

## 2. Architecture

```
config/assumptions.json      ← every price, SLO, policy weight, and threshold (the only file FinOps/Risk edit)
scripts/generate_data.py     ← deterministic generator (seeded); observed outcomes come FROM the engine's model
data/                        ← the three datasets (committed, diffable, reproducible)
simulations/scenario_engine.py
   ├─ simulate_telemetry()   circuit breaker → routing → shared-capacity model (TPM quota + queueing)
   ├─ simulate_retrieval()   none | post_filter | pre_filter ACL enforcement, with policy-as-code
   ├─ evaluate_portfolio()   risk-adjusted ROI, security/reliability gates, purgatory detection
   ├─ run_scenario()         one intervention set → ScenarioResult
   └─ compare()              the before/after decision table (direction-aware)
simulations/narrator.py      ← optional local-LLM brief (stdlib HTTP, no SDK dependency)
app.py                       ← Streamlit UI; holds no business logic
tests/                       ← executable acceptance criteria for each lever's claimed effect
```

### Design decisions that make the numbers defensible

1. **Demand is separate from outcome.** Telemetry stores what each caller *tried* to do (tokens, intrinsic `service_time_ms`, router score, tool result). Observed latency and status are model outputs. The engine re-simulates **both** baseline and intervention from demand through the same capacity model.
2. **Common random numbers.** Each event gets the same throttle draw and latency jitter in every scenario. Any delta therefore comes from the intervention, not from sampling noise. The `test_baseline_reproduces_stored_observations` test checks that the re-simulated baseline matches the logged telemetry exactly.
3. **Shared capacity is modeled explicitly.** Per minute, the engine computes demand against an org-wide TPM quota. Requests above the quota are throttled, and accepted load drives a soft queueing factor `1 + α·ρ²/(1−ρ)`. Requests that exceed the 30 s client timeout still get **billed**. This is how one tenant's loop pages every other tenant.
4. **Policy-as-code for ACLs.** Permission is derived from tenant, clearance and group policy at audit time. A log's `acl_check_result` field is not trusted.
5. **Trade-offs are always shown alongside the wins.** Each lever reports its cost: DLQ backlog (breaker), quality regression (routing), latency overhead (pre-filter).

---

## 3. The scenarios

### 3.1 Recursive retry loop (`INC-2026-1014-01`)
At T+22m the ERP API (`erp.post_journal`) starts returning 429/503. The invoice agent appends each error and the tool schema to its context (about 2.2k tokens per turn) and calls the model again, with no cap. When the context passes 128k, the orchestrator **re-enqueues a child trace** carrying 64k tokens of history. That is the recursion. Ten workers saturate, token velocity exceeds the 5M TPM org quota, and the copilots in other business units start receiving 429s.

Baseline over the 60-minute window:

| Metric | Value |
|---|---|
| Spend | $784, against a budgeted ~$60/hr |
| Peak burn | $1,681/hr (about 28× budget) |
| Spend on calls that made no business progress | 94% |
| Exposure if the loop runs undetected for 72h | ≈ $109k |
| Bystander-tenant P99 latency | pinned at the 30 s timeout |
| Bystander error rate | 21.5% |

**Circuit breaker.** The breaker enforces three limits: max consecutive tool failures per root trace, a per-trace token budget, and a per-service token-velocity limit. When any limit trips, the trace fails fast to a DLQ.

### 3.2 Multi-tenant vector leakage (OWASP LLM08:2025)
`hr-policy-bot` queries `enterprise-kb-shared`, a single HNSW index across three legal-entity tenants, with `filter_mode=none`. Questions like *"How is my bonus calculated?"* sit semantically closest to *Bonus Pool Allocation FY26 by Employee* and *Salary Review Export*. Similarity search has no concept of authorization.

Baseline results:
- 35% of queries leak at least one unauthorized chunk.
- 861 PII records are exposed.
- 161 chunks leak across tenants.

The engine compares three enforcement modes:

| Mode | Leaks | Context fill | Why it matters |
|---|---|---|---|
| `none` | 399 | 100% | The shipped bug. |
| `post_filter` | 0 | **69%** | Safe, but top-k gets emptied. Teams respond by raising k or quietly removing the filter. |
| `pre_filter` | 0 | 97% | ACL predicate inside the ANN query. Costs about +12% retrieval latency. |

### 3.3 Pilot Purgatory
Five pilots over 12 months. ROI is **risk-adjusted**: value × attribution confidence (A/B test = 1.0, survey = 0.3) − expected breach loss, divided by fully-loaded cost (inference + platform + engineering).

| Pilot | Baseline | After all levers | Lesson |
|---|---|---|---|
| P1 Support Copilot | SCALE | SCALE | A/B-proven value, and routing widens the margin. |
| P2 Contract Extractor | FIX | FIX | High value per unit, but 28% adoption. The problem is change management, not the model. |
| P3 HR Policy Assistant | **HALT** | **KILL** | Clearing the security gate shows the pilot was never worth scaling. |
| P4 Sales Email Gen | INSTRUMENT | INSTRUMENT | Users love it, but value is self-reported. Routing makes it cheaper to fail; it does not prove value. Run a holdout. |
| P5 Invoice Agent | FIX | **SCALE** | Finance-verified value. Retry overhead was the only blocker. |

The **reliability gate** blocks P5 from SCALE when routing is the only lever applied. Cheaper tokens hide a loop without bounding it (see `test_routing_alone_cannot_promote_a_looping_pilot`).

---

## 4. How a Head of AI reads the dashboard

| Metric | Read it as | Action threshold (default) |
|---|---|---|
| Peak burn $/hr vs. budget | Financial blast radius per hour of detection lag | > 3× budget pages FinOps and Platform |
| 72h exposure if undetected | Cost of a weekend with no alert. This is the number for the CFO. | Any loop path > $10k means hard caps are mandatory |
| Spend on non-progressing calls | Share of spend that bought nothing | > 20% in any service means a reliability defect, not a cost problem |
| Bystander P99 / error rate | Multi-tenant blast radius: who else got paged | Any breach puts per-tenant quotas on the roadmap |
| Compliance score | Directional, severity-weighted share of clean context | Use for trend only |
| **Reportable-exposure gate** | Binary: did restricted PII reach an LLM? | Any true means DPO/legal engagement, whatever the score |
| Quality regression (routing) | Share of routed traffic below the tier's capability on offline eval | > 2% means tightening thresholds or improving the router |
| Risk-adjusted ROI + attribution | Whether value is *proven*, not just claimed | SCALE needs ROI ≥ 1.5, adoption ≥ 40%, confidence ≥ 0.6 |

The composite risk score (financial 35%, reliability 25%, governance 40%) is a heuristic that fits on one slide, not a standard. The weights are in `config/assumptions.json`.

### Engineering lead ↔ Head of AI split
- **Engineering lead** owns `scenario_engine.py`, the generator and the tests. Every claim the Head of AI makes in a readout must map to a passing test.
- **Head of AI** owns `config/assumptions.json` (prices, SLOs, risk appetite, gate thresholds) and the decision log. Changing an assumption is a governance act, so it is recorded with the decision.
- **Shared ritual:** run the scenario, record the decision (owner, rationale, accepted trade-off), and export the JSON decision log as the audit trail.

---

## 5. Research basis and confidence

| Claim built into the model | Source | Confidence |
|---|---|---|
| Multi-tenant vector stores leak across access contexts; mitigate with permission-aware retrieval | OWASP Top 10 for LLM Apps 2025, **LLM08 Vector & Embedding Weaknesses** (Scenario #2) | Certain |
| ~95% of enterprise GenAI pilots show no measurable P&L impact | MIT NANDA, *The GenAI Divide: State of AI in Business 2025*. Methodology (interviews and surveys) is contested; use it as directional only | Likely |
| Learned routers cut cost 2–3.7× (≈35–85% depending on benchmark) while keeping ~95% of strong-model quality | Ong et al., **RouteLLM** (arXiv:2406.18665) | Certain for those benchmarks; Speculative for your traffic |
| Agent loops grow cost quadratically because full context is re-billed every turn; retry loops turn $50/day into $5k/day | Vendor engineering blogs and public postmortems (Augment, WunderGraph, community reports) | Likely (anecdotal tier) |
| Retries at multiple layers amplify load ("retry storms") | Google SRE book, *Addressing Cascading Failures* | Certain |

### Known limitations (read before quoting numbers)
- **Prices are illustrative tiers** ($5/$25, $1/$5, $0.15/$0.60 per 1M tokens in/out). Swap in your contract rates.
- **The DLQ count is a lower bound.** Telemetry contains only the traces that obtained a worker. With the breaker in place, the queued backlog would also fail fast.
- The queue model is a calibrated heuristic, not a fitted model of any one provider. Validate it against your gateway's P99-vs-utilization curve.
- Expected breach loss = records × $/record × P(exfiltration). All three are assumptions; replace them with your actuarial or insurance figures.
- Routing quality regression is measured against a noisy offline label. Treat it as a gate signal, not ground truth.

---

## 6. Extending
- **Real telemetry:** export OpenTelemetry `gen_ai.*` spans from your gateway into the `events` schema (field notes are in the JSON `metadata`). The engine needs only demand fields.
- **Local tier:** add a `local-mlx` tier in `assumptions.json` (marginal cost 0, low `capability`, capacity-bound) to model on-device offload on Apple Silicon.
- **CI gate:** run `pytest` on every change to `assumptions.json`, so a threshold edit that breaks a documented claim fails review.
