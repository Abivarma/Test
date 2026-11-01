"""
Deterministic generator for the three mock datasets.

Why a generator instead of hand-written files: the datasets must be internally
consistent (telemetry observed latency/status must come out of the SAME capacity
model the engine uses, pilot finance must reconcile with telemetry services), and
reproducible (same seed -> byte-identical files -> diffable in code review).

    python scripts/generate_data.py            # rewrite data/*
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulations.scenario_engine import (  # noqa: E402
    AGENT_SERVICE, BASELINE, load_assumptions, simulate_telemetry,
)

A = load_assumptions()
SEED = A["simulation_seed"]
DATA = ROOT / "data"
WINDOW_START = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)
WINDOW_MIN = 60
INCIDENT_START_MIN = 22      # ERP API begins returning 429
KILL_SWITCH_MIN = 55         # on-call engineer manually disables the agent (MTTR 33 min)
AGENT_WORKERS = 10           # orchestrator worker pool (bounded concurrency)
MAX_CONTEXT = 128_000
CONTEXT_GROWTH_PER_CALL = 2_200   # error payload + tool schema + scratchpad re-sent each turn
REENQUEUE_CARRY = 64_000          # orchestrator re-enqueues with truncated history -> child trace

TENANTS = ["acme-bank-uk", "acme-bank-us", "acme-insurance"]

# service -> (pilot, rpm, prompt mu/sd, completion mu/sd, beta(a,b) complexity, tier, endpoint)
SERVICES = {
    "support-copilot":    ("P1", 12.0, (2500, 700), (350, 120), (2.0, 4.0), "frontier", "/v1/chat/completions"),
    "contract-extractor": ("P2", 1.5, (14000, 4000), (600, 200), (5.0, 2.0), "frontier", "/v1/chat/completions"),
    "hr-policy-bot":      ("P3", 4.0, (3200, 800), (300, 100), (1.5, 5.0), "frontier", "/v1/chat/completions"),
    "sales-email-gen":    ("P4", 9.0, (1200, 300), (450, 150), (1.2, 6.0), "frontier", "/v1/chat/completions"),
}


def service_time_ms(prompt: int, completion: int, rng) -> float:
    """Frontier-tier intrinsic latency: TTFT + prefill + decode (~80 tok/s)."""
    return 600 + prompt * 0.015 + completion * 12.5 + rng.normal(0, 80)


def _ts(minute_float: float) -> str:
    return (WINDOW_START + timedelta(seconds=minute_float * 60)).isoformat().replace("+00:00", "Z")


def gen_telemetry(rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    ev: list[dict] = []
    seq = 0

    def add(**kw):
        nonlocal seq
        seq += 1
        kw["event_id"] = f"evt-{seq:06d}"
        ev.append(kw)

    # ---- steady interactive traffic (Poisson arrivals) ----------------------
    for svc, (pid, rpm, (pm, ps), (cm, cs), (ba, bb), tier, ep) in SERVICES.items():
        n = rng.poisson(rpm * WINDOW_MIN)
        for t in np.sort(rng.uniform(0, WINDOW_MIN, n)):
            p = int(max(200, rng.normal(pm, ps)))
            c = int(max(40, rng.normal(cm, cs)))
            cx = float(rng.beta(ba, bb))
            tid = f"tr-{rng.integers(1e12):012x}"
            add(timestamp=_ts(t), tenant_id=str(rng.choice(TENANTS)), service=svc, pilot_id=pid,
                endpoint=ep, trace_id=tid, root_trace_id=tid, parent_trace_id=None, attempt=1,
                consecutive_tool_failures=0, model_tier=tier, prompt_tokens=p, completion_tokens=c,
                eval_complexity_score=round(cx, 4),
                router_complexity_score=round(float(np.clip(cx + rng.normal(0, 0.08), 0, 1)), 4),
                service_time_ms=round(service_time_ms(p, c, rng), 1), tool_name=None, tool_status=None)

    # ---- invoice reconciliation agent: plan -> erp.post_journal -> summarise ------
    # Normal trace = 3 LLM calls. During the ERP outage each tool 429 is appended to
    # context and the agent "tries again" with no cap; at context limit the
    # orchestrator re-enqueues a CHILD trace carrying 64k tokens of history.
    t = 0.0
    arrivals = []
    while t < WINDOW_MIN:
        t += rng.exponential(1 / 1.5)
        arrivals.append(t)

    worker_free_at = [0.0] * AGENT_WORKERS
    for arr in arrivals:
        w = int(np.argmin(worker_free_at))
        now = max(arr, worker_free_at[w])
        if now >= WINDOW_MIN:
            break
        tenant = str(rng.choice(TENANTS))
        root = f"tr-{rng.integers(1e12):012x}"
        trace, parent = root, None
        ctx = int(rng.normal(6000, 800))
        attempt, fails = 0, 0
        cx = float(rng.beta(3, 3))
        while now < WINDOW_MIN:
            attempt += 1
            outage = INCIDENT_START_MIN <= now < KILL_SWITCH_MIN
            c = int(max(80, rng.normal(400, 90)))
            st = service_time_ms(ctx, c, rng)
            if attempt == 1:
                tool, tstat = "planner", 200
            elif outage:
                tool, tstat = "erp.post_journal", int(rng.choice([429, 429, 429, 503]))
            else:
                tool, tstat = "erp.post_journal", 200
            add(timestamp=_ts(now), tenant_id=tenant, service=AGENT_SERVICE, pilot_id="P5",
                endpoint="/v1/chat/completions", trace_id=trace, root_trace_id=root, parent_trace_id=parent,
                attempt=attempt, consecutive_tool_failures=fails, model_tier="frontier",
                prompt_tokens=ctx, completion_tokens=c, eval_complexity_score=round(cx, 4),
                router_complexity_score=round(float(np.clip(cx + rng.normal(0, 0.08), 0, 1)), 4),
                service_time_ms=round(st, 1), tool_name=tool, tool_status=tstat)
            now += (st + rng.uniform(1500, 3500)) / 60_000   # LLM time + tool round trip
            if tool == "erp.post_journal" and tstat == 200:
                # success path: one summarise call then done
                attempt += 1
                c2 = int(max(80, rng.normal(300, 60)))
                add(timestamp=_ts(now), tenant_id=tenant, service=AGENT_SERVICE, pilot_id="P5",
                    endpoint="/v1/chat/completions", trace_id=trace, root_trace_id=root, parent_trace_id=parent,
                    attempt=attempt, consecutive_tool_failures=0, model_tier="frontier",
                    prompt_tokens=ctx + 900, completion_tokens=c2, eval_complexity_score=round(cx, 4),
                    router_complexity_score=round(float(np.clip(cx + rng.normal(0, 0.08), 0, 1)), 4),
                    service_time_ms=round(service_time_ms(ctx + 900, c2, rng), 1),
                    tool_name="summarise", tool_status=200)
                now += service_time_ms(ctx + 900, c2, rng) / 60_000
                break
            if tool != "planner":
                fails += 1
            ctx += CONTEXT_GROWTH_PER_CALL + int(rng.normal(0, 200))
            if ctx > MAX_CONTEXT:  # recursive re-enqueue
                parent, trace = trace, f"tr-{rng.integers(1e12):012x}"
                ctx, attempt = REENQUEUE_CARRY, 0
            if now >= KILL_SWITCH_MIN and fails > 0:
                break  # kill switch: agent disabled, trace abandoned
        worker_free_at[w] = now

    df = pd.DataFrame(ev)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp_dt", "event_id"]).reset_index(drop=True)
    df["minute"] = ((df["timestamp_dt"] - pd.Timestamp(WINDOW_START)).dt.total_seconds() // 60).astype(int)
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]

    # Observed outcome = output of the engine's own capacity model (single source of truth).
    sim = simulate_telemetry(df, A, BASELINE)
    df["observed_latency_ms"] = sim["latency_ms"].round(1)
    df["http_status"] = sim["status"].map({"OK": 200, "THROTTLED": 429, "TIMEOUT": 504, "SHED": 0})
    df["billed_cost_usd"] = sim["cost_usd"].round(6)
    df = df.drop(columns=["timestamp_dt", "minute", "total_tokens"])

    meta = {
        "schema_version": "1.2",
        "source": "llm-gateway (OpenTelemetry gen_ai.* semantic conventions, flattened)",
        "window_start": WINDOW_START.isoformat(),
        "window_minutes": WINDOW_MIN,
        "org_tpm_limit": A["capacity"]["org_tpm_limit"],
        "incident": {
            "id": "INC-2026-1014-01", "severity": "SEV-1",
            "summary": "invoice-recon-agent: unbounded agent-level retry on ERP 429 with context inflation "
                       "and recursive orchestrator re-enqueue; exhausted shared org TPM quota.",
            "timeline": [
                {"minute": INCIDENT_START_MIN, "event": "ERP API (erp.post_journal) starts returning 429/503"},
                {"minute": INCIDENT_START_MIN + 6, "event": "Agent worker pool saturated; token velocity > 3M TPM"},
                {"minute": INCIDENT_START_MIN + 9, "event": "Org TPM quota exhausted; bystander tenants see 429s"},
                {"minute": 41, "event": "FinOps anomaly alert pages on-call (budget alert lag ~19 min)"},
                {"minute": KILL_SWITCH_MIN, "event": "Manual kill switch on stuck traces; ERP vendor lifts rate limit; backlog drains"},
            ],
        },
        "field_notes": {
            "service_time_ms": "intrinsic model latency at configured tier, no queueing",
            "eval_complexity_score": "offline LLM-judge task difficulty label (0-1), sampled eval",
            "router_complexity_score": "online router classifier prediction (noisy)",
            "consecutive_tool_failures": "tool failures in the ROOT trace before this call",
            "http_status": "0 = never sent, 200 ok, 429 provider throttle, 504 client timeout",
        },
    }
    return meta, df


def write_telemetry(meta: dict, df: pd.DataFrame) -> None:
    recs = json.loads(df.to_json(orient="records"))
    with open(DATA / "telemetry_logs.json", "w", encoding="utf-8") as fh:
        fh.write('{\n"metadata": ' + json.dumps(meta, indent=2) + ',\n"events": [\n')
        fh.write(",\n".join(json.dumps(r, separators=(",", ":")) for r in recs))
        fh.write("\n]\n}\n")


# --------------------------------------------------------------------------- #
# Vector audit log
# --------------------------------------------------------------------------- #
# (doc_id, title, tenant, classification, acl_groups, tags, pii_records_per_chunk, chunks)
DOCS = [
    ("DOC-HB-2026", "Employee Handbook 2026", "global", "internal", "ALL", "hr.policy", 0, 6),
    ("DOC-PTO", "PTO & Leave Policy", "global", "internal", "ALL", "hr.policy;hr.leave", 0, 3),
    ("DOC-PERF", "Performance Review Process", "global", "internal", "ALL", "hr.policy;hr.performance", 0, 3),
    ("DOC-LADDER", "Career Ladders & Promotion Framework", "global", "internal", "ALL", "hr.policy;hr.promotion", 0, 4),
    ("DOC-BEN", "Benefits Overview 2026", "global", "internal", "ALL", "hr.benefits", 0, 3),
    ("DOC-BUDGET", "Team Budget Planning Guide FY27", "global", "internal", "ALL", "finance.planning", 0, 3),
    ("DOC-EXP", "Travel & Expense Policy", "global", "public", "ALL", "finance.policy", 0, 2),
    ("DOC-HC-FY27", "Headcount Plan FY27 - Engineering", "acme-bank-uk", "confidential", "FINANCE;HR;EXEC", "finance.headcount", 0, 3),
    ("DOC-RESTRUCT", "Org Restructure Draft Q1 FY27", "acme-bank-uk", "restricted", "EXEC;HR", "hr.restructure;mnpi", 0, 2),
    ("DOC-COMPBAND", "Compensation Bands FY27 - Grades L3-L8", "acme-bank-uk", "restricted", "HR;EXEC", "hr.compensation", 0, 3),
    ("DOC-SALREV", "Salary Review Export - UK Engineering", "acme-bank-uk", "restricted", "HR", "hr.compensation;hr.pii.salary", 138, 3),
    ("DOC-BONUS", "Bonus Pool Allocation FY26 by Employee", "acme-bank-uk", "restricted", "HR;EXEC", "hr.compensation;hr.pii.salary", 89, 3),
    ("DOC-PIP", "Performance Improvement Plans - Active Cases", "acme-bank-uk", "restricted", "HR", "hr.performance;hr.pii.case", 12, 2),
    ("DOC-PAYADJ-US", "Payroll Adjustment Log Oct 2026", "acme-bank-us", "restricted", "HR", "hr.payroll;hr.pii.salary", 52, 3),
    ("DOC-CLAIMS-INS", "Claims Handler Compensation Review", "acme-insurance", "restricted", "HR", "hr.compensation;hr.pii.salary", 41, 2),
]

# intent -> {doc_id: semantic relevance}; unspecified docs sit at background relevance.
INTENTS = {
    "How is my bonus calculated?": {"DOC-BONUS": .86, "DOC-COMPBAND": .84, "DOC-BEN": .76, "DOC-PERF": .74, "DOC-SALREV": .78, "DOC-CLAIMS-INS": .72},
    "What is the salary range for an L5 engineer?": {"DOC-COMPBAND": .89, "DOC-SALREV": .85, "DOC-LADDER": .79, "DOC-PAYADJ-US": .74, "DOC-CLAIMS-INS": .73},
    "What do I need to get promoted to senior analyst?": {"DOC-LADDER": .88, "DOC-PERF": .82, "DOC-COMPBAND": .76, "DOC-PIP": .70},
    "Help me plan my team's budget for next year": {"DOC-BUDGET": .87, "DOC-HC-FY27": .84, "DOC-SALREV": .77, "DOC-COMPBAND": .75, "DOC-RESTRUCT": .73},
    "How many days of parental leave do I get?": {"DOC-PTO": .90, "DOC-BEN": .82, "DOC-HB-2026": .76},
    "Is my team affected by the reorg?": {"DOC-RESTRUCT": .88, "DOC-HC-FY27": .80, "DOC-HB-2026": .66},
    "What happens if I'm put on a performance plan?": {"DOC-PIP": .87, "DOC-PERF": .83, "DOC-HB-2026": .70},
    "Can I expense a client dinner?": {"DOC-EXP": .91, "DOC-HB-2026": .66},
    "When was my last pay adjustment?": {"DOC-PAYADJ-US": .86, "DOC-SALREV": .82, "DOC-BEN": .70},
    "What benefits do I get for dental?": {"DOC-BEN": .91, "DOC-HB-2026": .74},
}

ROLES = [  # role, clearance, department, count
    ("junior_analyst", 1, "OPERATIONS", 12), ("analyst", 2, "OPERATIONS", 8),
    ("engineering_manager", 2, "ENGINEERING", 5), ("hr_business_partner", 3, "HR", 3),
    ("finance_partner", 3, "FINANCE", 2), ("director", 3, "EXEC", 2),
]


def gen_vector_audit(rng: np.random.Generator) -> pd.DataFrame:
    chunks = [(d, f"{d[0]}#c{i:02d}") for d in DOCS for i in range(d[7])]
    users = []
    for role, clr, dept, n in ROLES:
        for _ in range(n):
            users.append((f"u-{rng.integers(1e6):06d}", role, clr, dept, str(rng.choice(TENANTS, p=[.6, .25, .15]))))
    rows, start = [], datetime(2026, 10, 1, tzinfo=timezone.utc)
    n_queries, top_k, pool = 260, 5, 15
    intents = list(INTENTS)
    for qi in range(n_queries):
        u = users[rng.integers(len(users))]
        # Juniors ask about pay/promotion more often - the realistic leak vector.
        weights = np.array([3, 2, 3, 1, 2, 1, 1, 1, 1, 1] if u[1] == "junior_analyst" else [1, .5, 1, 1, 3, .5, .5, 3, .5, 3], float)
        intent = intents[rng.choice(len(intents), p=weights / weights.sum())]
        rel = INTENTS[intent]
        ts = start + timedelta(minutes=float(rng.uniform(0, 14 * 24 * 60)))
        scored = []
        for (doc, cid) in chunks:
            base = rel.get(doc[0], rng.uniform(0.42, 0.60))
            scored.append((float(np.clip(base + rng.normal(0, 0.025), 0, 0.99)), doc, cid))
        scored.sort(key=lambda x: -x[0])
        lat = float(rng.gamma(6, 6))
        for rank, (sim, doc, cid) in enumerate(scored[:pool], start=1):
            rows.append({
                "event_ts": ts.isoformat().replace("+00:00", "Z"), "query_id": f"q-{qi:05d}",
                "session_id": f"s-{rng.integers(1e8):08d}", "tenant_id": u[4], "user_id": u[0],
                "user_role": u[1], "user_department": u[3], "user_clearance": u[2],
                "app_id": "hr-policy-bot", "query_intent": intent,
                "collection": "enterprise-kb-shared", "index_type": "hnsw", "filter_mode": "none",
                "top_k": top_k, "candidate_rank": rank, "chunk_id": cid, "doc_id": doc[0],
                "doc_title": doc[1], "doc_tenant_id": doc[2], "doc_classification": doc[3],
                "doc_acl_groups": doc[4], "metadata_tags": doc[5], "contains_pii": doc[6] > 0,
                "pii_record_count": doc[6], "similarity_score": round(sim, 4),
                "returned_to_llm": rank <= top_k, "retrieval_latency_ms": round(lat, 1),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pilot portfolio (12 months)
# --------------------------------------------------------------------------- #
def gen_pilots(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.period_range("2025-11", "2026-10", freq="M")
    specs = {
        # pid: name, BU, service, licensed, active(start,end), outcome, units(start,end), $/unit, attribution,
        #      inference(start,end), platform, eng
        "P1": ("Customer Support Copilot", "Retail Banking Ops", "support-copilot", 1500, (180, 1180),
               "tickets_deflected", (2500, 24000), 6.50, "ab_test", (2500, 19000), 6000, 46000),
        "P2": ("Contract Clause Extractor", "Legal", "contract-extractor", 40, (6, 13),
               "review_hours_saved", (40, 210), 210.0, "telemetry_proxy", (1200, 3100), 4200, 30000),
        "P3": ("HR Policy Assistant", "People & HR", "hr-policy-bot", 3000, (300, 620),
               "hr_tickets_deflected", (700, 1500), 9.00, "telemetry_proxy", (2400, 5200), 7000, 25000),
        "P4": ("Sales Email Generator", "Commercial", "sales-email-gen", 900, (420, 790),
               "self_reported_hours_saved", (1500, 4200), 55.0, "survey", (14000, 42000), 5000, 20000),
        "P5": ("Invoice Reconciliation Agent", "Finance Ops", "invoice-recon-agent", 60, (20, 48),
               "invoices_auto_reconciled", (1500, 15000), 4.20, "finance_verified", (1500, 8200), 3500, 28000),
    }
    rows = []
    for pid, (name, bu, svc, lic, (a0, a1), om, (u0, u1), vpu, attr, (i0, i1), plat, eng) in specs.items():
        for k, m in enumerate(months):
            f = k / (len(months) - 1)
            ease = f ** 0.8
            active = int(a0 + (a1 - a0) * ease + rng.normal(0, max(1, a1 * 0.02)))
            units = max(0, int(u0 + (u1 - u0) * ease + rng.normal(0, u1 * 0.03)))
            infer = (i0 + (i1 - i0) * ease) * rng.normal(1, 0.04)
            retry = float(np.clip(rng.normal(0.04, 0.01), 0.01, 0.08))
            phase = "pilot" if k < 7 else "limited_production"
            if pid == "P1" and k >= 5:
                phase = "limited_production" if k < 9 else "scaled"
            if pid in ("P3", "P4"):
                phase = "pilot"  # never left pilot: the purgatory signature
            if pid == "P5" and m >= pd.Period("2026-09", "M"):
                # ERP instability: recurring retry storms (INC-2026-1014-01 is the worst one)
                retry = 0.46 if m == pd.Period("2026-09", "M") else 0.78
                infer = infer / (1 - retry)
                units = int(units * (0.88 if retry > 0.5 else 0.95))
            records = 0
            crit = 0
            acl_open = 0
            if pid == "P3" and m >= pd.Period("2026-08", "M"):  # query-builder refactor dropped ACL filter
                records, crit, acl_open = int(rng.normal(330, 25)), 1, 1
            rows.append({
                "month": str(m), "pilot_id": pid, "pilot_name": name, "business_unit": bu, "service": svc,
                "phase": phase, "primary_model_tier": "frontier", "licensed_users": lic, "active_users": active,
                "requests": int(units * (3.2 if pid != "P5" else 4.1)),
                "inference_cost_usd": round(infer, 2), "platform_cost_usd": round(plat * rng.normal(1, 0.03), 2),
                "engineering_cost_usd": round(eng * rng.normal(1, 0.02), 2),
                "outcome_metric": om, "outcome_units": units, "value_per_unit_usd": vpu,
                "measured_value_usd": round(units * vpu, 2), "value_attribution": attr,
                "hallucination_rate": round(float(np.clip(rng.normal({"P4": .061, "P3": .034}.get(pid, .018), .004), 0, 1)), 4),
                "retry_overhead_pct": round(retry, 3), "records_at_risk": records,
                "security_findings_critical": crit, "acl_finding_open": acl_open,
            })
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta, tel = gen_telemetry(rng)
    write_telemetry(meta, tel)
    gen_vector_audit(rng).to_csv(DATA / "vector_audit_log.csv", index=False)
    gen_pilots(rng).to_csv(DATA / "pilot_metrics.csv", index=False)
    print(f"telemetry events: {len(tel):,}  ->  {DATA/'telemetry_logs.json'}")
    print(f"vector audit rows -> {DATA/'vector_audit_log.csv'}")
    print(f"pilot rows        -> {DATA/'pilot_metrics.csv'}")


if __name__ == "__main__":
    main()
