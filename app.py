"""
Enterprise AI War Room Simulator — Streamlit dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from simulations import narrator
from simulations.scenario_engine import (
    AGENT_SERVICE, BASELINE, FILTER_MODES, Interventions, compare, executive_summary,
    governance_metrics, load_all, load_assumptions, run_scenario, simulate_retrieval,
)

st.set_page_config(page_title="AI War Room", page_icon="🛰️", layout="wide")

# Categorical slots 1/2 of the validated reference palette (entity-stable across every chart).
AFTER, BEFORE = "After intervention", "Before (baseline)"
SCEN_SCALE = alt.Scale(domain=[AFTER, BEFORE], range=["#2a78d6", "#eb6834"])
RULE_COLOR = "#8a8984"


# --------------------------------------------------------------------------- #
# Data + scenario cache
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading telemetry, vector audit and pilot ledgers…")
def datasets():
    return load_all()


@st.cache_data(show_spinner="Re-simulating incident window…")
def scenario(iv_json: str, a_json: str):
    return run_scenario(datasets(), json.loads(a_json), Interventions(**json.loads(iv_json)))


def fmt(v: float, kind: str) -> str:
    if kind == "usd":
        return f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "score":
        return f"{v:.1f}"
    return f"{v:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar: executive levers + assumptions
# --------------------------------------------------------------------------- #
base_a = load_assumptions()
with st.sidebar:
    st.header("Executive interventions")
    cb = st.toggle("Hard token circuit breaker", help="Caps consecutive tool failures, per-trace token "
                   "budget and per-service token velocity. Fails fast to a DLQ.")
    with st.expander("Breaker parameters", expanded=cb):
        max_fail = st.slider("Max consecutive tool failures", 1, 10, 3)
        budget = st.slider("Per-trace token budget (k)", 20, 400, 60, step=10) * 1000
        vel = st.slider("Per-service velocity limit (k TPM)", 250, 5000, 1500, step=250) * 1000
    mr = st.toggle("Dynamic model routing", help="Router classifier sends low-complexity requests to "
                   "cheaper tiers. Router scores are noisy; quality regressions are measured against offline eval.")
    with st.expander("Routing thresholds", expanded=mr):
        small_max = st.slider("Route to SMALL if router score ≤", 0.0, 0.6, 0.35, 0.05)
        bal_max = st.slider("Route to BALANCED if router score ≤", small_max, 0.95, max(0.65, small_max), 0.05)
    vp = st.toggle("Vector metadata pre-filtering", help="ACL/tenant predicate pushed INTO the ANN query, "
                   "so top-k is computed over authorised chunks only.")

    st.divider()
    st.header("Assumptions")
    a = copy.deepcopy(base_a)
    a["capacity"]["org_tpm_limit"] = st.number_input("Org TPM quota", 1_000_000, 50_000_000,
                                                     a["capacity"]["org_tpm_limit"], step=500_000)
    a["slo"]["p99_latency_ms"] = st.number_input("P99 SLO (ms)", 2000, 60000, int(a["slo"]["p99_latency_ms"]), 1000)
    a["slo"]["budgeted_burn_usd_per_hr"] = st.number_input("Budgeted burn ($/hr)", 10.0, 1000.0,
                                                           a["slo"]["budgeted_burn_usd_per_hr"], 10.0)
    a["governance"]["cost_per_exposed_record_usd"] = st.number_input(
        "Cost per exposed record ($)", 10.0, 2000.0, a["governance"]["cost_per_exposed_record_usd"], 10.0)
    st.caption("Tier prices are illustrative — edit `config/assumptions.json` with your negotiated rates.")

iv = Interventions(circuit_breaker=cb, max_consecutive_failures=max_fail, trace_token_budget=budget,
                   service_velocity_limit_tpm=vel, model_routing=mr, routing_small_max=small_max,
                   routing_balanced_max=bal_max, vector_prefilter=vp)
a_json = json.dumps(a, sort_keys=True)
before = scenario(json.dumps(BASELINE.to_dict()), a_json)
after = scenario(json.dumps(iv.to_dict()), a_json)
cmp_df = compare(before, after)
ds = datasets()
meta = ds.metadata
inc = meta["incident"]


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def paired(col: str, bystander: bool = False) -> pd.DataFrame:
    src_b = before.minutes_bystander if bystander else before.minutes
    src_a = after.minutes_bystander if bystander else after.minutes
    return pd.concat([src_b[["minute", col]].assign(scenario=BEFORE),
                      src_a[["minute", col]].assign(scenario=AFTER)])


def before_after_chart(df: pd.DataFrame, col: str, title: str, y_title: str, y_fmt: str,
                       rules: list[tuple[float, str]] = (), upto: int | None = None, height: int = 260):
    if upto is not None:
        df = df[df["minute"] <= upto]
    hover = alt.selection_point(fields=["minute"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(df).encode(x=alt.X("minute:Q", title="Minute of incident window",
                                        scale=alt.Scale(domain=[0, meta["window_minutes"] - 1])))
    line = base.mark_line(strokeWidth=2).encode(
        y=alt.Y(f"{col}:Q", title=y_title, axis=alt.Axis(format=y_fmt)),
        color=alt.Color("scenario:N", scale=SCEN_SCALE, legend=alt.Legend(orient="top", title=None)))
    pts = base.mark_point(size=70, filled=True).encode(
        y=f"{col}:Q", color=alt.Color("scenario:N", scale=SCEN_SCALE, legend=None),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=["minute:Q", "scenario:N", alt.Tooltip(f"{col}:Q", format=y_fmt)]).add_params(hover)
    cross = base.mark_rule(color=RULE_COLOR, strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0))).transform_filter(hover)
    layers = [line, pts, cross]
    inc_rule = alt.Chart(pd.DataFrame({"minute": [inc["timeline"][0]["minute"]]})).mark_rule(
        strokeDash=[2, 3], color=RULE_COLOR).encode(x="minute:Q")
    layers.append(inc_rule)
    for val, label in rules:
        r = pd.DataFrame({"y": [val], "label": [label]})
        layers.append(alt.Chart(r).mark_rule(strokeDash=[6, 4], color=RULE_COLOR).encode(y="y:Q"))
        layers.append(alt.Chart(r).mark_text(align="left", dx=4, dy=-6, color=RULE_COLOR, fontSize=11)
                      .encode(y="y:Q", text="label:N", x=alt.value(4)))
    return alt.layer(*layers).properties(title=title, height=height)


# --------------------------------------------------------------------------- #
# Header + KPI strip
# --------------------------------------------------------------------------- #
st.title("Enterprise AI War Room")
st.markdown(f"**{inc['severity']} · {inc['id']}** — {inc['summary']}  \n"
            f"**Governance finding** — `hr-policy-bot` retrieves from a shared multi-tenant index with "
            f"`filter_mode=none` (OWASP LLM08:2025, vector & embedding weaknesses).")
st.caption(f"Active scenario: **{iv.label()}** · window {meta['window_start']} · "
           f"{meta['window_minutes']} min · {len(ds.telemetry):,} LLM calls · {ds.vector['query_id'].nunique()} RAG queries")

kpis = ["window_cost_usd", "peak_burn_usd_per_hr", "p99_bystander_latency_ms",
        "bystander_error_rate", "compliance_score", "composite_risk"]
cols = st.columns(len(kpis))
for c, key in zip(cols, kpis):
    r = cmp_df.set_index("metric").loc[key]
    delta = None if abs(r["delta"]) < 1e-9 else ("-" if r["delta"] < 0 else "+") + fmt(abs(r["delta"]), r["fmt"])
    c.metric(r["label"], fmt(r["after"], r["fmt"]), delta,
             delta_color="normal" if r["direction"] > 0 else "inverse",
             help=f"Baseline: {fmt(r['before'], r['fmt'])}")

tabs = st.tabs(["Executive summary", "Live replay", "Cost & tokens", "Latency & reliability",
                "Governance (vector ACL)", "Pilot portfolio", "Decision log & brief"])

# --------------------------------------------------------------------------- #
# 1. Executive summary
# --------------------------------------------------------------------------- #
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Before vs after")
        view = cmp_df.copy()
        view["Before"] = [fmt(v, f) for v, f in zip(view["before"], view["fmt"])]
        view["After"] = [fmt(v, f) for v, f in zip(view["after"], view["fmt"])]
        view["Δ"] = [("−" if d < 0 else "+") + fmt(abs(d), f) if abs(d) > 1e-9 else "—"
                     for d, f in zip(view["delta"], view["fmt"])]
        view["Status"] = ["▲ better" if i else ("▼ worse" if r else "· unchanged")
                          for i, r in zip(view["improved"], view["regressed"])]
        st.dataframe(view[["label", "Before", "After", "Δ", "Status"]].rename(columns={"label": "Metric"}),
                     hide_index=True, width="stretch", height=670)
    with right:
        st.subheader("Readout")
        for line in executive_summary(before, after):
            safe = line.replace("$", "\\$")  # stop Streamlit treating $…$ as LaTeX
            if line.startswith("TRADE-OFF"):
                st.warning(safe)
            else:
                st.markdown(f"- {safe}")
        st.subheader("Risk decomposition")
        rk = pd.DataFrame([{"component": k.replace("_risk", ""), "scenario": s, "score": v}
                           for s, res in ((BEFORE, before), (AFTER, after))
                           for k, v in res.risk.items() if k != "composite_risk"])
        st.altair_chart(alt.Chart(rk).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("component:N", title=None), yOffset="scenario:N",
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="Risk (0-100, lower is better)"),
            color=alt.Color("scenario:N", scale=SCEN_SCALE, legend=alt.Legend(orient="top", title=None)),
            tooltip=["component", "scenario", alt.Tooltip("score:Q", format=".1f")]).properties(height=220),
            width="stretch")

# --------------------------------------------------------------------------- #
# 2. Live replay
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.caption("Scrub or replay the incident minute by minute. Dashed vertical = ERP outage start.")
    c1, c2 = st.columns([4, 1])
    cursor = c1.slider("Minute", 0, meta["window_minutes"] - 1, meta["window_minutes"] - 1, key="cursor")
    play = c2.button("▶ Replay incident", width="stretch")
    tl = pd.DataFrame(inc["timeline"])
    tiles = st.empty()
    charts = st.empty()

    def render(m: int):
        bm, am = before.minutes.iloc[m], after.minutes.iloc[m]
        with tiles.container():
            t = st.columns(4)
            t[0].metric("Burn rate now ($/hr)", f"${am['burn_usd_per_hr']:,.0f}",
                        f"{am['burn_usd_per_hr'] - bm['burn_usd_per_hr']:+,.0f} vs baseline", delta_color="inverse")
            t[1].metric("Token velocity (TPM)", f"{am['billed_tokens']:,.0f}",
                        f"{am['billed_tokens'] - bm['billed_tokens']:+,.0f}", delta_color="inverse")
            p99b = before.minutes_bystander.iloc[m]["p99_latency_ms"]
            p99a = after.minutes_bystander.iloc[m]["p99_latency_ms"]
            t[2].metric("Bystander P99 (ms)", f"{p99a:,.0f}", f"{p99a - p99b:+,.0f}", delta_color="inverse")
            t[3].metric("Calls shed by breaker", f"{int(am['shed'])}")
            happened = tl[tl["minute"] <= m]
            if len(happened):
                st.info(" → ".join(f"**T+{r.minute}m** {r.event}" for r in happened.itertuples()))
        with charts.container():
            g1, g2 = st.columns(2)
            g1.altair_chart(before_after_chart(paired("burn_usd_per_hr"), "burn_usd_per_hr", "Burn rate",
                            "$/hr", "$,.0f", [(a["slo"]["budgeted_burn_usd_per_hr"], "budget")], upto=m),
                            width="stretch")
            g2.altair_chart(before_after_chart(paired("p99_latency_ms", True), "p99_latency_ms",
                            "Bystander-tenant P99 latency", "ms", ",.0f",
                            [(a["slo"]["p99_latency_ms"], "P99 SLO")], upto=m), width="stretch")

    if play:
        for m in range(0, meta["window_minutes"], 1):
            render(m)
            time.sleep(0.12)
    else:
        render(cursor)

# --------------------------------------------------------------------------- #
# 3. Cost & tokens
# --------------------------------------------------------------------------- #
with tabs[2]:
    g1, g2 = st.columns(2)
    g1.altair_chart(before_after_chart(paired("burn_usd_per_hr"), "burn_usd_per_hr", "Burn rate (all services)",
                    "$/hr", "$,.0f", [(a["slo"]["budgeted_burn_usd_per_hr"], "budget")]), width="stretch")
    g2.altair_chart(before_after_chart(paired("billed_tokens"), "billed_tokens", "Token velocity (billed tokens/min)",
                    "tokens / min", "~s", [(a["capacity"]["org_tpm_limit"], "org TPM quota")]), width="stretch")
    st.subheader("Spend by service")
    sv = pd.concat([before.telemetry.assign(scenario=BEFORE), after.telemetry.assign(scenario=AFTER)])
    by = sv.groupby(["service", "scenario"], as_index=False)["cost_usd"].sum()
    st.altair_chart(alt.Chart(by).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("service:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)), yOffset="scenario:N",
        x=alt.X("cost_usd:Q", title="Spend in window ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("scenario:N", scale=SCEN_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["service", "scenario", alt.Tooltip("cost_usd:Q", format="$,.2f")]).properties(height=280),
        width="stretch")
    with st.expander("Why the agent bill is quadratic"):
        ag = before.telemetry[before.telemetry["service"] == AGENT_SERVICE]
        st.markdown(
            f"Each failed `erp.post_journal` call appends the error + tool schema to context "
            f"(~2.2k tokens) and the agent re-calls the model. Cumulative billed input across N turns ≈ N²/2 × growth. "
            f"Worst root trace in this window: **{ag.groupby('root_trace_id')['attempt'].count().max()} LLM calls**, "
            f"**{ag.groupby('root_trace_id')['total_tokens'].sum().max():,.0f} tokens**. "
            f"Spend on calls that made no business progress: **{before.ops['wasted_spend_pct']:.0%}** of the window.")
        st.dataframe(after.telemetry.groupby("blocked_by").size().rename("calls").reset_index()
                     .replace({"blocked_by": {"": "(sent)"}}), hide_index=True)

# --------------------------------------------------------------------------- #
# 4. Latency & reliability
# --------------------------------------------------------------------------- #
with tabs[3]:
    g1, g2 = st.columns(2)
    g1.altair_chart(before_after_chart(paired("p99_latency_ms", True), "p99_latency_ms",
                    "P99 latency — bystander tenants (blast radius)", "ms", ",.0f",
                    [(a["slo"]["p99_latency_ms"], "P99 SLO")]), width="stretch")
    g2.altair_chart(before_after_chart(paired("error_rate", True), "error_rate",
                    "Error rate — bystander tenants (429 + 504)", "error rate", ".0%",
                    [(a["slo"]["max_error_rate"], "error budget")]), width="stretch")
    st.markdown("**Why bystanders suffer:** one runaway agent and the interactive copilots share one org-level "
                "TPM quota. Quota exhaustion is a *multi-tenant* failure: the agent's tenant is not the only one paged.")
    tier = pd.concat([before.telemetry.assign(scenario=BEFORE), after.telemetry.assign(scenario=AFTER)])
    tier = tier[tier["status"] != "SHED"].groupby(["scenario", "effective_tier"]).size().rename("calls").reset_index()
    st.dataframe(tier.pivot(index="effective_tier", columns="scenario", values="calls").fillna(0).astype(int))

# --------------------------------------------------------------------------- #
# 5. Governance
# --------------------------------------------------------------------------- #
with tabs[4]:
    g = before.gov
    k = st.columns(5)
    k[0].metric("Queries with an ACL violation", f"{g['queries_with_violation_pct']:.0%}")
    k[1].metric("Users exposed", g["users_exposed"])
    k[2].metric("PII records exposed", f"{g['pii_records_exposed']:,}")
    k[3].metric("Cross-tenant leaks", g["cross_tenant_leaks"])
    k[4].metric("Expected breach loss", f"${g['expected_breach_loss_usd']:,.0f}")
    if g["reportable_exposure"]:
        st.error("Baseline FAILS the regulatory gate: restricted PII reached LLM context. A compliance score "
                 "is directional; this gate is binary — engage DPO / privacy counsel regardless of the score.")

    st.subheader("Filter-mode comparison (same queries, same index)")
    modes = []
    for m in FILTER_MODES:
        gm = governance_metrics(simulate_retrieval(ds.vector, a, m), a)
        modes.append({"filter_mode": m, "ACL violations": gm["acl_violations"],
                      "PII records exposed": gm["pii_records_exposed"], "Compliance": round(gm["compliance_score"], 1),
                      "Context fill (recall proxy)": f"{gm['context_fill_rate']:.0%}",
                      "Retrieval P99 (ms)": round(gm["p99_retrieval_latency_ms"], 1)})
    st.dataframe(pd.DataFrame(modes), hide_index=True, width="stretch")
    st.caption("Post-filtering is ACL-safe but starves the context window (top-k emptied by the filter) — "
               "teams then raise k or drop the filter. Pre-filtering preserves recall at a small latency cost.")

    viol = before.retrieval[before.retrieval["violation"]]
    hm = viol.groupby(["user_role", "doc_classification"]).size().rename("violations").reset_index()
    c1, c2 = st.columns([3, 2])
    c1.altair_chart(alt.Chart(hm).mark_rect(cornerRadius=3).encode(
        x=alt.X("doc_classification:N", sort=["internal", "confidential", "restricted"], title="Document classification"),
        y=alt.Y("user_role:N", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("violations:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="violations")),
        tooltip=["user_role", "doc_classification", "violations"]).properties(
        title="Baseline violations: who saw what", height=260), width="stretch")
    top = (viol.groupby(["doc_title", "doc_classification"]).agg(
        hits=("chunk_id", "size"), users=("user_id", "nunique")).reset_index().sort_values("hits", ascending=False))
    c2.dataframe(top, hide_index=True, height=300)
    with st.expander("Sample leaked retrievals (junior roles)"):
        st.dataframe(viol[viol["user_role"] == "junior_analyst"][[
            "event_ts", "user_id", "tenant_id", "query_intent", "doc_title", "doc_tenant_id",
            "doc_classification", "similarity_score", "pii_record_count"]].head(25), hide_index=True)

# --------------------------------------------------------------------------- #
# 6. Pilot portfolio
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.caption("Trailing-3-month, risk-adjusted: value × attribution confidence − expected breach loss, "
               "over fully-loaded cost (inference + platform + engineering).")
    pb, pa = before.portfolio.assign(scenario=BEFORE), after.portfolio.assign(scenario=AFTER)
    pp = pd.concat([pb, pa])
    mx = float(max(pp["trailing_cost_usd"].max(), pp["trailing_risk_adj_value_usd"].max()) * 1.1)
    diag = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(
        strokeDash=[6, 4], color=RULE_COLOR).encode(x="x:Q", y="y:Q")
    dots = alt.Chart(pp).mark_point(size=160, filled=True, stroke="white", strokeWidth=2).encode(
        x=alt.X("trailing_cost_usd:Q", title="Fully-loaded cost, trailing 3 mo ($)", axis=alt.Axis(format="$~s"),
                scale=alt.Scale(domain=[0, mx])),
        y=alt.Y("trailing_risk_adj_value_usd:Q", title="Risk-adjusted value ($)", axis=alt.Axis(format="$~s"),
                scale=alt.Scale(domain=[0, mx])),
        color=alt.Color("scenario:N", scale=SCEN_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["pilot_name", "scenario", "decision", alt.Tooltip("risk_adjusted_roi:Q", format=".2f"),
                 alt.Tooltip("trailing_cost_usd:Q", format="$,.0f"),
                 alt.Tooltip("trailing_risk_adj_value_usd:Q", format="$,.0f")])
    moves = alt.Chart(pp).mark_line(color=RULE_COLOR, strokeWidth=1).encode(
        x="trailing_cost_usd:Q", y="trailing_risk_adj_value_usd:Q", detail="pilot_id:N")
    labels = alt.Chart(pa).mark_text(align="left", dx=10, fontSize=11).encode(
        x="trailing_cost_usd:Q", y="trailing_risk_adj_value_usd:Q", text="pilot_id:N")
    st.altair_chart((diag + moves + dots + labels).properties(height=380, title="Above the dashed line = ROI > 1"),
                    width="stretch")
    icon = {"SCALE": "🟢 SCALE", "FIX": "🟡 FIX", "INSTRUMENT": "🔵 INSTRUMENT", "HALT": "🛑 HALT", "KILL": "⚫ KILL"}
    out = pa[["pilot_id", "pilot_name", "business_unit", "risk_adjusted_roi", "adoption", "attribution",
              "months_in_pilot", "pilot_purgatory", "decision", "rationale", "intervention_effects"]].copy()
    out.insert(8, "baseline_decision", pb["decision"].map(icon).to_numpy())
    out["decision"] = out["decision"].map(icon)
    st.dataframe(out, hide_index=True, width="stretch", column_config={
        "risk_adjusted_roi": st.column_config.NumberColumn("Risk-adj ROI", format="%.2f"),
        "adoption": st.column_config.ProgressColumn("Adoption", min_value=0, max_value=1, format="percent"),
    })
    st.markdown("**Reading this:** routing makes a pilot *cheaper to fail*, it does not create value. "
                "P4 stays **INSTRUMENT** until an A/B holdout replaces survey attribution. P3 moves HALT → KILL "
                "once the security gate clears: fixing the leak reveals it was never worth scaling.")

# --------------------------------------------------------------------------- #
# 7. Decision log & brief
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.session_state.setdefault("decisions", [])
    owner = st.text_input("Decision owner", "Head of AI")
    note = st.text_area("Rationale / accepted risk", placeholder="e.g. Accept DLQ backlog for invoice agent; "
                        "Finance Ops to reconcile manually until ERP SLA restored.")
    if st.button("Record decision", type="primary"):
        st.session_state["decisions"].append({
            "recorded_at": datetime.now(timezone.utc).isoformat(), "owner": owner, "rationale": note,
            "incident": inc["id"], "interventions": iv.to_dict(),
            "metrics": {r.metric: {"before": r.before, "after": r.after}
                        for r in cmp_df.itertuples()},
            "assumptions_overrides": {"org_tpm_limit": a["capacity"]["org_tpm_limit"],
                                      "p99_slo_ms": a["slo"]["p99_latency_ms"]},
        })
        st.success("Recorded.")
    if st.session_state["decisions"]:
        st.dataframe(pd.DataFrame([{"recorded_at": d["recorded_at"], "owner": d["owner"],
                                    "scenario": Interventions(**d["interventions"]).label(),
                                    "rationale": d["rationale"]} for d in st.session_state["decisions"]]),
                     hide_index=True, width="stretch")
        st.download_button("Download decision log (JSON)", json.dumps(st.session_state["decisions"], indent=2),
                           file_name=f"decision_log_{inc['id']}.json", mime="application/json")

    st.divider()
    ok, info = narrator.available()
    st.subheader("Board brief")
    st.caption(f"Local LLM ({narrator.BASE_URL}): {'connected · ' + info if ok else info}. "
               "The LLM only narrates engine numbers; it is never in the decision path.")
    if st.button("Generate brief"):
        with st.spinner("Drafting…"):
            text, src = narrator.generate_brief(executive_summary(before, after))
        st.markdown(text)
        st.caption(f"source: {src}")
