"""
Enterprise AI War Room — core scenario engine.

Design contract
---------------
* Telemetry is stored as *demand* (what each caller tried to do) plus the
  *observed* outcome. The engine never trusts observed latency/status when it
  compares scenarios: it re-simulates BOTH baseline and intervention runs through
  the same capacity model with common random numbers (CRN), so every delta is
  attributable to the intervention, not to sampling noise.
* All economic / policy constants live in config/assumptions.json. Nothing here
  is a magic number you cannot audit.
* Pure functions over pandas DataFrames: easy to unit test, easy to call from a
  notebook, a CI gate, or the Streamlit app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "assumptions.json"

AGENT_SERVICE = "invoice-recon-agent"
FILTER_MODES = ("none", "post_filter", "pre_filter")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_assumptions(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class Interventions:
    """Executive levers. Defaults = 'do nothing' (the baseline)."""

    # Hard token circuit breaker
    circuit_breaker: bool = False
    max_consecutive_failures: int = 3
    trace_token_budget: int = 60_000
    service_velocity_limit_tpm: int = 1_500_000
    # Dynamic model routing (router score thresholds)
    model_routing: bool = False
    routing_small_max: float = 0.35
    routing_balanced_max: float = 0.65
    # Vector metadata pre-filtering (ACL enforced inside the ANN query)
    vector_prefilter: bool = False

    def label(self) -> str:
        on = [n for n, f in (("CircuitBreaker", self.circuit_breaker),
                             ("ModelRouting", self.model_routing),
                             ("VectorPrefilter", self.vector_prefilter)) if f]
        return " + ".join(on) if on else "Baseline (no intervention)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINE = Interventions()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_telemetry(path: Path | str = DATA_DIR / "telemetry_logs.json") -> tuple[dict, pd.DataFrame]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    df = pd.DataFrame(doc["events"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    start = pd.Timestamp(doc["metadata"]["window_start"])
    df["minute"] = ((df["timestamp"] - start).dt.total_seconds() // 60).astype(int)
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    return doc["metadata"], df


def load_vector_audit(path: Path | str = DATA_DIR / "vector_audit_log.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["event_ts"])
    df["doc_acl_groups"] = df["doc_acl_groups"].fillna("ALL")
    return df


def load_pilots(path: Path | str = DATA_DIR / "pilot_metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


@dataclass
class Datasets:
    metadata: dict
    telemetry: pd.DataFrame
    vector: pd.DataFrame
    pilots: pd.DataFrame


def load_all(data_dir: Path | str = DATA_DIR) -> Datasets:
    data_dir = Path(data_dir)
    meta, tel = load_telemetry(data_dir / "telemetry_logs.json")
    return Datasets(meta, tel, load_vector_audit(data_dir / "vector_audit_log.csv"),
                    load_pilots(data_dir / "pilot_metrics.csv"))


# --------------------------------------------------------------------------- #
# Economics primitives
# --------------------------------------------------------------------------- #
def token_cost(prompt: pd.Series, completion: pd.Series, tier: pd.Series, tiers: dict) -> pd.Series:
    pin = tier.map({k: v["input_per_1m"] for k, v in tiers.items()})
    pout = tier.map({k: v["output_per_1m"] for k, v in tiers.items()})
    return (prompt * pin + completion * pout) / 1_000_000


def queue_latency_factor(rho: np.ndarray | pd.Series, alpha: float, rho_cap: float) -> np.ndarray:
    """Soft M/M/1-style degradation: 1 + a*rho^2/(1-rho). ~1.04x at rho=0.3, ~5x at rho=0.9."""
    r = np.clip(np.asarray(rho, dtype=float), 0.0, rho_cap)
    return 1.0 + alpha * r ** 2 / (1.0 - r)


def route_tier(router_score: pd.Series, small_max: float, balanced_max: float) -> pd.Series:
    return pd.Series(np.select([router_score <= small_max, router_score <= balanced_max],
                               ["small", "balanced"], "frontier"), index=router_score.index)


# --------------------------------------------------------------------------- #
# Telemetry simulation
# --------------------------------------------------------------------------- #
def simulate_telemetry(events: pd.DataFrame, a: dict, iv: Interventions = BASELINE,
                       seed: int | None = None) -> pd.DataFrame:
    """Re-run the incident window under a given intervention set.

    Returns one row per *intended* LLM call with: effective tier, cost, latency,
    status ('OK', 'THROTTLED', 'TIMEOUT', 'SHED'), and blocked_by reason.
    """
    seed = a["simulation_seed"] if seed is None else seed
    tiers, cap = a["model_tiers"], a["capacity"]
    df = events.copy()
    n = len(df)

    # Common random numbers: identical draws per event across every scenario.
    rng = np.random.default_rng(seed)
    u_throttle = rng.random(n)
    jitter = rng.lognormal(mean=0.0, sigma=cap["latency_jitter_sigma"], size=n)

    df["blocked_by"] = ""
    df["breaker_open"] = False

    # ---- 1. Hard token circuit breaker -------------------------------------
    if iv.circuit_breaker:
        fail_cap = df["consecutive_tool_failures"] >= iv.max_consecutive_failures
        df.loc[fail_cap, "blocked_by"] = "max_consecutive_failures"

        # Per-root-trace token budget (counts only calls that actually went out).
        live = df["blocked_by"] == ""
        spent_before = (df["total_tokens"].where(live, 0)
                        .groupby(df["root_trace_id"]).cumsum() - df["total_tokens"].where(live, 0))
        over_budget = live & (spent_before + df["total_tokens"] > iv.trace_token_budget)
        df.loc[over_budget, "blocked_by"] = "trace_token_budget"

        # Per-service token-velocity breaker (tokens / minute).
        live = df["blocked_by"] == ""
        vel = (df["total_tokens"].where(live, 0)
               .groupby([df["service"], df["minute"]]).cumsum())
        velocity_trip = live & (vel > iv.service_velocity_limit_tpm)
        df.loc[velocity_trip, "blocked_by"] = "token_velocity"
        df["breaker_open"] = df["blocked_by"] != ""

    # ---- 2. Dynamic model routing ------------------------------------------
    df["effective_tier"] = df["model_tier"]
    if iv.model_routing:
        df["effective_tier"] = route_tier(df["router_complexity_score"],
                                          iv.routing_small_max, iv.routing_balanced_max)
    speed = {k: v["speed_factor"] for k, v in tiers.items()}
    capab = {k: v["capability"] for k, v in tiers.items()}
    df["service_time_eff_ms"] = (df["service_time_ms"]
                                 * df["effective_tier"].map(speed) / df["model_tier"].map(speed))
    df["quality_regression"] = df["eval_complexity_score"] > df["effective_tier"].map(capab)

    # ---- 3. Shared capacity model (org-wide TPM quota + queueing) -----------
    sent = df["blocked_by"] == ""
    demand = df["total_tokens"].where(sent, 0).groupby(df["minute"]).sum()
    limit = cap["org_tpm_limit"]
    throttle_frac = (1.0 - limit / demand.where(demand > 0, np.nan)).clip(lower=0).fillna(0.0)
    df["throttle_frac"] = df["minute"].map(throttle_frac).fillna(0.0)
    throttled = sent & (u_throttle < df["throttle_frac"].to_numpy())

    accepted_tokens = df["total_tokens"].where(sent & ~throttled, 0).groupby(df["minute"]).sum()
    rho = (accepted_tokens / limit).reindex(df["minute"]).to_numpy()
    factor = queue_latency_factor(rho, cap["queue_alpha"], cap["queue_rho_cap"])

    latency = df["service_time_eff_ms"].to_numpy() * factor * jitter
    timed_out = sent & ~throttled & (latency > cap["client_timeout_ms"])
    latency = np.where(throttled, cap["throttle_backoff_ms"] * jitter, latency)
    latency = np.minimum(latency, cap["client_timeout_ms"])

    df["status"] = np.select([~sent, throttled, timed_out], ["SHED", "THROTTLED", "TIMEOUT"], "OK")
    df["latency_ms"] = np.where(sent, latency, np.nan)
    df["rho"] = rho

    # Billing: provider bills accepted inference, including client-side timeouts.
    billable = df["status"].isin(["OK", "TIMEOUT"])
    df["cost_usd"] = token_cost(df["prompt_tokens"], df["completion_tokens"],
                                df["effective_tier"], tiers).where(billable, 0.0)
    df["billed_tokens"] = df["total_tokens"].where(billable, 0)
    # A call "progressed" the business task only if the model answered AND
    # (for agent calls) the downstream tool call it drove succeeded.
    tool_ok = df["tool_status"].isna() | (df["tool_status"] == 200)
    df["progressed"] = (df["status"] == "OK") & tool_ok
    return df


def minute_series(sim: pd.DataFrame, a: dict, bystanders_only: bool = False) -> pd.DataFrame:
    d = sim[sim["service"] != AGENT_SERVICE] if bystanders_only else sim
    g = d.groupby("minute")
    out = pd.DataFrame({
        "cost_usd": g["cost_usd"].sum(),
        "billed_tokens": g["billed_tokens"].sum(),
        "requests_sent": g["status"].apply(lambda s: (s != "SHED").sum()),
        "errors": g["status"].apply(lambda s: s.isin(["THROTTLED", "TIMEOUT"]).sum()),
        "p99_latency_ms": g["latency_ms"].quantile(0.99),
        "p50_latency_ms": g["latency_ms"].quantile(0.50),
        "shed": g["status"].apply(lambda s: (s == "SHED").sum()),
    }).reindex(range(int(sim["minute"].max()) + 1), fill_value=0)
    out["burn_usd_per_hr"] = out["cost_usd"] * 60
    out["error_rate"] = (out["errors"] / out["requests_sent"].replace(0, np.nan)).fillna(0)
    out.index.name = "minute"
    return out.reset_index()


def ops_metrics(sim: pd.DataFrame, a: dict) -> dict[str, float]:
    slo = a["slo"]
    sent = sim[sim["status"] != "SHED"]
    by = sent[sent["service"] != AGENT_SERVICE]
    ms = minute_series(sim, a)
    ms_by = minute_series(sim, a, bystanders_only=True)
    window_min = len(ms)
    agent = sim[sim["service"] == AGENT_SERVICE]
    progressed_cost = sim.loc[sim["progressed"], "cost_usd"].sum()
    total_cost = sim["cost_usd"].sum()
    peak_burn = ms["burn_usd_per_hr"].max()
    incident_burn = ms.loc[ms["burn_usd_per_hr"] > 2 * slo["budgeted_burn_usd_per_hr"], "burn_usd_per_hr"]
    sustained_burn = float(incident_burn.median()) if len(incident_burn) else float(ms["burn_usd_per_hr"].median())
    return {
        "window_cost_usd": float(total_cost),
        "peak_burn_usd_per_hr": float(peak_burn),
        "avg_burn_usd_per_hr": float(total_cost / window_min * 60),
        "exposure_72h_unmitigated_usd": float(sustained_burn * 72),
        "peak_token_velocity_tpm": float(ms["billed_tokens"].max()),
        "wasted_spend_usd": float(total_cost - progressed_cost),
        "wasted_spend_pct": float((total_cost - progressed_cost) / total_cost) if total_cost else 0.0,
        "p50_latency_ms": float(sent["latency_ms"].quantile(0.50)),
        "p99_latency_ms": float(sent["latency_ms"].quantile(0.99)),
        "p99_bystander_latency_ms": float(by["latency_ms"].quantile(0.99)),
        "slo_breach_minutes": int((ms_by["p99_latency_ms"] > slo["p99_latency_ms"]).sum()),
        "error_rate": float(sent["status"].isin(["THROTTLED", "TIMEOUT"]).mean()),
        "bystander_error_rate": float(by["status"].isin(["THROTTLED", "TIMEOUT"]).mean()),
        "calls_shed": int((sim["status"] == "SHED").sum()),
        "agent_traces_to_dlq": int(agent.loc[agent["status"] == "SHED", "root_trace_id"].nunique()),
        "quality_regression_rate": float(sent["quality_regression"].mean()),
        "requests_rerouted_pct": float((sent["effective_tier"] != sent["model_tier"]).mean()),
        "window_minutes": window_min,
    }


# --------------------------------------------------------------------------- #
# Governance: vector retrieval ACL simulation
# --------------------------------------------------------------------------- #
def acl_permitted(v: pd.DataFrame, a: dict) -> pd.Series:
    """Policy-as-code. The audit derives permission from policy, never from the log's opinion."""
    levels = a["governance"]["classification_levels"]
    same_tenant = (v["doc_tenant_id"] == v["tenant_id"]) | (v["doc_tenant_id"] == "global")
    clearance_ok = v["user_clearance"] >= v["doc_classification"].map(levels)
    groups = v["doc_acl_groups"].astype(str).str.split(";")
    group_ok = [("ALL" in g) or (dept in g) for g, dept in zip(groups, v["user_department"])]
    return same_tenant & clearance_ok & pd.Series(group_ok, index=v.index)


def simulate_retrieval(vector: pd.DataFrame, a: dict, mode: str = "none") -> pd.DataFrame:
    """Decide which candidate chunks reach the LLM context under a filter mode.

    none        -> ANN top-k, no ACL (the shipped bug)
    post_filter -> ANN top-k, then drop unauthorised chunks (safe, but starves context)
    pre_filter  -> ACL predicate inside the ANN query: top-k over authorised chunks only
    """
    if mode not in FILTER_MODES:
        raise ValueError(f"mode must be one of {FILTER_MODES}")
    v = vector.copy()
    v["permitted"] = acl_permitted(v, a)
    in_top_k = v["candidate_rank"] <= v["top_k"]
    if mode == "none":
        v["delivered"] = in_top_k
    elif mode == "post_filter":
        v["delivered"] = in_top_k & v["permitted"]
    else:
        auth_rank = (v[v["permitted"]].groupby("query_id")["similarity_score"]
                     .rank(ascending=False, method="first"))
        v["delivered"] = False
        v.loc[auth_rank.index, "delivered"] = auth_rank <= v.loc[auth_rank.index, "top_k"]
    v["violation"] = v["delivered"] & ~v["permitted"]
    v["cross_tenant"] = (v["doc_tenant_id"] != v["tenant_id"]) & (v["doc_tenant_id"] != "global")
    overhead = a["governance"]["prefilter_latency_overhead"] if mode == "pre_filter" else 0.0
    v["retrieval_latency_eff_ms"] = v["retrieval_latency_ms"] * (1 + overhead)
    v["filter_mode_simulated"] = mode
    return v


def governance_metrics(v: pd.DataFrame, a: dict) -> dict[str, Any]:
    g = a["governance"]
    w = v["doc_classification"].map(g["violation_weights"]).astype(float)
    w = w * np.where(v["cross_tenant"], g["cross_tenant_multiplier"], 1.0)
    d, viol = v[v["delivered"]], v[v["violation"]]
    weighted_delivered = float(w[v["delivered"]].sum())
    weighted_viol = float(w[v["violation"]].sum())
    compliance = 100.0 * (1 - weighted_viol / weighted_delivered) if weighted_delivered else 100.0
    exposed_records = int(viol.drop_duplicates("chunk_id")["pii_record_count"].sum())
    expected_loss = exposed_records * g["cost_per_exposed_record_usd"] * g["probability_exposed_record_exfiltrated"]
    q_fill = d.groupby("query_id").size().reindex(v["query_id"].unique(), fill_value=0)
    top_k = v.groupby("query_id")["top_k"].first()
    queries_with_viol = viol["query_id"].nunique()
    return {
        "chunks_delivered": int(len(d)),
        "acl_violations": int(len(viol)),
        "violation_rate": float(len(viol) / len(d)) if len(d) else 0.0,
        "queries_with_violation": int(queries_with_viol),
        "queries_with_violation_pct": float(queries_with_viol / v["query_id"].nunique()),
        "users_exposed": int(viol["user_id"].nunique()),
        "restricted_chunks_exposed": int((viol["doc_classification"] == "restricted").sum()),
        "pii_chunks_exposed": int(viol["contains_pii"].sum()),
        "pii_records_exposed": exposed_records,
        "cross_tenant_leaks": int(viol["cross_tenant"].sum()),
        "compliance_score": float(compliance),
        "reportable_exposure": bool(exposed_records > 0),
        "expected_breach_loss_usd": float(expected_loss),
        "context_fill_rate": float((q_fill / top_k.reindex(q_fill.index)).mean()),
        "p99_retrieval_latency_ms": float(d["retrieval_latency_eff_ms"].quantile(0.99)),
    }


# --------------------------------------------------------------------------- #
# Composite risk
# --------------------------------------------------------------------------- #
def risk_score(ops: dict, gov: dict, a: dict) -> dict[str, float]:
    """Heuristic 0-100 (higher = worse). Explainable on one slide; tune the weights."""
    budget = a["slo"]["budgeted_burn_usd_per_hr"]
    fin = float(np.clip((ops["peak_burn_usd_per_hr"] / budget - 1) * 100 / 9, 0, 100))  # 10x budget = 100
    rel = float(np.clip(100 * ops["slo_breach_minutes"] / ops["window_minutes"]
                        + 200 * ops["bystander_error_rate"], 0, 100))
    gov_r = 100 - gov["compliance_score"]
    if gov["reportable_exposure"]:
        gov_r = max(gov_r, 80.0)  # a reportable PII exposure is never 'low risk'
    w = a["risk_weights"]
    return {"financial_risk": fin, "reliability_risk": rel, "governance_risk": float(gov_r),
            "composite_risk": float(w["financial"] * fin + w["reliability"] * rel + w["governance"] * gov_r)}


# --------------------------------------------------------------------------- #
# Portfolio: Pilot Purgatory triage
# --------------------------------------------------------------------------- #
def routing_cost_ratio(events: pd.DataFrame, a: dict, iv: Interventions) -> dict[str, float]:
    """Steady-state (non-incident) cost ratio per service if routing were applied."""
    steady = events[events["consecutive_tool_failures"] == 0]
    tiers = a["model_tiers"]
    before = token_cost(steady["prompt_tokens"], steady["completion_tokens"], steady["model_tier"], tiers)
    routed = route_tier(steady["router_complexity_score"], iv.routing_small_max, iv.routing_balanced_max)
    after = token_cost(steady["prompt_tokens"], steady["completion_tokens"], routed, tiers)
    regress = steady["eval_complexity_score"] > routed.map({k: v["capability"] for k, v in tiers.items()})
    out = {}
    for svc, idx in steady.groupby("service").groups.items():
        out[svc] = {"cost_ratio": float(after[idx].sum() / before[idx].sum()),
                    "regression_rate": float(regress[idx].mean())}
    return out


def evaluate_portfolio(pilots: pd.DataFrame, events: pd.DataFrame, a: dict,
                       iv: Interventions = BASELINE) -> pd.DataFrame:
    p = a["portfolio"]
    conf_map = p["attribution_confidence"]
    last = pilots["month"].max()
    trail = pilots[pilots["month"] > last - p["trailing_months"]].copy()
    ratios = routing_cost_ratio(events, a, iv) if iv.model_routing else {}
    g = a["governance"]

    rows = []
    for pid, d in trail.groupby("pilot_id"):
        full = pilots[pilots["pilot_id"] == pid].sort_values("month")
        svc = d["service"].iloc[0]
        inference = d["inference_cost_usd"].copy()
        value = d["measured_value_usd"].copy()
        records = d["records_at_risk"].copy()
        crit = int(d["security_findings_critical"].iloc[-1])
        notes = []

        if iv.circuit_breaker:
            overhead = (d["retry_overhead_pct"] - p["residual_retry_overhead_after_breaker"]).clip(lower=0)
            saved = float((inference * overhead).sum())
            inference = inference * (1 - overhead)
            if saved > 1:
                notes.append(f"breaker removes ${saved:,.0f} retry overhead")
        if iv.model_routing and svc in ratios:
            r = ratios[svc]
            inference = inference * r["cost_ratio"]
            value = value * (1 - r["regression_rate"])
            notes.append(f"routing: inference x{r['cost_ratio']:.2f}, value -{r['regression_rate']:.1%}")
        if iv.vector_prefilter and records.sum() > 0:
            records = records * 0
            crit = max(0, crit - int(d["acl_finding_open"].iloc[-1]))
            notes.append("pre-filter closes ACL finding")

        conf = conf_map.get(d["value_attribution"].iloc[-1], 0.0)
        risk_adj_value = float((value * conf).sum())
        expected_loss = float(records.sum() * g["cost_per_exposed_record_usd"]
                              * g["probability_exposed_record_exfiltrated"])
        cost = float(inference.sum() + d["platform_cost_usd"].sum() + d["engineering_cost_usd"].sum())
        roi = (risk_adj_value - expected_loss) / cost if cost else 0.0
        adoption = float(d["active_users"].iloc[-1] / d["licensed_users"].iloc[-1])
        monthly_roi = (full["measured_value_usd"] * conf) / (
            full["inference_cost_usd"] + full["platform_cost_usd"] + full["engineering_cost_usd"])
        slope = float(np.polyfit(np.arange(len(monthly_roi)), monthly_roi.to_numpy(), 1)[0])
        months_pilot = int((full["phase"] == "pilot").sum())

        retry_last = float(d["retry_overhead_pct"].iloc[-1])
        if iv.circuit_breaker:
            retry_last = min(retry_last, p["residual_retry_overhead_after_breaker"])

        if crit > 0 or records.sum() > 0:
            decision, why = "HALT", "Security gate: open critical finding / records at risk. Fix before any spend."
        elif retry_last > p["max_retry_overhead_to_scale"]:
            decision, why = "FIX", (f"Reliability gate: {retry_last:.0%} of spend is retry overhead. Cheaper tokens "
                                    "(routing) would only mask the loop; bound it first.")
        elif roi >= p["scale_min_roi"] and adoption >= p["scale_min_adoption"] and conf >= p["scale_min_confidence"]:
            decision, why = "SCALE", "Risk-adjusted ROI, adoption and attribution all clear the bar."
        elif conf < p["scale_min_confidence"] and adoption >= 0.6:
            decision, why = "INSTRUMENT", "Users love it but value is unproven (weak attribution). Run an A/B holdout before more spend."
        elif roi >= p["fix_min_roi"] or slope > 0.05:
            decision, why = "FIX", "Economics are close or improving; fix unit cost/adoption with a dated exit criterion."
        else:
            decision, why = "KILL", "Below ROI floor with no improving trend. Redeploy the team."

        rows.append({
            "pilot_id": pid, "pilot_name": d["pilot_name"].iloc[0], "business_unit": d["business_unit"].iloc[0],
            "service": svc, "trailing_cost_usd": cost, "trailing_inference_usd": float(inference.sum()),
            "trailing_risk_adj_value_usd": risk_adj_value, "expected_breach_loss_usd": expected_loss,
            "risk_adjusted_roi": roi, "adoption": adoption, "attribution": d["value_attribution"].iloc[-1],
            "attribution_confidence": conf, "roi_trend_per_month": slope, "months_in_pilot": months_pilot,
            "retry_overhead_last_month": retry_last,
            "pilot_purgatory": bool(months_pilot >= p["purgatory_months"] and roi < 1.0),
            "decision": decision, "rationale": why, "intervention_effects": "; ".join(notes),
        })
    return pd.DataFrame(rows).sort_values("pilot_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration: one call = one scenario; compare() = the executive decision loop
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    interventions: Interventions
    telemetry: pd.DataFrame
    minutes: pd.DataFrame
    minutes_bystander: pd.DataFrame
    retrieval: pd.DataFrame
    ops: dict
    gov: dict
    risk: dict
    portfolio: pd.DataFrame

    def headline(self) -> dict[str, float]:
        return {**self.ops, **{k: v for k, v in self.gov.items() if not isinstance(v, bool)},
                **self.risk}


def run_scenario(ds: Datasets, a: dict, iv: Interventions = BASELINE) -> ScenarioResult:
    sim = simulate_telemetry(ds.telemetry, a, iv)
    ret = simulate_retrieval(ds.vector, a, "pre_filter" if iv.vector_prefilter else "none")
    ops, gov = ops_metrics(sim, a), governance_metrics(ret, a)
    return ScenarioResult(iv, sim, minute_series(sim, a), minute_series(sim, a, True), ret,
                          ops, gov, risk_score(ops, gov, a), evaluate_portfolio(ds.pilots, ds.telemetry, a, iv))


# Direction: +1 = higher is better, -1 = lower is better.
METRIC_SPEC: dict[str, tuple[str, int, str]] = {
    "window_cost_usd": ("Incident-window spend", -1, "usd"),
    "peak_burn_usd_per_hr": ("Peak burn rate ($/hr)", -1, "usd"),
    "exposure_72h_unmitigated_usd": ("72h exposure if undetected", -1, "usd"),
    "wasted_spend_pct": ("Spend on non-progressing calls", -1, "pct"),
    "peak_token_velocity_tpm": ("Peak token velocity (TPM)", -1, "int"),
    "p99_latency_ms": ("P99 latency, all (ms)", -1, "int"),
    "p99_bystander_latency_ms": ("P99 latency, bystander tenants (ms)", -1, "int"),
    "slo_breach_minutes": ("SLO breach minutes", -1, "int"),
    "bystander_error_rate": ("Bystander error rate", -1, "pct"),
    "quality_regression_rate": ("Quality regression (routing)", -1, "pct"),
    "agent_traces_to_dlq": ("Agent traces failed-fast to DLQ", -1, "int"),
    "acl_violations": ("ACL violations in LLM context", -1, "int"),
    "pii_records_exposed": ("PII records exposed", -1, "int"),
    "cross_tenant_leaks": ("Cross-tenant leaks", -1, "int"),
    "compliance_score": ("Compliance score", +1, "score"),
    "context_fill_rate": ("RAG context fill rate", +1, "pct"),
    "expected_breach_loss_usd": ("Expected breach loss", -1, "usd"),
    "composite_risk": ("Composite risk (0-100)", -1, "score"),
}


def compare(before: ScenarioResult, after: ScenarioResult) -> pd.DataFrame:
    b, a_ = before.headline(), after.headline()
    rows = []
    for key, (label, direction, fmt) in METRIC_SPEC.items():
        bv, av = float(b[key]), float(a_[key])
        delta = av - bv
        rel = delta / bv if bv else 0.0
        rows.append({"metric": key, "label": label, "before": bv, "after": av, "delta": delta,
                     "delta_pct": rel, "improved": (delta * direction) > 1e-12,
                     "regressed": (delta * direction) < -1e-12, "fmt": fmt, "direction": direction})
    return pd.DataFrame(rows)


def executive_summary(before: ScenarioResult, after: ScenarioResult) -> list[str]:
    """Plain-language bullets a Head of AI can paste into an incident channel."""
    b, a_ = before.headline(), after.headline()
    iv = after.interventions
    out = [f"Scenario: {iv.label()}."]
    saved = b["window_cost_usd"] - a_["window_cost_usd"]
    out.append(f"Incident-window spend ${b['window_cost_usd']:,.0f} -> ${a_['window_cost_usd']:,.0f} "
               f"(saves ${saved:,.0f}); 72h undetected exposure ${b['exposure_72h_unmitigated_usd']:,.0f} -> "
               f"${a_['exposure_72h_unmitigated_usd']:,.0f}.")
    out.append(f"Bystander P99 {b['p99_bystander_latency_ms']/1000:.1f}s -> {a_['p99_bystander_latency_ms']/1000:.1f}s; "
               f"bystander errors {b['bystander_error_rate']:.1%} -> {a_['bystander_error_rate']:.1%}; "
               f"SLO breach minutes {b['slo_breach_minutes']} -> {a_['slo_breach_minutes']}.")
    out.append(f"ACL violations {b['acl_violations']} -> {a_['acl_violations']}; PII records exposed "
               f"{b['pii_records_exposed']} -> {a_['pii_records_exposed']}; compliance "
               f"{b['compliance_score']:.1f} -> {a_['compliance_score']:.1f}.")
    # Trade-offs are mandatory, not optional, in an executive readout.
    if iv.circuit_breaker:
        out.append(f"TRADE-OFF: {a_['agent_traces_to_dlq']} invoice traces fail fast to DLQ and need human "
                   f"reconciliation until the ERP dependency recovers.")
    if iv.model_routing:
        out.append(f"TRADE-OFF: {a_['requests_rerouted_pct']:.0%} of traffic downgraded; offline-eval quality "
                   f"regression {a_['quality_regression_rate']:.2%}. Gate on eval set before prod.")
    if iv.vector_prefilter:
        out.append(f"TRADE-OFF: retrieval P99 +{after.retrieval['retrieval_latency_eff_ms'].quantile(.99) - before.retrieval['retrieval_latency_eff_ms'].quantile(.99):.0f}ms "
                   f"from filtered ANN; leaked data already in logs/caches still needs purge + DPO review.")
    out.append(f"Composite risk {b['composite_risk']:.0f} -> {a_['composite_risk']:.0f} / 100.")
    return out


if __name__ == "__main__":  # CLI smoke run: python -m simulations.scenario_engine
    assumptions = load_assumptions()
    data = load_all()
    base = run_scenario(data, assumptions)
    full = run_scenario(data, assumptions, Interventions(circuit_breaker=True, model_routing=True, vector_prefilter=True))
    pd.set_option("display.width", 160)
    print(compare(base, full)[["label", "before", "after", "delta_pct"]].to_string(index=False))
    print()
    print("\n".join(executive_summary(base, full)))
    print()
    print(full.portfolio[["pilot_id", "pilot_name", "risk_adjusted_roi", "decision"]].to_string(index=False))
