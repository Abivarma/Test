"""Executable acceptance criteria. If an intervention stops doing what the README claims, CI goes red."""
import pytest

from simulations.scenario_engine import (
    BASELINE, Interventions, compare, governance_metrics, load_all, load_assumptions,
    run_scenario, simulate_retrieval, simulate_telemetry,
)

A = load_assumptions()
DS = load_all()


@pytest.fixture(scope="module")
def base():
    return run_scenario(DS, A)


def test_baseline_reproduces_stored_observations():
    """Model validation: re-simulated baseline == telemetry as logged (same model, same CRN)."""
    sim = simulate_telemetry(DS.telemetry, A, BASELINE)
    status = sim["status"].map({"OK": 200, "THROTTLED": 429, "TIMEOUT": 504, "SHED": 0})
    assert (status.to_numpy() == DS.telemetry["http_status"].to_numpy()).all()
    assert abs(sim["cost_usd"].sum() - DS.telemetry["billed_cost_usd"].sum()) < 0.05


def test_baseline_is_an_incident(base):
    o = base.ops
    assert o["peak_burn_usd_per_hr"] > 10 * A["slo"]["budgeted_burn_usd_per_hr"]
    assert o["peak_token_velocity_tpm"] > A["capacity"]["org_tpm_limit"]
    assert o["bystander_error_rate"] > 0.05
    assert base.gov["reportable_exposure"] and base.gov["cross_tenant_leaks"] > 0


def test_circuit_breaker_contains_cost_and_blast_radius(base):
    r = run_scenario(DS, A, Interventions(circuit_breaker=True))
    assert r.ops["window_cost_usd"] < 0.15 * base.ops["window_cost_usd"]
    assert r.ops["bystander_error_rate"] < A["slo"]["max_error_rate"]
    assert r.ops["slo_breach_minutes"] < base.ops["slo_breach_minutes"]
    assert r.ops["agent_traces_to_dlq"] > 0  # the trade-off is surfaced, not hidden


def test_routing_alone_cuts_cost_but_does_not_stop_the_loop(base):
    r = run_scenario(DS, A, Interventions(model_routing=True))
    assert r.ops["window_cost_usd"] < base.ops["window_cost_usd"]
    assert r.ops["bystander_error_rate"] > A["slo"]["max_error_rate"]  # cheaper tokens != bounded loop
    assert 0 < r.ops["quality_regression_rate"] < 0.02


def test_aggressive_routing_costs_quality():
    lax = run_scenario(DS, A, Interventions(model_routing=True, routing_small_max=0.55, routing_balanced_max=0.9))
    std = run_scenario(DS, A, Interventions(model_routing=True))
    assert lax.ops["quality_regression_rate"] > std.ops["quality_regression_rate"]


def test_prefilter_eliminates_leakage_and_beats_postfilter_recall():
    pre = governance_metrics(simulate_retrieval(DS.vector, A, "pre_filter"), A)
    post = governance_metrics(simulate_retrieval(DS.vector, A, "post_filter"), A)
    assert pre["acl_violations"] == 0 and post["acl_violations"] == 0
    assert pre["compliance_score"] == 100.0
    assert pre["context_fill_rate"] > post["context_fill_rate"] + 0.15


def test_portfolio_triage(base):
    d = dict(zip(base.portfolio["pilot_id"], base.portfolio["decision"]))
    assert d == {"P1": "SCALE", "P2": "FIX", "P3": "HALT", "P4": "INSTRUMENT", "P5": "FIX"}
    full = run_scenario(DS, A, Interventions(circuit_breaker=True, model_routing=True, vector_prefilter=True))
    d2 = dict(zip(full.portfolio["pilot_id"], full.portfolio["decision"]))
    assert d2["P5"] == "SCALE" and d2["P3"] == "KILL" and d2["P4"] == "INSTRUMENT"


def test_routing_alone_cannot_promote_a_looping_pilot():
    r = run_scenario(DS, A, Interventions(model_routing=True))
    p5 = r.portfolio.set_index("pilot_id").loc["P5"]
    assert p5["decision"] == "FIX" and "Reliability gate" in p5["rationale"]


def test_compare_direction_semantics(base):
    full = run_scenario(DS, A, Interventions(circuit_breaker=True, model_routing=True, vector_prefilter=True))
    c = compare(base, full).set_index("metric")
    assert c.loc["compliance_score", "improved"] and c.loc["window_cost_usd", "improved"]
    assert c.loc["quality_regression_rate", "regressed"]
