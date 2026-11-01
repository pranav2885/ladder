import json
import math

import numpy as np
import pytest

from ladder.eval import baselines
from ladder.routing.calibrate import search
from ladder.routing.cascade import simulate
from ladder.routing.confidence_model import ConfidenceModel, out_of_fold
from ladder.routing.preroute import prompt_features, route, run_preroute, search_hybrid, search_preroute
from ladder.signals.logprobs import answer_tail, row_features, token_entropy

# 4 items x 3 tiers. Tier costs 1, 4, 10 J.
CORRECT = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
JOULES = np.tile([1.0, 4.0, 10.0], (4, 1))
SECONDS = np.tile([0.1, 0.4, 1.0], (4, 1))


def test_cascade_charges_rejected_attempts():
    probs = np.array([[0.9, 0.9, 1], [0.1, 0.9, 1], [0.1, 0.1, 1], [0.1, 0.1, 1]])
    out = simulate(probs, CORRECT, JOULES, SECONDS, [0.5, 0.5])
    assert out.final_tier.tolist() == [0, 1, 2, 2]
    assert out.joules.tolist() == [1, 5, 15, 15]  # wasted small-model work is counted
    assert out.wasted_joules.tolist() == [0, 1, 5, 5]
    assert out.seconds.tolist() == pytest.approx([0.1, 0.5, 1.5, 1.5])
    s = out.summary(3)
    assert s["accuracy"] == 0.75
    assert s["escalation_rate"] == [0.75, 2 / 3]
    assert s["accepted_wrong_rate"] == 0.0


def test_accepted_wrong_rate():
    probs = np.ones((4, 3))  # accept everything at tier 0
    s = simulate(probs, CORRECT, JOULES, SECONDS, [0.5, 0.5]).summary(3)
    assert s["accuracy"] == 0.25 and s["accepted_wrong_rate"] == 0.75


def test_baselines_always_and_oracle():
    big = baselines.always(CORRECT, JOULES, SECONDS, tier=2)
    assert big.correct.mean() == 0.75 and big.joules.sum() == 40
    small = baselines.always(CORRECT, JOULES, SECONDS, tier=0)
    assert small.correct.mean() == 0.25 and small.joules.sum() == 4
    o = baselines.oracle(CORRECT, JOULES, SECONDS)
    assert o.final_tier.tolist() == [0, 1, 2, 0]  # unanswerable item goes to the cheapest tier
    assert o.correct.mean() == 0.75 and o.joules.sum() == 1 + 4 + 10 + 1
    assert o.wasted_joules.sum() == 0


def test_random_escalation_matches_rates():
    r = baselines.random_escalation(CORRECT, JOULES, SECONDS, [0.0, 0.0], sims=10)
    assert r["accuracy"] == 0.25 and r["joules_total"] == 4  # never escalates = always smallest
    r = baselines.random_escalation(CORRECT, JOULES, SECONDS, [1.0, 1.0], sims=10)
    assert r["accuracy"] == 0.75 and r["joules_total"] == 60  # always climbs all the way


def test_calibration_meets_target_at_min_energy():
    probs = np.array([[0.9, 0.9, 1], [0.1, 0.9, 1], [0.1, 0.1, 1], [0.1, 0.1, 1]])
    best = search(probs, CORRECT, JOULES, SECONDS, target=0.0)
    out = simulate(probs, CORRECT, JOULES, SECONDS, best["thresholds"])
    assert out.correct.mean() >= 0.75  # matches always-largest on dev
    assert best["dev_joules"] == 36  # 1 + 5 + 15 + 15: cheapest way to stay at 75%
    loose = search(probs, CORRECT, JOULES, SECONDS, target=0.5)
    assert loose["dev_joules"] == 4 and loose["dev_accuracy"] == 0.25  # always-smallest allowed


def test_confidence_model_learns_and_handles_constant_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] > 0
    p = ConfidenceModel().fit(X, y).predict(X)
    assert ((p > 0.5) == y).mean() > 0.9
    assert ConfidenceModel().fit(X, np.ones(200, bool)).predict(X[:3]).tolist() == [1.0, 1.0, 1.0]
    oof = out_of_fold(X, y)
    assert oof.shape == (200,) and ((oof > 0.5) == y).mean() > 0.85
    assert out_of_fold(X[:5], np.array([1, 1, 1, 1, 0], bool)).tolist() == [0.8] * 5


def test_preroute_skips_small_tier_for_predicted_hard_items():
    # q: predicted P(correct) from the prompt alone
    q = np.array([[0.9, 0.9, 1], [0.1, 0.9, 1], [0.1, 0.1, 1], [0.1, 0.1, 1]])
    out = run_preroute(q, CORRECT, JOULES, SECONDS, [0.5, 0.5])
    assert out.final_tier.tolist() == [0, 1, 2, 2]
    assert out.joules.tolist() == [1, 4, 10, 10]  # no wasted attempts, unlike the cascade
    assert out.wasted_joules.sum() == 0
    best = search_preroute(q, CORRECT, JOULES, SECONDS, target=0.0)
    assert best["dev_joules"] == 25 and best["dev_accuracy"] == 0.75


def test_hybrid_starts_at_predicted_tier_then_cascades():
    q = np.array([[0.9, 0.9, 1], [0.9, 0.9, 1], [0.1, 0.1, 1], [0.1, 0.1, 1]])  # item 1 mispredicted easy
    p = np.array([[0.9, 0.9, 1], [0.1, 0.9, 1], [0.1, 0.1, 1], [0.1, 0.1, 1]])  # answer confidence catches it
    best = search_hybrid(q, p, CORRECT, JOULES, SECONDS, target=0.0)
    assert best["dev_accuracy"] == 0.75
    assert best["dev_joules"] == 1 + 5 + 10 + 10  # item 1 pays for the rejected tier-0 attempt only


def test_route_and_prompt_features():
    q = np.array([[0.2, 0.7, 1.0], [0.6, 0.1, 1.0]])
    assert route(q, [0.5, 0.5]).tolist() == [1, 0]
    assert route(q, [1.01, 1.01]).tolist() == [2, 2]
    f = prompt_features({"input": {"question": "Tom has 3 apples, then buys 2 more. How many in total?"}})
    assert f["n_numbers"] == 2 and f["n_step_words"] >= 3 and f["n_questions"] == 1
    assert "log_len_question" in f


def test_logprob_features():
    assert token_entropy([["a", 0.0]]) == 0.0
    assert token_entropy([["a", math.log(0.5)], ["b", math.log(0.5)]]) == pytest.approx(math.log(2))
    toks = [{"token": t, "logprob": lp, "top": [[t, lp]]}
            for t, lp in [("so", -0.1), ("Answer", -0.01), (":", -0.01), (" 4", -2.0)]]
    assert [t["token"] for t in answer_tail(toks)] == [":", " 4"]
    f = row_features(json.dumps(toks), json.dumps({"not_truncated": True}), 4, "stop")
    assert f["min_lp"] == -2.0 and f["tail_min_lp"] == -2.0 and f["frac_low"] == 0.25
    assert f["v_not_truncated"] == 1.0 and f["truncated"] == 0.0
    empty = row_features(None, "{}", 0, "length")
    assert empty["mean_lp"] == 0.0 and empty["truncated"] == 1.0
