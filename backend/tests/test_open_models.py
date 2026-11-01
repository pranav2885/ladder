from ladder.open_models import (
    MAJOR_LABS,
    annotate,
    detect_category,
    params_b,
    recommend,
    snapshot,
)


def test_snapshot_is_sourced_and_open_only():
    snap = snapshot()
    assert snap["source"].startswith("LMArena")
    for key, cat in snap["categories"].items():
        assert cat["url"].startswith("https://arena.ai/leaderboard/chat/text")
        assert cat["as_of"].startswith("2026-")
        entries = cat["top"] + cat["best_by_org"]
        assert entries, key
        assert all("proprietary" not in e["license"].lower() for e in entries)
        ranks = [e["rank"] for e in cat["best_by_org"]]
        assert ranks == sorted(ranks)  # best_by_org is in leaderboard order


def test_params_parsed_only_when_stated():
    assert params_b("Gemma 4 31B") == 31
    assert params_b("Qwen3.8-27B") == 27
    assert params_b("Qwen3.5-397B-A17B") == 397  # total, not active, parameters
    assert params_b("gpt-oss-120b") == 120
    assert params_b("SOLAR 10.7B Instruct v1.0") == 10.7
    assert params_b("Kimi K3 (Max)") is None
    assert params_b("Phi-4") is None


def test_fit_estimate():
    e = annotate({"model": "Granite 4.1 8B"}, ram_gb=23.4)
    assert e["est_q4_ram_gb"] == 5.8 and e["fits_this_machine"] is True
    assert annotate({"model": "gpt-oss-120b"}, ram_gb=23.4)["fits_this_machine"] is False
    assert annotate({"model": "Phi-4"}, ram_gb=23.4)["fits_this_machine"] is None  # unknown size


def test_category_mapping():
    assert detect_category("math") == "math"
    assert detect_category("extract") == "instruction_following"
    assert detect_category(None, "```python\ndef f(): pass\n```") == "coding"
    assert detect_category(None, "If I have 3 + 4 apples, how many?") == "math"
    assert detect_category(None, "Tell me about Paris") == "overall"


def test_recommend_math_on_this_laptop():
    r = recommend("math", ram_gb=23.4)
    assert r["category_label"] == "Math" and r["source_url"].endswith("/math")
    assert r["best"]["model"] == "GLM-5.3-Flash" and r["best"]["rank"] == 16
    fit = r["best_that_fits"]
    assert fit is not None and fit["fits_this_machine"] and fit["params_b"] <= 32
    orgs = [e["org"] for e in r["major_labs"]]
    assert orgs == [o for o in MAJOR_LABS if o in orgs]  # display order, major labs only
    assert "Google" in orgs and "OpenAI" in orgs
