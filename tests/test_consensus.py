import pandas as pd

from src.consensus import build_consensus_queue, summarize_queue
from src.fields import ADAS_LEVEL, ADS_LEVEL, REPORT_ID, REPORTED_AUTOMATION_LEVEL, REPORTING_ENTITY


def _phase1() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {REPORT_ID: "R-1", REPORTING_ENTITY: "A", REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
             "phase1_disagrees": True, "phase1_prediction_confidence": "0.80"},
            {REPORT_ID: "R-2", REPORTING_ENTITY: "B", REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
             "phase1_disagrees": True, "phase1_prediction_confidence": "0.70"},
            {REPORT_ID: "R-3", REPORTING_ENTITY: "C", REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
             "phase1_disagrees": False, "phase1_prediction_confidence": "0.55"},
            {REPORT_ID: "R-4", REPORTING_ENTITY: "D", REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
             "phase1_disagrees": False, "phase1_prediction_confidence": "0.10"},
            {REPORT_ID: "R-5", REPORTING_ENTITY: "E", REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
             "phase1_disagrees": True, "phase1_prediction_confidence": "0.90"},
        ]
    )


def _heuristics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {REPORT_ID: "R-1", "heuristic_disagrees": True, "heuristic_rules_triggered": "rule_a"},
            {REPORT_ID: "R-2", "heuristic_disagrees": False, "heuristic_rules_triggered": ""},
            {REPORT_ID: "R-3", "heuristic_disagrees": True, "heuristic_rules_triggered": "rule_b"},
            {REPORT_ID: "R-4", "heuristic_disagrees": False, "heuristic_rules_triggered": ""},
            {REPORT_ID: "R-5", "heuristic_disagrees": True, "heuristic_rules_triggered": "rule_c"},
        ]
    )


def _phase2() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {REPORT_ID: "R-1", "phase2_available": True, "phase2_disagrees": True,
             "phase2_llm_inferred_automation_level": ADAS_LEVEL},
            {REPORT_ID: "R-2", "phase2_available": True, "phase2_disagrees": True,
             "phase2_llm_inferred_automation_level": ADS_LEVEL},
        ]
    )


def test_consensus_score_tiers_and_ranking():
    queue = build_consensus_queue(_phase1(), _heuristics(), _phase2())

    scores = queue.set_index(REPORT_ID)["consensus_score"].to_dict()
    assert scores["R-1"] == 3  # heuristic + phase1 + phase2
    assert scores["R-2"] == 2  # phase1 + phase2
    assert scores["R-5"] == 2  # heuristic + phase1
    assert scores["R-3"] == 1  # heuristic only

    # R-4 has no disagreeing signal and must be excluded from the queue.
    assert "R-4" not in scores

    # Highest score first; ties broken by higher phase1 confidence (R-5 > R-2).
    assert queue[REPORT_ID].tolist() == ["R-1", "R-5", "R-2", "R-3"]
    tiers = queue.set_index(REPORT_ID)["consensus_tier"].to_dict()
    assert tiers["R-1"] == "three_signal_consensus"
    assert tiers["R-5"] == "two_signal_consensus"
    assert tiers["R-3"] == "single_signal"


def test_summarize_queue_counts():
    queue = build_consensus_queue(_phase1(), _heuristics(), _phase2())
    summary = summarize_queue(queue, total_records=5)

    assert summary["records_in_queue"] == 4
    assert summary["tier_counts"] == {
        "three_signal_consensus": 1,
        "two_signal_consensus": 2,
        "single_signal": 1,
    }
    assert summary["records_with_phase2_evidence"] == 2
    assert "review candidates only" in summary["responsible_use"]


def test_build_consensus_queue_handles_missing_phase2():
    # No Phase 2 evidence at all still yields a heuristic/phase1 queue.
    empty_phase2 = pd.DataFrame(columns=[REPORT_ID, "phase2_available", "phase2_disagrees"])
    queue = build_consensus_queue(_phase1(), _heuristics(), empty_phase2)

    assert queue[REPORT_ID].tolist() == ["R-5", "R-1", "R-2", "R-3"]
    assert queue.set_index(REPORT_ID)["consensus_score"].to_dict()["R-1"] == 2
    assert bool(queue["phase2_disagrees"].any()) is False
