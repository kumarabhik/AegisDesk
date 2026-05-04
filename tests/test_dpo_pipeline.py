from __future__ import annotations

import json

from training.build_aegisdesk_preference_corpus import _to_preference_row
from training.dpo_pair_generator import build_dpo_pair, generate_pairs


def _episode(task_id: str, score: float, raw_output: str) -> dict:
    return {
        "task_id": task_id,
        "fixture_id": task_id,
        "final_score": score,
        "trajectory": [
            {
                "step": 0,
                "prompt": "Task: investigate the case\nReply with a JSON action object.",
                "raw_output": raw_output,
                "action": {"action_type": "inspect_record"},
            }
        ],
    }


def test_build_dpo_pair_preserves_prompt_alignment() -> None:
    win = _episode("billing_seat_adjustment", 0.9, '{"action_type":"inspect_record","record_id":"acct_a"}')
    fail = _episode("billing_seat_adjustment", 0.1, '{"action_type":"open_ticket"}')
    pair = build_dpo_pair(win, fail, 0)
    assert pair is not None
    assert pair["task_id"] == "billing_seat_adjustment"
    assert pair["chosen_score"] == 0.9
    assert pair["rejected_score"] == 0.1


def test_generate_pairs_returns_task_matched_pairs() -> None:
    wins = [_episode("billing_seat_adjustment", 0.9, '{"action_type":"inspect_record","record_id":"acct_a"}')]
    fails = [_episode("billing_seat_adjustment", 0.1, '{"action_type":"open_ticket"}')]
    pairs = generate_pairs(wins, fails, max_pairs_per_task=5)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] != pairs[0]["rejected"]


def test_project_preference_row_contains_score_gap_and_metadata() -> None:
    pair = {
        "task_id": "api_partner_access_audit",
        "fixture_id": "api_partner_access_audit",
        "safety_slice": "security",
        "step": 0,
        "prompt": "Inspect the contract before acting",
        "chosen": '{"action_type":"inspect_record","record_id":"contract"}',
        "rejected": '{"action_type":"finalize_resolution"}',
        "chosen_action": {"action_type": "inspect_record"},
        "rejected_action": {"action_type": "finalize_resolution"},
        "chosen_score": 0.88,
        "rejected_score": 0.12,
    }
    row = _to_preference_row(pair)
    assert row["source"] == "aegisdesk-harvested-dpo"
    assert row["score_gap"] == 0.76
    assert row["safety_slice"] == "security"
    json.dumps(row)
