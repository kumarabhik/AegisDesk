"""CLI coverage for the oracle demo runner."""

from __future__ import annotations

import json

from oracle_demo import main


def test_oracle_demo_accepts_extended_pack(capsys) -> None:
    assert main(["--pack", "extended", "--seed", "11"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pack"] == "extended"
    assert [report["task_id"] for report in summary["reports"]] == [
        "admin_role_transfer_verification",
        "api_rate_limit_escalation",
        "tax_exemption_credit_review",
    ]
    assert all(report["track"] == "extended" for report in summary["reports"])
