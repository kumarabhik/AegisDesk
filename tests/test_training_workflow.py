"""Coverage for the numbered strongest-submission workflow helpers."""

from __future__ import annotations

import json

from training.check_training_readiness import main as readiness_main
from training.strongest_submission import build_stages, main as strongest_main


def test_training_readiness_passes_with_local_artifacts(tmp_path) -> None:
    output_path = tmp_path / "readiness.json"
    assert readiness_main(["--output", str(output_path)]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["paths"]["counts"]["support_sft_rows"] == 15124
    assert payload["paths"]["counts"]["support_pref_rows"] == 7119
    assert payload["manifest"]["held_out_count"] == 18
    assert payload["manifest"]["judged_count"] == 27


def test_strongest_submission_builds_ten_steps() -> None:
    stages = build_stages(
        env_url="https://i4mgr00t-meta.hf.space",
        model_name="Qwen/Qwen3-8B",
        results_path="training/benchmark_results.json",
    )
    assert len(stages) == 10
    assert [stage.number for stage in stages] == list(range(1, 11))
    assert stages[3].slug == "readiness-doctor"
    assert stages[9].slug == "benchmark-and-evidence"


def test_strongest_submission_step_ten_uses_plot_results_flag(capsys) -> None:
    assert strongest_main(["--step", "10", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["number"] == 10
    assert payload["commands"][1][1:] == [
        "training/plot_benchmark_results.py",
        "--results",
        "training/benchmark_results.json",
    ]
