"""Render per-track benchmark plots from training/benchmark_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="training/benchmark_results.json")
    parser.add_argument("--output-dir", default="training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    baseline_tracks = payload["summary"]["baseline"]["by_track"]
    champion_tracks = payload["summary"]["champion"]["by_track"]

    track_names = sorted(set(baseline_tracks) | set(champion_tracks))
    baseline_values = [baseline_tracks.get(track, {}).get("mean_score", 0.0) for track in track_names]
    champion_values = [champion_tracks.get(track, {}).get("mean_score", 0.0) for track in track_names]
    deltas = [champion - baseline for baseline, champion in zip(baseline_values, champion_values)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(track_names, baseline_values, label="baseline", alpha=0.75)
    axes[0].bar(track_names, champion_values, label="champion", alpha=0.75)
    axes[0].set_title("Mean Score by Track")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    colors = ["#0b6e4f" if delta >= 0 else "#a61b1b" for delta in deltas]
    axes[1].bar(track_names, deltas, color=colors)
    axes[1].axhline(0.0, color="#1f2933", linewidth=1)
    axes[1].set_title("Per-Track Delta")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    plot_path = output_dir / "per_track_delta.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "canonical_vs_held_out_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Canonical vs Held-out Summary",
                "",
                f"- Baseline canonical mean: `{payload['summary']['baseline']['canonical']['mean_score']:.4f}`",
                f"- Champion canonical mean: `{payload['summary']['champion']['canonical']['mean_score']:.4f}`",
                f"- Baseline held-out mean: `{payload['summary']['baseline']['held_out']['mean_score']:.4f}`",
                f"- Champion held-out mean: `{payload['summary']['champion']['held_out']['mean_score']:.4f}`",
                f"- Baseline security-slice mean: `{payload['summary']['baseline']['security_slice']['mean_score']:.4f}`",
                f"- Champion security-slice mean: `{payload['summary']['champion']['security_slice']['mean_score']:.4f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
