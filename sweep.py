
import json
import os
import re
import subprocess
import sys

VIDEO_IDS = ["test_001", "test_002", "test_003", "test_004", "test_005"]

def run_pipeline_for_video(video_id: str, env_overrides: dict) -> str:
    input_mp4 = f"videos/{video_id}.mp4"
    output_json = f"video_info/pred/{video_id}_pred.json"

    process_environment = os.environ.copy()
    process_environment.update(env_overrides)

    completed = subprocess.run(
        ["python3", "pipeline.py", input_mp4, "--output", output_json],
        capture_output=True,
        text=True,
        env=process_environment,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"pipeline failed on {video_id}")
    return output_json

def parse_compare_output(compare_stdout: str) -> dict:
    metrics = {}

    f1_match = re.search(
        r"Precision=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)\s*\n\s+Accuracy",
        compare_stdout,
    )
    if f1_match:
        metrics["per_second_precision"] = float(f1_match.group(1))
        metrics["per_second_recall"] = float(f1_match.group(2))
        metrics["per_second_f1"] = float(f1_match.group(3))

    accuracy_match = re.search(r"Accuracy=([\d.]+)", compare_stdout)
    if accuracy_match:
        metrics["accuracy"] = float(accuracy_match.group(1))

    iou_match = re.search(r"IoU \(non_content\): ([\d.]+)", compare_stdout)
    if iou_match:
        metrics["iou"] = float(iou_match.group(1))

    boundary_match = re.search(
        r"Boundary F1.*\n\s+Precision=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)",
        compare_stdout,
    )
    if boundary_match:
        metrics["boundary_precision"] = float(boundary_match.group(1))
        metrics["boundary_recall"] = float(boundary_match.group(2))
        metrics["boundary_f1"] = float(boundary_match.group(3))

    return metrics

def run_compare_for_video(video_id: str, prediction_path: str) -> dict:
    truth_path = f"video_info/ground/{video_id}.json"
    completed = subprocess.run(
        ["python3", "compare.py", prediction_path, truth_path],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"compare failed on {video_id}")

    metrics = parse_compare_output(completed.stdout)
    metrics["raw_output"] = completed.stdout

    with open(prediction_path) as prediction_file:
        prediction_payload = json.load(prediction_file)
    metrics["pred_segment_count"] = len(prediction_payload["segments"])
    return metrics

def evaluate_knob_combination(threshold: float, dedup_radius: float) -> dict:
    env_overrides = {
        "DISCONTINUITY_THRESHOLD": str(threshold),
        "CLUSTER_DEDUP_RADIUS_SECONDS": str(dedup_radius),
    }

    per_video_metrics = {}
    for video_id in VIDEO_IDS:
        prediction_path = run_pipeline_for_video(video_id, env_overrides)
        per_video_metrics[video_id] = run_compare_for_video(video_id, prediction_path)

    def mean_of(metric_key: str) -> float:
        values = [
            video_metrics.get(metric_key, 0.0)
            for video_metrics in per_video_metrics.values()
        ]
        return sum(values) / len(values)

    return {
        "threshold": threshold,
        "dedup_radius": dedup_radius,
        "mean_per_second_f1": mean_of("per_second_f1"),
        "mean_iou": mean_of("iou"),
        "mean_boundary_f1": mean_of("boundary_f1"),
        "mean_boundary_precision": mean_of("boundary_precision"),
        "mean_boundary_recall": mean_of("boundary_recall"),
        "mean_pred_segments": mean_of("pred_segment_count"),
        "per_video": per_video_metrics,
    }

def main() -> None:
    knob_grid = [
        (0.60, 2.0),
        (0.70, 2.0),
        (0.75, 2.0),
        (0.80, 2.0),
        (0.60, 5.0),
        (0.70, 5.0),
        (0.75, 5.0),
        (0.80, 5.0),
        (0.70, 8.0),
        (0.75, 8.0),
    ]

    print(
        f"{'threshold':>10} {'dedup_s':>8} | "
        f"{'sec_F1':>7} {'IoU':>6} {'bnd_F1':>7} {'bnd_P':>6} {'bnd_R':>6} | "
        f"{'avg_segs':>8}"
    )
    print("-" * 80)

    all_results = []
    for threshold, dedup_radius in knob_grid:
        aggregate = evaluate_knob_combination(threshold, dedup_radius)
        all_results.append(aggregate)
        print(
            f"{threshold:>10.2f} {dedup_radius:>8.1f} | "
            f"{aggregate['mean_per_second_f1']:>7.3f} "
            f"{aggregate['mean_iou']:>6.3f} "
            f"{aggregate['mean_boundary_f1']:>7.3f} "
            f"{aggregate['mean_boundary_precision']:>6.3f} "
            f"{aggregate['mean_boundary_recall']:>6.3f} | "
            f"{aggregate['mean_pred_segments']:>8.1f}"
        )

    print()
    best_by_boundary = max(all_results, key=lambda r: r["mean_boundary_f1"])
    best_by_combined = max(
        all_results,
        key=lambda r: r["mean_boundary_f1"] + r["mean_iou"],
    )
    print(
        f"Best mean boundary F1: threshold={best_by_boundary['threshold']:.2f}, "
        f"dedup={best_by_boundary['dedup_radius']:.1f}s "
        f"-> {best_by_boundary['mean_boundary_f1']:.3f}"
    )
    print(
        f"Best (boundary_F1 + IoU): threshold={best_by_combined['threshold']:.2f}, "
        f"dedup={best_by_combined['dedup_radius']:.1f}s "
        f"-> bnd_F1={best_by_combined['mean_boundary_f1']:.3f}, "
        f"IoU={best_by_combined['mean_iou']:.3f}"
    )

if __name__ == "__main__":
    main()
