
import json
import os
import re
import subprocess
import sys

VIDEO_IDS = ["test_001", "test_002", "test_003", "test_004", "test_005"]

PYTHON_INTERPRETER = sys.executable

def run_pipeline_for_video(video_id: str, env_overrides: dict) -> str:
    input_mp4 = f"videos/{video_id}.mp4"
    output_json = f"video_info/pred/{video_id}_ssm_pred.json"

    process_environment = os.environ.copy()
    process_environment.update(env_overrides)

    completed = subprocess.run(
        [PYTHON_INTERPRETER, "pipeline_ssm.py", input_mp4, "--output", output_json],
        capture_output=True,
        text=True,
        env=process_environment,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"pipeline_ssm failed on {video_id}")
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
        [PYTHON_INTERPRETER, "compare.py", prediction_path, truth_path],
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

def evaluate_knob_combination(
    novelty_threshold: float,
    kernel_half_seconds: int,
    multi_k_csv: str = "",
    label_n_clusters: int = 3,
    content_cluster_count: int = 1,
) -> dict:
    env_overrides = {
        "NOVELTY_THRESHOLD": str(novelty_threshold),
        "KERNEL_HALF_SECONDS": str(kernel_half_seconds),
        "KERNEL_HALF_SECONDS_MULTI": multi_k_csv,
        "LABEL_N_CLUSTERS": str(label_n_clusters),
        "CONTENT_CLUSTER_COUNT": str(content_cluster_count),
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

    def min_of(metric_key: str) -> float:
        values = [
            video_metrics.get(metric_key, 0.0)
            for video_metrics in per_video_metrics.values()
        ]
        return min(values)

    return {
        "novelty_threshold": novelty_threshold,
        "kernel_half_seconds": kernel_half_seconds,
        "multi_k_csv": multi_k_csv,
        "label_n_clusters": label_n_clusters,
        "content_cluster_count": content_cluster_count,
        "mean_recall": mean_of("per_second_recall"),
        "min_recall": min_of("per_second_recall"),
        "mean_iou": mean_of("iou"),
        "min_iou": min_of("iou"),
        "mean_per_second_f1": mean_of("per_second_f1"),
        "mean_pred_segments": mean_of("pred_segment_count"),
        "per_video": per_video_metrics,
    }

def print_per_video_breakdown(aggregate: dict) -> None:
    print(
        f"   {'video':<10} "
        f"{'recall':>7} {'IoU':>6} {'F1':>6} {'segs':>5}"
    )
    for video_id in VIDEO_IDS:
        video_metrics = aggregate["per_video"][video_id]
        print(
            f"   {video_id:<10} "
            f"{video_metrics.get('per_second_recall', 0.0):>7.3f} "
            f"{video_metrics.get('iou', 0.0):>6.3f} "
            f"{video_metrics.get('per_second_f1', 0.0):>6.3f} "
            f"{video_metrics.get('pred_segment_count', 0):>5d}"
        )

def main() -> None:
    knob_grid = [
        (0.40, 0, "60,90", 2, 1),
        (0.30, 0, "60,90", 2, 1),
        (0.30, 0, "30,60,90", 2, 1),
        (0.40, 0, "60,90", 3, 1),
        (0.30, 0, "60,90", 3, 1),
        (0.40, 0, "30,60,90", 3, 1),
        (0.30, 0, "30,60,90", 3, 1),
        (0.50, 0, "30,60,90", 3, 1),
        (0.40, 30, "", 3, 1),
        (0.30, 30, "", 3, 1),
        (0.30, 0, "60,90", 3, 2),
        (0.40, 0, "30,60,90", 3, 2),
        (0.40, 0, "30,60,90", 4, 1),
        (0.30, 0, "30,60,90", 4, 1),
    ]

    print(
        f"{'thresh':>6} {'K_label':>14} {'lbl':>5} | "
        f"{'mn_rec':>7} {'mn_rec_min':>10} "
        f"{'mn_IoU':>7} {'mn_IoU_min':>10} "
        f"{'mn_F1':>7} {'avg_segs':>8}"
    )
    print("-" * 105)

    all_results = []
    for (
        novelty_threshold,
        kernel_half_seconds,
        multi_k_csv,
        label_n_clusters,
        content_cluster_count,
    ) in knob_grid:
        aggregate = evaluate_knob_combination(
            novelty_threshold,
            kernel_half_seconds,
            multi_k_csv,
            label_n_clusters,
            content_cluster_count,
        )
        all_results.append(aggregate)
        k_display_label = (
            f"[{multi_k_csv}]" if multi_k_csv else f"K={kernel_half_seconds}"
        )
        labeling_tag = f"{label_n_clusters}/{content_cluster_count}"
        print(
            f"{novelty_threshold:>6.2f} {k_display_label:>14} {labeling_tag:>5} | "
            f"{aggregate['mean_recall']:>7.3f} "
            f"{aggregate['min_recall']:>10.3f} "
            f"{aggregate['mean_iou']:>7.3f} "
            f"{aggregate['min_iou']:>10.3f} "
            f"{aggregate['mean_per_second_f1']:>7.3f} "
            f"{aggregate['mean_pred_segments']:>8.1f}"
        )
        print_per_video_breakdown(aggregate)
        print()

    best_by_mean_recall = max(all_results, key=lambda r: r["mean_recall"])
    best_by_recall_floor = max(all_results, key=lambda r: r["min_recall"])
    best_by_recall_and_iou = max(
        all_results, key=lambda r: r["mean_recall"] + r["mean_iou"]
    )

    def format_config_for_summary(result: dict) -> str:
        k_label = (
            f"[{result['multi_k_csv']}]"
            if result["multi_k_csv"]
            else f"K={result['kernel_half_seconds']}"
        )
        labeling = f"{result['label_n_clusters']}/{result['content_cluster_count']}"
        return f"{k_label} lbl={labeling}"

    print()
    print(
        f"Best mean recall:        thresh={best_by_mean_recall['novelty_threshold']:.2f}, "
        f"{format_config_for_summary(best_by_mean_recall)} "
        f"-> mean_recall={best_by_mean_recall['mean_recall']:.3f}, "
        f"min_recall={best_by_mean_recall['min_recall']:.3f}"
    )
    print(
        f"Best worst-case recall:  thresh={best_by_recall_floor['novelty_threshold']:.2f}, "
        f"{format_config_for_summary(best_by_recall_floor)} "
        f"-> min_recall={best_by_recall_floor['min_recall']:.3f}, "
        f"mean_recall={best_by_recall_floor['mean_recall']:.3f}"
    )
    print(
        f"Best (recall + IoU):     thresh={best_by_recall_and_iou['novelty_threshold']:.2f}, "
        f"{format_config_for_summary(best_by_recall_and_iou)} "
        f"-> recall={best_by_recall_and_iou['mean_recall']:.3f}, "
        f"IoU={best_by_recall_and_iou['mean_iou']:.3f}"
    )

if __name__ == "__main__":
    main()
