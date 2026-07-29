
import argparse
import json
import sys
from math import floor

def normalize_truth_segments(truth_payload: dict) -> tuple:
    timeline_segments = truth_payload["timeline_segments"]
    if not timeline_segments:
        raise ValueError("truth has no timeline_segments")

    first_start = timeline_segments[0]["final_video_start_seconds"]
    if first_start != 0.0:
        raise ValueError(
            f"truth timeline must start at 0.0 (got {first_start})"
        )

    binary_segments = []
    for segment_index, raw_segment in enumerate(timeline_segments):
        raw_type = raw_segment["type"]
        non_content_types = {"ad", "intro", "outro", "self_promotion", "filler"}
        if raw_type in non_content_types:
            mapped_label = "non_content"
        elif raw_type == "video_content":
            mapped_label = "content"
        else:
            raise ValueError(
                f"unknown truth segment type at index {segment_index}: {raw_type}"
            )

        segment_start = float(raw_segment["final_video_start_seconds"])
        segment_end = float(raw_segment["final_video_end_seconds"])
        binary_segments.append({
            "start": segment_start,
            "end": segment_end,
            "label": mapped_label,
        })

    for segment_index in range(len(binary_segments) - 1):
        current_end = binary_segments[segment_index]["end"]
        next_start = binary_segments[segment_index + 1]["start"]
        if current_end != next_start:
            raise ValueError(
                f"truth timeline gap/overlap between segment {segment_index} "
                f"end={current_end} and segment {segment_index + 1} start={next_start}"
            )

    truth_duration_seconds = binary_segments[-1]["end"]
    return binary_segments, truth_duration_seconds

def label_at_time(segments: list, time_seconds: float, fallback_label: str) -> str:
    for segment in segments:
        within_segment = segment["start"] <= time_seconds < segment["end"]
        if within_segment:
            return segment["label"]
    return fallback_label

def collect_per_second_labels(
    truth_segments: list,
    prediction_segments: list,
    truth_duration_seconds: float,
) -> tuple:
    truth_labels_per_second = []
    prediction_labels_per_second = []

    last_integer_second = floor(truth_duration_seconds)
    for time_second in range(last_integer_second):
        truth_label = label_at_time(truth_segments, time_second, fallback_label="content")

        prediction_label = label_at_time(
            prediction_segments, time_second, fallback_label="content"
        )

        truth_labels_per_second.append(truth_label)
        prediction_labels_per_second.append(prediction_label)

    return truth_labels_per_second, prediction_labels_per_second

def compute_confusion_counts(
    truth_labels: list, prediction_labels: list, positive_label: str
) -> dict:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for truth_label, prediction_label in zip(truth_labels, prediction_labels):
        truth_is_positive = truth_label == positive_label
        prediction_is_positive = prediction_label == positive_label

        if truth_is_positive and prediction_is_positive:
            true_positive += 1
        elif (not truth_is_positive) and prediction_is_positive:
            false_positive += 1
        elif (not truth_is_positive) and (not prediction_is_positive):
            true_negative += 1
        else:
            false_negative += 1

    return {
        "TP": true_positive,
        "FP": false_positive,
        "TN": true_negative,
        "FN": false_negative,
    }

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> tuple:
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1

def compute_iou_for_label(
    truth_labels: list, prediction_labels: list, target_label: str
) -> float:
    intersection_seconds = 0
    union_seconds = 0
    for truth_label, prediction_label in zip(truth_labels, prediction_labels):
        truth_matches = truth_label == target_label
        prediction_matches = prediction_label == target_label

        if truth_matches and prediction_matches:
            intersection_seconds += 1
        if truth_matches or prediction_matches:
            union_seconds += 1
    return safe_divide(intersection_seconds, union_seconds)

def extract_internal_boundaries(segments: list) -> list:
    if len(segments) <= 1:
        return []
    boundary_times = []
    for segment_index in range(len(segments) - 1):
        boundary_times.append(float(segments[segment_index]["end"]))
    return boundary_times

def compute_boundary_f1(
    truth_boundaries: list,
    prediction_boundaries: list,
    tolerance_seconds: float = 2.0,
) -> tuple:
    sorted_truth_boundaries = sorted(truth_boundaries)
    available_predictions = list(prediction_boundaries)
    matched_prediction_indices = set()

    matched_pair_count = 0

    for truth_boundary in sorted_truth_boundaries:
        candidate_predictions = []
        for prediction_index, prediction_boundary in enumerate(available_predictions):
            if prediction_index in matched_prediction_indices:
                continue
            time_distance = abs(prediction_boundary - truth_boundary)
            if time_distance <= tolerance_seconds:
                candidate_predictions.append(
                    (time_distance, prediction_boundary, prediction_index)
                )

        if not candidate_predictions:
            continue

        candidate_predictions.sort(key=lambda triple: (triple[0], triple[1]))
        _best_distance, _best_value, best_index = candidate_predictions[0]
        matched_prediction_indices.add(best_index)
        matched_pair_count += 1

    true_positive_boundaries = matched_pair_count
    false_negative_boundaries = len(sorted_truth_boundaries) - matched_pair_count
    false_positive_boundaries = len(available_predictions) - matched_pair_count

    precision, recall, f1 = precision_recall_f1(
        true_positive_boundaries,
        false_positive_boundaries,
        false_negative_boundaries,
    )
    return precision, recall, f1

def render_text_timeline(
    truth_labels: list, prediction_labels: list, total_columns: int = 80
) -> tuple:
    total_seconds = len(truth_labels)
    if total_seconds == 0:
        return "", "", 0.0

    seconds_per_column = total_seconds / total_columns

    truth_row_chars = []
    prediction_row_chars = []

    for column_index in range(total_columns):
        column_start_second = int(column_index * seconds_per_column)
        column_end_second = int((column_index + 1) * seconds_per_column)
        if column_end_second <= column_start_second:
            column_end_second = column_start_second + 1
        column_end_second = min(column_end_second, total_seconds)

        truth_slice = truth_labels[column_start_second:column_end_second]
        prediction_slice = prediction_labels[column_start_second:column_end_second]

        truth_has_non_content = any(label == "non_content" for label in truth_slice)
        prediction_has_non_content = any(
            label == "non_content" for label in prediction_slice
        )

        truth_row_chars.append("#" if truth_has_non_content else ".")
        prediction_row_chars.append("#" if prediction_has_non_content else ".")

    return "".join(truth_row_chars), "".join(prediction_row_chars), seconds_per_column

def count_non_content_segments(segments: list) -> int:
    return sum(1 for segment in segments if segment["label"] == "non_content")

def run_comparison(prediction_json_path: str, truth_json_path: str) -> None:
    with open(prediction_json_path, "r") as prediction_file:
        prediction_payload = json.load(prediction_file)
    with open(truth_json_path, "r") as truth_file:
        truth_payload = json.load(truth_file)

    truth_segments, truth_duration_seconds = normalize_truth_segments(truth_payload)
    prediction_segments = prediction_payload["segments"]

    truth_labels_per_second, prediction_labels_per_second = collect_per_second_labels(
        truth_segments, prediction_segments, truth_duration_seconds
    )

    confusion = compute_confusion_counts(
        truth_labels_per_second, prediction_labels_per_second, positive_label="non_content"
    )
    precision, recall, f1 = precision_recall_f1(
        confusion["TP"], confusion["FP"], confusion["FN"]
    )
    total_seconds = len(truth_labels_per_second)
    correct_seconds = confusion["TP"] + confusion["TN"]
    overall_accuracy = safe_divide(correct_seconds, total_seconds)

    iou_non_content = compute_iou_for_label(
        truth_labels_per_second, prediction_labels_per_second, target_label="non_content"
    )

    truth_boundaries = extract_internal_boundaries(truth_segments)
    prediction_boundaries = extract_internal_boundaries(prediction_segments)
    boundary_precision, boundary_recall, boundary_f1 = compute_boundary_f1(
        truth_boundaries, prediction_boundaries
    )

    truth_row, prediction_row, seconds_per_column = render_text_timeline(
        truth_labels_per_second, prediction_labels_per_second
    )

    truth_segment_count = len(truth_segments)
    truth_non_content_segment_count = count_non_content_segments(truth_segments)
    prediction_segment_count = len(prediction_segments)
    prediction_non_content_segment_count = count_non_content_segments(prediction_segments)

    print(f"=== Comparison: {prediction_json_path} vs {truth_json_path} ===")
    print(f"Duration: {truth_duration_seconds:.1f}s")
    print()
    print("Confusion matrix (positive class = non_content):")
    print(
        f"  TP={confusion['TP']}  FP={confusion['FP']}  "
        f"TN={confusion['TN']}  FN={confusion['FN']}"
    )
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"  Accuracy={overall_accuracy:.3f}")
    print()
    print(f"IoU (non_content): {iou_non_content:.3f}")
    print()
    print("Segment counts:")
    print(
        f"  Predicted: {prediction_segment_count} total, "
        f"{prediction_non_content_segment_count} non-content"
    )
    print(
        f"  Truth:     {truth_segment_count} total, "
        f"{truth_non_content_segment_count} non-content"
    )
    print()
    print("Boundary F1 (±2s tolerance):")
    print(
        f"  Precision={boundary_precision:.3f}  "
        f"Recall={boundary_recall:.3f}  F1={boundary_f1:.3f}"
    )
    print()
    print(f"Timeline (each char = {seconds_per_column:.1f}s):")
    print(f"  TRUTH: {truth_row}")
    print(f"   PRED: {prediction_row}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a prediction JSON against a truth JSON."
    )
    parser.add_argument("prediction_json", help="Path to prediction JSON")
    parser.add_argument("truth_json", help="Path to truth JSON")
    args = parser.parse_args()

    try:
        run_comparison(args.prediction_json, args.truth_json)
    except ValueError as validation_error:
        print(f"Error: {validation_error}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
