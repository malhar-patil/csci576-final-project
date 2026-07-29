
import argparse
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

from features import extract_features
from pipeline import (
    ensure_wav_exists,
    enforce_minimum_segment_length,
    merge_adjacent_same_label,
)

KERNEL_HALF_SECONDS = int(float(os.environ.get("KERNEL_HALF_SECONDS", "30")))
KERNEL_HALF_SECONDS_MULTI = os.environ.get("KERNEL_HALF_SECONDS_MULTI", "60,90")
NOVELTY_THRESHOLD = float(os.environ.get("NOVELTY_THRESHOLD", "0.3"))
PEAK_NMS_RADIUS_SECONDS = float(os.environ.get("PEAK_NMS_RADIUS_SECONDS", "5"))
SENTINEL_DEDUP_RADIUS = float(os.environ.get("SENTINEL_DEDUP_RADIUS", "2"))
MINIMUM_SEGMENT_SECONDS = float(os.environ.get("MINIMUM_SEGMENT_SECONDS", "3"))

LABEL_N_CLUSTERS = int(float(os.environ.get("LABEL_N_CLUSTERS", "3")))
CONTENT_CLUSTER_COUNT = int(float(os.environ.get("CONTENT_CLUSTER_COUNT", "1")))

def build_feature_matrix(features: dict) -> np.ndarray:
    mfcc_per_window = features["mfcc"]
    centroid_column = features["spectral_centroid"].reshape(-1, 1)
    flatness_column = features["spectral_flatness"].reshape(-1, 1)
    zcr_column = features["zcr"].reshape(-1, 1)

    feature_matrix = np.concatenate(
        [mfcc_per_window, centroid_column, flatness_column, zcr_column],
        axis=1,
    ).astype(np.float32)

    column_means = feature_matrix.mean(axis=0)
    column_stds = feature_matrix.std(axis=0)
    feature_matrix_normalized = (feature_matrix - column_means) / (column_stds + 1e-8)
    return feature_matrix_normalized

def build_self_similarity_matrix(feature_matrix_normalized: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(feature_matrix_normalized, axis=1, keepdims=True)
    feature_matrix_unit = feature_matrix_normalized / (row_norms + 1e-8)
    similarity_matrix = feature_matrix_unit @ feature_matrix_unit.T
    return similarity_matrix.astype(np.float32)

def build_checkerboard_kernel(half_size_K: int, gaussian_sigma: float) -> np.ndarray:
    full_size = 2 * half_size_K

    sign_matrix = np.ones((full_size, full_size), dtype=np.float32)
    sign_matrix[:half_size_K, half_size_K:] = -1.0
    sign_matrix[half_size_K:, :half_size_K] = -1.0

    coordinate_axis = np.arange(full_size, dtype=np.float32) - (full_size - 1) / 2.0
    grid_y, grid_x = np.meshgrid(coordinate_axis, coordinate_axis, indexing="ij")
    gaussian_taper = np.exp(
        -(grid_x ** 2 + grid_y ** 2) / (2.0 * gaussian_sigma ** 2)
    ).astype(np.float32)

    kernel = sign_matrix * gaussian_taper
    return kernel

def compute_novelty_curve(
    similarity_matrix: np.ndarray,
    half_size_K: int,
) -> np.ndarray:
    num_windows = similarity_matrix.shape[0]
    novelty_curve = np.zeros(num_windows, dtype=np.float32)

    gaussian_sigma = max(1.0, half_size_K / 2.0)
    kernel = build_checkerboard_kernel(half_size_K, gaussian_sigma)

    if num_windows < 2 * half_size_K:
        return novelty_curve

    for window_index in range(half_size_K, num_windows - half_size_K):
        slice_start = window_index - half_size_K
        slice_end = window_index + half_size_K
        ssm_submatrix = similarity_matrix[slice_start:slice_end, slice_start:slice_end]
        novelty_value = float(np.sum(kernel * ssm_submatrix))
        novelty_curve[window_index] = novelty_value

    return novelty_curve

def smooth_and_normalize_novelty(novelty_curve: np.ndarray) -> np.ndarray:
    smoothing_kernel = np.ones(5, dtype=np.float32) / 5.0
    novelty_smoothed = np.convolve(novelty_curve, smoothing_kernel, mode="same")

    robust_max = float(np.percentile(novelty_smoothed, 95))
    if robust_max <= 0.0:
        return np.zeros_like(novelty_smoothed)

    novelty_normalized = np.clip(novelty_smoothed / robust_max, 0.0, 1.0)
    return novelty_normalized

def find_boundary_candidates(
    novelty_normalized: np.ndarray,
    duration_seconds: float,
) -> list:
    num_windows = len(novelty_normalized)

    raw_peaks = []
    for window_index in range(1, num_windows - 1):
        is_local_max = (
            novelty_normalized[window_index] > novelty_normalized[window_index - 1]
            and novelty_normalized[window_index] > novelty_normalized[window_index + 1]
        )
        is_above_threshold = novelty_normalized[window_index] > NOVELTY_THRESHOLD
        if is_local_max and is_above_threshold:
            raw_peaks.append((window_index, float(novelty_normalized[window_index])))

    peaks_by_strength = sorted(raw_peaks, key=lambda peak: peak[1], reverse=True)
    nms_kept_peak_times = []
    for peak_window_index, _strength in peaks_by_strength:
        candidate_time_seconds = float(peak_window_index)
        is_suppressed = False
        for already_kept_time in nms_kept_peak_times:
            if abs(candidate_time_seconds - already_kept_time) <= PEAK_NMS_RADIUS_SECONDS:
                is_suppressed = True
                break
        if not is_suppressed:
            nms_kept_peak_times.append(candidate_time_seconds)

    sentinel_filtered_peak_times = []
    for peak_time_seconds in nms_kept_peak_times:
        too_close_to_start = abs(peak_time_seconds - 0.0) <= SENTINEL_DEDUP_RADIUS
        too_close_to_end = (
            abs(peak_time_seconds - duration_seconds) <= SENTINEL_DEDUP_RADIUS
        )
        if too_close_to_start or too_close_to_end:
            continue
        sentinel_filtered_peak_times.append(peak_time_seconds)

    all_boundary_times = (
        [0.0] + sorted(sentinel_filtered_peak_times) + [duration_seconds]
    )
    return all_boundary_times

def compute_region_centroids(
    boundary_times: list,
    feature_matrix_normalized: np.ndarray,
) -> tuple:
    num_windows = feature_matrix_normalized.shape[0]
    region_centroids = []
    region_ranges = []

    for boundary_index in range(len(boundary_times) - 1):
        region_start_sec = boundary_times[boundary_index]
        region_end_sec = boundary_times[boundary_index + 1]

        start_window_index = int(round(region_start_sec))
        end_window_index = int(round(region_end_sec))

        start_window_index = max(0, min(start_window_index, num_windows))
        end_window_index = max(0, min(end_window_index, num_windows))
        if start_window_index >= end_window_index:
            collapsed_index = min(start_window_index, num_windows - 1)
            collapsed_index = max(0, collapsed_index)
            region_centroid_vector = feature_matrix_normalized[collapsed_index]
        else:
            region_centroid_vector = feature_matrix_normalized[
                start_window_index:end_window_index
            ].mean(axis=0)

        region_centroids.append(region_centroid_vector)
        region_ranges.append((region_start_sec, region_end_sec))

    return np.array(region_centroids, dtype=np.float32), region_ranges

def label_regions_by_clustering(
    region_centroids: np.ndarray,
    region_ranges: list,
) -> list:
    num_regions = len(region_ranges)

    effective_n_clusters = min(LABEL_N_CLUSTERS, num_regions)
    if num_regions <= 1 or effective_n_clusters < 2:
        return [
            {
                "start": region_ranges[region_index][0],
                "end": region_ranges[region_index][1],
                "label": "content",
                "mean_score": 0.0,
            }
            for region_index in range(num_regions)
        ]

    centroid_pairwise_separation = float(
        np.linalg.norm(region_centroids - region_centroids.mean(axis=0), axis=1).max()
    )
    if centroid_pairwise_separation < 1e-6:
        return [
            {
                "start": region_ranges[region_index][0],
                "end": region_ranges[region_index][1],
                "label": "content",
                "mean_score": 0.0,
            }
            for region_index in range(num_regions)
        ]

    kmeans_model = KMeans(
        n_clusters=effective_n_clusters, n_init=10, random_state=0
    )
    cluster_assignments = kmeans_model.fit_predict(region_centroids)

    cluster_total_duration = {
        cluster_id: 0.0 for cluster_id in range(effective_n_clusters)
    }
    for region_index in range(num_regions):
        region_start_sec, region_end_sec = region_ranges[region_index]
        region_duration = region_end_sec - region_start_sec
        cluster_total_duration[
            int(cluster_assignments[region_index])
        ] += region_duration

    effective_content_count = min(CONTENT_CLUSTER_COUNT, effective_n_clusters - 1)
    effective_content_count = max(1, effective_content_count)
    clusters_sorted_by_duration = sorted(
        cluster_total_duration.items(), key=lambda item: item[1], reverse=True
    )
    content_cluster_ids = {
        cluster_id
        for cluster_id, _duration in clusters_sorted_by_duration[
            :effective_content_count
        ]
    }

    segments = []
    for region_index in range(num_regions):
        region_start_sec, region_end_sec = region_ranges[region_index]
        is_content = int(cluster_assignments[region_index]) in content_cluster_ids
        segments.append(
            {
                "start": region_start_sec,
                "end": region_end_sec,
                "label": "content" if is_content else "non_content",
                "mean_score": 0.0 if is_content else 1.0,
            }
        )

    return segments

def parse_multi_k_setting() -> list:
    if not KERNEL_HALF_SECONDS_MULTI.strip():
        return []
    raw_values = KERNEL_HALF_SECONDS_MULTI.split(",")
    parsed_k_values = []
    for raw_value in raw_values:
        cleaned_value = raw_value.strip()
        if cleaned_value:
            parsed_k_values.append(int(float(cleaned_value)))
    return parsed_k_values

def compute_aggregated_novelty(
    similarity_matrix: np.ndarray,
    k_values: list,
) -> np.ndarray:
    num_windows = similarity_matrix.shape[0]
    aggregated_curve = np.zeros(num_windows, dtype=np.float32)
    for half_size_K in k_values:
        single_scale_novelty = compute_novelty_curve(similarity_matrix, half_size_K)
        single_scale_normalized = smooth_and_normalize_novelty(single_scale_novelty)
        aggregated_curve = np.maximum(aggregated_curve, single_scale_normalized)
    return aggregated_curve

def build_segments(
    feature_matrix_normalized: np.ndarray,
    similarity_matrix: np.ndarray,
    duration_seconds: float,
) -> list:
    multi_k_values = parse_multi_k_setting()
    if multi_k_values:
        novelty_normalized = compute_aggregated_novelty(
            similarity_matrix, multi_k_values
        )
    else:
        novelty_curve = compute_novelty_curve(similarity_matrix, KERNEL_HALF_SECONDS)
        novelty_normalized = smooth_and_normalize_novelty(novelty_curve)

    boundary_times = find_boundary_candidates(novelty_normalized, duration_seconds)

    region_centroids, region_ranges = compute_region_centroids(
        boundary_times, feature_matrix_normalized
    )
    raw_segments = label_regions_by_clustering(region_centroids, region_ranges)

    merged_segments = merge_adjacent_same_label(raw_segments)
    length_constrained_segments = enforce_minimum_segment_length(
        merged_segments,
        duration_seconds,
        minimum_length_seconds=MINIMUM_SEGMENT_SECONDS,
    )
    final_merged_segments = merge_adjacent_same_label(length_constrained_segments)

    final_segments = []
    for segment in final_merged_segments:
        final_segments.append(
            {
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "label": segment["label"],
            }
        )
    return final_segments

def run_pipeline(input_mp4_path: str, output_json_path: str) -> None:
    wav_path = ensure_wav_exists(input_mp4_path)

    features = extract_features(wav_path)
    num_windows = features["num_windows"]
    duration_seconds_floored = float(num_windows)

    print(
        f"Extracted features: {num_windows} windows "
        f"(audio_duration={features['duration_sec']:.3f}s, "
        f"floored_duration={duration_seconds_floored:.3f}s)",
        file=sys.stderr,
    )

    feature_matrix_normalized = build_feature_matrix(features)
    similarity_matrix = build_self_similarity_matrix(feature_matrix_normalized)

    multi_k_values = parse_multi_k_setting()
    k_description = (
        f"K_multi={multi_k_values}" if multi_k_values else f"K={KERNEL_HALF_SECONDS}"
    )
    print(
        f"SSM: shape={similarity_matrix.shape}, "
        f"diag_mean={float(np.diag(similarity_matrix).mean()):.3f}, "
        f"{k_description}, threshold={NOVELTY_THRESHOLD}",
        file=sys.stderr,
    )

    segments = build_segments(
        feature_matrix_normalized, similarity_matrix, duration_seconds_floored
    )

    output_payload = {
        "video_filename": os.path.basename(input_mp4_path),
        "duration_seconds": round(duration_seconds_floored, 3),
        "segments": segments,
    }

    with open(output_json_path, "w") as output_file:
        json.dump(output_payload, output_file, indent=2)

    print(
        f"Wrote {len(segments)} segments to {output_json_path}",
        file=sys.stderr,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2 audio segmentation via SSM novelty (spec §16)."
    )
    parser.add_argument("input_mp4", help="Path to input MP4 file")
    parser.add_argument(
        "--output", required=True, help="Path to write the segmentation JSON"
    )
    args = parser.parse_args()

    run_pipeline(args.input_mp4, args.output)

if __name__ == "__main__":
    main()
