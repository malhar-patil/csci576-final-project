
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import librosa
import soundfile
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans

HIGHPASS_HZ = float(os.environ.get("HIGHPASS_HZ", "4000"))
SMOOTH_WINDOW_SECONDS = int(float(os.environ.get("SMOOTH_WINDOW_SECONDS", "15")))
MINIMUM_SEGMENT_SECONDS = int(float(os.environ.get("MINIMUM_SEGMENT_SECONDS", "5")))
HOP_LENGTH = 512

def derive_wav_path(input_mp4_path):
    directory = os.path.dirname(input_mp4_path)
    base_name = os.path.basename(input_mp4_path)
    stem, _ext = os.path.splitext(base_name)
    return os.path.join(directory, stem + ".wav")

def cached_wav_is_valid(wav_path):
    if not os.path.exists(wav_path):
        return False
    try:
        info = soundfile.info(wav_path)
    except Exception:
        return False
    return info.samplerate == 16000 and info.channels == 1

def extract_audio_to_wav(input_mp4_path, wav_path):
    ffmpeg_command = [
        "ffmpeg", "-i", input_mp4_path,
        "-ac", "1", "-ar", "16000", "-vn", "-y", wav_path,
    ]
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

def ensure_wav_exists(input_mp4_path):
    wav_path = derive_wav_path(input_mp4_path)
    if cached_wav_is_valid(wav_path):
        return wav_path
    if os.path.exists(wav_path):
        os.remove(wav_path)
    extract_audio_to_wav(input_mp4_path, wav_path)
    return wav_path

def apply_highpass_filter(samples, sample_rate, cutoff_hz):
    nyquist_frequency = 0.5 * sample_rate
    normalized_cutoff = cutoff_hz / nyquist_frequency
    butter_order = 4
    numerator_coeffs, denominator_coeffs = butter(
        butter_order, normalized_cutoff, btype="highpass"
    )
    filtered_samples = filtfilt(numerator_coeffs, denominator_coeffs, samples)
    return filtered_samples.astype(np.float32)

def mean_pool_per_window(per_frame_values, frames_per_window, num_windows):
    is_two_dimensional = per_frame_values.ndim == 2
    if is_two_dimensional:
        coeffs = per_frame_values.shape[0]
        pooled = np.zeros((num_windows, coeffs), dtype=np.float32)
    else:
        pooled = np.zeros(num_windows, dtype=np.float32)
    for window_index in range(num_windows):
        frame_start = window_index * frames_per_window
        frame_end = frame_start + frames_per_window
        window_slice = per_frame_values[..., frame_start:frame_end]
        if window_slice.shape[-1] == 0:
            continue
        window_mean = window_slice.mean(axis=-1)
        if is_two_dimensional:
            pooled[window_index, :] = window_mean
        else:
            pooled[window_index] = window_mean
    return pooled

def extract_filtered_features(wav_path, highpass_hz):
    samples, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
    duration_seconds = float(len(samples)) / float(sample_rate)
    num_windows = int(duration_seconds // 1.0)

    filtered_samples = apply_highpass_filter(samples, sample_rate, highpass_hz)

    rms_per_frame = librosa.feature.rms(y=filtered_samples, hop_length=HOP_LENGTH)[0]
    centroid_per_frame = librosa.feature.spectral_centroid(
        y=filtered_samples, sr=sample_rate, hop_length=HOP_LENGTH
    )[0]
    flatness_per_frame = librosa.feature.spectral_flatness(
        y=filtered_samples, hop_length=HOP_LENGTH
    )[0]
    rolloff_per_frame = librosa.feature.spectral_rolloff(
        y=filtered_samples, sr=sample_rate, hop_length=HOP_LENGTH
    )[0]
    mfcc_per_frame = librosa.feature.mfcc(
        y=filtered_samples, sr=sample_rate, n_mfcc=13, hop_length=HOP_LENGTH
    )

    frames_per_window = int(sample_rate / HOP_LENGTH)

    rms_per_window = mean_pool_per_window(rms_per_frame, frames_per_window, num_windows)
    centroid_per_window = mean_pool_per_window(centroid_per_frame, frames_per_window, num_windows)
    flatness_per_window = mean_pool_per_window(flatness_per_frame, frames_per_window, num_windows)
    rolloff_per_window = mean_pool_per_window(rolloff_per_frame, frames_per_window, num_windows)
    mfcc_per_window = mean_pool_per_window(mfcc_per_frame, frames_per_window, num_windows)

    feature_matrix = np.concatenate(
        [
            rms_per_window.reshape(-1, 1),
            centroid_per_window.reshape(-1, 1),
            flatness_per_window.reshape(-1, 1),
            rolloff_per_window.reshape(-1, 1),
            mfcc_per_window,
        ],
        axis=1,
    ).astype(np.float32)

    return feature_matrix, duration_seconds, num_windows

def smooth_feature_matrix(feature_matrix, window_seconds):
    num_windows, num_features = feature_matrix.shape
    smoothed = np.zeros_like(feature_matrix)
    half_width = window_seconds // 2
    for window_index in range(num_windows):
        slice_start = max(0, window_index - half_width)
        slice_end = min(num_windows, window_index + half_width + 1)
        smoothed[window_index] = feature_matrix[slice_start:slice_end].mean(axis=0)
    return smoothed

def standardize_features(feature_matrix):
                                                                               
    matrix_64 = feature_matrix.astype(np.float64)
    column_means = matrix_64.mean(axis=0)
    column_stds = matrix_64.std(axis=0)
    standardized = (matrix_64 - column_means) / (column_stds + 1e-8)
                                                                       
    return np.clip(standardized, -5.0, 5.0)

def cluster_two_means(standardized_matrix):
    kmeans_estimator = KMeans(n_clusters=2, n_init=10, random_state=0)
    cluster_labels = kmeans_estimator.fit_predict(standardized_matrix)
    return cluster_labels

SATURATION_LOWER_FRACTION = 0.02                                          
SATURATION_UPPER_FRACTION = 0.50                                      

def rasterize_one_detector(pred_payload, num_windows):
    flags = []
    for time_second in range(num_windows):
        label_value = "content"
        for segment in pred_payload["segments"]:
            if segment["start"] <= time_second < segment["end"]:
                label_value = segment["label"]
                break
        flags.append(label_value == "non_content")
    return flags

def is_detector_in_sane_range(flags):
    if not flags:
        return False
    non_content_fraction = sum(int(flag) for flag in flags) / len(flags)
    return SATURATION_LOWER_FRACTION <= non_content_fraction <= SATURATION_UPPER_FRACTION

def load_consensus_non_content_flags(consensus_pred_paths, num_windows):
    flags_per_detector = []
    for pred_path in consensus_pred_paths:
        with open(pred_path) as fh:
            payload = json.load(fh)
        flags_per_detector.append(rasterize_one_detector(payload, num_windows))

    sane_flag_lists = [flags for flags in flags_per_detector if is_detector_in_sane_range(flags)]

    if not sane_flag_lists:
                                                                      
        return None

    consensus_flags = []
    majority_threshold = (len(sane_flag_lists) // 2) + 1
    for time_second in range(num_windows):
        agree_count = sum(int(detector_flags[time_second]) for detector_flags in sane_flag_lists)
        consensus_flags.append(agree_count >= majority_threshold)
    return consensus_flags

def count_agreement(flags_a, flags_b):
    return sum(1 for a, b in zip(flags_a, flags_b) if a == b)

def assign_content_label_by_consensus(cluster_labels, consensus_non_content_flags):
    flags_when_cluster_zero_is_content = [int(label) == 1 for label in cluster_labels]
    flags_when_cluster_one_is_content = [int(label) == 0 for label in cluster_labels]

    agreement_when_zero_content = count_agreement(
        flags_when_cluster_zero_is_content, consensus_non_content_flags
    )
    agreement_when_one_content = count_agreement(
        flags_when_cluster_one_is_content, consensus_non_content_flags
    )

    if agreement_when_zero_content >= agreement_when_one_content:
        is_content_per_window = cluster_labels == 0
    else:
        is_content_per_window = cluster_labels == 1
    return is_content_per_window

def assign_content_label_by_filtered_energy(cluster_labels, raw_feature_matrix):
    rms_per_window = raw_feature_matrix[:, 0]
    cluster_zero_mask = cluster_labels == 0
    cluster_one_mask = cluster_labels == 1

    cluster_zero_mean_rms = float(rms_per_window[cluster_zero_mask].mean()) if cluster_zero_mask.any() else 0.0
    cluster_one_mean_rms = float(rms_per_window[cluster_one_mask].mean()) if cluster_one_mask.any() else 0.0

    higher_rms_cluster = 0 if cluster_zero_mean_rms >= cluster_one_mean_rms else 1
    larger_diff_ratio = (
        abs(cluster_zero_mean_rms - cluster_one_mean_rms)
        / (max(cluster_zero_mean_rms, cluster_one_mean_rms) + 1e-12)
    )

    if larger_diff_ratio < 0.10:
                                                               
        cluster_zero_count = int(cluster_zero_mask.sum())
        cluster_one_count = int(cluster_one_mask.sum())
        content_cluster_id = 0 if cluster_zero_count >= cluster_one_count else 1
    else:
                                                                                
        content_cluster_id = 1 - higher_rms_cluster

    is_content_per_window = cluster_labels == content_cluster_id
    return is_content_per_window

def windows_to_segments(is_content_per_window, total_duration_seconds):
    if len(is_content_per_window) == 0:
        return []
    segments = []
    current_label = "content" if is_content_per_window[0] else "non_content"
    current_start = 0
    for window_index in range(1, len(is_content_per_window)):
        observed_label = "content" if is_content_per_window[window_index] else "non_content"
        if observed_label != current_label:
            segments.append({
                "start": float(current_start),
                "end": float(window_index),
                "label": current_label,
            })
            current_start = window_index
            current_label = observed_label
    segments.append({
        "start": float(current_start),
        "end": float(total_duration_seconds),
        "label": current_label,
    })
    return segments

def merge_adjacent_same_label(segments):
    if not segments:
        return []
    merged = [dict(segments[0])]
    for current in segments[1:]:
        if current["label"] == merged[-1]["label"]:
            merged[-1]["end"] = current["end"]
        else:
            merged.append(dict(current))
    return merged

def enforce_minimum_segment_length(segments, minimum_seconds):
    if not segments:
        return []
    cleaned = [dict(seg) for seg in segments]
    changed = True
    while changed:
        changed = False
        for index, segment in enumerate(cleaned):
            duration = segment["end"] - segment["start"]
            if duration >= minimum_seconds:
                continue
            if index > 0:
                cleaned[index]["label"] = cleaned[index - 1]["label"]
            elif len(cleaned) > 1:
                cleaned[index]["label"] = cleaned[index + 1]["label"]
            changed = True
            break
        cleaned = merge_adjacent_same_label(cleaned)
    return cleaned

def run_bgm_pipeline(input_mp4_path, output_json_path, consensus_pred_paths):
    wav_path = ensure_wav_exists(input_mp4_path)
    feature_matrix, duration_seconds, num_windows = extract_filtered_features(
        wav_path, HIGHPASS_HZ
    )
    smoothed_matrix = smooth_feature_matrix(feature_matrix, SMOOTH_WINDOW_SECONDS)
    standardized_matrix = standardize_features(smoothed_matrix)
    cluster_labels = cluster_two_means(standardized_matrix)

    consensus_flags = None
    if consensus_pred_paths:
        consensus_flags = load_consensus_non_content_flags(consensus_pred_paths, num_windows)

    if consensus_flags is not None:
        is_content_per_window = assign_content_label_by_consensus(
            cluster_labels, consensus_flags
        )
    else:
                                                                        
        cluster_zero_count = int((cluster_labels == 0).sum())
        cluster_one_count = int((cluster_labels == 1).sum())
        content_cluster_id = 0 if cluster_zero_count >= cluster_one_count else 1
        is_content_per_window = cluster_labels == content_cluster_id
    raw_segments = windows_to_segments(is_content_per_window, duration_seconds)
    merged_segments = merge_adjacent_same_label(raw_segments)
    final_segments = enforce_minimum_segment_length(merged_segments, MINIMUM_SEGMENT_SECONDS)

    payload = {
        "video_filename": os.path.basename(input_mp4_path),
        "duration_seconds": duration_seconds,
        "segments": final_segments,
    }
    with open(output_json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mp4_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--consensus-from", action="append", default=[],
                        help="Path to another detector's pred JSON; used to "
                             "disambiguate the 2-means cluster->label assignment. "
                             "Pass multiple times. If omitted, falls back to "
                             "the energy heuristic.")
    args = parser.parse_args()
    run_bgm_pipeline(args.input_mp4_path, args.output, args.consensus_from)

if __name__ == "__main__":
    main()
