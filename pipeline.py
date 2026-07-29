
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile

from features import extract_features

def derive_wav_path(input_mp4_path: str) -> str:
    directory = os.path.dirname(input_mp4_path)
    base_name = os.path.basename(input_mp4_path)
    stem, _ext = os.path.splitext(base_name)
    return os.path.join(directory, stem + ".wav")

def cached_wav_is_valid(wav_path: str) -> bool:
    if not os.path.exists(wav_path):
        return False
    try:
        info = soundfile.info(wav_path)
    except Exception:
        return False
    return info.samplerate == 16000 and info.channels == 1

def extract_audio_to_wav(input_mp4_path: str, wav_path: str) -> None:
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_mp4_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-y",
        wav_path,
    ]
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

def ensure_wav_exists(input_mp4_path: str) -> str:
    wav_path = derive_wav_path(input_mp4_path)

    if cached_wav_is_valid(wav_path):
        print(f"Cache hit: reusing {wav_path}", file=sys.stderr)
        return wav_path

    if os.path.exists(wav_path):
        os.remove(wav_path)

    extract_audio_to_wav(input_mp4_path, wav_path)
    return wav_path

def compute_silence_score(rms_per_window: np.ndarray) -> np.ndarray:
    num_windows = len(rms_per_window)

    noise_floor = float(np.percentile(rms_per_window, 10))
    silence_threshold = max(noise_floor * 2.0, 0.005)
    is_silent = rms_per_window < silence_threshold

    silence_score = np.zeros(num_windows, dtype=np.float32)
    for window_index in range(num_windows):
        is_first_window = window_index == 0
        is_last_window = window_index == num_windows - 1

        if is_first_window:
            if num_windows >= 2 and is_silent[0] and is_silent[1]:
                silence_score[window_index] = 1.0
            elif num_windows == 1 and is_silent[0]:
                silence_score[window_index] = 1.0
        elif is_last_window:
            if is_silent[window_index - 1] and is_silent[window_index]:
                silence_score[window_index] = 1.0
        else:
            neighborhood_silent_count = (
                int(is_silent[window_index - 1])
                + int(is_silent[window_index])
                + int(is_silent[window_index + 1])
            )
            if neighborhood_silent_count >= 2:
                silence_score[window_index] = 1.0

    return silence_score

def in_band(value: float, lo: float, hi: float) -> float:
    band_width = hi - lo
    if lo <= value <= hi:
        return 1.0
    falloff_margin = band_width * 0.2
    if value < lo:
        return max(0.0, 1.0 - (lo - value) / falloff_margin)
    return max(0.0, 1.0 - (value - hi) / falloff_margin)

def compute_music_score(
    zcr_per_window: np.ndarray,
    centroid_per_window: np.ndarray,
    flatness_per_window: np.ndarray,
) -> np.ndarray:
    num_windows = len(zcr_per_window)
    music_score = np.zeros(num_windows, dtype=np.float32)

    for window_index in range(num_windows):
        zcr_score = in_band(float(zcr_per_window[window_index]), 0.03, 0.20)
        centroid_score = in_band(float(centroid_per_window[window_index]), 1000.0, 3500.0)
        flatness_score = in_band(float(flatness_per_window[window_index]), 0.1, 0.4)

        speech_likelihood = (
            zcr_score * 0.4
            + centroid_score * 0.3
            + flatness_score * 0.3
        )
        music_score[window_index] = 1.0 - speech_likelihood

    return music_score

DISCONTINUITY_THRESHOLD = float(os.environ.get("DISCONTINUITY_THRESHOLD", "0.6"))
CLUSTER_DEDUP_RADIUS_SECONDS = float(
    os.environ.get("CLUSTER_DEDUP_RADIUS_SECONDS", "2.0")
)

def compute_mfcc_discontinuities(mfcc_per_window: np.ndarray) -> list:
    num_windows = mfcc_per_window.shape[0]
    delta = np.zeros(num_windows, dtype=np.float32)

    for window_index in range(1, num_windows):
        diff_vector = mfcc_per_window[window_index] - mfcc_per_window[window_index - 1]
        delta[window_index] = float(np.linalg.norm(diff_vector))

    smoothing_kernel = np.ones(3, dtype=np.float32) / 3.0
    delta_smoothed = np.convolve(delta, smoothing_kernel, mode="same")

    robust_max = float(np.percentile(delta_smoothed, 95))
    if robust_max <= 0.0:
        return []

    delta_normalized = np.clip(delta_smoothed / robust_max, 0.0, 1.0)

    discontinuity_events = []
    for window_index in range(num_windows):
        if delta_normalized[window_index] > DISCONTINUITY_THRESHOLD:
            discontinuity_events.append(
                (window_index, float(delta_normalized[window_index]))
            )
    return discontinuity_events

def median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    half = kernel_size // 2
    n = len(values)
    output = np.zeros_like(values)
    for i in range(n):
        window_start = max(0, i - half)
        window_end = min(n, i + half + 1)
        output[i] = float(np.median(values[window_start:window_end]))
    return output

def consolidate_boundary_candidates(
    discontinuity_events: list,
    duration_seconds: float,
) -> list:
    events_by_strength = sorted(
        discontinuity_events, key=lambda event: event[1], reverse=True
    )

    kept_discontinuity_times = []
    for window_index, _strength in events_by_strength:
        candidate_time = float(window_index)

        too_close_to_start = abs(candidate_time - 0.0) <= CLUSTER_DEDUP_RADIUS_SECONDS
        too_close_to_end = abs(candidate_time - duration_seconds) <= CLUSTER_DEDUP_RADIUS_SECONDS
        if too_close_to_start or too_close_to_end:
            continue

        is_duplicate = False
        for kept_time in kept_discontinuity_times:
            if abs(candidate_time - kept_time) <= CLUSTER_DEDUP_RADIUS_SECONDS:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        kept_discontinuity_times.append(candidate_time)

    all_boundary_times = [0.0] + sorted(kept_discontinuity_times) + [duration_seconds]
    return all_boundary_times

def label_segments_from_boundaries(
    boundary_times: list,
    nc_score_smoothed: np.ndarray,
) -> list:
    segments = []
    num_windows = len(nc_score_smoothed)

    for boundary_index in range(len(boundary_times) - 1):
        segment_start_sec = boundary_times[boundary_index]
        segment_end_sec = boundary_times[boundary_index + 1]

        start_idx = int(round(segment_start_sec))
        end_idx = int(round(segment_end_sec))

        start_idx = max(0, min(start_idx, num_windows))
        end_idx = max(0, min(end_idx, num_windows))

        if start_idx == end_idx:
            single_index = min(start_idx, num_windows - 1)
            mean_score = float(nc_score_smoothed[single_index])
        else:
            mean_score = float(nc_score_smoothed[start_idx:end_idx].mean())

        label = "non_content" if mean_score > 0.5 else "content"
        segments.append({
            "start": segment_start_sec,
            "end": segment_end_sec,
            "label": label,
            "mean_score": mean_score,
        })

    return segments

def merge_adjacent_same_label(segments: list) -> list:
    if not segments:
        return []

    merged = [dict(segments[0])]
    for current_segment in segments[1:]:
        last_segment = merged[-1]
        if current_segment["label"] == last_segment["label"]:
            last_segment["end"] = current_segment["end"]
            last_duration = last_segment["end"] - last_segment["start"]
            previous_duration = last_duration - (
                current_segment["end"] - current_segment["start"]
            )
            current_duration = current_segment["end"] - current_segment["start"]
            total_duration = previous_duration + current_duration
            if total_duration > 0:
                weighted_mean = (
                    last_segment["mean_score"] * previous_duration
                    + current_segment["mean_score"] * current_duration
                ) / total_duration
                last_segment["mean_score"] = weighted_mean
        else:
            merged.append(dict(current_segment))
    return merged

def enforce_minimum_segment_length(
    segments: list,
    duration_seconds: float,
    minimum_length_seconds: float = 5.0,
) -> list:
    if duration_seconds < minimum_length_seconds:
        return segments
    if len(segments) <= 1:
        return segments

    working_segments = [dict(segment) for segment in segments]
    output_segments = []
    index = 0

    while index < len(working_segments):
        current_segment = working_segments[index]
        current_duration = current_segment["end"] - current_segment["start"]

        is_too_short = current_duration < minimum_length_seconds
        if not is_too_short:
            output_segments.append(current_segment)
            index += 1
            continue

        has_left_neighbor = len(output_segments) > 0
        has_right_neighbor = index + 1 < len(working_segments)

        if has_left_neighbor and has_right_neighbor:
            left_neighbor = output_segments[-1]
            right_neighbor = working_segments[index + 1]
            left_score_distance = abs(
                left_neighbor["mean_score"] - current_segment["mean_score"]
            )
            right_score_distance = abs(
                right_neighbor["mean_score"] - current_segment["mean_score"]
            )
            merge_into_left = left_score_distance <= right_score_distance

            if merge_into_left:
                left_neighbor["end"] = current_segment["end"]
                index += 1
            else:
                right_neighbor["start"] = current_segment["start"]
                index += 1
        elif has_left_neighbor:
            left_neighbor = output_segments[-1]
            left_neighbor["end"] = current_segment["end"]
            index += 1
        elif has_right_neighbor:
            right_neighbor = working_segments[index + 1]
            right_neighbor["start"] = current_segment["start"]
            index += 1
        else:
            output_segments.append(current_segment)
            index += 1

    return output_segments

def build_segments(
    silence_score: np.ndarray,
    music_score: np.ndarray,
    discontinuity_events: list,
    duration_seconds: float,
) -> list:
    nc_score = 0.5 * silence_score + 0.5 * music_score
    nc_score_smoothed = median_filter_1d(nc_score, kernel_size=9)

    boundary_times = consolidate_boundary_candidates(
        discontinuity_events, duration_seconds
    )

    raw_segments = label_segments_from_boundaries(boundary_times, nc_score_smoothed)
    merged_segments = merge_adjacent_same_label(raw_segments)
    length_constrained_segments = enforce_minimum_segment_length(
        merged_segments, duration_seconds
    )
    final_merged_segments = merge_adjacent_same_label(length_constrained_segments)

    final_segments = []
    for segment in final_merged_segments:
        final_segments.append({
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "label": segment["label"],
        })
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

    silence_score = compute_silence_score(features["rms"])
    music_score = compute_music_score(
        features["zcr"], features["spectral_centroid"], features["spectral_flatness"]
    )
    discontinuity_events = compute_mfcc_discontinuities(features["mfcc"])

    print(
        f"Detectors: silence_windows={int(silence_score.sum())}, "
        f"mean_music_score={float(music_score.mean()):.3f}, "
        f"discontinuity_events={len(discontinuity_events)}",
        file=sys.stderr,
    )

    segments = build_segments(
        silence_score, music_score, discontinuity_events, duration_seconds_floored
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
        description="Audio-based content/non-content video segmentation."
    )
    parser.add_argument("input_mp4", help="Path to input MP4 file")
    parser.add_argument(
        "--output", required=True, help="Path to write the segmentation JSON"
    )
    args = parser.parse_args()

    run_pipeline(args.input_mp4, args.output)

if __name__ == "__main__":
    main()
