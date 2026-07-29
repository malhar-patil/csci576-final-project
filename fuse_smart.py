
import argparse
import json
from math import ceil

SATURATION_LOWER_FRACTION = 0.02
SATURATION_UPPER_FRACTION = 0.50
ANCHOR_WINDOW_SECONDS = 5
MIN_NONCONTENT_SECONDS = 5
BRIDGE_MAX_GAP_SECONDS = 15
EXPAND_FRAGMENT_BELOW_SECONDS = 5
EXPAND_MAX_RESULT_SECONDS = 60

def label_at_second(segments, time_second):
    for segment in segments:
        if segment["start"] <= time_second < segment["end"]:
            return segment["label"]
    return "content"

def collect_boundary_seconds(pred_payload, total_duration_seconds):
    boundary_set = {0, int(round(total_duration_seconds))}
    for segment in pred_payload["segments"]:
        boundary_set.add(int(round(segment["start"])))
        boundary_set.add(int(round(segment["end"])))
    return boundary_set

def rasterize_non_content_flags(pred_payload, total_seconds):
    flags_per_second = []
    for time_second in range(total_seconds):
        label_value = label_at_second(pred_payload["segments"], time_second)
        flags_per_second.append(label_value == "non_content")
    return flags_per_second

def detector_non_content_fraction(flags_per_second):
    if not flags_per_second:
        return 0.0
    return sum(int(flag) for flag in flags_per_second) / len(flags_per_second)

def is_detector_in_sane_range(flags_per_second):
    fraction = detector_non_content_fraction(flags_per_second)
    return SATURATION_LOWER_FRACTION <= fraction <= SATURATION_UPPER_FRACTION

def per_second_majority_flags(video_flags, sane_audio_flag_lists):
    audio_length = (
        min(len(flags) for flags in sane_audio_flag_lists)
        if sane_audio_flag_lists else len(video_flags)
    )
    flags_length = min(len(video_flags), audio_length) if sane_audio_flag_lists else len(video_flags)
    audio_majority_threshold = (
        (len(sane_audio_flag_lists) // 2) + 1 if sane_audio_flag_lists else None
    )
    majority_flags = []
    for time_second in range(flags_length):
        if video_flags[time_second]:
            majority_flags.append(True)
            continue
        if not sane_audio_flag_lists:
            majority_flags.append(False)
            continue
        agree_count = sum(int(flags[time_second]) for flags in sane_audio_flag_lists)
        majority_flags.append(agree_count >= audio_majority_threshold)
    return majority_flags

def label_segment_by_majority(majority_flags, segment_start, segment_end):
    voted_non_content_count = 0
    total_seconds_in_segment = 0
    flags_length = len(majority_flags)
    for time_second in range(segment_start, segment_end):
        if time_second >= flags_length:
            break
        if majority_flags[time_second]:
            voted_non_content_count += 1
        total_seconds_in_segment += 1
    if total_seconds_in_segment == 0:
        return "content"
    fraction_non_content = voted_non_content_count / total_seconds_in_segment
    return "non_content" if fraction_non_content > 0.5 else "content"

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

def has_video_boundary_near(video_boundary_seconds, target_second, window_seconds):
    for boundary_second in video_boundary_seconds:
        if abs(boundary_second - target_second) <= window_seconds:
            return True
    return False

def split_run_by_video_flags(segment, video_non_content_flags):
    start_int = int(round(segment["start"]))
    end_int = int(round(segment["end"]))
    flags_length = len(video_non_content_flags)

    sub_segments = []
    cursor = start_int
    current_label = "non_content" if (
        cursor < flags_length and video_non_content_flags[cursor]
    ) else "content"
    for time_second in range(start_int + 1, end_int):
        observed_label = "non_content" if (
            time_second < flags_length and video_non_content_flags[time_second]
        ) else "content"
        if observed_label != current_label:
            sub_segments.append({
                "start": float(cursor),
                "end": float(time_second),
                "label": current_label,
            })
            cursor = time_second
            current_label = observed_label
    sub_segments.append({
        "start": float(cursor),
        "end": float(end_int),
        "label": current_label,
    })
    return sub_segments

def anchor_non_content_to_video(
    segments,
    video_non_content_flags,
    video_boundary_seconds,
    anchor_window_seconds,
):
    anchored = []
    for segment in segments:
        if segment["label"] != "non_content":
            anchored.append(dict(segment))
            continue

        segment_start_int = int(round(segment["start"]))
        segment_end_int = int(round(segment["end"]))

        start_anchored = has_video_boundary_near(
            video_boundary_seconds, segment_start_int, anchor_window_seconds,
        )
        end_anchored = has_video_boundary_near(
            video_boundary_seconds, segment_end_int, anchor_window_seconds,
        )

        if start_anchored and end_anchored:
            anchored.append(dict(segment))
        else:
            anchored.extend(split_run_by_video_flags(segment, video_non_content_flags))
    return merge_adjacent_same_label(anchored)

def bridge_close_non_content_runs(segments, max_gap_seconds, video_cut_boundaries):
    if not segments:
        return []
    bridged = [dict(segments[0])]
    for current in segments[1:]:
        previous = bridged[-1]
        is_bridge_candidate = (
            previous["label"] == "non_content"
            and current["label"] == "content"
            and len(bridged) >= 1
        )
                                                                        
        bridged.append(dict(current))

    refined = []
    index = 0
    while index < len(bridged):
        current = bridged[index]
        is_nc_content_nc_triple = (
            index + 2 < len(bridged)
            and current["label"] == "non_content"
            and bridged[index + 1]["label"] == "content"
            and bridged[index + 2]["label"] == "non_content"
        )
        if is_nc_content_nc_triple:
            content_gap = bridged[index + 1]
            gap_length = content_gap["end"] - content_gap["start"]
            if gap_length <= max_gap_seconds:
                merged_run = {
                    "start": current["start"],
                    "end": bridged[index + 2]["end"],
                    "label": "non_content",
                }
                refined.append(merged_run)
                index += 3
                continue
        refined.append(dict(current))
        index += 1
    return merge_adjacent_same_label(refined)

def expand_short_non_content_fragments(
    segments, video_cut_boundaries, fragment_below_seconds, max_result_seconds,
):
    if not segments:
        return []
    expanded = []
    for segment in segments:
        run_length = segment["end"] - segment["start"]
        is_short_nc_fragment = (
            segment["label"] == "non_content"
            and run_length < fragment_below_seconds
        )
        if not is_short_nc_fragment:
            expanded.append(dict(segment))
            continue

        start_int = int(round(segment["start"]))
        end_int = int(round(segment["end"]))

        cuts_strictly_before = [b for b in video_cut_boundaries if b < start_int]
        cuts_at_or_after = [b for b in video_cut_boundaries if b >= end_int]
        previous_cut = max(cuts_strictly_before) if cuts_strictly_before else start_int
        next_cut = min(cuts_at_or_after) if cuts_at_or_after else end_int

        expanded_length = next_cut - previous_cut
        if expanded_length <= max_result_seconds:
            expanded.append({
                "start": float(previous_cut),
                "end": float(next_cut),
                "label": "non_content",
            })
        else:
            expanded.append(dict(segment))
    return merge_adjacent_same_label(expanded)

def subtract_overlaps(segments):
    sorted_segments = sorted(segments, key=lambda s: s["start"])
    rebuilt = []
    for segment in sorted_segments:
        if not rebuilt:
            rebuilt.append(dict(segment))
            continue
        previous = rebuilt[-1]
        if segment["start"] >= previous["end"]:
            rebuilt.append(dict(segment))
            continue
                                        
        if segment["label"] == "non_content" and previous["label"] == "content":
            previous["end"] = segment["start"]
            if previous["end"] <= previous["start"]:
                rebuilt.pop()
            rebuilt.append(dict(segment))
        elif segment["label"] == "content" and previous["label"] == "non_content":
            segment_copy = dict(segment)
            segment_copy["start"] = previous["end"]
            if segment_copy["end"] > segment_copy["start"]:
                rebuilt.append(segment_copy)
        else:
            previous["end"] = max(previous["end"], segment["end"])
    return merge_adjacent_same_label(rebuilt)

def remove_short_noncontent_runs(segments, minimum_run_seconds):
    cleaned = []
    for segment in segments:
        run_length = segment["end"] - segment["start"]
        if segment["label"] == "non_content" and run_length < minimum_run_seconds:
            cleaned.append({**segment, "label": "content"})
        else:
            cleaned.append(dict(segment))
    return merge_adjacent_same_label(cleaned)

def fuse_smart(video_pred_path, audio_detector_paths, output_path):
    with open(video_pred_path) as fh:
        video_payload = json.load(fh)

    audio_payloads = []
    for detector_path in audio_detector_paths:
        with open(detector_path) as fh:
            audio_payloads.append(json.load(fh))

    all_payloads = [video_payload] + audio_payloads
    total_duration_seconds = max(payload["duration_seconds"] for payload in all_payloads)
    total_seconds_int = int(ceil(total_duration_seconds))

    video_non_content_flags = rasterize_non_content_flags(video_payload, total_seconds_int)
    audio_flag_lists = [
        rasterize_non_content_flags(payload, total_seconds_int)
        for payload in audio_payloads
    ]

    sane_audio_flag_lists = []
    dropped_count = 0
    for index, flags in enumerate(audio_flag_lists):
        if is_detector_in_sane_range(flags):
            sane_audio_flag_lists.append(flags)
        else:
            dropped_count += 1
            fraction = detector_non_content_fraction(flags)
            print(
                f"[fuse_smart] dropping saturated audio detector "
                f"{audio_detector_paths[index]} (non_content fraction={fraction:.2f})"
            )

    majority_flags = per_second_majority_flags(video_non_content_flags, sane_audio_flag_lists)

    boundary_set = set()
    for payload in all_payloads:
        boundary_set |= collect_boundary_seconds(payload, total_duration_seconds)
    sorted_boundaries = sorted(boundary_set)

    segmented_output = []
    for boundary_index in range(len(sorted_boundaries) - 1):
        segment_start_second = sorted_boundaries[boundary_index]
        segment_end_second = sorted_boundaries[boundary_index + 1]
        if segment_end_second <= segment_start_second:
            continue
        chosen_label = label_segment_by_majority(
            majority_flags, segment_start_second, segment_end_second,
        )
        segmented_output.append({
            "start": float(segment_start_second),
            "end": float(segment_end_second),
            "label": chosen_label,
        })

    merged_segments = merge_adjacent_same_label(segmented_output)

    if "cut_boundaries" in video_payload:
        video_boundary_seconds = sorted(int(b) for b in video_payload["cut_boundaries"])
    else:
        video_boundary_seconds = sorted(
            collect_boundary_seconds(video_payload, total_duration_seconds)
        )
    anchored_segments = anchor_non_content_to_video(
        merged_segments,
        video_non_content_flags,
        video_boundary_seconds,
        ANCHOR_WINDOW_SECONDS,
    )

    expanded_segments = expand_short_non_content_fragments(
        anchored_segments,
        video_boundary_seconds,
        EXPAND_FRAGMENT_BELOW_SECONDS,
        EXPAND_MAX_RESULT_SECONDS,
    )
    deconflicted_segments = subtract_overlaps(expanded_segments)

    bridged_segments = bridge_close_non_content_runs(
        deconflicted_segments, BRIDGE_MAX_GAP_SECONDS, video_boundary_seconds,
    )

    cleaned_segments = remove_short_noncontent_runs(bridged_segments, MIN_NONCONTENT_SECONDS)

    fused_payload = {
        "video_filename": video_payload["video_filename"],
        "duration_seconds": total_duration_seconds,
        "segments": cleaned_segments,
    }
    with open(output_path, "w") as fh:
        json.dump(fused_payload, fh, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True,
                        help="Path to the video segmenter's pred JSON. Its "
                             "boundaries anchor non_content runs.")
    parser.add_argument("--detector", action="append", required=True,
                        help="Path to an audio detector pred JSON; pass multiple times.")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    fuse_smart(arguments.video, arguments.detector, arguments.output)

if __name__ == "__main__":
    main()
