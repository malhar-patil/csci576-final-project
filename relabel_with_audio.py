
import argparse
import json
from math import ceil

def rasterize_non_content_flags(prediction_payload, total_seconds_int):
    flags_per_second = []
    for time_second in range(total_seconds_int):
        current_label = "content"
        for segment in prediction_payload["segments"]:
            within_segment = segment["start"] <= time_second < segment["end"]
            if within_segment:
                current_label = segment["label"]
                break
        flags_per_second.append(current_label == "non_content")
    return flags_per_second

def relabel_segment(audio_flag_lists, segment_start, segment_end, vote_fraction_threshold):
    seconds_with_majority_non_content = 0
    seconds_in_segment = 0
    flags_length = min(len(flags) for flags in audio_flag_lists)
    for time_second in range(segment_start, segment_end):
        if time_second >= flags_length:
            break
        non_content_vote_count = sum(int(flags[time_second]) for flags in audio_flag_lists)
        if non_content_vote_count >= 1:
            seconds_with_majority_non_content += 1
        seconds_in_segment += 1
    if seconds_in_segment == 0:
        return "content"
    fraction_non_content = seconds_with_majority_non_content / seconds_in_segment
    return "non_content" if fraction_non_content >= vote_fraction_threshold else "content"

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", required=True,
                        help="Prediction JSON whose segment boundaries we keep")
    parser.add_argument("--audio", action="append", required=True,
                        help="Audio detector JSON; pass multiple times")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vote-fraction", type=float, default=0.4,
                        help="Fraction of seconds in segment that must vote non_content")
    parser.add_argument("--min-noncontent-seconds", type=int, default=5)
    arguments = parser.parse_args()

    with open(arguments.boundaries) as fh:
        boundaries_payload = json.load(fh)

    audio_payloads = []
    for audio_path in arguments.audio:
        with open(audio_path) as fh:
            audio_payloads.append(json.load(fh))

    total_duration_seconds = boundaries_payload["duration_seconds"]
    total_seconds_int = int(ceil(total_duration_seconds))

    audio_flag_lists = [
        rasterize_non_content_flags(payload, total_seconds_int)
        for payload in audio_payloads
    ]

    relabeled_segments = []
    for segment in boundaries_payload["segments"]:
        segment_start_int = int(round(segment["start"]))
        segment_end_int = int(round(segment["end"]))
        new_label = relabel_segment(
            audio_flag_lists, segment_start_int, segment_end_int,
            arguments.vote_fraction,
        )
        relabeled_segments.append({
            "start": float(segment_start_int),
            "end": float(segment_end_int),
            "label": new_label,
        })

    merged_segments = merge_adjacent_same_label(relabeled_segments)

    cleaned_segments = []
    for segment in merged_segments:
        run_length = segment["end"] - segment["start"]
        if segment["label"] == "non_content" and run_length < arguments.min_noncontent_seconds:
            cleaned_segments.append({**segment, "label": "content"})
        else:
            cleaned_segments.append(dict(segment))
    cleaned_segments = merge_adjacent_same_label(cleaned_segments)

    output_payload = {
        "video_filename": boundaries_payload["video_filename"],
        "duration_seconds": total_duration_seconds,
        "segments": cleaned_segments,
    }
    with open(arguments.output, "w") as fh:
        json.dump(output_payload, fh, indent=2)

if __name__ == "__main__":
    main()
