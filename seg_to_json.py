
import argparse
import json
import os
import re

def parse_seg_file(seg_file_path):
    with open(seg_file_path, "r") as seg_file_handle:
        seg_text = seg_file_handle.read()

    name_match = re.search(r'name:\s*"([^"]+)"', seg_text)
    video_filename = name_match.group(1) if name_match else os.path.basename(seg_file_path).replace(".seg", ".mp4")

    duration_match = re.search(r"duration_seconds:\s*(\d+)", seg_text)
    duration_seconds_int = int(duration_match.group(1))

    segment_block_pattern = re.compile(
        r"start_seconds:\s*(\d+)\s+category:\s*(CONTENT|NON_CONTENT)"
    )
    raw_entries = [
        {"start_seconds": int(start_str), "category": category_str}
        for start_str, category_str in segment_block_pattern.findall(seg_text)
    ]

    return {
        "video_filename": video_filename,
        "duration_seconds": float(duration_seconds_int),
        "raw_entries": raw_entries,
    }

def build_unified_segments(parsed_seg):
    raw_entries = parsed_seg["raw_entries"]
    duration_seconds = parsed_seg["duration_seconds"]

    if not raw_entries:
        return []

    last_entry = raw_entries[-1]
    if last_entry["start_seconds"] >= duration_seconds:
        raw_entries = raw_entries[:-1]

    unified_segments = []
    for entry_index, current_entry in enumerate(raw_entries):
        is_last_entry = entry_index == len(raw_entries) - 1
        segment_start = float(current_entry["start_seconds"])
        if is_last_entry:
            segment_end = duration_seconds
        else:
            segment_end = float(raw_entries[entry_index + 1]["start_seconds"])

        if segment_end <= segment_start:
            continue

        label_lower = "content" if current_entry["category"] == "CONTENT" else "non_content"
        unified_segments.append({
            "start": segment_start,
            "end": segment_end,
            "label": label_lower,
        })

    merged_segments = []
    for segment in unified_segments:
        if merged_segments and merged_segments[-1]["label"] == segment["label"]:
            merged_segments[-1]["end"] = segment["end"]
        else:
            merged_segments.append(dict(segment))
    return merged_segments

def collect_raw_cut_boundaries(parsed_seg):
    raw_entries = parsed_seg["raw_entries"]
    duration_seconds = parsed_seg["duration_seconds"]
    boundary_set = {0, int(round(duration_seconds))}
    for entry in raw_entries:
        boundary_set.add(int(entry["start_seconds"]))
    return sorted(boundary_set)

def convert_seg_to_pred_json(seg_file_path, output_json_path):
    parsed_seg = parse_seg_file(seg_file_path)
    unified_segments = build_unified_segments(parsed_seg)
    raw_cut_boundaries = collect_raw_cut_boundaries(parsed_seg)
    pred_payload = {
        "video_filename": parsed_seg["video_filename"],
        "duration_seconds": parsed_seg["duration_seconds"],
        "segments": unified_segments,
        "cut_boundaries": raw_cut_boundaries,
    }
    with open(output_json_path, "w") as output_handle:
        json.dump(pred_payload, output_handle, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg_file_path")
    parser.add_argument("--output", required=True)
    parsed_args = parser.parse_args()
    convert_seg_to_pred_json(parsed_args.seg_file_path, parsed_args.output)

if __name__ == "__main__":
    main()
