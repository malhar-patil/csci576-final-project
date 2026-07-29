
import argparse
import json
from math import ceil, floor

MIN_NONCONTENT_SECONDS = 3

def label_at_second(segments, time_second):
    for segment in segments:
        if segment["start"] <= time_second < segment["end"]:
            return segment["label"]
    return "content"

def rasterize_predictions(pred_payload, total_seconds):
    is_non_content_per_second = []
    for time_second in range(total_seconds):
        label_value = label_at_second(pred_payload["segments"], time_second)
        is_non_content_per_second.append(label_value == "non_content")
    return is_non_content_per_second

def fuse_per_second(video_flags, audio_v1_flags, audio_ssm_flags):
    fused_flags = []
    for video_flag, audio_flag, ssm_flag in zip(video_flags, audio_v1_flags, audio_ssm_flags):
        audio_intersection = audio_flag and ssm_flag
        fused_flags.append(video_flag or audio_intersection)
    return fused_flags

def remove_short_noncontent_runs(fused_flags, minimum_run_length):
    cleaned_flags = list(fused_flags)
    run_start_index = None
    for index, flag in enumerate(cleaned_flags + [False]):
        if flag and run_start_index is None:
            run_start_index = index
        elif (not flag) and run_start_index is not None:
            run_length = index - run_start_index
            if run_length < minimum_run_length:
                for fill_index in range(run_start_index, index):
                    cleaned_flags[fill_index] = False
            run_start_index = None
    return cleaned_flags

def flags_to_segments(fused_flags, total_duration_seconds):
    segments = []
    if not fused_flags:
        return segments

    current_label = "non_content" if fused_flags[0] else "content"
    current_start_second = 0
    for time_second in range(1, len(fused_flags)):
        observed_label = "non_content" if fused_flags[time_second] else "content"
        if observed_label != current_label:
            segments.append({
                "start": float(current_start_second),
                "end": float(time_second),
                "label": current_label,
            })
            current_start_second = time_second
            current_label = observed_label
    segments.append({
        "start": float(current_start_second),
        "end": float(total_duration_seconds),
        "label": current_label,
    })
    return segments

def fuse_three_predictions(video_pred_path, audio_pred_path, ssm_pred_path, output_path):
    with open(video_pred_path) as fh:
        video_payload = json.load(fh)
    with open(audio_pred_path) as fh:
        audio_payload = json.load(fh)
    with open(ssm_pred_path) as fh:
        ssm_payload = json.load(fh)

    duration_seconds = max(
        video_payload["duration_seconds"],
        audio_payload["duration_seconds"],
        ssm_payload["duration_seconds"],
    )
    total_seconds_int = int(ceil(duration_seconds))

    video_flags = rasterize_predictions(video_payload, total_seconds_int)
    audio_flags = rasterize_predictions(audio_payload, total_seconds_int)
    ssm_flags = rasterize_predictions(ssm_payload, total_seconds_int)

    fused_flags = fuse_per_second(video_flags, audio_flags, ssm_flags)
    cleaned_flags = remove_short_noncontent_runs(fused_flags, MIN_NONCONTENT_SECONDS)
    fused_segments = flags_to_segments(cleaned_flags, duration_seconds)

    fused_payload = {
        "video_filename": video_payload["video_filename"],
        "duration_seconds": duration_seconds,
        "segments": fused_segments,
    }
    with open(output_path, "w") as fh:
        json.dump(fused_payload, fh, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ssm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fuse_three_predictions(args.video, args.audio, args.ssm, args.output)

if __name__ == "__main__":
    main()
