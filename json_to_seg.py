
import argparse
import json

def label_to_category(label_value):
    if label_value == "content":
        return "CONTENT"
    return "NON_CONTENT"

def write_seg_file(pred_payload, output_seg_path, video_width=0, video_height=0):
    video_filename = pred_payload["video_filename"]
    duration_seconds_int = int(round(float(pred_payload["duration_seconds"])))
    segments_list = pred_payload["segments"]

    with open(output_seg_path, "w") as seg_file_handle:
        seg_file_handle.write("{\n")
        seg_file_handle.write(f'  name: "{video_filename}",\n')
        seg_file_handle.write(f"  duration_seconds: {duration_seconds_int},\n")
        seg_file_handle.write(f"  width: {video_width},\n")
        seg_file_handle.write(f"  height: {video_height},\n")
        seg_file_handle.write("  segments: [\n")

        last_segment_index = len(segments_list) - 1
        for segment_index, current_segment in enumerate(segments_list):
            start_seconds_int = int(round(float(current_segment["start"])))
            category_string = label_to_category(current_segment["label"])
            is_last_segment = segment_index == last_segment_index
            trailing_comma = "" if is_last_segment else ","

            seg_file_handle.write("    {\n")
            seg_file_handle.write(f"      start_seconds: {start_seconds_int}\n")
            seg_file_handle.write(f"      category: {category_string}\n")
            seg_file_handle.write(f"    }}{trailing_comma}\n")

        seg_file_handle.write("  ]\n")
        seg_file_handle.write("}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert fused pred JSON to .seg for the Qt player."
    )
    parser.add_argument("input_json_path")
    parser.add_argument("--output", required=True, help="Path to write .seg")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parsed_args = parser.parse_args()

    with open(parsed_args.input_json_path) as input_file_handle:
        pred_payload = json.load(input_file_handle)

    write_seg_file(
        pred_payload,
        parsed_args.output,
        video_width=parsed_args.width,
        video_height=parsed_args.height,
    )

if __name__ == "__main__":
    main()
