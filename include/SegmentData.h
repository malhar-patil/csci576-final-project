#pragma once

enum class SegmentType{
    content, 
    non_content
};

struct Segment{
    SegmentType type = SegmentType::content;
    double start_sec = 0.0;
    double end_sec = 0.0;
    bool enabled = true;
};