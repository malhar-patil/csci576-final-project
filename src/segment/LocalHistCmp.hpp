#pragma once

extern "C"
{
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include <vector>
#include <cstdint>

using namespace std;

constexpr int BLOCKS_PER_DIM = 8;

enum class Category
{
  CONTENT,
  NON_CONTENT
};

string to_string(Category category);

struct Segment
{
  int64_t start_seconds;
  Category category;
};

struct SegmentData
{
  unsigned long long hist_sums[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256]{};
  int frames = 0;
  int64_t start_pts;
};

class LocalHistCmp
{
public:
  LocalHistCmp(int width,
               int height,
               AVPixelFormat src_format,
               AVRational time_base,
               double s_threshold,
               double c_threshold,
               bool debug);

  ~LocalHistCmp();

  void process_frame(AVFrame *frame);
  vector<Segment> finish_segmentation();

private:
  SwsContext *sws_context;
  AVRational time_base;
  unsigned long long prev_hists[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256];
  unsigned long long hist_sums[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256]{};
  int frames = 0;
  vector<SegmentData> segment_data;
  double s_threshold;
  double c_threshold;
  bool debug;

  bool chi_squared_test(unsigned long long e[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256],
                        unsigned long long o[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256],
                        double threshold);
  void calculate_hist_avgs(int count,
                           unsigned long long in[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256],
                           unsigned long long out[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256]);
};
