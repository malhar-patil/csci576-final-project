extern "C"
{
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include <iostream>
#include <cstdint>
#include <vector>

#include "LocalHistCmp.hpp"

using namespace std;

string to_string(Category category)
{
  switch (category)
  {
  case Category::CONTENT:
  {
    return "CONTENT";
  }
  case Category::NON_CONTENT:
  {
    return "NON_CONTENT";
  }
  }
}

LocalHistCmp::LocalHistCmp(int width,
                           int height,
                           AVPixelFormat src_format,
                           AVRational time_base,
                           double s_threshold,
                           double c_threshold,
                           bool debug) : time_base(time_base), s_threshold(s_threshold), c_threshold(c_threshold), debug(debug)
{
  sws_context = sws_getContext(width,
                               height,
                               src_format,
                               width,
                               height,
                               AV_PIX_FMT_GBRP,
                               SWS_BICUBIC,
                               nullptr,
                               nullptr,
                               nullptr);
}

LocalHistCmp::~LocalHistCmp()
{
  sws_freeContext(sws_context);
}

void LocalHistCmp::process_frame(AVFrame *frame)
{
  AVFrame *gbr_frame = av_frame_alloc();

  sws_scale_frame(sws_context, gbr_frame, frame);

  int block_width = lroundf(ceilf(1.f * gbr_frame->width / BLOCKS_PER_DIM));
  int block_height = lroundf(ceilf(1.f * gbr_frame->height / BLOCKS_PER_DIM));
  unsigned long long hists[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256]{};
  for (int i = 0; i < BLOCKS_PER_DIM; i++)
  {
    for (int j = 0; j < BLOCKS_PER_DIM; j++)
    {
      for (int c = 0; c < 3; c++)
      {
        for (int y = i * block_height; y < (i + 1) * block_height; y++)
        {
          for (int x = j * block_width; x < (j + 1) * block_width; x++)
          {
            if (y < 0 || y >= gbr_frame->height || x < 0 || x >= gbr_frame->width)
            {
              continue;
            }

            uint8_t value = gbr_frame->data[c][y * gbr_frame->linesize[c] + x];
            hists[i * BLOCKS_PER_DIM + j][c][value]++;
          }
        }
      }
    }
  }

  av_frame_free(&gbr_frame);

  if (frames == 0 || chi_squared_test(prev_hists, hists, s_threshold))
  {
    if (frames == 0 || frame->pts - segment_data.back().start_pts > 1. * time_base.den / time_base.num)
    {
      segment_data.emplace_back();
    }
    segment_data.back().start_pts = frame->pts;

    if (debug)
    {
      cout << "Frame: " << frames << ", time (s): " << 1. * frame->pts * time_base.num / time_base.den << endl;
    }
  }

  for (int i = 0; i < BLOCKS_PER_DIM * BLOCKS_PER_DIM; i++)
  {
    for (int c = 0; c < 3; c++)
    {
      for (int j = 0; j < 256; j++)
      {
        prev_hists[i][c][j] = hists[i][c][j];
        hist_sums[i][c][j] += hists[i][c][j];
        segment_data.back().hist_sums[i][c][j] += hists[i][c][j];
      }
    }
  }

  frames++;
  segment_data.back().frames++;
}

vector<Segment> LocalHistCmp::finish_segmentation()
{
  vector<Segment> segments;
  unsigned long long overall_hist_avgs[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256];
  unsigned long long segment_hist_avgs[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256];
  calculate_hist_avgs(frames, hist_sums, overall_hist_avgs);
  for (SegmentData sd : segment_data)
  {
    segments.emplace_back();
    segments.back().start_seconds = llround(1. * sd.start_pts * time_base.num / time_base.den);
    calculate_hist_avgs(sd.frames, sd.hist_sums, segment_hist_avgs);
    if (!chi_squared_test(overall_hist_avgs, segment_hist_avgs, c_threshold))
    {
      segments.back().category = Category::CONTENT;
    }
    else
    {
      segments.back().category = Category::NON_CONTENT;
    }

    if (debug)
    {
      cout << "Category: " << to_string(segments.back().category) << endl;
    }
  }

  return segments;
}

bool LocalHistCmp::chi_squared_test(unsigned long long e[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256],
                                    unsigned long long o[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256],
                                    double threshold)
{
  double score = 0;
  for (int i = 0; i < BLOCKS_PER_DIM * BLOCKS_PER_DIM; i++)
  {
    for (int c = 0; c < 3; c++)
    {
      for (int j = 0; j < 256; j++)
      {
        if (e[i][c][j] + o[i][c][j] == 0)
        {
          continue;
        }

        unsigned long long abs_diff = max(e[i][c][j], o[i][c][j]) - min(e[i][c][j], o[i][c][j]);
        score += 1. * abs_diff * abs_diff / (e[i][c][j] + o[i][c][j]);
      }
    }
  }
  score /= sws_context->dst_w * sws_context->dst_h;

  if (debug)
  {
    cout << "Score: " << score << endl;
  }

  return score > threshold;
}

void LocalHistCmp::calculate_hist_avgs(int count,
                                       unsigned long long in[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256],
                                       unsigned long long out[BLOCKS_PER_DIM * BLOCKS_PER_DIM][3][256])
{
  for (int i = 0; i < BLOCKS_PER_DIM * BLOCKS_PER_DIM; i++)
  {
    for (int c = 0; c < 3; c++)
    {
      for (int j = 0; j < 256; j++)
      {
        out[i][c][j] = llround(1. * in[i][c][j] / count);
      }
    }
  }
}
