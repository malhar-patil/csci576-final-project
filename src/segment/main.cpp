extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <vector>

#include "LocalHistCmp.hpp"

using namespace std;
using namespace filesystem;

int main(int argc, char *argv[])
{
  AVFormatContext *format_context = avformat_alloc_context();
  avformat_open_input(&format_context, argv[1], nullptr, nullptr);
  avformat_find_stream_info(format_context, nullptr);

  const AVCodec *codec;
  int video_stream_index = av_find_best_stream(format_context, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  AVCodecContext *codec_context = avcodec_alloc_context3(codec);
  AVStream *video_stream = format_context->streams[video_stream_index];
  AVCodecParameters *codec_parameters = video_stream->codecpar;
  avcodec_parameters_to_context(codec_context, codec_parameters);
  avcodec_open2(codec_context, codec, nullptr);

  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();

  double s_threshold = 5;
  double c_threshold = 2;
  bool debug = false;
  if (argc > 2)
  {
    s_threshold = stod(argv[2]);
  }
  if (argc > 3)
  {
    c_threshold = stod(argv[3]);
  }
  if (argc > 4)
  {
    debug = stoi(argv[4]);
  }

  LocalHistCmp local_hist_cmp(codec_context->width,
                              codec_context->height,
                              codec_context->pix_fmt,
                              video_stream->time_base,
                              s_threshold,
                              c_threshold,
                              debug);

  while (av_read_frame(format_context, packet) == 0)
  {
    if (packet->stream_index != video_stream_index)
    {
      continue;
    }

    avcodec_send_packet(codec_context, packet);
    while (avcodec_receive_frame(codec_context, frame) == 0)
    {
      local_hist_cmp.process_frame(frame);
    }

    av_packet_unref(packet);
  }

  vector<Segment> segments = local_hist_cmp.finish_segmentation();

  path video_path(argv[1]);
  path video_name = video_path.filename();
  ofstream video_seg_file(video_path.replace_extension("seg"));
  video_seg_file << "{\n";
  video_seg_file << "  name: " << video_name << ",\n";
  int64_t video_duration = llround(1. * video_stream->duration * video_stream->time_base.num / video_stream->time_base.den);
  video_seg_file << "  duration_seconds: " << video_duration << ",\n";
  video_seg_file << "  width: " << codec_context->width << ",\n";
  video_seg_file << "  height: " << codec_context->height << ",\n";
  video_seg_file << "  segments: [\n";
  for (int i = 0; i < segments.size(); i++)
  {
    video_seg_file << "    {\n";
    video_seg_file << "      start_seconds: " << segments[i].start_seconds << "\n";
    video_seg_file << "      category: " << to_string(segments[i].category) << "\n";
    video_seg_file << "    }" << (i < segments.size() - 1 ? ",\n" : "\n");
  }
  video_seg_file << "  ]\n";
  video_seg_file << "}\n";
  video_seg_file.close();

  av_frame_free(&frame);
  av_packet_free(&packet);

  avcodec_free_context(&codec_context);

  avformat_close_input(&format_context);
  avformat_free_context(format_context);
}
