# Multimodal Segmentation of Long-Form Online Video into Content and Non-Content

<video src="https://github.com/user-attachments/assets/45921752-3b8c-474b-b8d8-c89d9c557ae4"></video>

Given a long-form video with ads spliced into the main content, the system labels every second of the video as either `content` or `non_content` (ads, intros, outros, self-promotion, filler), then plays the video back with the ad regions marked on the timeline and skippable.

There is no training data and no learned model here. For every video we used unsupervised signal processing: histogram-based shot detection on the video side, spectral and MFCC-based clustering on the audio side, and a voting step that fuses the detectors together.

## Contributions

**[Justin](https://github.com/usc-jmshi):** Worked on C++ video segmenter. ffmpeg decoding, the local histogram chi-squared shot-boundary detector, the segment-vs-global-average categorisation stage, threshold parameters, and the `.seg` output format.

**[Bhuvan](https://github.com/BHUVAN-RJ):** Worked on audio side and the fusion layer. Shared feature extraction, the v1 detector, the self-similarity-matrix detector, the high-pass BGM detector and its cluster-labelling logic, the scoring harness and parameter sweeps, both fusion implementations, the format adapters, the end-to-end driver script, and the technical report.

**[Malhar](https://github.com/malhar-patil):** Worked on Qt6 player. `.seg` parsing and loading, the content/non-content timeline bar, the timestamp sidebar with current-segment highlighting, the skip-non-content playback mode, and the CMake build for Windows and macOS.

## Working

The pipeline has three parts that were developed independently and joined by a common JSON schema.

**Video segmenter (C++/ffmpeg).** Decodes frames, splits each into an 8x8 grid, and compares per-block RGB histograms between consecutive frames with a chi-squared score. A score above the threshold opens a new segment, with a one-second guard against over-fragmenting. Each segment is then compared against the whole-video's mean histogram. Segments that look visually anomalous are labelled non-content.

**Audio detectors (Python).** Audio is extracted to 16 kHz mono WAV and reduced to per-second features (RMS, spectral centroid, flatness, rolloff, ZCR, 13 MFCCs). Three detectors run on top of that:

- `pipeline.py` — silence, a music-vs-speech heuristic, and MFCC discontinuity peaks.
- `pipeline_ssm.py` — a Foote self-similarity matrix with a checkerboard novelty kernel, then KMeans. Catches multi-minute regime changes rather than brief acoustic events.
- `pipeline_bgm.py` — a 4 kHz high-pass drops narration formants so background music dominates; features on the filtered signal are smoothed and 2-means clustered. Which cluster is "ads" is decided by agreement with the other detectors.

**Fusion.** `fuse_smart.py` pools boundary points from every detector and flips a segment to non-content when a majority of its seconds have at least two detectors agreeing. Saturated detectors (predicting almost nothing or almost everything as non-content) are dropped from the vote. `fuse_predictions.py` is the earlier, simpler rule and is kept for comparison.

**Player (Qt6).** Loads a video, finds the matching `.seg` beside it, and draws a timeline bar coloured by segment type. A sidebar lists each block with its timestamp and highlights the current one. A toggle skips non-content regions during playback.

---

Predictions share one schema so any stage can be swapped or scored independently:

```json
{ "video_filename": "...", "duration_seconds": 1458.0,
  "segments": [{ "start": 0.0, "end": 41.0, "label": "content" }] }
```

## Building

The video segmenter needs ffmpeg development libraries. The submodule is pinned in `.gitmodules`; a prebuilt `install/` prefix also works.

```
git submodule update --init --recursive
cd src/segment && make
```

The player needs Qt6 (Widgets, Multimedia, MultimediaWidgets) and CMake 3.20+. `CMakePresets.json` has a Windows/MinGW preset; adjust the paths for your install.

```
cmake --preset default
cmake --build build
```

## Running

End to end on one video:

```
./run_pipeline.sh videos/test_001.mp4
```

That runs the video segmenter, both audio detectors, fuses them, and writes `.seg` next to the video. Open the video in the player and it picks that file up automatically.
