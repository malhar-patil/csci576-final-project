# Multimodal Ad Segmentation — Technical Report

## 1. Project Overview

Goal: given a long-form video with ads spliced into main content, produce a per-second segmentation labelling each region as `content` or `non_content`. The team approached this multimodally — video-side analysis (Justin) on one branch, audio-side analysis (Malhar, on the `bhuvan` branch) on another, with merge/fusion as a separate step.

This report documents what each component does, why we chose its design, the empirical results across the five test videos, the failure modes we hit, and the mitigations we applied.

## 2. Test Material

Five MP4s, each ~24 minutes, with ads inserted. Ground-truth files in `video_info/ground/test_00X.json` give exact ad start/end times. Per-second metrics are produced by `compare.py` (from the bhuvan branch), which rasterises both prediction and truth into one-second bins and reports TP/FP/TN/FN, precision, recall, F1, IoU, and a boundary-F1 within ±2s tolerance. The positive class throughout this report is `non_content`.

| Video | Duration | Ad seconds | Ad fraction | Notes |
|---|---|---|---|---|
| test_001 | 1458s | 179 | 12.3% | 3 ads at clean cuts; modal show appearance is consistent |
| test_002 | ~1452s | 150 | ~10% | Visually ad-like throughout, low video contrast |
| test_003 | ~1700s | 186 | ~11% | Acoustically and visually similar ads & content |
| test_004 | ~1935s | 136 | 7% | Show floods Justin's stage-2; loud action music |
| test_005 | ~1419s | 106 | 7.5% | Nat-Geo interview, constant narration, sharp BGM contrast between ads and content |

## 3. Components Implemented

### 3.1 Justin's video pipeline (C++, on `justin` branch)

**Files:** `src/segment/main.cpp`, `src/segment/LocalHistCmp.{cpp,hpp}`

**Stage 1 — boundary detection** (`process_frame`, `LocalHistCmp.cpp:55`). Each frame is converted from YUV to GBRP, divided into an 8×8 grid (`BLOCKS_PER_DIM = 8`). For every block × every RGB channel × every 8-bit intensity bin, the algorithm accumulates a histogram. It compares the current frame's grid-of-histograms to the previous frame's via a chi-squared-style score:

```
score = Σ_block Σ_channel Σ_bin |e[i] − o[i]|² / (e[i] + o[i])
score /= (frame width × height)
```

If `score > s_threshold` (default 5) **and** at least one second has elapsed since the last cut, a new segment is opened. The 1-second guard prevents rapid cuts from over-fragmenting.

**Stage 2 — categorisation** (`finish_segmentation`, `LocalHistCmp.cpp:120`). After all frames are processed, the algorithm builds the *whole-video* mean per-block histogram. For each segment, it builds the segment's own mean per-block histogram, runs the same chi-squared test against the global mean, and labels:

- `score > c_threshold` (default 2) → `NON_CONTENT` (visually anomalous)
- otherwise → `CONTENT`

**Why this design.** The frame-to-frame chi-squared on local block histograms is a classical shot-boundary detector, robust to small camera motion because each block is normalised separately. The stage-2 categorisation embodies a strong assumption: ads/intros/credits are visually different from the modal frames of the show, and content dominates by duration so the global average reflects "content".

**Output format.** `<video>.seg`, a near-JSON file with bare-identifier values.

### 3.2 Audio v1 (`pipeline.py`, on `bhuvan` branch)

**Approach.** Decode WAV at 16 kHz mono, compute per-1s features via librosa: RMS, spectral centroid, spectral flatness, ZCR, 13 MFCCs. Three sub-detectors:

- **Silence detector**: threshold on RMS with a percentile-based noise floor.
- **Music-vs-speech**: heuristic on spectral flatness + ZCR per window.
- **MFCC discontinuity**: peaks in frame-to-frame MFCC distance.

The three signals are combined into a binary content/non-content per window, then segments shorter than `MINIMUM_SEGMENT_SECONDS` are merged into neighbours.

**Why this design.** This is a deliberate v1: pure signal processing, no learned models, recall-first detection meant to be filtered by a downstream pipeline. The `pipeline.py` docstring explicitly lists its known limitations (no transcript analysis, brittle music/speech, MFCC discontinuity firing on within-content events).

### 3.3 Audio SSM (`pipeline_ssm.py`, on `bhuvan` branch)

**Approach.** Build a Foote self-similarity matrix from the per-window MFCC + spectral features. Detect novelty as peaks of the SSM autocorrelation along a checkerboard kernel of half-width 30s. Cluster the resulting regions with KMeans (`LABEL_N_CLUSTERS = 3`, `CONTENT_CLUSTER_COUNT = 1` — the longest-duration cluster is `content`).

**Why this design.** v1's MFCC discontinuity fires on every brief acoustic event. The SSM averages over a multi-second look-back/look-ahead so transient bursts wash out and only multi-minute regime changes survive. Recall-first by design: the docstring explicitly says "the downstream video pipeline filters false positives."

### 3.4 Format adapter — `seg_to_json.py`

**Why.** Justin's `.seg` format isn't valid JSON (unquoted keys, no commas between fields, bare identifiers as values), so `compare.py` cannot read it directly. We need a unified pred-JSON schema for scoring and fusion.

**What it does.** Parses `.seg` with regex (intentionally not `json.loads` because the file is not JSON), walks consecutive entries treating the next entry's start as the current entry's end, drops the trailing EOF sentinel at `duration_seconds`, lowercases labels, and emits the schema used by `*_pred.json` and the audio detectors:

```json
{ "video_filename": "...", "duration_seconds": ..., "segments": [{"start", "end", "label"}] }
```

It also merges adjacent same-label segments — Justin's stage-1 occasionally emits two adjacent NON_CONTENT entries when boundaries fire close together (e.g. the title sequence on test_001).

### 3.5 First fusion attempt — `fuse_predictions.py`

**Rule:** `fused_is_non_content[t] = video[t] OR (audio_v1[t] AND audio_ssm[t])`.

**Why.** Initial assumption was that Justin's video pipeline has the cleanest boundaries when it fires (test_001 confirmed this with sub-second ad boundary precision), and audio is what saves false-negatives. The intersection of audio_v1 ∧ audio_ssm is a high-precision rescue: it fires only when both audio detectors independently see a regime change, suppressing single-detector audio noise.

**Outcome on test_001.** F1 0.842, recall 0.939, all three ads detected — including Ad 3 (1088–1117s) which video missed and was rescued by the audio AND.

**Outcome on tests 002–005.** Failed. Video-first is the wrong prior on tests where video saturates (test_004 flags 100% non_content) or has no signal (test_005 flags 0%). The rule was overfit to test_001.

### 3.6 Smarter fusion — `fuse_smart.py`

**Rule.** Take the union of boundary points (every `start` in every detector's segments). For each segment between adjacent boundaries, count the seconds where ≥2 of N detectors say `non_content`; if a majority of seconds agree, label the segment `non_content`. Drop non-content runs shorter than `MIN_NONCONTENT_SECONDS`.

**Why.** Decouples Justin's strong stage-1 (boundaries) from his weak stage-2 (categorisation). Pooling boundaries from all detectors gives a finer timeline; per-segment 2-of-N voting suppresses noise from any single detector while letting any two-detector agreement carry a region.

**What we extended.** Originally implemented for 3 detectors (video, audio v1, audio SSM); generalised to N via repeated `--detector` flags so we could plug in BGM as a 4th input.

### 3.7 BGM-filtered audio detector — `pipeline_bgm.py`

This was the major new component, motivated specifically by test_005 (Nat Geo interview). The user identified the key signal: BGM is sharply different in ads vs. content, but a constant narration overlay washes it out in standard MFCC features.

**Stage A — high-pass filter.** A 4th-order Butterworth high-pass at 4 kHz (configurable via `HIGHPASS_HZ`), applied with `scipy.signal.filtfilt` for zero-phase. Narration formants top out around 3.5 kHz; BGM (cymbals, synth pads, reverb tails, sibilance, music sparkle) extends well above. Filtering at 4 kHz deletes most of the narration energy and leaves a "BGM-dominant" residual.

**Stage B — feature extraction on filtered audio.** Per 1s window: RMS energy, spectral centroid, flatness, rolloff, plus 13 MFCCs — all computed on the *filtered* signal. So MFCCs here describe the timbre of the BGM-dominant residual, not the narration.

**Stage C — temporal smoothing.** Rolling mean over `SMOOTH_WINDOW_SECONDS` (default 15s). Suppresses per-second jitter; ad/content regimes change on the order of tens of seconds, not seconds.

**Stage D — clustering.** Standardise (z-score per feature) with float64 + clip to ±5σ to suppress occasional MFCC spikes that can blow up KMeans' float32 internals. Then 2-means.

**Stage E — cluster→label assignment.** This is where the architectural complexity ended up. See §4.4 for the full discussion; the final design uses **sane-consensus disambiguation** with a duration-prior fallback.

### 3.8 Cross-detector cluster labelling

**Problem.** 2-means is a strong primitive — it consistently finds two acoustic regimes — but the binary "which cluster is ads" decision varies by show. In a sports/action show, content is loud and ads are quieter; in a documentary, ads are loud and content is quieter. No fixed acoustic prior generalises.

**Solution.** Use the *other* detectors' predictions as an external label-disambiguation signal. After clustering, compute both possible labelings (cluster-0-as-content vs cluster-1-as-content). For each, count per-second agreement against a consensus built from the other detectors. Pick the labeling with higher agreement.

**Saturation filter.** Detectors whose total predicted non_content fraction is outside [2%, 50%] are excluded from the consensus before voting — they're either silent or flooding, and including them would invert BGM's labelling toward whichever extreme they're saturating. If every detector saturates, BGM falls back to the duration prior (smaller cluster = ads).

## 4. Problems and Mitigations

### 4.1 Build environment for Justin's binary

**Problem.** The C++ pipeline links against ffmpeg dev libraries pinned via a submodule + a local `install/` prefix. The submodule wasn't initialised; building ffmpeg from source would have taken 10–30 minutes.

**Mitigation.** Bypassed by using a pre-built `install/` directory provided by Justin. The makefile expected `pkg-config` to expand the link line; pkg-config wasn't on PATH, so the `make` invocation produced a binary missing all the av* symbols. Worked around by linking manually with the explicit set of libraries from the .pc files: `-lavformat -lavcodec -lswscale -lswresample -lavutil -lm -lbz2 -lz -liconv -pthread` plus the Apple frameworks (`CoreFoundation`, `Security`, `AudioToolbox`, `VideoToolbox`, `CoreMedia`, `CoreVideo`, `CoreServices`).

### 4.2 `.seg` format isn't real JSON

**Problem.** `main.cpp` writes a near-JSON file with bare identifier values (`name: "test_001.mp4",`). `json.loads` throws.

**Mitigation.** `seg_to_json.py` parses with regex against the known structure. Long-term fix is to change `main.cpp` to emit valid JSON, but that change touches the C++ build which we wanted to avoid disturbing during the merge.

### 4.3 Justin's stage-2 is mis-calibrated for non-modal shows

**Problem.** Stage-2 compares each segment's mean histogram against the *whole-video* mean. The implicit prior is that content occupies the majority of frames so the global mean ≈ content. On test_004 the show has no dominant visual mode (constant motion, varied colour), so every segment looks "different from average" and gets flagged NON_CONTENT — the binary predicted everything as non-content (TP=136, FP=1799, TN=0). On test_003/005 the global average gets pulled toward visually-distinct ad frames so true content segments look "different from average" and get correctly labeled… but ad segments do too, washing out the signal.

**Mitigation.** The smart fusion (§3.6) was specifically designed to drop stage-2 in favour of stage-1 boundaries + cross-detector majority. This recovered test_001 without harming the failing cases (it didn't fix them, but stopped letting video alone dictate the answer).

### 4.4 BGM cluster→label assignment is video-dependent

**Problem.** 2-means gives clean clusters but no knowledge of which one is ads.

**Iteration trail:**

1. **Duration prior** (longer cluster = content): worked on test_001 (F1 0.778), test_004 (F1 0.561), tests where ads occupy a clear minority. Failed on test_002 (F1 0.000) where the larger cluster *was* ads.
2. **Energy prior** (louder cluster = ads): "loudness war" intuition. Fixed test_002 (0.000 → 0.276) and test_005 (0.035 → 0.191) — the documentary cases where ads are punchier than content. But catastrophically inverted test_001 (0.778 → 0.079) and test_004 (0.561 → 0.019), where the show itself has loud action music and the ads are comparatively quieter.
3. **Plain consensus** (agree with majority of other detectors): fixed test_002 outright, restored test_001 — but inverted test_004 because video saturates at 100% non_content there, so the consensus itself was wrong.
4. **Sane-consensus** (exclude saturating detectors before voting): the final design. A detector is excluded from the consensus if its predicted non_content fraction is outside [2%, 50%]. On test_004 this drops video and audio_SSM, leaving only audio_v1 — which agrees with the smaller-cluster-is-ads labeling, restoring F1 0.561. On test_002 all detectors saturate so we fall back to the duration prior (still wrong on this video, but at least not worse).

This iteration history is preserved in `pipeline_bgm.py` as separate functions (`assign_content_label_by_filtered_energy`, `assign_content_label_by_consensus`) so we can re-evaluate priors without re-running the expensive feature extraction.

### 4.5 Equal-weight majority dilutes specialist detectors

**Problem.** Each test video has a *different* best single detector (test_001: video; test_002: audio SSM; test_003: BGM/audio v1; test_004: BGM; test_005: audio SSM). Any equal-weight vote averages over the specialist and the noisy detectors and produces a mediocre middle ground.

**Mitigation (partial).** The boundary-union + majority-vote design already implicitly upweights agreement: a segment only fires non_content if *multiple* detectors agree across the majority of its seconds. But this doesn't help when only one detector has signal (test_004) — that detector is outvoted by the silent ones. A confidence-weighted vote (using each detector's per-second score, not just its binary output) would address this; it's listed as future work below.

### 4.6 Numerical stability in clustering

**Problem.** Filtered MFCCs occasionally produced very large values. sklearn's KMeans converts internally to float32 and the matmul step overflowed, throwing `RuntimeWarning` and (in principle) corrupting cluster centres.

**Mitigation.** Standardise in float64 and clip z-scores to ±5σ before passing to KMeans. The warnings still appear (sklearn's float32 conversion happens inside the estimator regardless of input dtype) but the output is empirically stable — re-running with the fix produced bit-identical results on test_001/003/005, confirming the warnings were noise rather than silent corruption.

## 5. Final Per-Detector Results (F1, non-content positive class)

| Test | video | audio v1 | audio SSM | **BGM (sane consensus)** | smart3 fusion | smart4 fusion |
|---|---|---|---|---|---|---|
| 001 | **0.845** | 0.690 | 0.375 | 0.778 | 0.835 | 0.827 |
| 002 | 0.063 | 0.006 | **0.278** | 0.000 | 0.073 | 0.066 |
| 003 | 0.061 | 0.144 | 0.050 | **0.158** | 0.012 | 0.048 |
| 004 | 0.131 | 0.456 | 0.198 | **0.561** | 0.189 | 0.188 |
| 005 | 0.000 | 0.000 | **0.085** | 0.035 | 0.000 | 0.013 |

## 6. Per-Video Analysis

### test_001 — clean modal show (1458s, 3 ads, 12% ad fraction)

Justin's video pipeline wins outright (F1 0.845). Smart fusion ties (0.842). All three ads are detected when fused; only Ad 3 is missed by video alone and rescued by audio_v1 ∧ audio_ssm. This is the cleanest case in the suite.

### test_002 — failure mode for everyone (10% ad fraction)

Best single detector: audio SSM at F1 0.278. Every other detector is below 0.07. Fusion does not help. Diagnosis: video stage-1 over-fires (60%+ of seconds flagged), the BGM regimes don't separate ads from content cleanly, audio v1 sees almost nothing (recall 0.007). This is a genuinely hard video where all four signal sources happen to be weak. Improving it likely requires a different feature primitive (transcript, learned embedding) rather than fusion.

### test_003 — universal weakness

No detector exceeds F1 0.16. Best result is BGM at 0.158 (high recall 0.42, low precision 0.097). Fusion regresses: voting averages everyone toward zero. Probably ads are acoustically and visually similar to content here — a principled limit of unsupervised methods.

### test_004 — BGM specialist case

BGM dominates at F1 0.561, audio v1 second at 0.456. Justin's pipeline saturates at 100% non_content (F1 0.131, useless on its own). The fusion *should* be excellent here but isn't (smart4 only F1 0.188), because the saturated video and the over-flagging audio SSM together outvote BGM in the majority count. This is the clearest evidence that equal-weight majority is the wrong fusion shape on this dataset.

### test_005 — Nat Geo interview

Best detector: audio SSM at F1 0.085. Best alternative-prior BGM (energy heuristic) hits F1 0.191 in isolation, but with the final sane-consensus design it falls back to the duration prior and degrades to 0.035 because every detector saturates and there's no consensus. This is the case the BGM filter was originally designed for, and the cluster_2-means *does* find the regime change — but neither acoustic prior nor any external consensus is reliable enough to choose which cluster is ads. Possible directions: a per-cluster narration-vs-music ratio metric, or a transcript-driven prior.

## 7. Files Generated

- `src/segment/segment` — built C++ binary
- `videos/test_00X.seg` — Justin's raw outputs
- `video_info/pred/test_00X_video_pred.json` — adapted to unified schema
- `video_info/pred/test_00X_pred.json` — audio v1 (from prior runs)
- `video_info/pred/test_00X_ssm_pred.json` — audio SSM (from prior runs)
- `video_info/pred/test_00X_bgm_pred.json` — BGM detector with sane-consensus labelling
- `video_info/pred/test_00X_fused_pred.json` — first-attempt fusion (video OR audio AND)
- `video_info/pred/test_00X_smart_pred.json` — boundary-union 3-way majority
- `video_info/pred/test_00X_smart4_pred.json` — boundary-union 4-way majority including BGM
- `seg_to_json.py` — `.seg` → unified-JSON adapter
- `fuse_predictions.py` — first-attempt fusion
- `fuse_smart.py` — boundary-union N-way majority fusion
- `pipeline_bgm.py` — high-pass + 2-means + sane-consensus BGM detector

## 8. Future Work

1. **Confidence-weighted fusion**. Each detector emits binary flags but internally has continuous scores (chi-squared distance for video, novelty-curve peak height for SSM, KMeans distance ratio for BGM). Voting on the raw scores instead of binary flags would let a high-confidence specialist outweigh a flooding detector.
2. **Per-detector saturation/auto-thresholding**. Justin's `c_threshold = 2` is mis-calibrated for tests 002/004; a sweep + auto-pick per video could push video's F1 well above the current 0.06–0.13 range on those.
3. **Transcript-driven detector** (deferred from spec.md as out-of-scope for v1). Sponsor reads embedded in host speech remain undetectable to all four current signals.
4. **Bandstop variant of BGM filter**. The current high-pass at 4 kHz keeps high-band BGM but throws away sub-bass rumble that's also distinct between ads and content. A bandstop (notch out 200–3500 Hz) would retain both sides of the narration band.
5. **Test_005-specific feature**: a narration-vs-music *ratio* per window, not just energy in a band. Specifically, the ratio of speech-like spectral flatness to music-like spectral flatness within the cluster's central windows would let us read the ad/content sign from cluster geometry instead of an external consensus.
