
import sys

import numpy as np
import librosa

DEFAULT_HOP_LENGTH = 512

def _mean_pool_per_window(per_frame_values, frames_per_window, num_windows):
    is_two_dimensional = per_frame_values.ndim == 2
    if is_two_dimensional:
        n_coeffs = per_frame_values.shape[0]
        pooled = np.zeros((num_windows, n_coeffs), dtype=np.float32)
    else:
        pooled = np.zeros(num_windows, dtype=np.float32)

    for window_index in range(num_windows):
        frame_start = window_index * frames_per_window
        frame_end = frame_start + frames_per_window
        window_slice = per_frame_values[..., frame_start:frame_end]

        if window_slice.shape[-1] == 0:
            continue

        window_mean = window_slice.mean(axis=-1)
        if is_two_dimensional:
            pooled[window_index, :] = window_mean
        else:
            pooled[window_index] = window_mean

    return pooled

def extract_features(wav_path: str, window_sec: float = 1.0) -> dict:
    samples, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
    duration_sec = float(len(samples)) / float(sample_rate)

    num_windows = int(duration_sec // window_sec)

    rms_per_frame = librosa.feature.rms(y=samples, hop_length=DEFAULT_HOP_LENGTH)[0]
    centroid_per_frame = librosa.feature.spectral_centroid(
        y=samples, sr=sample_rate, hop_length=DEFAULT_HOP_LENGTH
    )[0]
    flatness_per_frame = librosa.feature.spectral_flatness(
        y=samples, hop_length=DEFAULT_HOP_LENGTH
    )[0]
    zcr_per_frame = librosa.feature.zero_crossing_rate(
        y=samples, hop_length=DEFAULT_HOP_LENGTH
    )[0]
    mfcc_per_frame = librosa.feature.mfcc(
        y=samples, sr=sample_rate, n_mfcc=13, hop_length=DEFAULT_HOP_LENGTH
    )

    frames_per_window = int(sample_rate / DEFAULT_HOP_LENGTH)

    rms_per_window = _mean_pool_per_window(
        rms_per_frame, frames_per_window, num_windows
    )
    centroid_per_window = _mean_pool_per_window(
        centroid_per_frame, frames_per_window, num_windows
    )
    flatness_per_window = _mean_pool_per_window(
        flatness_per_frame, frames_per_window, num_windows
    )
    zcr_per_window = _mean_pool_per_window(
        zcr_per_frame, frames_per_window, num_windows
    )
    mfcc_per_window = _mean_pool_per_window(
        mfcc_per_frame, frames_per_window, num_windows
    )

    return {
        "sample_rate": int(sample_rate),
        "duration_sec": float(duration_sec),
        "num_windows": int(num_windows),
        "rms": rms_per_window,
        "spectral_centroid": centroid_per_window,
        "spectral_flatness": flatness_per_window,
        "zcr": zcr_per_window,
        "mfcc": mfcc_per_window,
    }

def _smoke_test(wav_path):
    features = extract_features(wav_path)
    print(f"sample_rate={features['sample_rate']}")
    print(f"duration_sec={features['duration_sec']:.3f}")
    print(f"num_windows={features['num_windows']}")
    print(f"rms shape={features['rms'].shape}")
    print(f"spectral_centroid shape={features['spectral_centroid'].shape}")
    print(f"spectral_flatness shape={features['spectral_flatness'].shape}")
    print(f"zcr shape={features['zcr'].shape}")
    print(f"mfcc shape={features['mfcc'].shape}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python features.py <wav_path>", file=sys.stderr)
        sys.exit(1)
    _smoke_test(sys.argv[1])
