import librosa
import numpy as np
import scipy.stats

def find_mp4_audio_boundaries(video_path, n_segments=6):
    # 1. Load the audio stream directly from the MP4
    # Librosa uses audioread/ffmpeg under the hood for video containers
    y, sr = librosa.load(video_path, sr=None) 
    duration = librosa.get_duratio
    
    # 2. Extract Timbral Features (MFCCs)
    # We use 20 MFCCs to capture the "texture" of the scene
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # 3. Standardize the features (important for varied video clips)
    # This prevents volume spikes from skewing the segmentation
    mfcc_scaled = scipy.stats.zscore(mfcc, axis=1)
    
    # 4. Agglomerative Clustering for Segmentation
    # This groups "like-sounding" time blocks together
    # Use 'ward' linkage to minimize variance within each scene segment
    boundaries = librosa.segment.agglomerative(mfcc_scaled, n_segments)
    
    # 5. Convert frame indices to timestamps
    boundary_times = librosa.frames_to_time(boundaries, sr=sr)
    
    print(f"--- Audio Analysis for: {video_path} ---")
    for i, t in enumerate(boundary_times):
        # We skip the very first boundary (0.0s)
        if i == 0: continue
        print(f"Acoustic Shift {i}: {t:.3f} seconds")

def find_dynamic_boundaries(video_path, seconds_per_segment=5):
    # 1. Load audio and get total duration
    y, sr = librosa.load(video_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 2. Calculate n_segments: Duration / your desired interval
    # We use max(2, ...) to ensure at least one split exists
    n_segments = max(2, int(duration / seconds_per_segment))
    
    print(f"Video Duration: {duration:.2f}s | Targeted Segments: {n_segments}")

    # 3. Extract Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_scaled = scipy.stats.zscore(mfcc, axis=1)
    
    # 4. Run the Cluster-based Segmentation
    # The 'axis=1' tells it to cluster over the time dimension
    boundaries = librosa.segment.agglomerative(mfcc_scaled, n_segments)
    
    # 5. Convert to time
    boundary_times = librosa.frames_to_time(boundaries, sr=sr)
    
    return boundary_times[1:] # Drop the 0.0s start point


# Example usage
video_file = "/users/a/l/alshen/VideoTree/VideoTree/data/VideoMME/data/_aVHf_jmWk8.mp4"

cuts = find_dynamic_boundaries(video_file, seconds_per_segment=5)
print(cuts)
