import os
import cv2
import glob
import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow/MediaPipe warnings
os.environ['MEDIAPIPE_DISABLE_GL_CONTEXT'] = '3' # Suppリ Suppress GL context messages

# --- 1. CONFIGURATION ---
# List of directories containing your video files
real_video_directories = ['data_subset/real']
fake_video_directories = ['data_subset/fake']

# Path to save the output plots
plot_path = 'output/plots'

# Path to save the output CSV files with scores
output_path = 'output/scores'

# --- Parallel Processing Settings ---
# Set the number of CPU cores to use.
# Using multiprocessing.cpu_count() - 1 is a safe choice to leave one core for the OS.
# Set to 1 to disable parallel processing.
NUM_WORKERS = multiprocessing.cpu_count() - 4
# --- END OF CONFIGURATION ---


# --- Constants ---
TARGET_FPS = 30.0
HEART_RATE_BPM_RANGE = [40, 240]  # Min and Max heart rate in BPM

# --- Helper Functions (Signal Processing, etc.) ---

def get_face_bounding_box(frame, face_detector):
    """Detects a face in the frame and returns its bounding box."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    if not results.detections: return None
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    return x, y, w, h

def divide_into_regions(bounding_box):
    """Divides a bounding box into a 3x3 grid."""
    x, y, w, h = bounding_box
    region_w, region_h = w // 3, h // 3
    regions = []
    for i in range(3):
        for j in range(3):
            regions.append((x + j * region_w, y + i * region_h, region_w, region_h))
    return regions

def extract_green_channel_mean(frame, region):
    """Extracts the mean green channel intensity from a region."""
    rx, ry, rw, rh = region
    roi = frame[ry:ry+rh, rx:rx+rw]
    if roi.size == 0: return 0.0
    return np.mean(roi[:, :, 1])

def plot_rppg_signals(signals, timestamps, save_path):
    """Plots the 9 rPPG signals and saves the figure to a file."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    fig.suptitle('rPPG Signals from 9 Facial Regions (GREEN Method)', fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.plot(timestamps, signals[i])
        ax.set_title(f'Region {i+1} (R{i+1})')
        ax.grid(True, linestyle='--', alpha=0.6)
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Average Green Channel Intensity', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.07, 0.05, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

def process_rppg_for_hr(signal, fps):
    """
    Processes a raw rPPG signal to estimate a time-series of heart rate values.
    Includes safety checks to prevent crashes on short signals.
    """
    # 1. Detrending
    detrend_window_size = int(fps) # 1-second moving average
    if len(signal) <= detrend_window_size: return []
    moving_avg = np.convolve(signal, np.ones(detrend_window_size)/detrend_window_size, mode='valid')
    detrended_signal = signal[len(signal) - len(moving_avg):] - moving_avg

    # 2. Bandpass Filter (with safety check)
    # The filter requires the signal to be longer than its padding length.
    # We establish a minimum meaningful length based on the HR analysis window.
    hr_window_size = int(5 * fps) # 5-second window for HR calculation
    if len(detrended_signal) < hr_window_size:
        return [] # Signal is too short for a reliable HR calculation

    nyquist = 0.5 * fps
    low = HEART_RATE_BPM_RANGE[0] / 60.0 / nyquist
    high = HEART_RATE_BPM_RANGE[1] / 60.0 / nyquist
    if low >= high: return []
    b, a = butter(2, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)

    # 3. Estimate HR using FFT on a sliding window
    hr_values = []
    step_size = int(1 * fps) # 1-second step
    for i in range(0, len(filtered_signal) - hr_window_size + 1, step_size):
        window = filtered_signal[i:i + hr_window_size]
        fft_data = np.fft.rfft(window)
        fft_freq = np.fft.rfftfreq(len(window), 1.0 / fps)
        valid_indices = np.where((fft_freq >= HEART_RATE_BPM_RANGE[0]/60.0) & (fft_freq <= HEART_RATE_BPM_RANGE[1]/60.0))
        if len(valid_indices[0]) == 0: continue
        peak_idx = valid_indices[0][np.argmax(np.abs(fft_data[valid_indices]))]
        hr_values.append(fft_freq[peak_idx] * 60.0)
    return hr_values

def calculate_feat1(hr_values):
    """Calculates the feat#1 metric."""
    if not hr_values or len(hr_values) < 2: return 0.0
    return np.std(np.array(hr_values))

# --- Main Video Analysis Pipeline ---

def analyze_single_video(video_path, face_detector, plot_save_path=None):
    """
    Analyzes one video file to extract rPPG signals and calculate feat#1 scores.
    Returns a dictionary of scores. Returns None if face is not detected or processing fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0: return None
        frame_skip = max(1, int(original_fps / TARGET_FPS))

        rppg_signals, timestamps = [[] for _ in range(9)], []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_skip == 0:
                bbox = get_face_bounding_box(frame, face_detector)
                if bbox:
                    regions = divide_into_regions(bbox)
                    for i, region in enumerate(regions):
                        rppg_signals[i].append(extract_green_channel_mean(frame, region))
                    timestamps.append(frame_count / original_fps)
            frame_count += 1
        cap.release()

        if not timestamps: return None

        if plot_save_path:
            plot_rppg_signals(rppg_signals, timestamps, plot_save_path)

        scores = {}
        all_signals_np = np.array(rppg_signals)
        for i in range(9):
            hr_values = process_rppg_for_hr(all_signals_np[i, :], TARGET_FPS)
            scores[f'r{i+1}'] = calculate_feat1(hr_values)
        avg_signal = np.mean(all_signals_np, axis=0)
        hr_values_avg = process_rppg_for_hr(avg_signal, TARGET_FPS)
        scores['avg'] = calculate_feat1(hr_values_avg)
        
        return {'path': video_path, **scores}
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# --- Worker function for parallel processing ---
def process_video_worker(args):
    """
    A wrapper function for multiprocessing. Each worker initializes its own face detector.
    """
    video_path, plot_save_path = args
    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    result = analyze_single_video(video_path, face_detector, plot_save_path)
    face_detector.close()
    return result

# --- Batch Processing Function ---
def run_batch_analysis(video_dirs, video_type, base_plot_path, base_output_path):
    """
    Processes all videos in a list of directories in parallel and saves the results.
    """
    plot_dir = os.path.join(base_plot_path, video_type)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(base_output_path, exist_ok=True)

    all_video_paths = []
    for directory in video_dirs:
        for ext in ('*.mp4', '*.mov', '*.avi', '*.mkv'):
            all_video_paths.extend(glob.glob(os.path.join(directory, ext)))
    print(f"\nFound {len(all_video_paths)} videos for category: '{video_type}'")
    if not all_video_paths: return

    # Prepare arguments for each worker process
    tasks = []
    for video_path in all_video_paths:
        video_filename = os.path.basename(video_path)
        plot_save_path = os.path.join(plot_dir, f"{os.path.splitext(video_filename)[0]}.png")
        tasks.append((video_path, plot_save_path))

    # Run tasks in parallel
    results = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(tasks), desc=f"Processing {video_type} videos") as pbar:
            for result in pool.imap_unordered(process_video_worker, tasks):
                if result:
                    results.append(result)
                pbar.update(1)

    # Save scores to CSV
    if results:
        csv_path = os.path.join(base_output_path, f'{video_type}.csv')
        df = pd.DataFrame(results)
        cols = ['path', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'avg']
        df = df[cols]
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved scores for {len(results)} {video_type} videos to {csv_path}")

if __name__ == '__main__':
    print(f"Starting batch analysis using {NUM_WORKERS} worker processes.")
    
    print("\n--- Processing REAL videos ---")
    run_batch_analysis(real_video_directories, 'real', plot_path, output_path)

    print("\n--- Processing FAKE videos ---")
    run_batch_analysis(fake_video_directories, 'fake', plot_path, output_path)
    
    print("\nBatch processing complete.")