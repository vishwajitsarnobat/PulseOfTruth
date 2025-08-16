import os
import cv2
import glob
import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from tqdm import tqdm
import multiprocessing

# --- 1. CONFIGURATION ---
# List of directories containing your video files
real_video_directories = ['data_subset/real']
fake_video_directories = ['data_subset/fake']

# Path to save the output plots
plot_path = 'output/plots'

# Path to save the final CSV dataset
output_path = 'output/dataset_subset'

# --- Parallel Processing Settings ---
# Set the number of CPU cores to use.
# Using multiprocessing.cpu_count() - 1 is a safe choice to leave one core for the OS.
# Set to 1 to disable parallel processing.
NUM_WORKERS = multiprocessing.cpu_count() - 4
# --- END OF CONFIGURATION ---


# --- Constants ---
TARGET_FPS = 10.0
HEART_RATE_BPM_RANGE = [40, 240]

# --- Helper Functions (Signal Processing, etc.) ---

def get_face_bounding_box(frame, face_detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    if not results.detections: return None
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    return x, y, w, h

def divide_into_regions(bounding_box):
    x, y, w, h = bounding_box
    region_w, region_h = w // 3, h // 3
    regions = []
    for i in range(3):
        for j in range(3):
            regions.append((x + j * region_w, y + i * region_h, region_w, region_h))
    return regions

def extract_green_channel_mean(frame, region):
    rx, ry, rw, rh = region
    roi = frame[ry:ry+rh, rx:rx+rw]
    if roi.size == 0: return 0.0
    return np.mean(roi[:, :, 1])

def plot_rppg_signals(signals, timestamps, save_path):
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
    Processes a raw rPPG signal using a robust PEAK DETECTION method
    with an added SMOOTHING step to improve peak quality.
    """
    # 1. Detrending
    detrend_window_size = int(fps)
    if len(signal) <= detrend_window_size: return []
    moving_avg = np.convolve(signal, np.ones(detrend_window_size)/detrend_window_size, mode='valid')
    detrended_signal = signal[len(signal) - len(moving_avg):] - moving_avg

    # 2. Bandpass Filter
    nyquist = 0.5 * fps
    low = HEART_RATE_BPM_RANGE[0] / 60.0 / nyquist
    high = HEART_RATE_BPM_RANGE[1] / 60.0 / nyquist
    if low >= high or len(detrended_signal) < 16:
        return []
    b, a = butter(2, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)

    # --- START OF THE NEW STEP ---
    # 3. Smooth the signal to make peaks more prominent
    # A Savitzky-Golay filter is excellent for this.
    # We use a small window (e.g., 5 frames) and a low-order polynomial (e.g., 3)
    # The window_length must be an odd number.
    window_length = 5
    if len(filtered_signal) < window_length:
        return [] # Signal is too short to smooth
    smoothed_signal = savgol_filter(filtered_signal, window_length, polyorder=3)
    # --- END OF THE NEW STEP ---

    # 4. Find Peaks in the SMOOTHED Signal
    distance_between_peaks = (60 / HEART_RATE_BPM_RANGE[1]) * fps
    peaks, _ = find_peaks(smoothed_signal, distance=distance_between_peaks)

    if len(peaks) < 2:
        return []

    # 5. Calculate Inter-Beat Intervals (IBI) and convert to BPM
    inter_beat_intervals = np.diff(peaks) / fps
    hr_values = 60.0 / inter_beat_intervals

    # 6. Filter out implausible BPM values
    plausible_hr_values = hr_values[(hr_values >= HEART_RATE_BPM_RANGE[0]) & (hr_values <= HEART_RATE_BPM_RANGE[1])]
    
    return plausible_hr_values.tolist()

# --- Feature Calculation Functions ---

def calculate_feat1(hr_values):
    """Calculates overall variability (SDNN-inspired)."""
    if len(hr_values) < 2: return 0.0
    return np.std(np.array(hr_values))

def calculate_feat2(hr_values):
    """Calculates short-term jitter (RMSSD-inspired)."""
    if len(hr_values) < 2: return 0.0
    return np.sqrt(np.mean(np.square(np.diff(hr_values))))

def calculate_feat3(hr_values):
    """Calculates jitter consistency (SDSD-inspired)."""
    if len(hr_values) < 2: return 0.0
    return np.std(np.diff(hr_values))

# --- Main Video Analysis Pipeline ---

def analyze_single_video(video_path, face_detector, plot_save_path=None):
    """
    Analyzes one video file using the advanced CHROM method to extract a clean rPPG signal,
    then calculates all feat#1/2/3 scores.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0: return None
        frame_skip = max(1, int(original_fps / TARGET_FPS))

        # --- Data collection for all regions ---
        # Instead of just green, we now need all 3 color channels
        rgb_signals_per_region = [[[] for _ in range(3)] for _ in range(9)] # 9 regions, 3 channels (R,G,B)
        timestamps = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_skip == 0:
                bbox = get_face_bounding_box(frame, face_detector)
                if bbox:
                    regions = divide_into_regions(bbox)
                    for i, region in enumerate(regions):
                        rx, ry, rw, rh = region
                        roi = frame[ry:ry+rh, rx:rx+rw]
                        if roi.size == 0:
                            # Append 0 if region is empty to keep lists in sync
                            for c in range(3): rgb_signals_per_region[i][c].append(0)
                            continue
                        
                        # BGR to RGB, then get mean for each channel
                        avg_rgb = np.mean(roi, axis=(0, 1))
                        rgb_signals_per_region[i][0].append(avg_rgb[2]) # R
                        rgb_signals_per_region[i][1].append(avg_rgb[1]) # G
                        rgb_signals_per_region[i][2].append(avg_rgb[0]) # B

                    timestamps.append(frame_count / original_fps)
            frame_count += 1
        cap.release()

        if not timestamps: return None

        # --- CHROM Signal Processing ---
        final_rppg_signals = []
        l = len(timestamps) # Number of frames
        
        for region_idx in range(9):
            rgb_signal = np.array(rgb_signals_per_region[region_idx])
            
            # 1. Detrend the RGB signals
            win = int(TARGET_FPS) # 1-second window
            if l <= win: 
                final_rppg_signals.append(np.zeros(l))
                continue

            mean_color = np.mean(rgb_signal, axis=1)
            detrended_rgb = rgb_signal / (mean_color[:, None] + 1e-9) # Add epsilon to avoid division by zero

            # 2. Calculate Chrominance Signals (X_chrom, Y_chrom)
            X = 3 * detrended_rgb[0] - 2 * detrended_rgb[1]
            Y = 1.5 * detrended_rgb[0] + detrended_rgb[1] - 1.5 * detrended_rgb[2]
            
            # 3. Bandpass filter X and Y
            nyquist = 0.5 * TARGET_FPS
            low = HEART_RATE_BPM_RANGE[0] / 60.0 / nyquist
            high = HEART_RATE_BPM_RANGE[1] / 60.0 / nyquist
            b, a = butter(2, [low, high], btype='band')
            
            Xf = filtfilt(b, a, X)
            Yf = filtfilt(b, a, Y)

            # 4. Create the final rPPG signal (S)
            alpha = np.std(Xf) / (np.std(Yf) + 1e-9)
            S = Xf - alpha * Yf
            final_rppg_signals.append(S)

        if plot_save_path:
            # We plot the final processed CHROM signal now, not the raw green channel
            plot_rppg_signals(final_rppg_signals, timestamps, plot_save_path)

        # --- Comprehensive Feature Extraction (using the clean CHROM signals) ---
        scores = {}
        all_signals_np = np.array(final_rppg_signals)
        signal_sources = {f'r{i+1}': all_signals_np[i, :] for i in range(9)}
        signal_sources['avg'] = np.mean(all_signals_np, axis=0)

        for name, signal in signal_sources.items():
            hr_values = process_rppg_for_hr(signal, TARGET_FPS) # This function is now fed a much cleaner signal
            scores[f'{name}_feat1'] = calculate_feat1(hr_values)
            scores[f'{name}_feat2'] = calculate_feat2(hr_values)
            scores[f'{name}_feat3'] = calculate_feat3(hr_values)

        return {'path': video_path, **scores}
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# --- Worker function for parallel processing ---
def process_video_worker(args):
    """A wrapper function for multiprocessing to handle setup for each worker."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['MEDIAPIPE_DISABLE_GL_CONTEXT'] = '1'
    
    video_path, plot_save_path = args
    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    result = analyze_single_video(video_path, face_detector, plot_save_path)
    face_detector.close()
    return result

# --- Batch Processing Function ---
def run_batch_analysis(video_dirs, video_type, base_plot_path, base_output_path):
    """Processes all videos in a list of directories in parallel and saves the results."""
    plot_dir = os.path.join(base_plot_path, video_type)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(base_output_path, exist_ok=True)

    all_video_paths = []
    for directory in video_dirs:
        for ext in ('*.mp4', '*.mov', '*.avi', '*.mkv'):
            all_video_paths.extend(glob.glob(os.path.join(directory, ext)))
    print(f"\nFound {len(all_video_paths)} videos for category: '{video_type}'")
    if not all_video_paths: return

    tasks = []
    for video_path in all_video_paths:
        video_filename = os.path.basename(video_path)
        plot_save_path = os.path.join(plot_dir, f"{os.path.splitext(video_filename)[0]}.png")
        tasks.append((video_path, plot_save_path))

    results = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(tasks), desc=f"Processing {video_type} videos") as pbar:
            for result in pool.imap_unordered(process_video_worker, tasks):
                if result:
                    results.append(result)
                pbar.update(1)

    if results:
        csv_path = os.path.join(base_output_path, f'{video_type}.csv')
        df = pd.DataFrame(results)
        
        # Define the full, ordered list of columns for the CSV
        cols = ['path']
        prefixes = [f'r{i+1}' for i in range(9)] + ['avg']
        for prefix in prefixes:
            cols.extend([f'{prefix}_feat1', f'{prefix}_feat2', f'{prefix}_feat3'])
        
        df = df[cols] # Ensure columns are in the correct order
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved features for {len(results)} {video_type} videos to {csv_path}")

if __name__ == '__main__':
    # freeze_support() is necessary for multiprocessing on Windows/macOS
    multiprocessing.freeze_support()

    print(f"Starting batch analysis using {NUM_WORKERS} worker processes.")
    
    print("\n--- Processing REAL videos ---")
    run_batch_analysis(real_video_directories, 'real', plot_path, output_path)

    print("\n--- Processing FAKE videos ---")
    run_batch_analysis(fake_video_directories, 'fake', plot_path, output_path)
    
    print("\nBatch processing complete. Your training dataset is ready.")