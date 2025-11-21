import os
import glob
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple
import pathlib as Path
import matplotlib.pyplot as plt
import librosa.display

# =========================
# Configuration générale
# =========================

# --- Paths ---
INPUT_PATH = "Tasks/Task1/task1.wav"
GOOD_OUT = "Tasks/WmTest/audio_good_fm.wav" # Changed filename to reflect FM
BAD_OUT = "Tasks/WmTest/audio_bad.wav"

# --- BAD Watermark Parameters (Kept from original) ---
BAD_TONE_FREQ_HZ = 1000.0   # 1 kHz (audible)
BAD_TONE_SCALE = 0.05       # stronger

# --- GOOD FM Watermark Parameters (New, Inaudible) ---
FM_FC = 18500.0             # Carrier Frequency: High/Inaudible (>16kHz)
FM_MOD_F = 0.2              # Modulation Frequency: Very low (data rate, e.g., 5s cycle)
FM_DELTA_F = 1000.0         # Frequency Deviation: 1 kHz swing
FM_SCALE = 0.05             # Watermark Amplitude Scale

# =========================
# Task 1 : génération de watermarks
# =========================

def task1():
    os.makedirs(os.path.dirname(GOOD_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(BAD_OUT), exist_ok=True)

    # Note: Use a try/except here if INPUT_PATH might not exist
    try:
        audio, sr = sf.read(INPUT_PATH, always_2d=True)
    except Exception as e:
        print(f"[Task1 ERROR] Could not load input file {INPUT_PATH}: {e}")
        return

    audio = audio.astype(np.float64)
    print(f"[Task1] Loaded '{INPUT_PATH}' | sr={sr} Hz | shape={audio.shape}")

    n, ch = audio.shape
    t = np.arange(n, dtype=np.float64) / sr

    # ------------------------------------------------------------------
    # --- BAD watermark : sinus 1 kHz (Kept from original) ---
    # ------------------------------------------------------------------
    sine = np.sin(2 * np.pi * BAD_TONE_FREQ_HZ * t)[:, None]     # (n,1)
    sine = np.repeat(sine, ch, axis=1)                          # (n,ch)
    audio_bad = audio + BAD_TONE_SCALE * sine
    audio_bad = np.clip(audio_bad, -1.0, 1.0).astype(np.float32)
    sf.write(BAD_OUT, audio_bad, sr)
    print(f"[Task1] Wrote BAD watermark -> {BAD_OUT}")
    
    plot_spectrogram(np.abs(librosa.stft(audio_bad.mean(axis=1), n_fft=4096, hop_length=256)), sr, 256, "Tasks/plots/audio_bad_spectrogram.png", "Bad Watermarked Audio")

    # ------------------------------------------------------------------
    # --- GOOD watermark : FM tone (Replaces random noise) ---
    # ------------------------------------------------------------------
    
    # 1. Calculate Modulation Index (beta)
    # beta = Delta_f / Fm
    beta = FM_DELTA_F / FM_MOD_F
    
    # 2. Calculate the instantaneous phase phi(t)
    # This phase integrates the frequency function: 2*pi*FC*t + beta * sin(2*pi*FM*t)
    phase = (2 * np.pi * FM_FC * t) + (beta * np.sin(2 * np.pi * FM_MOD_F * t))
    
    # 3. Generate the watermark signal
    watermark_mono = np.cos(phase) * FM_SCALE
    
    # 4. Expand to stereo if needed
    watermark = watermark_mono[:, None] # (n,1)
    watermark = np.repeat(watermark, ch, axis=1) # (n,ch)

    # 5. Embed and save
    audio_good = audio + watermark
    audio_good = np.clip(audio_good, -1.0, 1.0).astype(np.float32)
    sf.write(GOOD_OUT, audio_good, sr)
    print(f"[Task1] Wrote GOOD FM watermark -> {GOOD_OUT}")
    plot_spectrogram(np.abs(librosa.stft(audio_good.mean(axis=1), n_fft=4096, hop_length=256)), sr, 256, "Tasks/plots/audio_good_spectrogram.png", "Good FM Watermarked Audio")


# =========================
# PARAMETERS
# =========================
# N_FFT: Window size for STFT
N_FFT = 16384 # Increased for better frequency resolution in analysis
# HOP: Hop length for STFT (frame shift)
HOP = 256
# FMIN, FMAX: Frequency band for high-frequency watermark extraction
FMIN = 16000.0
FMAX = 20000.0

# Directory where watermarked files are expected to be found
TASK2_DIR = "Tasks/Task2"
# Directory where the generated plots will be saved
PLOTS_DIR = "Tasks/plots"

# --- Classification Thresholds ---
# Assuming one group is high-freq FM (~18.5kHz), one is low-freq tone (~1kHz), and one is unknown.
FM_THRESHOLD_HZ = 10000.0  # Frequencies above this are considered high-frequency (FM)
MOD_THRESHOLD_HZ = 0.05    # Modulation frequencies above this are considered modulated

# ---------------------------------------
# PLOT SPECTROGRAM
# ---------------------------------------
def plot_spectrogram(S_mag: np.ndarray, sr: int, hop: int, output_path: str, title: str):
    """
    Generates, saves, and closes a full magnitude spectrogram plot.
    """
    # Convert magnitude spectrogram to dB for visualization
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(12, 6)) 
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='hz', ax=ax) 
    
    ax.set(title=f"Spectrogram of {title}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    plt.tight_layout()
    # Ensure directory exists before saving plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig) 

# ---------------------------------------
# 1) EXTRACT f0 (carrier frequency)
# ---------------------------------------
def get_f0(S_mag: np.ndarray, sr: int, n_fft: int, fmin: float, fmax: float) -> float:
    """
    Return the mean frequency (spectral centroid) inside the [fmin, fmax] band
    over the entire duration, approximating the carrier frequency f0.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # Use provided fmin/fmax arguments instead of globals
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[band_mask]

    if freqs_band.size == 0:
        return 0.0

    # Mean magnitude per bin over time
    mean_spectrum = np.mean(S_mag[band_mask, :], axis=1)

    total_energy = np.sum(mean_spectrum)
    if total_energy <= 1e-12:
        return float(np.mean(freqs_band))

    # Weighted mean frequency (spectral centroid within the band)
    mean_freq = np.sum(mean_spectrum * freqs_band) / total_energy
    return float(mean_freq)


# ---------------------------------------
# 2) TRACK f_peak(t) (Instantaneous frequency)
# ---------------------------------------
def track_peak(S_mag: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    """
    Tracks the instantaneous peak frequency of the watermark signal over time
    by finding the highest energy bin within the [FMIN, FMAX] band for each frame.
    
    Returns: The frequency trajectory (peak_freqs).
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fmin_bin = np.argmin(np.abs(freqs - FMIN))
    fmax_bin = np.argmin(np.abs(freqs - FMAX))
    
    band_mask = np.arange(fmin_bin, fmax_bin + 1)
    S_band = S_mag[band_mask, :]
    
    local_peak_indices = np.argmax(S_band, axis=0)
    global_peak_indices = band_mask[local_peak_indices]
    
    peak_freqs = freqs[global_peak_indices]
    
    return peak_freqs

# ---------------------------------------
# 3) EXTRACT fm (modulation frequency)
# ---------------------------------------
def get_fm(peak_freqs: np.ndarray, sr: int, hop: int) -> float:
    """
    Uses a standard FFT on the frequency trajectory (peak_freqs) to find the 
    dominant modulation frequency (fm).
    """
    frame_rate = float(sr) / float(hop)
    
    x = peak_freqs - np.mean(peak_freqs)
    
    if len(x) < 4:
         return 0.0

    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    
    freqs_mod = np.fft.rfftfreq(len(x), d=1.0 / frame_rate)
    
    mag_search = mag[1:]
    freqs_mod_search = freqs_mod[1:]
    
    if len(mag_search) == 0:
         return 0.0

    idx_local = np.argmax(mag_search)
    fm = freqs_mod_search[idx_local]
    
    return float(fm)


# ---------------------------------------
# EXTRACT ALL PARAMETERS FOR A FILE
# ---------------------------------------
def extract_parameters(path: str) -> Tuple[float, float, float]:
    """
    Reads an audio file, computes the spectrogram, and extracts 
    the carrier frequency (f0), the modulation frequency (fm), 
    and the frequency deviation (delta_f) of the watermark. It also saves the spectrogram plot.
    """
    try:
        audio, sr = sf.read(path)
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        return 0.0, 0.0, 0.0

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, window="hann")
    S_mag = np.abs(S)
    
    # --- PLOT SPECTROGRAM ---
    filename = os.path.basename(path).replace(".wav", "_spectrogram.png")
    plot_path = os.path.join(PLOTS_DIR, filename) 
    
    plot_spectrogram(S_mag, sr, HOP, plot_path, os.path.basename(path)) 
    # ------------------------

    # 1) Carrier Frequency (f0) - Search in high-frequency band [FMIN, FMAX]
    f0 = get_f0(S_mag, sr, N_FFT, FMIN, FMAX)

    # 2) Track modulation movement (Also in high-frequency band)
    peak_freqs = track_peak(S_mag, sr, N_FFT)
    
    modulation_signal = peak_freqs - np.mean(peak_freqs)
    delta_f = np.max(np.abs(modulation_signal))

    # 3) Extract fm
    fm = get_fm(peak_freqs, sr, HOP)

    return f0, fm, delta_f

# =========================
# NEW: Classification and Reconstruction
# =========================

def classify_audio(f0: float, fm: float, delta_f: float) -> str:
    """
    Classifies the audio into groups based on extracted FM parameters.
    """
    if f0 > FM_THRESHOLD_HZ:
        if delta_f > 100.0: # High deviation suggests active FM data encoding
            return "Group_A_HighFreq_FM"
        else:
            return "Group_B_HighFreq_Tone" # Pure tone/low deviation
    else:
        if fm > MOD_THRESHOLD_HZ:
             return "Group_C_LowFreq_Mod" # E.g., tone in audible range with slight modulation
        else:
            return "Group_D_LowFreq_Tone" # E.g., simple 1kHz tone or noise

def reconstruct_watermark(extracted_params: Dict[str, float], n: int, sr: int, ch: int, output_filepath: str):
    """
    Reconstructs the time-domain watermark based on extracted parameters 
    and saves its spectrogram.
    """
    f0 = extracted_params['f0']
    fm = extracted_params['fm']
    delta_f = extracted_params['delta_f']
    
    # Estimate a constant amplitude scale from a typical FM watermark (e.g., 0.05)
    # In a real scenario, this would also need to be extracted, but we use a fixed estimate.
    A_w = 0.05 
    
    t = np.arange(n) / sr
    
    # Calculate beta, ensuring we handle near-zero fm for pure tones
    if fm < MOD_THRESHOLD_HZ:
        # Case 1: Pure tone (f0)
        beta = 0.0
        phase = 2 * np.pi * f0 * t 
        group_name = "Recon_Tone"
    else:
        # Case 2: FM Modulated Tone
        beta = delta_f / fm
        # Ensure beta is not excessively large due to very small fm values
        beta = np.clip(beta, 0, 100000.0)
        phase = 2 * np.pi * f0 * t + beta * np.sin(2 * np.pi * fm * t)
        group_name = "Recon_FM"
        
    watermark_mono = A_w * np.cos(phase)
    
    # Expand to match channels (ch=1 for simplicity in most analysis)
    watermark = np.tile(watermark_mono[:, np.newaxis], (1, ch)).astype(np.float32)

    # Save Spectrogram of the Reconstructed Watermark
    S_mag = np.abs(librosa.stft(watermark_mono, n_fft=N_FFT, hop_length=HOP, window="hann"))
    
    plot_path = os.path.join(PLOTS_DIR, "reconstructed", f"{group_name}_{os.path.basename(output_filepath).replace('.wav', '.png')}")
    plot_spectrogram(S_mag, sr, HOP, plot_path, f"Reconstructed Watermark: {group_name} ({f0:.0f} Hz)")
    
    # We won't save the reconstructed audio itself, just the visualization.
    # sf.write(output_filepath, watermark, sr)


def task3_analysis():
    """
    Orchestrates the classification, extraction, and reconstruction for Task 3.
    """
    # 1. Setup
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PLOTS_DIR, "reconstructed"), exist_ok=True)
    
    search_path = os.path.join(TASK2_DIR, "*_watermarked.wav")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No watermarked files found in {TASK2_DIR}.")
        return

    print("=== Task 3: Watermark Classification and Reconstruction ===\n")
    results: List[Dict[str, Any]] = []

    # 2. Extract Parameters and Classify
    for path in files:
        f0, fm, delta_f = extract_parameters(path) 
        group = classify_audio(f0, fm, delta_f)
        
        results.append({
            'path': path,
            'name': os.path.basename(path),
            'f0': f0,
            'fm': fm,
            'delta_f': delta_f,
            'group': group
        })

    # 3. Grouping and Display
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    
    print("{:<25} {:>10} {:>10} {:>10} {:<20}".format("File", "f0 (Hz)", "fm (Hz)", "Df (Hz)", "Group"))
    print("-" * 75)
    
    for res in results:
        print("{:<25} {:>10.1f} {:>10.3f} {:>10.2f} {:<20}".format(
            res['name'], res['f0'], res['fm'], res['delta_f'], res['group']
        ))
        if res['group'] not in grouped_results:
            grouped_results[res['group']] = []
        grouped_results[res['group']].append(res)
        
    print("-" * 75)
    print("\n[INFO] Classification Complete. Starting Watermark Reconstruction and Plotting.")

    # 4. Reconstruction and Plotting (One visualization per group)
    reconstructed_groups = {}
    
    for group_name, group_list in grouped_results.items():
        if not group_list: continue

        # Use the parameters from the first file in the group for reconstruction
        sample = group_list[0]
        
        # Estimate properties for reconstruction (requires reading the audio length)
        try:
            audio, sr = sf.read(sample['path'])
        except Exception:
            continue
            
        n, ch = audio.shape if audio.ndim == 2 else (audio.size, 1)

        extracted_params = {'f0': sample['f0'], 'fm': sample['fm'], 'delta_f': sample['delta_f']}
        
        # Reconstruct the function and save its spectrogram
        reconstruct_watermark(
            extracted_params, 
            n, sr, ch, 
            output_filepath=f"reconstructed_{group_name}_{sample['name']}"
        )
        print(f"[RECON] Saved spectrogram for reconstructed function from {group_name}")

# ---------------------------------------
# TASK 3
# ---------------------------------------
TASK3_PATH1 = "Tasks/Task3/task3_watermarked_method1.wav"
TASK3_PATH2 = "Tasks/Task3/task3_watermarked_method2.wav"


def task3():
    #load both audio files
    try:
        audio1, sr1 = sf.read(TASK3_PATH1)
        audio2, sr2 = sf.read(TASK3_PATH2)
    except Exception as e:
        print(f"[Task3 ERROR] Could not load input files: {e}")
        
    # Convert to mono if stereo
    if audio1.ndim == 2:
        audio1 = audio1.mean(axis=1)
    if audio2.ndim == 2:
        audio2 = audio2.mean(axis=1)
        
    S1 = librosa.stft(audio1, n_fft=N_FFT, hop_length=HOP, window="hann")
    S2 = librosa.stft(audio2, n_fft=N_FFT, hop_length=HOP, window="hann")
    
    plot_spectrogram(np.abs(S1), sr1, HOP, "Tasks/plots/task3_method1_spectrogram.png", "Task 3 Method 1 Watermarked Audio")
    plot_spectrogram(np.abs(S2), sr2, HOP, "Tasks/plots/task3_method2_spectrogram.png", "Task 3 Method 2 Watermarked Audio")

    f0_1 = get_f0(S1, sr1, N_FFT, 4000, 5000)
    f0_2 = get_f0(S2, sr2, N_FFT, 15500, 20000)

    print(f"[Task3] Estimated main tone f0_1: {f0_1:.2f} Hz  ({os.path.basename(TASK3_PATH1)})")
    print(f"[Task3] Estimated main tone f0_2: {f0_2:.2f} Hz  ({os.path.basename(TASK3_PATH2)})")

    # lower one is time-domain slowed, higher is freq-domain
    f_low, f_high = sorted([f0_1, f0_2])
    x = f_high / f_low

    print(f"[Task3] Slowdown factor (f_high / f_low): {x:.4f}")



# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------
def main():
    # """
    # Locates watermarked files in TASK2_DIR and prints the extracted f0, fm, and delta_f.
    # It also ensures the plot output directory exists.
    # """
    # # Create plots directory if it doesn't exist
    # # Using os.makedirs is more reliable than pathlib.Path in some environments
    # try:
    #     os.makedirs(PLOTS_DIR, exist_ok=True)
    # except Exception as e:
    #     print(f"Error creating directory {PLOTS_DIR}: {e}")
    #     print("Please ensure you have write permissions for the target directory.")
    #     return
    
    # # Use glob to find all files ending in _watermarked.wav inside the directory
    # search_path = os.path.join(TASK2_DIR, "*_watermarked.wav")
    # files = sorted(glob.glob(search_path))
    
    # if not files:
    #     print(f"No watermarked files found in {TASK2_DIR}. Please ensure the path is correct.")
    #     return

    # print("=== Extracted f0, fm, and Delta_f for all files ===\n")
    # print("{:<25} {:>10} {:>10} {:>10}".format("File", "f0 (Hz)", "fm (Hz)", "Df (Hz)"))
    # print("-" * 55)

    # for path in files:
    #     f0, fm, delta_f = extract_parameters(path) 
    #     name = os.path.basename(path)
    #     print("{:<25} {:>10.1f} {:>10.3f} {:>10.2f}".format(name, f0, fm, delta_f))
        
    # print(f"\nAll spectrograms saved to the '{PLOTS_DIR}' directory.")
    
    # task1()  # Call task1 to generate watermarks
    
    task3()  # Call task3 to generate plots for task 3

if __name__ == "__main__":
    main()