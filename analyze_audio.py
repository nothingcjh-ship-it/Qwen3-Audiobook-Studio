
import os
import soundfile as sf
import numpy as np

def analyze_audio(file_path, target_time=9.0, window=0.5):
    print(f"Analyzing {file_path} @ {target_time}s...")
    
    try:
        data, sr = sf.read(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    duration = len(data) / sr
    print(f"Duration: {duration:.2f}s")
    
    if target_time > duration:
        print("Target time exceeds duration.")
        return

    start_sample = int((target_time - window) * sr)
    end_sample = int((target_time + window) * sr)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(data), end_sample)
    
    segment = data[start_sample:end_sample]
    
    # Analyze
    max_amp = np.max(np.abs(segment))
    rms = np.sqrt(np.mean(segment**2))
    
    print(f"Segment ({target_time-window}s - {target_time+window}s):")
    print(f"  Max Amplitude: {max_amp:.4f}")
    print(f"  RMS Power: {rms:.4f}")
    
    # Simple anomaly detection
    if max_amp > 0.95:
        print("  WARNING: Possible Clipping detected!")
    elif max_amp < 0.001:
        print("  WARNING: Silence detected (possible dropout).")
    else:
        print("  Status: Normal levels range.")
        
    # Check for sudden DC offset (pop/click)
    mean_amp = np.mean(segment)
    print(f"  DC Offset: {mean_amp:.4f}")
    
if __name__ == "__main__":
    analyze_audio("outputs/audiobook_20260131_171818/final_audiobook.wav")
