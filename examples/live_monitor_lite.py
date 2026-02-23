import sys
import os
import time
import numpy as np
import pyaudio
import csv

# ==========================================
# Path Configuration
# ==========================================
# Add project root to path (so src/extractor.py can be imported from anywhere)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Use ".." if placed in examples/ folder, or "." if in root
project_root = os.path.join(current_dir, "..") 
sys.path.insert(0, project_root)

from src.extractor import GuitarFeatureExtractor

def main():
    print("Guitar Performance Analyzer - Live Monitor (CLI)")
    print("Initializing... (Loading TensorFlow may take a few seconds)")

    # Resolve absolute path to config.yaml
    config_path = os.path.join(project_root, "config.yaml")
    
    try:
        extractor = GuitarFeatureExtractor(config_path=config_path)
        print("Success: Model and Extractor loaded successfully.")
    except Exception as e:
        print(f"Error: Initialization failed: {e}")
        return

    # ==========================================
    # Audio Buffer and PyAudio Configuration
    # ==========================================
    sr = extractor.sr
    phrase_sec = extractor.cfg['analysis']['phrase_length_sec']
    buffer_size = int(phrase_sec * sr)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)

    # PyAudio callback function for background recording
    def audio_callback(in_data, frame_count, time_info, status):
        nonlocal audio_buffer
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        # Shift the buffer to the left and append new data to the right end
        audio_buffer = np.roll(audio_buffer, -len(audio_data))
        audio_buffer[-len(audio_data):] = audio_data
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sr,
                        input=True,
                        frames_per_buffer=2048,
                        stream_callback=audio_callback)
    except Exception as e:
        print(f"Error: Audio device error: {e}")
        p.terminate()
        return

    # ==========================================
    # CSV Output Configuration
    # ==========================================
    csv_filename = f"guitar_features_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
    csv_writer = None
    print(f"Info: Analysis results will be saved to {csv_filename}.")

    # ==========================================
    # Main Loop
    # ==========================================
    print("Recording started. Press Ctrl+C to stop.")
    stream.start_stream()

    try:
        # Wait until the initial buffer is filled
        time.sleep(phrase_sec)

        while stream.is_active():
            loop_start = time.time()
            
            # Copy buffer to prevent overwriting during analysis
            current_buffer = audio_buffer.copy()
            
            # Extract 35D features
            features = extractor.extract_all(current_buffer)

            # Write to CSV (Auto-generate header on first run)
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=features.keys())
                csv_writer.writeheader()
            csv_writer.writerow(features)
            csv_file.flush() # Flush to save data in case of unexpected termination

            # Update real-time console display
            sys.stdout.write("\033[K") # Clear the current line
            output = (f"Time: {time.strftime('%H:%M:%S')} | "
                      f"Dead:{features.get('tech_dead', 0.0):.2f} "
                      f"PM:{features.get('tech_pm', 0.0):.2f} "
                      f"Open:{features.get('tech_open', 0.0):.2f} | "
                      f"Changes:{int(features.get('tech_change_count', 0))}")
            print(f"\r{output}", end="", flush=True)

            # Accurate sleep accounting for processing time
            process_time = time.time() - loop_start
            sleep_time = max(0.0, phrase_sec - process_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        # Cleanup (Release resources)
        csv_file.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Cleanup complete. CSV saved successfully.")

if __name__ == "__main__":
    main()