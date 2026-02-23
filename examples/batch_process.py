import os
import sys
import glob
import argparse
import librosa
import pandas as pd
from tqdm import tqdm

# ==========================================
# Path Configuration
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.extractor import GuitarFeatureExtractor

def batch_process(input_dir, output_file):
    print("Guitar Performance Analyzer - Batch Processor")
    print(f"Input Directory: {input_dir}")
    print(f"Output File: {output_file}")
    
    config_path = os.path.join(project_root, "config.yaml")
    
    try:
        extractor = GuitarFeatureExtractor(config_path=config_path)
        print("Success: Model and Extractor loaded successfully.\n")
    except Exception as e:
        print(f"Error: Initialization failed: {e}")
        return

    phrase_sec = extractor.cfg['analysis']['phrase_length_sec']
    sr = extractor.sr

    # Search for all WAV files in the target directory and subdirectories
    search_pattern = os.path.join(input_dir, "**", "*.wav")
    input_files = glob.glob(search_pattern, recursive=True)
    
    if not input_files:
        print(f"Warning: No .wav files found in the specified directory ({input_dir}).")
        return

    all_results = []
    error_files = []

    for file_path in tqdm(input_files, desc="Processing Audio Files"):
        file_name = os.path.basename(file_path)
        
        try:
            y, _ = librosa.load(file_path, sr=sr)
            samples_per_phrase = int(phrase_sec * sr)
            
            for i in range(0, len(y), samples_per_phrase):
                phrase_y = y[i : i + samples_per_phrase]
                
                # Skip fragments shorter than half of the specified window length
                if len(phrase_y) < samples_per_phrase * 0.5:
                    continue

                features = extractor.extract_all(phrase_y)
                features["file_name"] = file_name
                features["start_sec"] = round(i / sr, 3)
                all_results.append(features)
                
        except Exception as e:
            error_files.append((file_name, str(e)))

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["file_name", "start_sec"] + [c for c in df.columns if c not in ["file_name", "start_sec"]]
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df[cols].to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nProcessing complete. Extracted features from a total of {len(all_results)} phrases.")
        print(f"Saved to: {os.path.abspath(output_file)}")
    else:
        print("\nWarning: No valid phrase data could be extracted.")

    if error_files:
        print(f"\nWarning: Errors occurred in the following {len(error_files)} files and were skipped:")
        for err_file, err_msg in error_files:
            print(f"  - {err_file}: {err_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 35-dimensional features from a directory of WAV files.")
    parser.add_argument("-i", "--input", type=str, default=".", help="Directory containing input .wav files")
    parser.add_argument("-o", "--output", type=str, default="analysis_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    batch_process(args.input, args.output)