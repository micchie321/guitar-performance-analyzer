import os
import sys
import glob
import argparse
import librosa
import pandas as pd
from tqdm import tqdm

# ==========================================
# ⚙️ パス設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.extractor import GuitarFeatureExtractor

def batch_process(input_dir, output_file):
    print(f"🎸 Guitar Performance Analyzer - Batch Processor")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ファイル: {output_file}")
    
    config_path = os.path.join(project_root, "config.yaml")
    
    try:
        extractor = GuitarFeatureExtractor(config_path=config_path)
        print("✅ モデルとExtractorのロード完了\n")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return

    phrase_sec = extractor.cfg['analysis']['phrase_length_sec']
    sr = extractor.sr

    # 指定ディレクトリおよびサブディレクトリ内のWAVをすべて検索
    search_pattern = os.path.join(input_dir, "**", "*.wav")
    input_files = glob.glob(search_pattern, recursive=True)
    
    if not input_files:
        print(f"⚠️ 指定されたディレクトリ ({input_dir}) に .wav ファイルが見つかりません。")
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
                
                # 指定秒数の半分未満の端数はスキップ
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
        print(f"\n✅ 処理完了! 合計 {len(all_results)} フレーズの特徴量を抽出しました。")
        print(f"📁 保存先: {os.path.abspath(output_file)}")
    else:
        print("\n⚠️ 抽出可能なフレーズデータがありませんでした。")

    if error_files:
        print(f"\n⚠️ 以下の {len(error_files)} ファイルでエラーが発生し、スキップされました:")
        for err_file, err_msg in error_files:
            print(f"  - {err_file}: {err_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 35-dimensional features from a directory of WAV files.")
    parser.add_argument("-i", "--input", type=str, default=".", help="Directory containing input .wav files")
    parser.add_argument("-o", "--output", type=str, default="analysis_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    batch_process(args.input, args.output)