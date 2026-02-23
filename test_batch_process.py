import os
import sys
import subprocess
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import time

# 出力先をデスクトップに指定
TEST_OUTPUT_CSV = os.path.join(os.path.expanduser("~"), "Desktop", "test_batch_results.csv")
TEST_INPUT_DIR = os.path.join(os.getcwd(), "test_wav_input")

def create_dummy_wav(filename, duration=5.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = (audio * 32767).astype(np.int16)
    wav.write(filename, sr, audio)

def run_end_to_end_test():
    print("=== 🧪 batch_process.py エンドツーエンドテスト ===\n")

    if not os.path.exists(TEST_INPUT_DIR):
        os.makedirs(TEST_INPUT_DIR)
    
    create_dummy_wav(os.path.join(TEST_INPUT_DIR, "test_1.wav"))
    create_dummy_wav(os.path.join(TEST_INPUT_DIR, "test_2.wav"))
    print(f"✅ テスト用WAVを作成しました: {TEST_INPUT_DIR}")

    # ★ 修正ポイント: テストスクリプト自身の絶対パスを取得して、そこから辿る
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_script_dir, "examples", "batch_process.py")

    # 実際にユーザーが実行するCLIコマンドをシミュレート
    cmd = [
        sys.executable,  # 現在のPythonのパス
        script_path,
        "-i", TEST_INPUT_DIR,
        "-o", TEST_OUTPUT_CSV
    ]

    print(f"🚀 コマンドライン実行をテストします:\n > {' '.join(cmd)}\n")
    
    try:
        # cwd を指定して、プロジェクトルートで実行しているのと同じ状態を作る
        result = subprocess.run(cmd, check=True, text=True, cwd=current_script_dir)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ バッチスクリプトの実行中にエラーが発生しました。終了コード: {e.returncode}")
        return

    time.sleep(1) # ファイル書き込み待機
    
    # CSVの検証
    print("\n=== 📊 出力結果の検証 ===")
    if os.path.exists(TEST_OUTPUT_CSV):
        print(f"✨ 成功！デスクトップにCSVファイルが生成されました。")
        df = pd.read_csv(TEST_OUTPUT_CSV)
        print(f"✅ データ行数: {len(df)} 行 (期待値: 4行)")
        print(f"✅ カラム数: {len(df.columns)}")
        print("\n[サンプルデータ]")
        print(df[["file_name", "start_sec", "tech_open", "spectral_centroid"]].head(2))
    else:
        print(f"⚠️ コマンドは成功しましたが、CSVファイルが見つかりません: {TEST_OUTPUT_CSV}")

if __name__ == "__main__":
    run_end_to_end_test()