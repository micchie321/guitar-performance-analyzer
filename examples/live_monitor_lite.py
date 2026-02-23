import sys
import os
import time
import numpy as np
import pyaudio
import csv

# ==========================================
# ⚙️ パス設定
# ==========================================
# プロジェクトルートにパスを通す（どこから実行されてもsrc/extractor.pyを読み込めるようにする）
current_dir = os.path.dirname(os.path.abspath(__file__))
# もし examples/ フォルダ内に置く場合は ".."、ルートに置く場合は "." にします
project_root = os.path.join(current_dir, "..") 
sys.path.insert(0, project_root)

from src.extractor import GuitarFeatureExtractor

def main():
    print("🎸 Guitar Performance Analyzer - Live Monitor (CLI)")
    print("初期化中... (TensorFlowのロードに数秒かかります)")

    # config.yamlの絶対パスを解決
    config_path = os.path.join(project_root, "config.yaml")
    
    try:
        extractor = GuitarFeatureExtractor(config_path=config_path)
        print("✅ モデルとExtractorのロード完了")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return

    # ==========================================
    # 🔊 オーディオバッファとPyAudio設定
    # ==========================================
    sr = extractor.sr
    phrase_sec = extractor.cfg['analysis']['phrase_length_sec']
    buffer_size = int(phrase_sec * sr)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)

    # PyAudioのバックグラウンド録音用コールバック関数
    def audio_callback(in_data, frame_count, time_info, status):
        nonlocal audio_buffer
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        # バッファを左にシフトし、新しいデータを右端に追加
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
        print(f"❌ オーディオデバイスのエラー: {e}")
        p.terminate()
        return

    # ==========================================
    # 📝 CSV出力設定
    # ==========================================
    csv_filename = f"guitar_features_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
    csv_writer = None
    print(f"📝 解析結果を {csv_filename} に記録します。")

    # ==========================================
    # 🔄 メインループ
    # ==========================================
    print("🎤 録音を開始しました。終了するには Ctrl+C を押してください。")
    stream.start_stream()

    try:
        # 最初のバッファがたまるまで待機
        time.sleep(phrase_sec)

        while stream.is_active():
            loop_start = time.time()
            
            # 解析中に裏でバッファが書き換わるのを防ぐためコピーを取得
            current_buffer = audio_buffer.copy()
            
            # 35次元特徴量の抽出
            features = extractor.extract_all(current_buffer)

            # CSV書き込み (初回のみヘッダーを自動生成)
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=features.keys())
                csv_writer.writeheader()
            csv_writer.writerow(features)
            csv_file.flush() # 強制終了に備えて都度書き込み

            # コンソールへのリアルタイム表示更新
            sys.stdout.write("\033[K") # 行をクリア
            output = (f"Time: {time.strftime('%H:%M:%S')} | "
                      f"Dead:{features.get('tech_dead', 0.0):.2f} "
                      f"PM:{features.get('tech_pm', 0.0):.2f} "
                      f"Open:{features.get('tech_open', 0.0):.2f} | "
                      f"Changes:{int(features.get('tech_change_count', 0))}")
            print(f"\r{output}", end="", flush=True)

            # 処理時間を差し引いた正確なスリープ
            process_time = time.time() - loop_start
            sleep_time = max(0.0, phrase_sec - process_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 モニターを停止しました。")
    finally:
        # 終了処理 (リソースの解放)
        csv_file.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("✅ 終了処理が完了し、CSVが保存されました。")

if __name__ == "__main__":
    main()