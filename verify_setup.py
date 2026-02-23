import os
import sys
import time
import numpy as np

def run_tests():
    print("=== 🎸 Guitar Performance Analyzer - 動作確認テスト ===\n")
    
    # 1. パスの解決テスト
    print("[テスト 1/4] パスとモジュールの解決...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from src.extractor import GuitarFeatureExtractor
        print("  ✅ モジュールのインポートに成功しました。")
    except Exception as e:
        print(f"  ❌ インポートエラー: {e}")
        return False

    # 2. 設定ファイルとモデルのロードテスト
    print("\n[テスト 2/4] ExtractorとAIモデルの初期化...")
    try:
        config_path = os.path.join(current_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} が見つかりません。")
            
        extractor = GuitarFeatureExtractor(config_path=config_path)
        print("  ✅ config.yaml と TensorFlowモデルの読み込みに成功しました。")
    except Exception as e:
        print(f"  ❌ 初期化エラー: {e}")
        return False

    # 3. 擬似オーディオデータの生成
    print("\n[テスト 3/4] テスト用オーディオデータの生成...")
    try:
        sr = extractor.sr
        phrase_sec = extractor.cfg['analysis']['phrase_length_sec']
        
        # 4秒間のサイン波（A4 = 440Hz）とわずかなノイズを合成した擬似ギター音
        t = np.linspace(0, phrase_sec, int(sr * phrase_sec), False)
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t) # サイン波
        dummy_audio += np.random.normal(0, 0.05, dummy_audio.shape) # ホワイトノイズ
        dummy_audio = dummy_audio.astype(np.float32)
        print(f"  ✅ {phrase_sec}秒間 ({len(dummy_audio)} サンプル) の擬似データを生成しました。")
    except Exception as e:
        print(f"  ❌ データ生成エラー: {e}")
        return False

    # 4. 特徴量抽出の実行テスト (速度の検証)
    print("\n[テスト 4/4] 35次元特徴量の抽出とAI推論の実行 (速度検証)...")
    try:
        expected_keys = ["tech_dead", "tech_pm", "tech_open", "spectral_centroid", "tonal_stability", "dissonance"]
        
        # 初回のコンパイル遅延（コールドスタート）と2回目の速度を比較するため、2回ループします
        for i in range(1, 3):
            print(f"\n  --- 実行 {i}回目 ---")
            start_time = time.time()
            features = extractor.extract_all(dummy_audio)
            process_time = time.time() - start_time
            
            print(f"  ✅ 処理時間: {process_time:.3f} 秒")
            
            # 1回目の実行時のみ、中身の正当性をチェックして表示
            if i == 1:
                missing_keys = [k for k in expected_keys if k not in features]
                if missing_keys:
                    print(f"  ❌ 抽出されたデータに以下のキーが不足しています: {missing_keys}")
                    return False
                    
                print(f"  ✅ 合計 {len(features)} 次元の特徴量が正常に出力されました。")
                print("\n  [抽出データ サンプル]")
                print(f"   - Dead Note確率 : {features.get('tech_dead', 0.0):.3f}")
                print(f"   - Open Note確率 : {features.get('tech_open', 0.0):.3f}")
                print(f"   - 音の明るさ(Centroid): {features.get('spectral_centroid', 0.0):.1f} Hz")
                print(f"   - 安定性(Tonal Stability): {features.get('tonal_stability', 0.0):.3f}")
                print(f"   - 不協和度(Dissonance) : {features.get('dissonance', 0.0):.3f}")
                
    except Exception as e:
        print(f"  ❌ 抽出処理中にエラーが発生しました: {e}")
        return False

    print("\n🎉 すべてのテストをクリアしました！システムは完璧に動作しています！")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)