# Guitar Performance Analyzer

A Python library and toolkit for the **analysis, semantic labeling, and vectorization of guitar performances**. 
It integrates the physical characteristics of audio signals with playing techniques (emotional indicators) to extract them as a comprehensive 35-dimensional feature vector.

## Key Features
* **35-Dimensional Semantic Vectorization:** Extracts high-level features classified into 4 categories: Technique, Timbre, Tonal, and Rhythm.
* **Hybrid Onset Detection:** Combines real-time instantaneous attack detection with a rolling buffer for precise statistical and semantic analysis.
* **Ready-to-use CLI Tools:** Includes a Real-time Monitor for live performance tracking and a Batch Processor for massive dataset generation.

## The 35-Dimensional Feature Vector Breakdown
The core extractor analyzes phrases and outputs a 35D dictionary containing the following metrics:

1. **Technique (4 dimensions):** Ratios of playing techniques in the phrase (`tech_dead`, `tech_pm`, `tech_open`) and the number of technique changes (`tech_change_count`).
2. **Timbre (17 dimensions):** Physical sound qualities including `spectral_centroid` (brightness), `spectral_rolloff` (sharpness), `zcr` (noisiness), `mid_scoop`, and `mfcc_1` to `mfcc_13` (vocal tract/body resonance).
3. **Tonal (7 dimensions):** Pitch and harmonic relationships such as `polyphony_score`, `dissonance`, `pitch_variance`, `tonal_stability`, and `minorness`.
4. **Rhythm (7 dimensions):** Timing and groove indicators including `complexity`, `rhythm_ratio_average` / `rhythm_ratio_std` (Inter-Onset Interval variability), `tempo`, `note_density`, and `articulation`.

### Application to Emotional Metrics (Valence-Arousal)
The 35D features provided by this library are designed based on "Mid-level features" proposed in music emotion recognition research (e.g., Kirke & Miranda, 2013). 
By applying regression models or weighted calculations, users can estimate the emotional state of a performance:
* **Arousal (Energy):** Highly correlated with `spectral_centroid`, `note_density`, and `tempo`.
* **Valence (Positivity):** Derivable from the balance of `tonal_stability`, `minorness`, and `dissonance`.

## Research Reliability & Practical Performance
This project is based on research from Digital Hollywood University Graduate School. 

### 1. Model Accuracy
The playing technique identification model (Open/Palm Mute/Dead Note) has been evaluated using a dataset derived from YouTube audio sources.
- **Test Accuracy:** 90.97%
- **Macro-F1 Score:** 0.910
- **Model:** CNN (Convolutional Neural Network) with Mel-spectrogram input.

### 2. Real-world Performance & Tips
While the model shows high accuracy on test datasets, performance in a live environment can vary depending on your guitar's pickups, strings, and interface gain.
- **Optimization Tips:**
    - **Gain Staging:** Ensure your input signal is loud enough but **never clipping**.
    - **Exaggerate Nuance:** For the most stable 35D vector extraction, try to emphasize the difference between Open, Mute, and Dead notes during performance.
- **Data Efficiency:** By vectorizing performance, data size is reduced to approximately **1/315** compared to raw PCM audio, making it ideal for long-term logging.

### Future Roadmap for Practical Musical Application
Currently, the analyzer extracts features using a fixed-length rolling window (e.g., 4.0 seconds). To increase practical utility for music production and semantic analysis, the following structural enhancements are planned for future implementation:
* **Tempo-Synced Windowing:** Dynamically determining the phrase length based on the detected BPM rather than a fixed time duration.
* **Downbeat Alignment:** Aligning the start point of the extraction window with the musical downbeat (the beginning of a measure) to ensure that the extracted features accurately represent a coherent musical phrase.

## Configuration (`config.yaml`)
You can easily customize the analysis behavior by editing the `config.yaml` file:
* **Audio Setup:** Change the sample rate (`sr`) and the length of the analysis window (`phrase_length_sec`).
* **Onset Detection Sensitivity:** Adjust parameters like `onset_backtrack_ms` and thresholds.
* **Model Paths:** Specify the location of the pre-trained `.h5` TensorFlow model.

## Installation
```bash
pip install -r requirements.txt
python verify_setup.py
```

## Usage
* **Real-time Live Monitor:** `python examples/live_monitor_lite.py`
* **Batch Processor:** `python examples/batch_process.py -i ./my_dataset -o ./results/feature_dataset.csv`

---

# Guitar Performance Analyzer (日本語)

ギター演奏の物理的特徴と奏法（感性指標）を統合し、演奏表現を「35次元の特徴量ベクトル」として意味的ラベリング（ベクトル化）する解析ライブラリです。

## 35次元特徴量の内訳
コアエンジンはフレーズを解析し、以下の指標を含む35次元のデータを出力します。

1. **奏法 / Technique (4次元):** 各奏法の割合（`tech_dead`, `tech_pm`, `tech_open`）および奏法の変化回数。
2. **音色 / Timbre (17次元):** 音の明るさ（`spectral_centroid`）、鋭さ（`spectral_rolloff`）、楽器の響きを表す `mfcc_1` 〜 `mfcc_13` など。
3. **音程 / Tonal (7次元):** 和音の複雑さ（`polyphony_score`）、不協和度（`dissonance`）、調的安定性（`tonal_stability`）、マイナー感（`minorness`）など。
4. **リズム / Rhythm (7次元):** リズムの複雑度（`complexity`）、テンポ（`tempo`）、音符の密度（`note_density`）、アーティキュレーション（`articulation`）、`rhythm_ratio_average` など。

### 感性指標（Valence-Arousal）への応用について
本ライブラリの特徴量は、先行研究（Kirke & Miranda等）で提唱されている「感性推定に寄与する中間特徴量」をベースに設計されています。 
重み付け計算や機械学習を用いることで、演奏の感情価（Valence）や覚醒度（Arousal）を推定することが可能です。

## 研究の信頼性と実運用上の留意点
本プロジェクトは、デジタルハリウッド大学大学院における研究成果に基づいています。

### 1. 識別モデルの精度
奏法識別モデルは、YouTube音源から抽出された約10,000サンプルのデータセットで評価されています。
- **テスト正解率:** 90.97%
- **Macro-F1 スコア:** 0.910
- **手法:** Melスペクトログラムを入力とするCNNを採用。

### 2. 実演奏における性能と使いこなしのヒント
実際の生演奏では、使用機器や録音環境によって精度が変動する場合があります。
- **安定させるためのコツ:**
    - **適切なゲイン設定:** 入力信号が十分な大きさを持ちつつ、音割れ（クリッピング）しないように調整してください。
    - **ニュアンスの強調:** 奏法の違いを意識的に強調して弾くことで、より安定した特徴ベクトルが抽出されます。
- **データ効率:** 演奏のベクトル化により、生波形データと比較してデータ量を約 1/315 に圧縮。大規模な演奏ログの蓄積に最適化されています。

### 今後の展望：より実践的な音楽解析に向けて
現在の仕様では、固定長（例：4.0秒間隔）のローリングウィンドウで特徴量を抽出していますが、音楽制作やデータ分析における実用性をさらに高めるため、以下のような拡張を視野に入れています。
* **テンポに同期したフレーズ長の決定:** 固定の秒数ではなく、検出されたテンポ（BPM）に基づいて、解析するフレーズの長さを動的に決定するアプローチ。
* **小節の頭（ダウンビート）への同期:** 解析の開始地点を「小節の頭」に自動で合わせることで、抽出されたベクトルが「意味のあるひとつの音楽的フレーズ」をより正確に表現できるようにする設計。

## 設定ファイルの編集 (`config.yaml`)
`config.yaml` を編集することで、解析の挙動をカスタマイズできます。
* **オーディオ設定:** サンプリングレート (`sr`) や、解析単位となるフレーズ長 (`phrase_length_sec`) の変更。
* **CNNロジック:** 使用する学習済みモデル (`model_path`) の指定、およびモデル入力用の音声切り出し設定。
* **オンセット検出:** 検出感度 (`threshold`) や帯域制限 (`fmin`, `fmax`) の調整。
* **音色特徴:** MFCCの次元数 (`n_mfcc`) や Mid-Scoop（中音域の削れ具合）の計算帯域の設定。

## インストールと使用方法
* **インストール:** `pip install -r requirements.txt`
* **動作確認:** `python verify_setup.py`
* **リアルタイム・モニター:** `python examples/live_monitor_lite.py`
* **バッチ処理:** `python examples/batch_process.py -i ./input_dir -o ./result.csv`