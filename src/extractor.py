import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import yaml
from scipy.stats import entropy

class GuitarFeatureExtractor:
    """
    Core class for extracting 35-dimensional (variable) features from guitar performances.
    """
    def __init__(self, config_path="config.yaml"):
        import os
        print(f"Debug: Loading config from {config_path}...")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        # Build absolute path for the model
        conf_dir = os.path.dirname(os.path.abspath(config_path))
        model_rel_path = self.cfg['cnn_logic']['model_path']
        model_path = os.path.abspath(os.path.join(conf_dir, model_rel_path))
        
        print(f"Debug: Attempting to load model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error: Model file not found: {model_path}")

        print("Debug: Loading TensorFlow model (This may take a few seconds)...")
        self.model = tf.keras.models.load_model(model_path)
        print("Debug: Model loaded successfully.")

        self.sr = self.cfg['analysis']['sr']
        self.n_mfcc = self.cfg['timbre']['n_mfcc']

    def extract_all(self, y):
        """
        Extracts all features from the given audio waveform and returns them as a dictionary.
        """
        # A. Onset Detection (Bandpass 200Hz - 8000Hz)
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, 
            fmin=self.cfg['onset']['fmin'], 
            fmax=self.cfg['onset']['fmax']
        )
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
        
        # B. Feature Extraction per Category
        tech_feats = self._get_technique_features(y, onset_frames)
        timbre_feats = self._get_timbre_features(y)
        tonal_feats = self._get_tonal_features(y)
        rhythm_feats = self._get_rhythm_features(y, onset_env, onset_frames)

        # Integration
        return {**tech_feats, **timbre_feats, **tonal_feats, **rhythm_feats}

    def _get_technique_features(self, y, onset_frames):
        """Calculate technique ratios and change counts (150ms Active + 350ms Padding)"""
        onset_samples = librosa.frames_to_samples(onset_frames)
        backtrack = int(self.cfg['cnn_logic']['backtrack_sec'] * self.sr)
        active_len = int(self.cfg['cnn_logic']['active_audio_sec'] * self.sr)
        total_len = int(self.cfg['cnn_logic']['total_input_sec'] * self.sr)
        
        labels = []
        for s in onset_samples:
            start = max(0, s - backtrack)
            # Extract 150ms
            chunk = y[start : start + active_len]
            # Pad with zeros for 350ms (Total 0.5s)
            padded = np.pad(chunk, (0, max(0, total_len - len(chunk))))[:total_len]
            
            # Mel-spectrogram conversion and normalization
            mels = librosa.feature.melspectrogram(y=padded, sr=self.sr, n_mels=128, hop_length=512)
            X = (librosa.power_to_db(mels, ref=np.max) + 80.0) / 80.0
            p = self.model.predict(X[np.newaxis, ..., np.newaxis], verbose=0)[0]
            labels.append(np.argmax(p))

        total = len(labels) if len(labels) > 0 else 1
        return {
            "tech_dead": labels.count(0) / total,
            "tech_pm": labels.count(1) / total,
            "tech_open": labels.count(2) / total,
            "tech_change_count": int(np.sum(np.diff(labels) != 0)) if len(labels) > 1 else 0
        }

    def _get_timbre_features(self, y):
        """Timbre-related features (Variable MFCC dimensions)"""
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Mid-scoop (Low + High / Mid)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_idx = freqs < self.cfg['timbre']['mid_scoop_low']
        high_idx = freqs >= self.cfg['timbre']['mid_scoop_high']
        mid_idx = ~(low_idx | high_idx)
        mid_scoop = (np.sum(S[low_idx]) + np.sum(S[high_idx])) / (np.sum(S[mid_idx]) + 1e-9)

        feats = {
            "spectral_centroid": float(centroid),
            "spectral_rolloff": float(rolloff),
            "zcr": float(zcr),
            "mid_scoop": float(mid_scoop)
        }
        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc), axis=1)
        for i, val in enumerate(mfccs):
            feats[f"mfcc_{i+1}"] = float(val)
        return feats

    def _get_tonal_features(self, y):
        """Tonal and harmonic features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        # pitch_variance (Standard deviation of spectral centroid)
        pitch_variance = np.std(librosa.feature.spectral_centroid(y=y, sr=self.sr))
        
        return {
            "polyphony_score": float(np.mean(entropy(chroma + 1e-9, axis=0))),
            "tonal_stability": float(1.0 / (np.std(chroma) + 1e-9)),
            "pitch_variance": float(pitch_variance),
            "pitch_confidence": float(np.mean(np.max(chroma, axis=0))),
            "dissonance": float(np.mean(np.diff(chroma, axis=0)**2)),
            "melodiousness": float(1.0 - entropy(np.mean(chroma, axis=1) + 1e-9)),
            "minorness": float(np.mean(chroma[3, :] + chroma[8, :]))
        }

    def _get_rhythm_features(self, y, onset_env, onset_frames):
        """Rhythm-related features (Supports IOI ratio)"""
        iois = np.diff(librosa.frames_to_samples(onset_frames) / self.sr)
        
        r_mean, r_std = 1.0, 0.0
        if len(iois) > 1:
            ratios = iois[:-1] / (iois[1:] + 1e-5)
            r_mean, r_std = np.mean(ratios), np.std(ratios)

        # Complexity (Entropy of IOI)
        complexity = entropy(pd.Series(np.round(iois/0.05)*0.05).value_counts()) if len(iois) > 0 else 0.0
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        return {
            "rhythmic_complexity": float(complexity),
            "rhythm_ratio_mean": float(r_mean),
            "rhythm_ratio_std": float(r_std),
            "tempo": float(tempo[0] if isinstance(tempo, np.ndarray) else tempo),
            "note_density": float(len(onset_frames) / (len(y)/self.sr + 1e-9)),
            "articulation": float(np.mean(onset_env[onset_frames])) if len(onset_frames) > 0 else 0.0,
            "rhythmic_stability": float(1.0 / (np.std(np.diff(iois)) + 1e-9)) if len(iois) > 1 else 0.0
        }