import librosa
import numpy as np
from feature_extraction import extract_features

def load_audio(file_path, sample_rate=16000):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def prepare_data(audio_files, transcriptions, char2idx):
    features = []
    labels = []
    for audio_file, transcription in zip(audio_files, transcriptions):
        audio = load_audio(audio_file)
        feature = extract_features(audio)
        features.append(feature)
        label = [char2idx[char] for char in transcription.lower() if char in char2idx]
        labels.append(label)
    return features, labels