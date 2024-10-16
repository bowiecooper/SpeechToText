import librosa
import numpy as np

def extract_features(audio, n_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    return mfccs.T