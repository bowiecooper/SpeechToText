import torch
from feature_extraction import extract_features
from data_processing import load_audio

def transcribe(model, audio_file, idx2char):
    print(f"Transcribing file: {audio_file}")
    audio = load_audio(audio_file)
    features = extract_features(audio)
    print(f"Features shape: {features.shape}")
    
    with torch.no_grad():
        features = torch.FloatTensor(features).unsqueeze(0)
        output = model(features)
        prediction = torch.argmax(output, dim=-1)
    
    transcription = ''.join([idx2char[idx.item()] for idx in prediction[0] if idx != 0])
    print(f"Raw prediction: {prediction}")
    print(f"Transcription length: {len(transcription)}")
    return transcription