import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer



audio, rate = librosa.load("skills.wav", sr = 16000)

print("audio array:", audio)
print("\n")
print("my rate:", rate)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

input_values = tokenizer(audio, return_tensors = "pt").input_values
logits = model(input_values).logits
prediction = torch.argmax(logits, dim = -1)

transcription = tokenizer.batch_decode(prediction)[0]

print(transcription)