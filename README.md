# SpeechToText
----------------------------------------------------------
A simple speech to text model utilizing librosa to extract mfcc features, a simple LSTM recurrent neural network for recognition, and audio clips from the movie "Taken" (great movie) for training

Feature Extraction.py: We extract MFCC (Mel-Frequency Cepstral Coefficients) features from audio files using the librosa library.
Data Processing.py: Prepares the data by loading audio files and generating feature and label mappings.
myModel.py: A bidirectional LSTM (Long Short-Term Memory) network processes the audio features and outputs character predictions.
train.py: The model is trained using CTC Loss (Connectionist Temporal Classification) to align the predicted outputs with transcriptions
inference.py: Handles the transcription of our audio files and outputs the shape of our feeatures, our raw prediction in tensors, transcription length, and transcription
utils.py: Utility function program to include methods such as character mappings
main.py: Ties all of these functions together

Example usage:

python train.py

output:
Epoch 100/100, File 1/15, Loss: 0.5656
Epoch 100/100, File 2/15, Loss: 0.1493
...
Epoch 100/100, Average Loss: 0.1599

python inference.py skills.wav

output:

Transcribing file: skills.wav
Features shape: (339, 13)
Raw prediction: tensor([[23,  0, ...]])
Transcription length: 275

Transcription: whhhhaattt  i  ddo    hhaaveee  aaree   aa  vveerry  paartticuullaarrr  ssseett   off skkiilllls   skkiiillllsss  ii   haavvee aacqquiiirreeedd  oovver aaa vveeryy llonng ccarreeeeeerrr   sskilllllss   thhat mmakee mme aa niighttmaarree  foorr   ppeeopplee  liikkkee   yyyoou

Current issues:
My model is prone to overfitting due to a lack of post-processing to address overfitting. Future implementations to fix this might include using reularization techniques such as dropout or weight decay
Furthermore, my model is trained on an extremely limited dataset leading to a general weak model due to a lack of enough quality data
