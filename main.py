import torch
from data_processing import prepare_data
from myModel import TranscriptionModel
from train import train_model
from inference import transcribe
from utils import create_char_mappings

def main():
    # Setup
    char2idx, idx2char = create_char_mappings()
    
    # Prepare data (replace with your actual data)
    audio_files = ['Conditions.wav', 'DontRemember.wav', 'GoingToTakeYou.wav', 'Goodluck.wav', 'IDontKnow.wav', 'LetMyDaughter.wav',
                    'Lied.wav', 'Negotiating.wav', 'NoIdea.wav', 'Sacrifice.wav', 'skills.wav', 'Space.wav', 'Stayfocus.wav', 'Taken.wav', 'Think.wav']
    
    transcriptions = ['Three conditions i want the address and phone number of where youre staying ok if you move i want to know where and with whom youll be staying you call me when you land you call me every night before you go to sleep its international my number is programmed ok ok awesome kimmy youre not focused yes i am what did i say you said that to call you when i land and every night before i go to sleep and your phones international the numbers programmed in ok one last thing i get to take you to the airport ok',
                      'you dont remember me we spoke on the phone two days ago i told i would find you',
                      'now the next part is very important they are going to take you',
                      'good luck good luck good luck good luck good luck good luck',
                      'i dont know who you are i dont know what you want if you are looking for a ransom i can tell you i dont have money',
                      'if you let my daughter go now that will be the end of it iw ill not look for you i will not pursue you',
                      'lenor do you know about this shes not just going to paris i know she lied to me yes because she cant be honest with you',
                      'why are you bothering the girl its none of your business she is my business and if you are not spending money you are costing money i was just negotiating there is no negotiating the price is the prices',
                      'you have no idea what the world is like yes and neither will she unless she goes out and experiences it',
                      'i dont get you what you sacrificed our marriage in the service of the country you made a mess of your life in the service of the country cant you sacrifice just a little this one time for your own daughter',
                      'what i do have are a very particular set of skills skills i have acquired over a very long career skills that make me a nightmare for people like you',
                      'hello its me has kim called you ryan shes seventeen shes in paris give her some space shell call',
                      'im scareed i know you stay focused kimmy you have to hold it together',
                      'i will look for you i will find you and i will kill you',
                      'ill think about oh god ryan everyone at this table knows what that means']
    
    train_data, train_labels = prepare_data(audio_files, transcriptions, char2idx)
    
    # Initialize and train the model
    model = TranscriptionModel(input_dim=13, hidden_dim=256, output_dim=len(char2idx))
    train_model(model, train_data, train_labels, epochs=100)
    
    # Save the model
    torch.save(model.state_dict(), 'trained_model.pth')
    
    # Inference
    test_audio = 'skills.wav'
    transcription = transcribe(model, test_audio, idx2char)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()