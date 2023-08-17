import torch
import torchaudio
from models import ConvNet
from dataset import RespiratorySoundDataset

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

class_mapping = {
    1:'URTI',
    2:'Healthy',
    3:'Asthma',
    4:'COPD',
    5:'LRTI',
    6:'Bronchiectasis',
    7:'Pneumonia',
    8:'Bronchiolitis'
}

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        pred = model(input)
        predicted_index = pred[0].argmax(0).item()
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    
    return predicted, expected

if __name__ == "__main__":
    # load back the model
    cnn = ConvNet()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    rsd = RespiratorySoundDataset(
        target_sr=SAMPLE_RATE,
        target_length=NUM_SAMPLES,
        device='cpu',
        path_to_csv='data/final.csv',
        transformation=mel_spectrogram
    )


    # get a sample from the urban sound dataset for inference
    input, target = rsd[0][0], rsd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make inference on test set
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")