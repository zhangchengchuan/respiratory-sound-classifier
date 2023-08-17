import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from dataset import RespiratorySoundDataset
from models import ConvNet

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def train(dataset, epochs, model, loss_fn, optimizer, device):

    # Create Data Loader
    train_data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Train
    for i in range(epochs):
        print(f"Epoch {i+1}")
        for input, target in train_data_loader:
            input, target = input.to(device), torch.tensor(target)

            pred = model(input)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
        
        print("End of Epoch ----------------------------------")
    
    print("Training Finished")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    rsd = RespiratorySoundDataset(
        target_sr=SAMPLE_RATE,
        target_length=NUM_SAMPLES,
        device=device,
        path_to_csv='data/final.csv',
        transformation=mel_spectrogram
    )
    cnn = ConvNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    train(dataset=rsd, model=cnn, loss_fn=loss_fn, optimizer=optimizer, epochs=EPOCHS, device=device)

    # save model
    torch.save(cnn.state_dict(), "model.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")