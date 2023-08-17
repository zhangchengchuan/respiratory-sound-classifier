import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd


class RespiratorySoundDataset(Dataset):
    def __init__(self, target_sr, target_length, device, path_to_csv, transformation):
        self.target_sr = target_sr
        self.target_length = target_length
        self.csv = self._process_csv(path_to_csv)
        self.device = device
        self.transformation = transformation
    
    # Main Methods
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        signal, sr = torchaudio.load('data/train/'+row[1])
        signal = signal.to(self.device)
        signal = self._process_signal(signal, sr)
        label = row[2]
        return signal, label

    def __len__(self):
        return len(self.csv)
    
    # Return the rows labelled as training in CSV file
    def _process_csv(self, path_to_csv):
        csv = pd.read_csv(path_to_csv)
        return csv[csv['train_or_test'] == 1]
    
    # Operations to normalize signal
    def _process_signal(self, signal, sr):
        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
        
        # Mix down if number of channels > 1
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Cut or Right pad to normalize length of signal
        if signal.shape[1] > self.target_length:
            signal = signal[:, :self.target_length]
        elif signal.shape[1] < self.target_length:
            pad_right = self.target_length - signal.shape[1]
            signal = torch.nn.functional.Pad(signal, (0, pad_right))
        
        # Transformation
        signal = self.transformation(signal)

        return signal


# Testing
if __name__ == "__main__":
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

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

    print(f"There are {len(rsd)} samples in the dataset.")
    signal, label = rsd[0]
    print(signal.size())
