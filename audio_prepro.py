import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from matplotlib import pyplot as plt

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 96
sample_rate = 16000

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=1.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    window_fn=torch.hamming_window
)

def resample(source_sr, target_sr):
    resample_transform = T.Resample(source_sr, target_sr)
    return resample_transform

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

    plt.savefig("spec.png")

def pad(tensor, sampe_rate):
    # 0-Pad 10 sec at fs hz and add little noise
    z = torch.zeros(10*sample_rate, dtype=torch.float32)
    z[:tensor.size(0)] = tensor
    z = z + 5*1e-4*torch.rand(z.size(0))
    return z

def preprocess_audio(audio="/content/test_file.wav", transform=mel_spectrogram):
    if isinstance(audio, str):
        audio, sr = torchaudio.load(audio)
        audio = resample(sr, sample_rate)(audio)
        # downmix to mono
        audio = torch.mean(audio, dim=0)
    else:
        pass
    # audio = audio[:sample_rate]
    audio = pad(audio, sample_rate)
    if transform is not None:
        audio = transform(audio)[:96, :96]
        audio = torch.log(audio + torch.finfo(torch.float32).eps)
    return audio
