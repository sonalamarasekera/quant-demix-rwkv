import torch

from rwkv_ss.transforms.stft import STFTProcessor


def test_stft_output_shape():
    processor = STFTProcessor(n_fft=512, hop_length=128)
    wav = torch.randn(2, 1, 48000)
    mag, phase = processor.stft(wav)

    assert mag.shape == phase.shape
    assert mag.shape[0] == 2
    assert mag.shape[1] == 257  # n_fft // 2 + 1
    assert mag.ndim == 3


def test_istft_multi_source_roundtrip_shape():
    processor = STFTProcessor(n_fft=512, hop_length=128)
    wav = torch.randn(2, 1, 48000)
    mag, phase = processor.stft(wav)
    t_wav = wav.shape[-1]

    sep_mag = mag.unsqueeze(1).expand(2, 2, mag.shape[1], mag.shape[2])
    sep_wav = processor.istft(sep_mag, phase, length=t_wav)

    assert sep_wav.shape == (2, 2, t_wav)


def test_stft_istft_single_source_roundtrip():
    processor = STFTProcessor(n_fft=256, hop_length=64)
    wav = torch.randn(1, 1, 16000)
    t_wav = wav.shape[-1]
    mag, phase = processor.stft(wav)
    recon = processor.istft(mag, phase, length=t_wav)

    assert recon.shape == (1, t_wav)
