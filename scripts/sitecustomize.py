"""
Local-only compatibility shim (Windows).

torchaudio >= 2.9 routes torchaudio.load/save/info through TorchCodec, whose
Windows wheels fail at import time because libtorchcodec_image.dll cannot be
loaded without FFmpeg shared libraries. On Linux (the HPC target) TorchCodec
imports fine and this shim does nothing at all.

We patch only when TorchCodec is genuinely unavailable, and we back the calls
with soundfile, which bundles libsndfile and needs no external DLLs. For PCM
WAV/FLAC (everything this project reads) soundfile returns bit-identical
float32 samples, so results are unchanged.

Delete this file to disable.
"""

def _install():
    try:
        import torchcodec  # noqa: F401
        import torchcodec._core.ops  # noqa: F401
        return  # TorchCodec works -> leave torchaudio alone (Linux/HPC path)
    except Exception:
        pass

    try:
        import torch
        import torchaudio
        import soundfile as sf
    except Exception:
        return

    def load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
        kw = {"dtype": "float32" if normalize else "int16",
              "always_2d": True, "start": int(frame_offset)}
        if num_frames is not None and num_frames > 0:
            kw["frames"] = int(num_frames)
        data, sr = sf.read(str(uri), **kw)          # (frames, channels)
        t = torch.from_numpy(data)
        return (t.T.contiguous() if channels_first else t), sr

    def save(uri, src, sample_rate, channels_first=True, format=None,
             encoding=None, bits_per_sample=None, buffer_size=4096, backend=None):
        t = src.detach().cpu()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        data = (t.T if channels_first else t).numpy()   # -> (frames, channels)
        subtype = None
        if bits_per_sample == 16:
            subtype = "PCM_16"
        elif bits_per_sample == 24:
            subtype = "PCM_24"
        elif bits_per_sample == 32:
            subtype = "PCM_32"
        sf.write(str(uri), data, int(sample_rate), subtype=subtype)

    def info(uri, format=None, buffer_size=4096, backend=None):
        i = sf.info(str(uri))
        return type("AudioMetaData", (), {
            "sample_rate": i.samplerate,
            "num_frames": i.frames,
            "num_channels": i.channels,
            "bits_per_sample": 0,
            "encoding": i.subtype or "",
        })()

    torchaudio.load, torchaudio.save, torchaudio.info = load, save, info
    torchaudio._EPA_SOUNDFILE_SHIM = True


_install()
del _install
