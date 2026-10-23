from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import argparse
import math
import numpy as np
from pathlib import Path
import json
import os
from moviepy import VideoFileClip
import tempfile
import subprocess
import librosa

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
    ".aac", ".wma", ".opus", ".aiff", ".aif"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mpeg", ".mpg", ".m4v", ".wmv"
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "Audio_Model"
if not DEFAULT_MODEL_DIR.is_dir():
    DEFAULT_MODEL_DIR = SCRIPT_DIR

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_audio(input_path, target_sr):
    input_path = Path(input_path)

    # Extract audio to a temporary WAV file using FFmpeg.
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(target_sr),
            "-c:a", "pcm_s16le",
            str(wav_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed:\n{result.stderr}"
            )

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty audio file.")

        # Load the extracted WAV at the model's expected sample rate.
        audio, sr = librosa.load(
            str(wav_path),
            sr=target_sr,
            mono=True,
        )

    if audio.size == 0:
        raise ValueError(f"No audio samples found in {input_path}")

    return audio.astype(np.float32), sr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help = "audio or video path")
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    model = Wav2Vec2ForSequenceClassification.from_pretrained(DEFAULT_MODEL_DIR)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(DEFAULT_MODEL_DIR)
    device = choose_device()
    model.to(device)
    model.eval()
    audio, sr = load_audio(input_path, feature_extractor.sampling_rate)
    inputs = feature_extractor(audio, sampling_rate = sr, return_tensors = "pt", padding = True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    prob_real = probs[0][0].item()
    prob_fake = probs[0][1].item()
    prediction = "fake" if prob_fake > 0.5 else "real"

    print(f"Prediction: {prediction}")
    print(f"Confidence: {max(prob_real, prob_fake):.2%}")
    print(f"Probabilities - Real: {prob_real:.2%}, Fake: {prob_fake:.2%}")

if __name__ == "__main__":
    main()