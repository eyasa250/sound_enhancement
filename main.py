import os
import librosa
import noisereduce as nr
import numpy as np
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, HighShelfFilter, Gain
import soundfile as sf
#from moviepy.editor import VideoFileClip
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from moviepy.editor import VideoFileClip  # Import spécifique pour VideoFileClip

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_audio(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".mp4":
        audio_path = os.path.join(UPLOAD_DIR, "audio.wav")
        video = VideoFileClip(input_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        return audio_path
    return input_path

def process_audio(input_path: str, output_filename: str) -> str:
    sr = 22050  # adapté pour la voix
    audio_path = extract_audio(input_path)
    audio_data, sr = librosa.load(audio_path, sr=sr, mono=True)

    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr, stationary=True, prop_decrease=0.4)

    board = Pedalboard([
        NoiseGate(threshold_db=-45, ratio=1.2, release_ms=300),
        Compressor(threshold_db=-24, ratio=2.0),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=3.0, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=4000, gain_db=2.0, q=0.7),
        Gain(gain_db=2.0)
    ])

    effected = board(reduced_noise, sr)
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    sf.write(output_path, effected, sr)
    return output_path

@app.post("/process-audio/")
async def process_audio_endpoint(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    output_filename = f"enhanced_{file.filename.rsplit('.', 1)[0]}.wav"
    processed_path = process_audio(input_path, output_filename)

    return FileResponse(processed_path, media_type="audio/wav", filename=output_filename)
