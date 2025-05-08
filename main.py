import os
import librosa
import noisereduce as nr
import numpy as np
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, HighShelfFilter, Gain
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

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

    # Réduction du bruit
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr, stationary=True, prop_decrease=0.4)

    # Traitement audio avec Pedalboard
    board = Pedalboard([
        NoiseGate(threshold_db=-45, ratio=1.2, release_ms=300),
        Compressor(threshold_db=-24, ratio=2.0),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=3.0, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=4000, gain_db=2.0, q=0.7),
        Gain(gain_db=2.0)
    ])

    effected = board(reduced_noise, sr) 
    
    # Sauvegarder l'audio traité
    processed_audio_path = os.path.join(PROCESSED_DIR, output_filename + ".wav")
    sf.write(processed_audio_path, effected, sr)
    
    return processed_audio_path

def create_video_with_enhanced_audio(video_path: str, audio_path: str, output_filename: str) -> str:
    # Charger la vidéo et l'audio traités
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    
    # Remplacer l'audio original par l'audio traité
    video = video.set_audio(audio)
    
    # Sauvegarder la vidéo avec le nouvel audio
    output_video_path = os.path.join(PROCESSED_DIR, output_filename + "_enhanced.mp4")
    video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=False)
    
    return output_video_path

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Sauvegarder le fichier vidéo téléchargé
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Nom du fichier de sortie
    output_filename = f"enhanced_{file.filename.rsplit('.', 1)[0]}"
    
    # Traiter l'audio
    processed_audio_path = process_audio(input_path, output_filename)
    
    # Créer une vidéo avec l'audio amélioré
    processed_video_path = create_video_with_enhanced_audio(input_path, processed_audio_path, output_filename)
    
    # Retourner le fichier vidéo traité
    return FileResponse(processed_video_path, media_type="video/mp4", filename=processed_video_path)
