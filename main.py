import os
import librosa
import noisereduce as nr
import numpy as np
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, HighShelfFilter, Gain
import soundfile as sf
from moviepy import *
# --- 1. Définir le fichier d'entrée ---
input_path = 'p232_021.wav'  # ou testvideo.mp4 ou audio.wav
sr = 44100  # Fréquence d’échantillonnage cible

# --- 2. Extraire ou charger l'audio ---
def extract_audio(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == ".mp4":
        # Extraire l’audio du fichier vidéo
        video = VideoFileClip(input_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        return audio_path
    elif ext in [".mp3", ".wav"]:
        return input_path
    else:
        raise ValueError("Format de fichier non pris en charge. Utilisez mp3, wav ou mp4.")

audio_path = extract_audio(input_path)

# --- 3. Charger l’audio ---
audio_data, sr = librosa.load(audio_path, sr=sr, mono=True)
print(f"Audio chargé : {audio_data.shape}, Sample rate : {sr}")

# --- 4. Réduction de bruit ---
reduced_noise = nr.reduce_noise(y=audio_data, sr=sr, stationary=False, prop_decrease=1)
print("Réduction de bruit appliquée.")

# --- 5. Appliquer les effets avec Pedalboard ---
board = Pedalboard([
    # Couper les silences faibles (souffle micro)
    NoiseGate(threshold_db=-45, ratio=1.2, release_ms=300),
    
    # Compression douce : égalise la voix sans l'écraser
    Compressor(threshold_db=-24, ratio=2.0),
    
    # Ajout de basses pour une voix plus chaude
    LowShelfFilter(cutoff_frequency_hz=120, gain_db=3.0, q=0.7),
    
    # Légère clarté dans les aigus (améliore l'articulation)
    HighShelfFilter(cutoff_frequency_hz=4000, gain_db=2.0, q=0.7),
    
    # Gain final (ajuster seulement si le volume est trop faible)
    Gain(gain_db=2.0)
])


effected = board(reduced_noise, sr)
print("Effets appliqués.")

# --- 6. Sauvegarder l’audio final ---
output_path = 'enhanced_audiop232_021.wav'
sf.write(output_path, effected, sr)
print(f"Fichier sauvegardé : {output_path}")
