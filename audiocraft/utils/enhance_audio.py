# enhance_audio.py
import librosa
import soundfile as sf
import numpy as np
from pedalboard import Pedalboard
from pedalboard import (
    Compressor,
    Limiter,
    NoiseGate,
    Pedalboard,
    Reverb
)

def enhance_audio(input_file: str, output_file: str):
    """
    Loads an audio file, checks if it's mono or stereo,
    resamples it to 44.1kHz, and applies a mastering chain.
    """
    print(f"Enhancing {input_file}...")
    
    # 1. Load audio, preserving original channels and SR
    try:
        # Load with sr=None to get original SR, mono=False to get original channels
        audio, sr = librosa.load(input_file, sr=None, mono=False)
        print(f"Loaded audio at {sr} Hz. Shape: {audio.shape}")
    except Exception as e:
        print(f"Error loading audio file {input_file}: {e}")
        return

    # 2. Check channels
    is_mono = False
    if audio.ndim == 1:
        is_mono = True
        print("Detected mono audio.")
    else:
        is_mono = False
        print(f"Detected stereo audio (channels={audio.shape[0]}).")

    # 3. Resample to 44.1kHz
    target_sr = 44100
    # librosa.resample works correctly on (samples,) for mono
    # and (channels, samples) for stereo.
    audio_resampled = librosa.resample(
        audio, 
        orig_sr=sr, 
        target_sr=target_sr,
        res_type='kaiser_best'
    )
    
    # 4. Correct shape for pedalboard after resampling
    # Pedalboard expects (channels, samples)
    if audio_resampled.ndim == 1:
         # Reshape mono (samples,) to (1, samples)
        audio_data_resampled = audio_resampled.reshape(1, -1)
    else:
        # Stereo is already (channels, samples)
        audio_data_resampled = audio_resampled
        
    print(f"Resampled audio to {target_sr} Hz.")
    print(f"Resampled audio shape for pedalboard: {audio_data_resampled.shape}")

    # 5. Define mastering chain
    
    # --- Define Reverb ---
    # We use different settings for mono (to create stereo) vs. stereo (to add space)
    if is_mono:
        print("Applying mono-to-stereo reverb.")
        reverb = Reverb(
            room_size=0.25,     # Small room
            damping=0.5,
            wet_level=0.3,      # 30% reverb - this creates the stereo image
            dry_level=0.7,
            width=1.0           # Max stereo width
        )
    else:
        # Input is already stereo, just add a bit of "space"
        print("Applying subtle stereo reverb for space.")
        reverb = Reverb(
            room_size=0.15,     # Tighter room
            damping=0.5,
            wet_level=0.15,     # Much less reverb
            dry_level=0.85,
            width=1.0           # Keep existing width
        )

    # --- Define the full processing board ---
    board = pedalboard.Pedalboard([
    NoiseGate(threshold_db=-50, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-14, ratio=2),
    Limiter(threshold_db=-1, release_ms=50),
    ])

    print("Applying mastering chain...")
    
    # 6. Process the audio
    effected_audio = board(audio_data_resampled, sample_rate=target_sr)

    # 7. Save the final file
    # soundfile.write expects (samples, channels)
    # Pedalboard outputs (channels, samples)
    # So, we transpose with .T
    print(f"Saving enhanced audio to {output_file}...")
    sf.write(output_file, effected_audio.T, samplerate=target_sr)
    print("Enhancement done.")
