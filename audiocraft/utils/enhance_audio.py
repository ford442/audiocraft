import numpy as np
import soundfile as sf
import pedalboard
from pedalboard.pedal import (
    Compressor,
    # HighShelf, # Keep or remove based on your previous fix
    Limiter,
    # LowShelf,  # Keep or remove based on your previous fix
    NoiseGate,
    Reverb
)
import librosa # Make sure librosa is imported if used for resampling

# --- Your existing board definition should be here ---
# Example (adjust based on your previous fixes):
board = pedalboard.Pedalboard([
    NoiseGate(threshold_db=-50, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-14, ratio=2),
    # LowShelf(cutoff_frequency_hz=400, gain_db=2, q=0.707),
    # HighShelf(cutoff_frequency_hz=3000, gain_db=3, q=0.707),
    Limiter(threshold_db=-1, release_ms=50),
])
# --- End board definition ---


def enhance(input_path: str, output_path: str = None, target_sr: int = 44100):
    if output_path is None:
        output_path = input_path

    # --- Add TRY block ---
    try:
        print(f"Enhancing {input_path}...")
        audio, sr = sf.read(input_path)
        print(f"Loaded audio at {sr} Hz. Shape: {audio.shape}")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.ndim == 1:
            print("Detected mono audio.")
            channels = 1
            # Pedalboard expects (C, T) or (T,) for mono? Let's keep it (T,) for librosa, convert later
            audio_for_resample = audio
        elif audio.ndim == 2:
            # Assuming shape is (T, C), transpose to (C, T) for librosa/pedalboard
            audio_for_resample = audio.T
            channels = audio_for_resample.shape[0]
            print(f"Detected {channels}-channel audio (transposed). Shape: {audio_for_resample.shape}")
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        # Resample if necessary using librosa
        if sr != target_sr:
             # librosa.resample works best with mono (T,) or stereo (2, T)
             # If input was (T,C), we transposed to (C,T)
            if audio_for_resample.ndim > 1:
                 # Resample each channel individually if multi-channel
                 resampled_list = [librosa.resample(y=ch, orig_sr=sr, target_sr=target_sr) for ch in audio_for_resample]
                 audio_resampled = np.stack(resampled_list, axis=0)
            else: # Mono case
                 audio_resampled = librosa.resample(y=audio_for_resample, orig_sr=sr, target_sr=target_sr)

            sr = target_sr
            print(f"Resampled audio to {sr} Hz. Shape: {audio_resampled.shape}")
        else:
            audio_resampled = audio_for_resample # No resampling needed

        # Ensure shape is (C, T) for pedalboard processing
        if audio_resampled.ndim == 1:
             audio_pb = audio_resampled[np.newaxis, :] # Add channel dim -> (1, T)
        else:
             audio_pb = audio_resampled

        print(f"Audio shape for pedalboard: {audio_pb.shape}")

        # Apply Pedalboard effects chain
        processed_audio = board(audio_pb, sr)

        # Apply mono-to-stereo reverb if input was mono
        if channels == 1:
            print("Applying mono-to-stereo reverb.")
            reverb_effect = Reverb(room_size=0.75, wet_level=0.5, dry_level=0.5)
            # Apply reverb - needs shape (C, T)
            reverb_audio = reverb_effect(processed_audio, sr)

            # Handle reverb output channels
            if reverb_audio.shape[0] == 1:
               processed_audio = np.concatenate([reverb_audio, reverb_audio], axis=0) # Manual stereo
            elif reverb_audio.shape[0] == 2:
               processed_audio = reverb_audio # Use stereo output
            else:
               print(f"Warning: Reverb produced unexpected channels: {reverb_audio.shape[0]}. Taking first two.")
               processed_audio = reverb_audio[:2, :]

        # --- Ensure output shape is (T, C) for soundfile.write ---
        # Soundfile expects (frames, channels)
        if processed_audio.shape[0] == channels or processed_audio.shape[0] == 2: # If shape is (C, T)
             processed_audio_for_sf = processed_audio.T
        elif processed_audio.shape[1] == channels or processed_audio.shape[1] == 2: # If shape is already (T, C)
             processed_audio_for_sf = processed_audio
        elif processed_audio.ndim == 1: # Mono case (T,)
            processed_audio_for_sf = processed_audio # soundfile handles mono (T,)
        else:
             raise ValueError(f"Unexpected shape after processing: {processed_audio.shape}")


        # Save the enhanced audio
        sf.write(output_path, processed_audio_for_sf, sr)
        print(f"Enhanced audio saved to {output_path}")

    # --- Add EXCEPT block ---
    except Exception as e:
        print(f"!!! FAILED TO ENHANCE AUDIO in enhance_audio.py !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        # We don't re-raise, allowing the main script to continue if needed,
        # but the error will be clearly printed.

# --- End of enhance function ---
