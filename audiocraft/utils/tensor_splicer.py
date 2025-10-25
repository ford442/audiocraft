import torch
import torchaudio

from audiocraft.models import EncodecModel
from audiocraft.data.audio import audio_write
from torchaudio.transforms import Resample # <-- ADDED IMPORT

def load_and_prepare_audio(file_path, model, target_length_sec=10):
    """
    Loads an audio file, resamples it to the model's sample rate,
    converts it to mono, and pads/truncates it to a target length.
    """
    # Load the audio file
    wav, sr = torchaudio.load(file_path)

    # Resample if necessary
    if sr != model.sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model.sample_rate)(wav)

    # Convert to mono by averaging channels
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Pad or truncate to the target length
    target_samples = model.sample_rate * target_length_sec
    current_samples = wav.shape[1]

    if current_samples < target_samples:
        # Pad with silence
        padding = target_samples - current_samples
        wav = torch.nn.functional.pad(wav, (0, padding))
    elif current_samples > target_samples:
        # Truncate
        wav = wav[:, :target_samples]

    # Add a batch dimension (required by the model)
    return wav.unsqueeze(0) # Shape: [1, 1, num_samples]

# --- Main Execution ---

print("Loading EnCodec model...")
# Load the pre-trained EnCodec model
# 'facebook/encodec_32khz' is a good high-quality model
model = EncodecModel.get_pretrained('facebook/encodec_32khz')
MODEL_SR = model.sample_rate # <-- FIXED: Define MODEL_SR right after loading

# --- 1. Load and Prepare Audio ---
# We must make the songs the same length. Let's use 10 seconds.
TARGET_DURATION_SEC = 300

print("Loading and preparing content song (song_A)...")
# This song will provide the MELODY and STRUCTURE
wav_A = load_and_prepare_audio('song_content.mp3', model, TARGET_DURATION_SEC)

print("Loading and preparing style song (song_B)...")
# This song will provide the TIMBRE and STYLE
wav_B = load_and_prepare_audio('song_style.mp3', model, TARGET_DURATION_SEC)
TARGET_BANDWIDTH = 6.0 # <-- ADDED VARIABLE

# 'codes' is a tensor of shape [batch, num_quantizers, time]
# We pass the bandwidth directly to the encode method.
codes_A = model.encode(wav_A )[0] # <-- CHANGED
codes_B = model.encode(wav_B )[0] # <-- CHANGED

# The number of quantizers is the 2nd dimension (dim=1)
num_quantizers = codes_A.shape[1]
print(f"Tensor shape: {codes_A.shape}") # e.g., [1, 8, 750]
print(f"Total quantizers available (based on {TARGET_BANDWIDTH}kbps): {num_quantizers}")

# --- 3. Splice the Tensors ---
print("Splicing tensors...")

# This is your main creative control!
# We take the first 'k' from Song A (content) and the rest from Song B (style).
# A good starting point is 2 or 4 (out of 8).
SPLICE_POINT = 3

if SPLICE_POINT >= num_quantizers:
    print(f"Warning: SPLICE_POINT ({SPLICE_POINT}) is >= total quantizers ({num_quantizers}).")
    print("The output will just be song_content.wav.")
    SPLICE_POINT = num_quantizers

# Create a new empty tensor to hold the combined codes
codes_C = torch.zeros_like(codes_A)

# Copy the "content" (first k quantizers) from Song A
codes_C[:, :SPLICE_POINT, :] = codes_A[:, :SPLICE_POINT, :]

# Copy the "style" (remaining quantizers) from Song B
codes_C[:, SPLICE_POINT:, :] = codes_B[:, SPLICE_POINT:, :]

# --- 4. Decode Back to Audio ---
print("Decoding spliced tensor back to audio...")
with torch.no_grad():
    wav_C = model.decode(codes_C)

# --- 5. Save the Output ---
output_filename = f'output_splice_at_{SPLICE_POINT}.wav'
audio_write(
    output_filename, 
    wav_C.squeeze(0).cpu(), # Remove batch dim and move to CPU
    model.sample_rate
)
resampler = Resample(orig_freq=MODEL_SR, new_freq=TARGET_SR)
wav_output_44k = resampler(wav_C.squeeze(0)) # Remove batch dim
torchaudio.save(
    output_filename,
    wav_output_44k.cpu(), # Move to CPU for saving
    TARGET_SR,
    format="flac" # <-- EXPLICITLY SET FORMAT
)
print(f"Done! Your new song is saved as: {output_filename}")
