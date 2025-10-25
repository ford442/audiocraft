import torch
import torchaudio
from audiocraft.models import EnCodecModel
from audiocraft.data.audio import audio_write

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
model = EnCodecModel.from_pretrained('facebook/encodec_32khz')
model.set_target_bandwidth(6.0) # Set a reasonable bandwidth (6 kbps)

# --- 1. Load and Prepare Audio ---
# We must make the songs the same length. Let's use 10 seconds.
TARGET_DURATION_SEC = 10 

print("Loading and preparing content song (song_A)...")
# This song will provide the MELODY and STRUCTURE
wav_A = load_and_prepare_audio('song_content.wav', model, TARGET_DURATION_SEC)

print("Loading and preparing style song (song_B)...")
# This song will provide the TIMBRE and STYLE
wav_B = load_and_prepare_audio('song_style.wav', model, TARGET_DURATION_SEC)

# --- 2. Encode to "Tensor Level" ---
print("Encoding both songs to their tensor representations...")
# 'codes' is a tensor of shape [batch, num_quantizers, time]
# This is the "tensor level" you were talking about!
# We only need the codes, not the scale (hence [0])
codes_A = model.encode(wav_A)[0]
codes_B = model.encode(wav_B)[0]

print(f"Tensor shape: {codes_A.shape}") # e.g., [1, 8, 750] (Batch, Quantizers, Time)

# --- 3. Splice the Tensors ---
print("Splicing tensors...")

# This is your main creative control!
# The model has model.quantizer.n_q quantizers (e.g., 8).
# We take the first 'k' from Song A (content) and the rest from Song B (style).
# A good starting point is 2 or 4.
SPLICE_POINT = 2 

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

print(f"Done! Your new song is saved as: {output_filename}")
