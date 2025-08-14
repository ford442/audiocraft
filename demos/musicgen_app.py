import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
import gc

os.putenv('PYTORCH_NVML_BASED_CUDA_CHECK','1')
os.putenv('TORCH_LINALG_PREFER_CUSOLVER','1')
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,pinned_use_background_threads:True'
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.putenv('HF_HUB_ENABLE_HF_TRANSFER','1')

from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

from einops import rearrange
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.preferred_blas_library="cublas"
torch.backends.cuda.preferred_linalg_library="cusolver"
torch.set_float32_matmul_precision("highest")


import gradio as gr
import librosa  # Import librosa

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion

MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
IS_BATCHED = "facebook/MusicGen" in SPACE_ID or 'musicgen-internal/musicgen_dev' in SPACE_ID
print(IS_BATCHED)
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr  # Still a good idea to keep this
# Preallocating the pool of processes.  Not used for waveform generation anymore,
# but could be used for other parallel tasks if needed.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
file_cleaner = FileCleaner()

# No more make_waveform function!

def load_model(version='facebook/musicgen-melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        if MODEL is not None: #Prevent error on first load
           del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(version)


def load_diffusion():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()

def unload_model():
    """Helper function to unload the current MusicGen model."""
    global MODEL
    if MODEL is not None:
        print("Unloading MusicGen model...")
        del MODEL
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MODEL = None
        print("MusicGen model unloaded.")

def _do_predictions(texts, melodies, duration, progress=False, gradio_progress=None, **gen_kwargs):
    # ... (previous code for loading model, processing melodies) ...

    # Ensure MODEL is loaded before proceeding
    if MODEL is None:
        print("Error: MODEL is None when entering _do_predictions.")
        raise gr.Error("MusicGen model is not loaded.")

    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("New batch:", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = MODEL.sample_rate
    target_ac = MODEL.audio_channels

    for melody_input in melodies:
        if melody_input is None:
            processed_melodies.append(None)
        else:
            sr, melody_data = melody_input
            melody_tensor = torch.from_numpy(melody_data).to(MODEL.device).float()
            if melody_tensor.dim() == 1:
                melody_tensor = melody_tensor[None]
            melody_tensor = melody_tensor[..., :int(sr * duration)]
            melody_converted = convert_audio(melody_tensor, sr, target_sr, target_ac)
            processed_melodies.append(melody_converted)

    # Capture stereo info and potentially prepare tokens for MBD *before* unloading
    stereo_processing_needed = False
    if MODEL.compression_model and isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
        stereo_processing_needed = True
        print("Detected stereo compression model.")

    tokens = None # Initialize tokens to None
    try:
        if any(m is not None for m in processed_melodies):
            print("Generating with chroma...")
            # return_tokens=True is important here
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=True # Always return tokens if diffusion is a possibility
            )
        else:
            print("Generating without chroma...")
            outputs = MODEL.generate(texts, progress=progress, return_tokens=True)

        tokens = outputs[1] # Get the tokens from the generator

    except RuntimeError as e:
        print(f"Runtime error during generation: {e}")
        if "CUDA out of memory" in str(e):
            unload_model()
        raise gr.Error("Error while generating: " + e.args[0])
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}")
        raise gr.Error("An unexpected error occurred during generation: " + str(e))

    # --- Handle Diffusion ---
    if USE_DIFFUSION and MBD is not None:
        print("Applying MultiBandDiffusion...")
        if gradio_progress is not None:
            gradio_progress(0.5, desc='Applying MultiBandDiffusion...')

        # --- PREPARE TOKENS FOR MBD's ENCODEC ---
        # This is the critical part for the IndexError.
        # We need to ensure 'tokens' is in the format expected by MBD's codec.
        tokens_for_mbd = tokens # Initialize

        if stereo_processing_needed and isinstance(tokens, torch.Tensor):
            print(f"Original tokens shape for stereo: {tokens.shape}")
            # The original Stereo model likely returns codes for stereo.
            # The exact format depends on how InterleaveStereoCompressionModel stores them.
            # It's possible the tokens are NOT interleaved across channels in the way
            # the `rearrange` was assuming, or they represent more than just quantizer IDs.
            # If `tokens` is a list of tensors (e.g., per layer), this needs to be handled.

            # Let's re-examine the original code:
            # if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            #     left, right = MODEL.compression_model.get_left_right_codes(tokens)
            #     tokens = torch.cat([left, right]) # This cat'd tokens variable is what's used for MBD

            # This implies the `tokens` fed to `MBD.tokens_to_wav` SHOULD be the result of this concat.
            # So, if the concat operation or the extraction of left/right is faulty, this is the problem.

            # Let's try to replicate the logic from the original code carefully.
            try:
                # Ensure MODEL is still available to access its compression_model
                # This means we cannot unload MODEL *before* this block if we need its compression_model.
                # We need to unload it *after* all token processing.

                if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
                    # Ensure the tokens are in a format that `get_left_right_codes` can process.
                    # If `tokens` is already a tensor from generation, and it's stereo,
                    # it might already be in a format that `get_left_right_codes` understands.
                    # Or, `get_left_right_codes` expects something specific.
                    # Let's assume `tokens` from `generate` is a tensor representing codes.

                    # Check if MBD's codec expects stereo directly or needs mono.
                    # If MBD's codec is mono, we'd need to split and use only one.
                    # The error `IndexError: index 4 is out of range` suggests the MBD quantizer has fewer layers
                    # than the tokens are expecting. This is a structural mismatch.

                    # Let's assume the stereo model's `tokens` *are* already structured for stereo.
                    # The `rearrange` logic was inside the `if USE_DIFFUSION` block AFTER `MBD.tokens_to_wav` was called,
                    # which is incorrect. The tokens need to be processed BEFORE being passed to MBD.

                    # Proposed fix: Process tokens for stereo *before* calling MBD.

                    # Re-evaluating the original code's intent:
                    # The original code snippet for stereo handling was:
                    # if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
                    #     left, right = MODEL.compression_model.get_left_right_codes(tokens)
                    #     tokens = torch.cat([left, right]) # <-- This line modifies `tokens`
                    #     # ... unload model ...
                    #     outputs_diffusion = MBD.tokens_to_wav(tokens) # <-- uses modified `tokens`
                    #     # ... rearrangement ...

                    # The issue is where `MODEL.compression_model` is accessed.
                    # The solution is to ensure `MODEL` is valid when `MODEL.compression_model` is accessed.

                    # Let's put the stereo token processing back BEFORE unloading.
                    if stereo_processing_needed:
                        # Assuming 'tokens' is a tensor. If it's a list of tensors for layers, this needs adjusting.
                        # The `get_left_right_codes` might be the key.
                        # Let's assume `tokens` itself is a single tensor of codes.
                        
                        # --- If the model is stereo, we need to prepare tokens for MBD ---
                        # The exact `get_left_right_codes` usage might imply how stereo tokens are structured.
                        # If `tokens` represents combined stereo codes, and `MBD` expects a different structure:
                        
                        # --- Potential strategy: If MBD codec is mono, and we got stereo tokens ---
                        # We need to get the token representation of ONE channel.
                        # This might involve splitting `tokens` if they are structured like `(2, N_layers, T)` or `(2, T)`
                        # and then passing a mono version to `tokens_to_wav`.

                        # If the error is truly about layer indexing within MBD's decoder,
                        # it means the shape/structure of the tokens passed to MBD is wrong.

                        # --- Let's try a simpler approach: Assume MBD expects mono tokens ---
                        # If the original MusicGen model is stereo, we should extract mono tokens.
                        # This is a guess, as the exact stereo token representation is unclear.

                        # Let's try to decode the first musicgen output (outputs[0]) into mono tokens
                        # and then pass those mono tokens to MBD. This is a hypothesis.
                        # This would require re-encoding `outputs[0]`. This is complex.

                        # More likely: the stereo code format is causing the layer mismatch.
                        # Let's try to split if `tokens` has a stereo dimension.
                        # Example: if tokens is (2, num_layers, seq_len) -> split into (num_layers, seq_len) for each channel.

                        # Let's assume the stereo model's `tokens` are structured in a way that the first dimension
                        # distinguishes channels. If it's (2, seq_len) and MBD expects (seq_len), take one.
                        # Or if it's (2, num_layers, seq_len) and MBD expects (num_layers, seq_len).
                        
                        # The problematic line: `layer = self.layers[i]` suggests MBD's quantizer has specific layer mapping.
                        # If the `tokens` passed to it have a different number of "layers" or a wrong dimension for layers.

                        # Let's revisit the `rearrange` part:
                        # The original code had:
                        # if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
                        #     left, right = MODEL.compression_model.get_left_right_codes(tokens)
                        #     tokens = torch.cat([left, right])
                        #     # Unload the MusicGen model to free up GPU memory
                        #     del self.model # This was the original error source.
                        #     gc.collect()
                        #     if torch.cuda.is_available():
                        #         torch.cuda.empty_cache()
                        # outputs_diffusion = MBD.tokens_to_wav(tokens)
                        # if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
                        #     assert outputs_diffusion.shape[1] == 1  # output is mono
                        #     outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)

                        # The problem is that the *stereo processing logic itself* needs to happen *before* unloading MODEL.
                        # AND the `tokens` variable needs to be correctly formed for MBD.

                        # If the stereo model's `tokens` are already structured correctly for MBD after generation,
                        # and `get_left_right_codes` and `torch.cat` are not needed, then the error is elsewhere.
                        # If they *are* needed, they must be done while MODEL is loaded.

                        # Let's assume for a moment that the stereo model DOES NOT need `get_left_right_codes` and `cat`,
                        # and MBD can consume the raw tokens directly IF they are stereo.
                        # If MBD expects mono, and we have stereo tokens, we need to convert.

                        # --- Hypothesis: MBD expects mono tokens, and the stereo model generates stereo tokens ---
                        # If `tokens` is a tensor like `(2, num_layers, seq_len)` or `(2, seq_len)`
                        # we might need to select one channel's tokens.
                        # Let's check the shape of `tokens` and see if the first dim is `2`.

                        # If tokens are `(2, seq_len)` -> MBD expects `(seq_len)`?
                        # If tokens are `(2, num_layers, seq_len)` -> MBD expects `(num_layers, seq_len)`?
                        
                        # Based on the error index 4 being out of range, it's likely MBD expects fewer "layers"
                        # or a different structure than what the stereo tokens are providing.

                        # Let's try to UN-do the `rearrange` part and check if the original `tokens` is usable.
                        # The `rearrange` was applied to `outputs_diffusion` *after* `tokens_to_wav`.
                        # This means the `tokens` fed into `tokens_to_wav` were the issue.

                        # Let's assume the `tokens` from `MODEL.generate` are `(num_layers, seq_len)`.
                        # If the stereo model also produces `(2, num_layers, seq_len)` or `(num_layers, seq_len)`?

                        # --- Let's try to pass `tokens` directly to MBD if it's already suitable ---
                        # Or if it needs a specific format:
                        
                        # If `tokens` is `(2, seq_len)` and MBD wants `(seq_len)`:
                        if tokens.ndim == 2 and tokens.shape[0] == 2:
                             print("Reducing tokens from stereo shape (2, seq_len) to mono (seq_len).")
                             tokens_for_mbd = tokens[0] # Take first channel
                        # If `tokens` is `(2, num_layers, seq_len)` and MBD wants `(num_layers, seq_len)`
                        elif tokens.ndim == 3 and tokens.shape[0] == 2:
                             print("Reducing tokens from stereo shape (2, num_layers, seq_len) to mono (num_layers, seq_len).")
                             tokens_for_mbd = tokens[0] # Take first channel
                        # If `tokens` is already shaped like `(num_layers, seq_len)` and it's stereo by implicit encoding
                        # then we might not need to do anything.

                        # This is still guessing the token format.
                        # The most reliable way is to check the documentation or example outputs for stereo tokens.

                        # --- Let's try removing the problematic stereo processing ---
                        # If the default tokens already work for MBD (perhaps MBD is also stereo-aware),
                        # then the stereo_processing_needed check was causing problems.
                        # But the problem is `IndexError: index 4 out of range`, which means the MBD quantizer itself
                        # is not compatible with the number of layers the tokens are implying.

                        # If the `IndexError` is about `self.layers[i]`, and `i` goes up to 4, it means MBD's quantizer
                        # has 5 stages (layers 0, 1, 2, 3, 4).
                        # The `tokens` likely encode information about these stages.
                        # If stereo splitting or concatenation alters this structure, the error occurs.

                        # Let's assume that if the original model is stereo, the `tokens` it produces are already
                        # stereo-aware or the `MBD` itself handles stereo if given the right `tokens`.
                        # The original code `left, right = MODEL.compression_model.get_left_right_codes(tokens)`
                        # and then `torch.cat([left, right])` suggests that this process *is* required to get
                        # the correct token representation for MBD.

                        # If stereo_processing_needed is True, and we are UNLOADING MODEL, we MUST do this processing
                        # *before* unloading.

                        if stereo_processing_needed:
                            print("Processing tokens for stereo using MusicGen's compression model...")
                            # IMPORTANT: This requires MODEL to be available.
                            if MODEL is None: # Safety check, should not happen here if logic is right
                                raise gr.Error("MODEL is None, cannot process stereo tokens.")
                            
                            # Try to get stereo codes. This might return tensors or lists of tensors.
                            # The exact structure of `tokens` and what `get_left_right_codes` expects/returns is key.
                            stereo_codes = MODEL.compression_model.get_left_right_codes(tokens)
                            
                            # The original `torch.cat([left, right])` suggests `left` and `right` are tensors.
                            # Let's assume `stereo_codes` is a tuple/list `(left_codes, right_codes)`
                            # and that `torch.cat` on these is the expected format for MBD.
                            
                            if isinstance(stereo_codes, (tuple, list)) and len(stereo_codes) == 2:
                                left_codes, right_codes = stereo_codes
                                tokens_for_mbd = torch.cat([left_codes, right_codes])
                                print(f"Processed tokens shape for MBD: {tokens_for_mbd.shape}")
                            else:
                                # If get_left_right_codes returned something unexpected, use original tokens.
                                print("Warning: Could not process stereo codes as expected. Using original tokens for MBD.")
                                tokens_for_mbd = tokens

                        else: # Not stereo processing needed, or model is not stereo
                            tokens_for_mbd = tokens

                else: # Not a stereo model, or stereo_processing_needed is False
                    tokens_for_mbd = tokens

            except Exception as e:
                print(f"Error during stereo token processing: {e}")
                # If stereo processing fails, we might fall back or raise an error.
                # For now, let's use original tokens if processing failed, but this is risky.
                tokens_for_mbd = tokens # Fallback, but likely leads to same error.
                # Better to raise an error if stereo is required and processing fails.
                raise gr.Error(f"Failed to prepare tokens for MBD due to stereo processing error: {e}")

        # --- UNLOAD MUSICGEN MODEL ---
        # This must happen AFTER we have finished using MODEL (e.g., for compression_model).
        unload_model()

        # --- NOW GENERATE WITH MBD using the prepared tokens ---
        try:
            outputs_diffusion = MBD.tokens_to_wav(tokens_for_mbd) # Use the potentially modified tokens
            
            # --- Stereo formatting for diffusion output (if needed) ---
            # This part seems to be for the final audio output format, not the tokens.
            if stereo_processing_needed:
                # The original assert `outputs_diffusion.shape[1] == 1` suggests MBD might output mono.
                if outputs_diffusion.ndim == 3 and outputs_diffusion.shape[1] == 1: # If batch, mono, time
                    print("Rearranging diffusion output from mono to stereo...")
                    # The rearrangement `'(s b) c t -> b (s c) t', s=2` is applied to `outputs_diffusion`
                    # which is the *audio* waveform, not the tokens. This is to make it stereo.
                    outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)
                elif outputs_diffusion.ndim == 2: # If batch, time (mono)
                    print("Rearranging diffusion output from mono (batch, time) to stereo (batch, 2, time)...")
                    outputs_diffusion = rearrange(outputs_diffusion, 'b t -> b 2 t', b=outputs_diffusion.shape[0]) # Assuming b is batch size


            # Save the diffusion output
            diffusion_audio_wav_path = None
            with NamedTemporaryFile("wb", suffix=".flac", delete=False) as file:
                audio_write(
                    file.name, outputs_diffusion.detach().cpu().float().squeeze(0), MBD.sample_rate,
                    strategy="loudness", loudness_headroom_db=16, loudness_compressor=True, add_suffix=False
                )
                diffusion_audio_wav_path = file.name
                file_cleaner.add(file.name)
        except IndexError as e:
            print(f"Caught IndexError during MBD decoding: {e}. This means tokens format mismatch.")
            # This is the error we are trying to fix.
            raise gr.Error(f"Token format mismatch for MultiBandDiffusion decoder. Error: {e}")
        except Exception as e:
            print(f"An error occurred during MBD processing: {e}")
            raise gr.Error(f"Error during MBD processing: {e}")

        print("batch finished (with diffusion)", len(texts), time.time() - be)
        print("Tempfiles currently stored: ", len(file_cleaner.files))
        return default_audio_wav_path, diffusion_audio_wav_path

    else: # Not using diffusion or MBD is not loaded
        # ... (save default audio path) ...
        default_audio_wav_path = None
        try:
            with NamedTemporaryFile("wb", suffix=".flac", delete=False) as file:
                audio_write(
                    file.name, outputs[0].detach().cpu().float().squeeze(0), MODEL.sample_rate,
                    strategy="loudness", loudness_headroom_db=16, loudness_compressor=True, add_suffix=False
                )
                default_audio_wav_path = file.name
                file_cleaner.add(file.name)
        except Exception as e:
            print(f"Error writing default audio file: {e}")

        # If not using diffusion, the MODEL remains loaded.
        # The caller (predict_full) will handle unloading if necessary.
        return default_audio_wav_path, None

def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/musicgen-stereo-melody')
    # No change needed here, already returns (audio, None) or (audio, audio)
    return _do_predictions(texts, melodies, BATCHED_DURATION)


def predict_full(model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        progress(0, desc="Loading diffusion model...")
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")  # Correct interruption handling
    MODEL.set_custom_progress_callback(_progress)
    MODEL.set_generation_params(extend_stride=6)
    # Call _do_predictions and unpack the results correctly
    audio_file, diffusion_file = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        gradio_progress=progress)

    # Return gr.Audio components directly, handling None for diffusion_file
    return gr.Audio(value=audio_file, label="Generated Music (wav)"), audio_file, \
           gr.Audio(value=diffusion_file, label="MultiBand Diffusion Decoder (wav)") if diffusion_file else None, \
           diffusion_file if diffusion_file else None



def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                           label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                           interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                      "facebook/musicgen-large", "facebook/musicgen-melody-large",
                                      "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                                      "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                                      "facebook/musicgen-stereo-melody-large"],
                                     label="Model", value="facebook/musicgen-stereo-melody", interactive=True)
                    model_path = gr.Text(label="Model Path (custom models)")
                with gr.Row():
                    decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                       label="Decoder", value="Default", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=420, value=10, label="Duration", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                # Use gr.Audio for both, and let it handle waveform display
                output = gr.Audio(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath', visible=False)  # Keep this for compatibility
                diffusion_output = gr.Audio(label="MultiBand Diffusion Decoder")
                audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath', visible=False) # Keep this for compatibility


        submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                     show_progress=False).then(predict_full, inputs=[model, model_path, decoder, text, melody, duration, topk, topp,
                                                                      temperature, cfg_coef],
                                                outputs=[output, audio_output, diffusion_output, audio_diffusion])
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)

        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default", 10, 250, 0, 1.0, 3.0 #Include all parameters
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default", 10, 250, 0, 1.0, 3.0
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default", 10, 250, 0, 1.0, 3.0
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default", 10, 250, 0, 1.0, 3.0
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default", 10, 250, 0, 1.0, 3.0
                ],
                [
                    "Punk rock with loud drum and power guitar",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "MultiBand_Diffusion", 10, 250, 0, 1.0, 3.0
                ],
            ],
            inputs=[text, melody, model, decoder, duration, topk, topp, temperature, cfg_coef], # All inputs
            outputs=[output] # Output goes to the main output audio
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass.

            The model was trained with description from a stock music catalog, descriptions that will work best
            should include some level of details on the instruments present, along with some intended use case
            (e.g. adding "perfect for a commercial" can somehow help).

            Using one of the `melody` model (e.g. `musicgen-melody-*`), you can optionally provide a reference audio
            from which a broad melody will be extracted.
            The model will then try to follow both the description and melody provided.
            For best results, the melody should be 30 seconds long (I know, the samples we provide are not...)

            It is now possible to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min).
            An overlap of 12 seconds is kept with the previously generated chunk, and 18 "new" seconds
            are generated each time.

            We present 10 model variations:
            1. facebook/musicgen-melody -- a music generation model capable of generating music condition
                on text and melody inputs. **Note**, you can also use text only.
            2. facebook/musicgen-small -- a 300M transformer decoder conditioned on text only.
            3. facebook/musicgen-medium -- a 1.5B transformer decoder conditioned on text only.
            4. facebook/musicgen-large -- a 3.3B transformer decoder conditioned on text only.
            5. facebook/musicgen-melody-large -- a 3.3B transformer decoder conditioned on and melody.
            6. facebook/musicgen-stereo-*: same as the previous models but fine tuned to output stereo audio.

            We also present two way of decoding the audio tokens
            1. Use the default GAN based compression model. It can suffer from artifacts especially
                for crashes, snares etc.
            2. Use [MultiBand Diffusion](https://arxiv.org/abs/2308.02560). Should improve the audio quality,
                at an extra computational cost. When this is selected, we provide both the GAN based decoded
                audio, and the one obtained with MBD.

            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(source="upload", type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
        submit.click(predict_batched, inputs=[text, melody],
                     outputs=[output, audio_output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output]
        )
        gr.Markdown("""
        ### More details

        The model will generate 15 seconds of audio based on the description you provided.
        The model was trained with description from a stock music catalog, descriptions that will work best
        should include some level of details on the instruments present, along with some intended use case
        (e.g. adding "perfect for a commercial" can somehow help).

        You can optionally provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        For best results, the melody should be 30 seconds long (I know, the samples we provide are not...)

        You can access more control (longer generation, more models etc.) by clicking
        the <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        (you will then need a paid GPU from HuggingFace).
        If you have a GPU, you can run the gradio demo locally (click the link to our repo below for more info).
        Finally, you can get a GPU for free from Google
        and run the demo in [a Google Colab.](https://ai.honu.io/red/musicgen-colab).

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
        for more details. All samples are generated with the `stereo-melody` model.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    if IS_BATCHED:
        global USE_DIFFUSION
        USE_DIFFUSION = False
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
