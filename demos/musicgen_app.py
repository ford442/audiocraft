import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
import gc

os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCH_LINALG_PREFER_CUSOLVER] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,pinned_use_background_threads:True'
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ['HF_HUB_ENABLE_HF_TRANSFER] = '1'

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

def _do_predictions(texts, melodies, duration, progress=False, gradio_progress=None, chunk_len=1024, overlap_len=128, **gen_kwargs):
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

    # Initialize return variables to None
    default_audio_wav_path = None
    diffusion_audio_wav_path = None

    # Capture stereo info and potentially prepare tokens for MBD *before* unloading
    stereo_processing_needed = False
    if MODEL.compression_model and isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
        stereo_processing_needed = True
        print("Detected stereo compression model.")

    tokens = None
    try:
        if any(m is not None for m in processed_melodies):
            print("Generating with chroma...")
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=True
            )
        else:
            print("Generating without chroma...")
            outputs = MODEL.generate(texts, progress=progress, return_tokens=True)

        tokens = outputs[1]

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

        tokens_for_mbd = tokens
        try:
            if stereo_processing_needed:
                print("Processing tokens for stereo using MusicGen's compression model...")
                if MODEL is None:
                    raise gr.Error("MODEL is None, cannot process stereo tokens.")
                
                stereo_codes = MODEL.compression_model.get_left_right_codes(tokens)
                
                if isinstance(stereo_codes, (tuple, list)) and len(stereo_codes) == 2:
                    left_codes, right_codes = stereo_codes
                    tokens_for_mbd = torch.cat([left_codes, right_codes])
                    print(f"Processed tokens shape for MBD: {tokens_for_mbd.shape}")
                else:
                    print("Warning: Could not process stereo codes as expected. Using original tokens for MBD.")
                    tokens_for_mbd = tokens
            else:
                tokens_for_mbd = tokens

        except Exception as e:
            print(f"Error during stereo token processing: {e}")
            raise gr.Error(f"Failed to prepare tokens for MBD due to stereo processing error: {e}")

        # --- UNLOAD MUSICGEN MODEL ---
        unload_model()

        # --- NOW GENERATE WITH MBD using the prepared tokens ---
        try:
            outputs_diffusion = MBD.tokens_to_wav(tokens_for_mbd)
            
            # --- Stereo formatting for diffusion output (if needed) ---
            if stereo_processing_needed:
                if outputs_diffusion.ndim == 3 and outputs_diffusion.shape[1] == 1: # If batch, mono, time
                    print("Rearranging diffusion output from mono to stereo...")
                    outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)
                elif outputs_diffusion.ndim == 2: # If batch, time (mono)
                    print("Rearranging diffusion output from mono (batch, time) to stereo (batch, 2, time)...")
                    # Assuming b is batch size, and we want to insert a channel dimension
                    outputs_diffusion = rearrange(outputs_diffusion, 'b t -> b 1 t') # Add channel dim first
                    # Now if MBD expects (batch, 2, time), we might need another step.
                    # The original was `(s b) c t -> b (s c) t`. If MBD outputs (2, T) or (B, 2, T), this might be fine.
                    # If MBD outputs (B, 1, T) and expects (B, 2, T), then we'd need to duplicate channels.
                    # Let's stick to the original `rearrange` if it implies (2, T) or (2, 1, T) from MBD.
                    # The previous fix was trying to match the original rearrange:
                    # If `outputs_diffusion` is `(2, T)` and expected `(1, 2, T)` or similar
                    # The original code implies `outputs_diffusion` *is* stereo when it's `(s b) c t`.
                    # If `outputs_diffusion` is `(2, T)` from MBD, and that's stereo, it's fine.
                    # If it's `(1, T)` (mono), we need to make it stereo.
                    if outputs_diffusion.shape[1] == 1: # If shape is (Batch, 1, Time)
                        print("Duplicating mono channel to create stereo output...")
                        outputs_diffusion = outputs_diffusion.repeat(1, 2, 1) # Duplicate channel

            # Save the diffusion output
            with NamedTemporaryFile("wb", suffix=".flac", delete=False) as file:
                audio_write(
                    file.name, outputs_diffusion.detach().cpu().float().squeeze(0), MBD.sample_rate,
                    strategy="loudness", loudness_headroom_db=16, loudness_compressor=True, add_suffix=False
                )
                diffusion_audio_wav_path = file.name
                file_cleaner.add(file.name)
        except IndexError as e:
            print(f"Caught IndexError during MBD decoding: {e}. This means tokens format mismatch.")
            raise gr.Error(f"Token format mismatch for MultiBandDiffusion decoder. Error: {e}")
        except Exception as e:
            print(f"An error occurred during MBD processing: {e}")
            raise gr.Error(f"Error during MBD processing: {e}")

        print("batch finished (with diffusion)", len(texts), time.time() - be)
        print("Tempfiles currently stored: ", len(file_cleaner.files))
        # Now both variables are guaranteed to have a value (either a path or None)
        return default_audio_wav_path, diffusion_audio_wav_path

    else: # Not using diffusion or MBD is not loaded
        # Save the default output
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

        # If not using diffusion, MODEL remains loaded.
        # The caller (predict_full) will handle unloading if necessary.
        return default_audio_wav_path, None # diffusion_audio_wav_path is still None here, which is correct.

def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/musicgen-stereo-melody')
    # No change needed here, already returns (audio, None) or (audio, audio)
    return _do_predictions(texts, melodies, BATCHED_DURATION)


def predict_full(model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, chunk_len, overlap_len, progress=gr.Progress()):
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

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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
        chunk_len=chunk_len, overlap_len=overlap_len,  # Pass the new values
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
                    chunk_len = gr.Slider(minimum=128, maximum=2048, value=1024, step=128, label="Chunk Length", interactive=True)
                    overlap_len = gr.Slider(minimum=16, maximum=512, value=128, step=16, label="Overlap Length", interactive=True)

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
                                                                      temperature, cfg_coef, chunk_len, overlap_len],  # Add the new inputs
                                                outputs=[output, audio_output, diffusion_output, audio_diffusion])
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)

        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default", 10, 250, 0, 1.0, 3.0, 1024, 128 # Add default chunk/overlap values
                ],
            ],
            inputs=[text, melody, model, decoder, duration, topk, topp, temperature, cfg_coef, chunk_len, overlap_len], # All inputs
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
