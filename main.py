import gradio as gr
import torch
import os

# Import the MusicGen model and related functions from audiocraft
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- 1. Model Loading and Generation (Unchanged) ---

MODEL = None  # Global variable to hold the model

def load_model(version):
    """Loads a MusicGen model based on the version string."""
    print("Loading model", version)
    # NOTE: If you baked models into the image, change this path.
    # For now, we are letting it download on first run for simplicity.
    return MusicGen.get_pretrained(version)

def predict(model, text, melody, duration, topk, topp, temperature, cfg_coef):
    """Main prediction function that generates audio."""
    global MODEL
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)

    if duration > MODEL.lm.cfg.dataset.segment_duration:
        raise gr.Error("MusicGen currently supports durations of up to 30 seconds!")

    MODEL.set_generation_params(
        use_sampling=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        duration=duration,
        cfg_coef=cfg_coef,
    )

    if melody:
        sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
        output = MODEL.generate_with_chroma(
            descriptions=[text],
            melody_wavs=melody,
            melody_sample_rate=sr,
            progress=True
        )
    else:
        output = MODEL.generate(descriptions=[text], progress=True)

    output = output.detach().cpu().float()
    out_files = []
    # Create a temporary directory for the output files
    os.makedirs("/tmp/gradio", exist_ok=True)
    for idx, one_wav in enumerate(output):
        file_path = f"/tmp/gradio/gen_{idx}.flac"
        audio_write(file_path, one_wav, MODEL.sample_rate, strategy="loudness", loudness_compressor=True)
        out_files.append(file_path)
    return out_files

# --- 2. Gradio Interface (Unchanged) ---

with gr.Blocks() as ui:
    gr.Markdown(
        """
        # MusicGen on Vertex AI
        This is your private demo for MusicGen, a model for generating music from text.
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="Input Text", interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
            with gr.Row():
                submit = gr.Button("Generate")
            with gr.Row():
                model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
            with gr.Row():
                duration = gr.Slider(minimum=1, maximum=30, value=10, label="Duration", interactive=True)
            with gr.Row():
                topk = gr.Number(label="Top-k", value=250, interactive=True)
                topp = gr.Number(label="Top-p", value=0, interactive=True)
                temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
        with gr.Column():
            output = gr.File(label="Generated Music")

    submit.click(predict, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef], outputs=[output])

# --- 3. Launch the App ---
# We tell Gradio to listen on all network interfaces on the required port 8080.
ui.launch(server_name="0.0.0.0", server_port=8080)
