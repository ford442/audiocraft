import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, Request

# Import the MusicGen model and related functions from audiocraft
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- 1. Model Loading and Generation (Unchanged) ---

MODEL = None  # Global variable to hold the model

def load_model(version):
    """Loads a MusicGen model based on the version string."""
    print("Loading model", version)
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
    for idx, one_wav in enumerate(output):
        file_path = audio_write(f'gen_{idx}', one_wav, MODEL.sample_rate, strategy="loudness", loudness_compressor=True)
        out_files.append(file_path)
    return out_files

# --- 2. Gradio Interface (Unchanged) ---

def create_gradio_ui():
    """Builds and returns the Gradio web interface."""
    with gr.Blocks() as ui:
        gr.Markdown(
            """
            # MusicGen
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
    return ui

# --- 3. FastAPI Application Setup and Mounting ---

# Initialize the FastAPI app
app = FastAPI()

# Health check endpoint for Vertex AI
@app.get("/healthz", status_code=200)
def health_check():
    return {"status": "ok"}

# Prediction endpoint for Vertex AI
@app.post("/predict")
async def api_predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    duration = data.get("duration", 10)
    
    if MODEL is None:
        MODEL = load_model("melody")
    MODEL.set_generation_params(duration=duration)
    output = MODEL.generate(descriptions=[text])
    
    return {"status": "prediction_complete", "input_text": text}

# --- FIX: Create the Gradio UI and mount it onto the FastAPI app ---
ui = create_gradio_ui()
app = gr.mount_gradio_app(app, ui, path="/")


# Main entry point for uvicorn server (this is now the only server)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
