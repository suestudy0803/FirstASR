import gradio as gr
from transformers import pipeline
import numpy as np
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import torch
import librosa

TARGET_SR = 16000
TAG = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

d = ModelDownloader()
device = "cuda" if torch.cuda.is_available() else "cpu"
speech2text = Speech2Text(
    **d.download_and_unpack(TAG),
    device=device
)

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    
    if y.dtype != np.float32:
        y = y.astype(np.float32) / 32768.0

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    a = speech2text(y)[0][0]

    if stream is None:
        stream = a
    else:
        stream += ' ' + a
    
    return stream, stream 

with gr.Blocks() as demo:
    state = gr.State()
    text = gr.Textbox()

    audio = gr.Audio(sources=["microphone"])

    audio.stream(fn=transcribe, inputs=[state, audio], outputs=[state, text], stream_every=2)

demo.launch()

# demo = gr.Interface(
#     transcribe,
#     ["state",  gr.Audio(sources=["microphone"], every=2)],
#     ["state", "text"],
#     live=True,
# )
# demo.launch()
