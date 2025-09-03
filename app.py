# app.py
import string
import numpy as np
import gradio as gr
import torch
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
#from sue_asr_inference_speech2text import Speech2Text

TARGET_SR = 16000
TAG = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

def text_normalizer(text: str) -> str:
    text = text.upper()
    return text.translate(str.maketrans("", "", string.punctuation))

print(">>> Loading ASR model (first run may take a while)...")
d = ModelDownloader()
device = "cuda" if torch.cuda.is_available() else "cpu"
speech2text = Speech2Text(
    **d.download_and_unpack(TAG),
    device=device,
    # ctc_weight=0.7,
    # beam_size=5,
    # batch_size=0,
    # nbest=1,
    # minlenratio=0.0,
    # maxlenratio=0.0,
)
print(f">>> Model ready on: {device}")

wav = []

def transcribe_chunk(file_path: str | None, running_text: str):
    if running_text is None:
        running_text = ""
    if not file_path:
        return running_text, running_text

    # try:
        # Gradio가 wav로 넘겨주므로 그대로 읽으면 됨
    y, sr = librosa.load(file_path)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0  # int16 -> float32
    # elif y.dtype == np.float32:
    #     y = y / 32768.0  # float32 -> (정규화)
        
    # 온전한 데이터
    global wav
    wav = np.concatenate((wav, y))
    print(f"shape: {wav.shape} -> second: {wav.shape[0]/sr:.2f} | dtype: {wav.dtype}")

    # except Exception as e:
    #     print("[WARN] load failed:", e)
    #     return running_text, running_text

    # try:
    nbests = speech2text(wav)
    hyp = nbests[0][0]
    print('inf', hyp)
    # hyp, *_ = nbests[0]
    # piece = text_normalizer(hyp)
    # print(hyp)

    return hyp, hyp
    # except Exception as e:
    #     print("[WARN] ASR failed:", e)
    #     piece = ""

    # if piece.strip():
    #     running_text = (running_text + " " if running_text else "") + piece

    # return running_text, running_text

def clear_state():
    return "", ""

with gr.Blocks(title="Real-time ASR (ESPnet + Gradio)") as demo:
    gr.Markdown("## 🎤 Real-time-ish ASR Demo\n마이크를 켜면 조각 단위로 인식 결과가 아래에 누적됩니다.")

    state = gr.State("")
    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="filepath",   # 파일 경로로 받음
        format="wav",      # Gradio가 내부적으로 wav로 변환
        label="Microphone (streaming)",
        every=3
    )
    out = gr.Textbox(label="Transcript (running)", lines=8, interactive=False)

    mic.stream(fn=transcribe_chunk, inputs=[mic, state], outputs=[state, out])
    gr.Button("Clear").click(fn=clear_state, inputs=None, outputs=[state, out])

if __name__ == "__main__":
    demo.queue().launch()  # 공유 원하면 launch(share=True)
# FirstASR
