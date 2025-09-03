# app.py
import numpy as np
import gradio as gr
import torch
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
#from sue_asr_inference_speech2text import Speech2Text

# ====== 설정 ======
TARGET_SR = 16000
TAG = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

# 링버퍼 길이(초)와 최소 디코딩 길이
RING_SECONDS = 8          # 최근 8초만 유지
MIN_DECODE_SECONDS = 1.0  # 최소 1초 이상일 때만 디코딩

RING_SAMPLES = TARGET_SR * RING_SECONDS
MIN_DECODE_SAMPLES = int(TARGET_SR * MIN_DECODE_SECONDS)

# ====== 모델 로드 ======
print(">>> Loading ASR model (first run may take a while)...")
d = ModelDownloader()
device = "cuda" if torch.cuda.is_available() else "cpu"
speech2text = Speech2Text(
    **d.download_and_unpack(TAG),
    device=device,
    # 필요 시 조정:
    # ctc_weight=0.7,
    # beam_size=5,
    # nbest=1,
    # minlenratio=0.0,
    # maxlenratio=0.0,
)
print(f">>> Model ready on: {device}")

# ====== 전역 버퍼: 최근 오디오만 유지 ======
wav_buf = np.array([], dtype=np.float32)

def _resample_to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """원본 SR 유지해서 읽고, 여기서만 16k로 1회 리샘플"""
    if sr == TARGET_SR:
        return y.astype(np.float32, copy=False)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="polyphase")
    return y16.astype(np.float32, copy=False)

def _append_ring(buf: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """buf 뒤에 chunk를 붙이고, 최근 RING_SECONDS만 유지"""
    if buf.size == 0:
        cat = chunk
    else:
        cat = np.concatenate([buf, chunk])
    if cat.shape[0] > RING_SAMPLES:
        cat = cat[-RING_SAMPLES:]  # 뒤에서 RING_SECONDS만 유지
    return cat

def transcribe_chunk(file_path: str | None, running_text: str):
    """Gradio 오디오 스트림 콜백 (주기적으로 호출)"""
    global wav_buf

    if running_text is None:
        running_text = ""
    if not file_path:
        return running_text, running_text

    # 1) 파일→파형 (원본 SR 유지)
    y, sr = librosa.load(file_path, sr=None, mono=True)
    # 2) 16k로 한 번만 리샘플
    y = _resample_to_16k(y, sr)

    # 3) 링버퍼에 누적
    wav_buf = _append_ring(wav_buf, y)
    cur_sec = wav_buf.shape[0] / TARGET_SR
    print(f"[buf] {cur_sec:.2f}s @16k, dtype={wav_buf.dtype}")

    # 4) 너무 짧으면 디코딩 생략(불필요 호출/잡음 출력 방지)
    if wav_buf.shape[0] < MIN_DECODE_SAMPLES:
        return running_text, running_text

    # 5) 최근 구간만 디코딩
    nbests = speech2text(wav_buf)
    hyp = nbests[0][0] if nbests and nbests[0] else ""

    # 필요하다면 누적 로직 사용:
    # running_text = (running_text + " " if running_text else "") + hyp

    return hyp, hyp

def clear_state():
    """UI와 버퍼 동시 초기화"""
    global wav_buf
    wav_buf = np.array([], dtype=np.float32)
    return "", ""

with gr.Blocks(title="Real-time ASR (ESPnet + Gradio)") as demo:
    gr.Markdown("## 🎤 Real-time-ish ASR Demo\n마이크를 켜면 최근 8초만 유지하는 링버퍼로 디코딩합니다.")

    state = gr.State("")
    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="filepath",  # 파일 경로로 콜백에 전달
        format="wav",
        label="Microphone (streaming)",
        every = 3,
    )
    out = gr.Textbox(label="Transcript (running)", lines=8, interactive=False)

    # 주기 설정은 stream()에!
    mic.stream(fn=transcribe_chunk, inputs=[mic, state], outputs=[state, out])

    gr.Button("Clear").click(fn=clear_state, inputs=None, outputs=[state, out])

if __name__ == "__main__":
    # 전역 버퍼 충돌을 피하려면 concurrency_count=1 권장
    demo.queue().launch()  # 공유 필요 시 launch(share=True)
