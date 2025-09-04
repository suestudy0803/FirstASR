# app.py
import numpy as np
import gradio as gr
import torch
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
#from sue_asr_inference_speech2text import Speech2Text
from utils import merge_with_cosine

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

# def merge_guard(prev_committed: str, prev_pending: str, cur_text: str, guard_words=1):
#     merged_prev = (prev_committed + (" " + prev_pending if prev_pending else "")).strip()
#     pw = merged_prev.split()
#     cw = cur_text.strip().split()

#     # LCP(최장 공통 접두사) 길이
#     i = 0
#     while i < len(pw) and i < len(cw) and pw[i] == cw[i]:
#         i += 1

#     added = cw[i:]                   # 새로 추가된 단어들
#     if len(added) <= guard_words:    # 아직 확정 X
#         return prev_committed, " ".join(added)

#     stable = " ".join(added[:-guard_words])   # 확정 구간
#     pending = " ".join(added[-guard_words:])  # 보류(마지막 1단어)
#     committed = ( " ".join(pw + stable.split()) ).strip()
#     return committed, pending

def transcribe_chunk(file_path: str | None, running_text: str):
    global wav_buf

    if running_text is None:
        running_text = ""
    if not file_path:
        return running_text, running_text

    y, sr = librosa.load(file_path, sr=None, mono=True)
    y = _resample_to_16k(y, sr)

    wav_buf = _append_ring(wav_buf, y)
    if wav_buf.shape[0] < MIN_DECODE_SAMPLES:
        return running_text, running_text

    nbests = speech2text(wav_buf)
    hyp = nbests[0][0] if nbests and nbests[0] else ""

    # ✅ 이전 누적(running_text)과 새 가설(hyp)을 코사인 유사도 기반으로 병합
    merged = merge_with_cosine(running_text, hyp,
                               max_overlap_words=8,
                               min_overlap_words=1,
                               ngram=3,
                               threshold=0.55)

    return merged, merged


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

