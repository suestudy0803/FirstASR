import numpy as np
import gradio as gr
import torch
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

# ====== 설정 ======
TARGET_SR = 16000
TAG = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

# 링버퍼 길이(초)와 최소 디코딩 길이
RING_SECONDS = 4      # 최근 8초만 유지
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
    # 필요 시 조정:s
    # ctc_weight=0.7,
    # beam_size=5,
    # nbest=1,
    # minlenratio=0.0,
    # maxlenratio=0.0,
)
print(f">>> Model ready on: {device}")

# ====== 전역 버퍼: 최근 오디오만 유지 ======
wav_buf = np.array([], dtype=np.float32)

# ---------------------------
# 오디오 전처리 & 링버퍼 유틸
# ---------------------------
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

# ---------------------------
# 소리(파형) 겹침 감지/제거 (NCC 계열)
# ---------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-8:
        return x * 0.0
    return (x - m) / s

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # 두 벡터가 길이가 다르면 앞쪽을 자름
    L = min(a.size, b.size)
    if L == 0:
        return 0.0
    a = a[-L:]
    b = b[:L]
    an = _zscore(a)
    bn = _zscore(b)
    denom = np.linalg.norm(an) * np.linalg.norm(bn)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(an, bn) / denom)

def trim_audio_overlap(prev_buf: np.ndarray,
                       new_chunk: np.ndarray,
                       sr: int,
                       max_overlap_sec: float = 1.0,
                       min_overlap_sec: float = 0.08,
                       hop_ms: int = 10,
                       sim_threshold: float = 0.75):
    """
    prev_buf의 '마지막 최대 max_overlap_sec'과 new_chunk의 '처음'을 겹침으로 가정하고
    L을 크게→작게로 스캔하며 코사인 유사도가 sim_threshold 이상인 가장 긴 L을 찾음.
    찾으면 new_chunk[:L]를 제거하여 반환.
    반환: (trimmed_chunk, detected_overlap_samples, best_sim)
    """
    if prev_buf.size == 0 or new_chunk.size == 0:
        return new_chunk, 0, 0.0

    max_L = int(max_overlap_sec * sr)
    min_L = int(min_overlap_sec * sr)
    hop = int(hop_ms * sr / 1000)

    tail = prev_buf[-max_L:] if prev_buf.size >= max_L else prev_buf.copy()
    head = new_chunk[:max_L] if new_chunk.size >= max_L else new_chunk.copy()

    # 스캔 시작 길이: tail/head의 공통 가능한 최대 길이
    start_L = min(tail.size, head.size)
    if start_L < min_L:
        return new_chunk, 0, 0.0

    best_L = 0
    best_sim = -1.0

    # L을 크게->작게로 내려가며 유사도 검사 (연산량 줄이려 hop 간격 사용)
    for L in range(start_L, min_L - 1, -hop):
        sim = _cos_sim(tail[-L:], head[:L])
        if sim > best_sim:
            best_sim = sim
            best_L = L
        if sim >= sim_threshold:
            # 첫번째로 임계값을 넘는 가장 긴 L을 채택하고 중단
            break

    if best_sim >= sim_threshold and best_L >= min_L:
        # 겹침 구간 제거
        trimmed = new_chunk[best_L:]
        print(f"[overlap] detected {best_L/sr:.3f}s (sim={best_sim:.3f}) -> trimming")
        return trimmed, best_L, best_sim
    else:
        return new_chunk, 0, float(best_sim)

# ---------------------------
# 스트리밍 콜백
# ---------------------------
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

    # 2-1) '소리 차원' 겹침 제거 (prev tail vs new head)
    y, ovl_samps, ovl_sim = trim_audio_overlap(
        prev_buf=wav_buf,
        new_chunk=y,
        sr=TARGET_SR,
        max_overlap_sec=1.0,   # 필요에 따라 0.5~1.5s 조정
        min_overlap_sec=0.08,  # 80ms 이상일 때만 겹침으로 인정
        hop_ms=10,             # 10ms 스텝
        sim_threshold=0.75     # 0.7~0.85 사이 조절
    )

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

    if not hyp.strip():
        return running_text, running_text

    # 6) 텍스트는 단순히 이어붙임 (오디오에서 이미 겹침 제거됨)
    merged = (running_text + " " + hyp).strip() if running_text.strip() else hyp

    # 너무 긴 텍스트는 뒷부분만 유지 (메모리 관리)
    if len(merged.split()) > 100:
        words = merged.split()
        merged = ' '.join(words[-80:])
        print(f"[transcribe] Text truncated to last 80 words")

    print(f"[transcribe] New hyp: '{hyp}'")
    print(f"[transcribe] Merged: '{merged[:50]}{'...' if len(merged) > 50 else ''}'")

    return merged, merged

def clear_state():
    """UI와 버퍼 동시 초기화"""
    global wav_buf
    wav_buf = np.array([], dtype=np.float32)
    return "", ""

# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="Real-time ASR (Audio-level Overlap Trimming)") as demo:
    gr.Markdown("## 🎤 Real-time ASR (Audio-level Overlap Trimming)\n오디오 파형에서 겹침을 감지/제거한 뒤 인식합니다.")

    state = gr.State("")
    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="filepath",  # 파일 경로로 콜백에 전달
        format="wav",
        label="Microphone (streaming)",
        every=2,          # 콜백 주기(초) - 필요 시 조정
    )
    out = gr.Textbox(label="Transcript", lines=8, interactive=False)

    mic.stream(fn=transcribe_chunk, inputs=[mic, state], outputs=[state, out])
    gr.Button("Clear").click(fn=clear_state, inputs=None, outputs=[state, out])

if __name__ == "__main__":
    # 전역 버퍼 충돌을 피하려면 concurrency_count=1 권장
    demo.queue().launch()
