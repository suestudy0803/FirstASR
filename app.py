# # app.py
# import numpy as np
# import gradio as gr
# import torch
# import librosa
# from espnet_model_zoo.downloader import ModelDownloader
# from espnet2.bin.asr_inference import Speech2Text
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# # ====== 설정 ======
# TARGET_SR = 16000
# TAG = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

# # 링버퍼 길이(초)와 최소 디코딩 길이
# RING_SECONDS = 8          # 최근 8초만 유지
# MIN_DECODE_SECONDS = 1.0  # 최소 1초 이상일 때만 디코딩

# RING_SAMPLES = TARGET_SR * RING_SECONDS
# MIN_DECODE_SAMPLES = int(TARGET_SR * MIN_DECODE_SECONDS)

# # 스마트 병합 설정
# COSINE_THRESHOLD = 0.3    # 코사인 유사도 임계값

# # ====== 모델 로드 ======
# print(">>> Loading ASR model (first run may take a while)...")
# d = ModelDownloader()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# speech2text = Speech2Text(
#     **d.download_and_unpack(TAG),
#     device=device,
#     # 필요 시 조정:
#     # ctc_weight=0.7,
#     # beam_size=5,
#     # nbest=1,
#     # minlenratio=0.0,
#     # maxlenratio=0.0,
# )
# print(f">>> Model ready on: {device}")

# # ====== 전역 버퍼: 최근 오디오만 유지 ======
# wav_buf = np.array([], dtype=np.float32)

# def _resample_to_16k(y: np.ndarray, sr: int) -> np.ndarray:
#     """원본 SR 유지해서 읽고, 여기서만 16k로 1회 리샘플"""
#     if sr == TARGET_SR:
#         return y.astype(np.float32, copy=False)
#     y16 = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="polyphase")
#     return y16.astype(np.float32, copy=False)

# def _append_ring(buf: np.ndarray, chunk: np.ndarray) -> np.ndarray:
#     """buf 뒤에 chunk를 붙이고, 최근 RING_SECONDS만 유지"""
#     if buf.size == 0:
#         cat = chunk
#     else:
#         cat = np.concatenate([buf, chunk])
#     if cat.shape[0] > RING_SAMPLES:
#         cat = cat[-RING_SAMPLES:]  # 뒤에서 RING_SECONDS만 유지
#     return cat

# def do_cosimilarity(text1, text2):
#     """두 텍스트의 코사인 유사도 계산"""
#     if not text1.strip() or not text2.strip():
#         return 0.0
    
#     try:
#         # TF-IDF 벡터화
#         vectorizer = TfidfVectorizer().fit([text1, text2])
#         tfidf = vectorizer.transform([text1, text2])
#         # 코사인 유사도 계산
#         cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
#         return cosine_sim
#     except:
#         return 0.0

# def find_overlap_and_merge(text1, text2):
#     """두 텍스트 간의 겹치는 부분을 찾아서 병합"""
#     if not text1.strip():
#         return text2
#     if not text2.strip():
#         return text1
    
#     # 단어 단위로 분할
#     words1 = text1.strip().split()
#     words2 = text2.strip().split()
    
#     max_overlap = 0
#     best_overlap_pos = -1
    
#     # text1의 뒷부분과 text2의 앞부분에서 겹치는 부분 찾기
#     min_len = min(len(words1), len(words2))
#     for i in range(1, min_len + 1):
#         if words1[-i:] == words2[:i]:
#             if i > max_overlap:
#                 max_overlap = i
#                 best_overlap_pos = i
    
#     if best_overlap_pos > 0:
#         # 겹치는 부분이 있으면 병합
#         merged = words1 + words2[best_overlap_pos:]
#         return ' '.join(merged)
#     else:
#         # 겹치는 부분이 없으면 단순 연결
#         return text1 + ' ' + text2

# def smart_concat(text1, text2):
#     """스마트 텍스트 연결: 코사인 유사도 기반으로 overlap 여부 결정"""
#     if not text1.strip():
#         return text2
#     if not text2.strip():
#         return text1
    
#     # 코사인 유사도 계산
#     similarity = do_cosimilarity(text1, text2)
#     print(f"[smart_concat] Cosine similarity: {similarity:.3f}")
    
#     if similarity > COSINE_THRESHOLD:
#         # 유사도가 높으면 overlap을 찾아서 병합
#         merged = find_overlap_and_merge(text1, text2)
#         print(f"[smart_concat] Overlap merge applied")
#         return merged
#     else:
#         # 유사도가 낮으면 단순 연결
#         merged = text1 + ' ' + text2
#         print(f"[smart_concat] Simple concatenation applied")
#         return merged

# def transcribe_chunk(file_path: str | None, running_text: str):
#     """Gradio 오디오 스트림 콜백 (주기적으로 호출)"""
#     global wav_buf

#     if running_text is None:
#         running_text = ""
#     if not file_path:
#         return running_text, running_text

#     # 1) 파일→파형 (원본 SR 유지)
#     y, sr = librosa.load(file_path, sr=None, mono=True)
#     # 2) 16k로 한 번만 리샘플
#     y = _resample_to_16k(y, sr)

#     # 3) 링버퍼에 누적
#     wav_buf = _append_ring(wav_buf, y)
#     cur_sec = wav_buf.shape[0] / TARGET_SR
#     print(f"[buf] {cur_sec:.2f}s @16k, dtype={wav_buf.dtype}")

#     # 4) 너무 짧으면 디코딩 생략(불필요 호출/잡음 출력 방지)
#     if wav_buf.shape[0] < MIN_DECODE_SAMPLES:
#         return running_text, running_text

#     # 5) 최근 구간만 디코딩
#     nbests = speech2text(wav_buf)
#     hyp = nbests[0][0] if nbests and nbests[0] else ""
    
#     if not hyp.strip():
#         return running_text, running_text

#     # 6) 스마트 병합 적용
#     merged = smart_concat(running_text, hyp)
    
#     print(f"[transcribe] Previous: '{running_text}'")
#     print(f"[transcribe] New hyp: '{hyp}'")
#     print(f"[transcribe] Merged: '{merged}'")
    
#     return merged, merged

# def clear_state():
#     """UI와 버퍼 동시 초기화"""
#     global wav_buf
#     wav_buf = np.array([], dtype=np.float32)
#     return "", ""

# with gr.Blocks(title="Real-time ASR with Smart Concatenation") as demo:
#     gr.Markdown("## 🎤 Real-time ASR with Smart Text Merging\n마이크를 켜면 코사인 유사도 기반으로 스마트하게 텍스트를 병합합니다.")

#     state = gr.State("")
#     mic = gr.Audio(
#         sources=["microphone"],
#         streaming=True,
#         type="filepath",  # 파일 경로로 콜백에 전달
#         format="wav",
#         label="Microphone (streaming)",
#         every = 3,
#     )
#     out = gr.Textbox(label="Transcript (smart concatenated)", lines=8, interactive=False)

#     # 주기 설정은 stream()에!
#     mic.stream(fn=transcribe_chunk, inputs=[mic, state], outputs=[state, out])

#     gr.Button("Clear").click(fn=clear_state, inputs=None, outputs=[state, out])

# if __name__ == "__main__":
#     # 전역 버퍼 충돌을 피하려면 concurrency_count=1 권장
#     demo.queue().launch()  # 공유 필요 시 launch(share=True)


# app.py
import numpy as np
import gradio as gr
import torch
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import jellyfish

# ====== 설정 ======
TARGET_SR = 16000
TAG = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

# 링버퍼 길이(초)와 최소 디코딩 길이
RING_SECONDS = 8          # 최근 8초만 유지
MIN_DECODE_SECONDS = 1.0  # 최소 1초 이상일 때만 디코딩

RING_SAMPLES = TARGET_SR * RING_SECONDS
MIN_DECODE_SAMPLES = int(TARGET_SR * MIN_DECODE_SECONDS)

# 스마트 병합 설정
#COSINE_THRESHOLD = 0.3    # 코사인 유사도 임계값

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

# def do_cosimilarity(text1, text2):
#     """두 텍스트의 코사인 유사도 계산"""
#     if not text1.strip() or not text2.strip():
#         return 0.0
    
#     try:
#         # TF-IDF 벡터화
#         vectorizer = TfidfVectorizer().fit([text1, text2])
#         tfidf = vectorizer.transform([text1, text2])
#         # 코사인 유사도 계산
#         cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
#         return cosine_sim
#     except:
#         return 0.0

def phonetic_similarity(text1: str, text2: str) -> bool:
    """
    문장 전체를 대상으로 Soundex/Metaphone 변환 후 같으면 유사하다고 간주.
    참고: jellyfish는 단어 단위에서 더 신뢰도가 높음. 필요하면
    마지막 k-단어 vs 첫 k-단어 비교로 확장 가능.
    """
    t1 = text1.strip()
    t2 = text2.strip()
    if not t1 or not t2:
        return False

    try:
        sdx1 = jellyfish.soundex(t1)
        sdx2 = jellyfish.soundex(t2)
    except Exception:
        sdx1 = sdx2 = ""

    try:
        meta1 = jellyfish.metaphone(t1)
        meta2 = jellyfish.metaphone(t2)
    except Exception:
        meta1 = meta2 = ""

    return (sdx1 == sdx2) or (meta1 == meta2)


def find_overlap_and_merge(text1, text2):
    """두 텍스트 간의 겹치는 부분을 찾아서 병합"""
    if not text1.strip():
        return text2
    if not text2.strip():
        return text1
    
    # 단어 단위로 분할
    words1 = text1.strip().split()
    words2 = text2.strip().split()
    
    max_overlap = 0
    best_overlap_pos = -1
    
    # text1의 뒷부분과 text2의 앞부분에서 겹치는 부분 찾기
    min_len = min(len(words1), len(words2))
    for i in range(1, min_len + 1):
        if words1[-i:] == words2[:i]:
            if i > max_overlap:
                max_overlap = i
                best_overlap_pos = i
    
    if best_overlap_pos > 0:
        # 겹치는 부분이 있으면 병합
        merged = words1 + words2[best_overlap_pos:]
        return ' '.join(merged)
    else:
        # 겹치는 부분이 없으면 단순 연결
        return text1 + ' ' + text2

# def smart_concat(text1, text2):
#     """스마트 텍스트 연결: 코사인 유사도 기반으로 overlap 여부 결정"""
#     if not text1.strip():
#         return text2
#     if not text2.strip():
#         return text1
    
#     # 코사인 유사도 계산
#     similarity = do_cosimilarity(text1, text2)
#     print(f"[smart_concat] Cosine similarity: {similarity:.3f}")
    
#     if similarity > COSINE_THRESHOLD:
#         # 유사도가 높으면 overlap을 찾아서 병합
#         merged = find_overlap_and_merge(text1, text2)
#         print(f"[smart_concat] Overlap merge applied")
#         return merged
#     else:
#         # 유사도가 낮으면 단순 연결
#         merged = text1 + ' ' + text2
#         print(f"[smart_concat] Simple concatenation applied")
#         return merged

def smart_concat(text1, text2):
    """발음 기반 유사도(jellyfish)로 overlap 병합 여부 결정"""
    if not text1.strip():
        return text2
    if not text2.strip():
        return text1

    similar = phonetic_similarity(text1, text2)
    print(f"[smart_concat] Phonetic similar?: {bool(similar)}")

    if similar:
        merged = find_overlap_and_merge(text1, text2)
        print(f"[smart_concat] Phonetic-overlap merge applied")
        return merged
    else:
        merged = text1 + ' ' + text2
        print(f"[smart_concat] Simple concatenation applied")
        return merged

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
    
    if not hyp.strip():
        return running_text, running_text

    # 6) 스마트 병합 적용
    merged = smart_concat(running_text, hyp)
    
    # 너무 긴 텍스트는 뒷부분만 유지 (메모리 관리)
    if len(merged.split()) > 100:
        words = merged.split()
        merged = ' '.join(words[-80:])  # 마지막 80단어만 유지
        print(f"[transcribe] Text truncated to last 80 words")
    
    print(f"[transcribe] Previous: '{running_text[:50]}{'...' if len(running_text) > 50 else ''}'")
    print(f"[transcribe] New hyp: '{hyp}'")
    print(f"[transcribe] Merged: '{merged[:50]}{'...' if len(merged) > 50 else ''}'")
    
    return merged, merged

def clear_state():
    """UI와 버퍼 동시 초기화"""
    global wav_buf
    wav_buf = np.array([], dtype=np.float32)
    return "", ""

with gr.Blocks(title="Real-time ASR with Smart Concatenation") as demo:
    gr.Markdown("## 🎤 Real-time ASR with Smart Text Merging\n마이크를 켜면 코사인 유사도 기반으로 스마트하게 텍스트를 병합합니다.")

    state = gr.State("")
    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="filepath",  # 파일 경로로 콜백에 전달
        format="wav",
        label="Microphone (streaming)",
        every = 3,
    )
    out = gr.Textbox(label="Transcript (smart concatenated)", lines=8, interactive=False)

    # 주기 설정은 stream()에!
    mic.stream(fn=transcribe_chunk, inputs=[mic, state], outputs=[state, out])

    gr.Button("Clear").click(fn=clear_state, inputs=None, outputs=[state, out])

if __name__ == "__main__":
    # 전역 버퍼 충돌을 피하려면 concurrency_count=1 권장
    demo.queue().launch()  # 공유 필요 시 launch(share=True)

