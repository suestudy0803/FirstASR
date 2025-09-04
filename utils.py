# import math
# import re
# from collections import Counter

# _word_re = re.compile(r"\S+")

# def _normalize_text(s: str) -> str:
#     # 대소문자/공백 표준화 (원한다면 구두점 제거도 가능)
#     return " ".join(s.strip().split()).lower()

# def _char_ngrams(s: str, n: int = 3) -> Counter:
#     # 공백 포함한 연속 문자 n-그램(잘린 단어에도 강함)
#     if not s:
#         return Counter()
#     grams = [s[i:i+n] for i in range(max(0, len(s)-n+1))]
#     return Counter(grams)

# def _cosine_from_counts(a: Counter, b: Counter) -> float:
#     if not a or not b:
#         return 0.0
#     # dot
#     dot = sum(a[k] * b.get(k, 0) for k in a)
#     # norms
#     na = math.sqrt(sum(v*v for v in a.values()))
#     nb = math.sqrt(sum(v*v for v in b.values()))
#     return (dot / (na * nb)) if na > 0 and nb > 0 else 0.0

# def _first_k_words(s: str, k: int) -> str:
#     if k <= 0:
#         return ""
#     ws = _word_re.findall(s)
#     return " ".join(ws[:k])

# def _last_k_words(s: str, k: int) -> str:
#     if k <= 0:
#         return ""
#     ws = _word_re.findall(s)
#     return " ".join(ws[-k:])

# def merge_with_cosine(prev: str, cur: str, *,
#                       max_overlap_words: int = 8,
#                       min_overlap_words: int = 1,
#                       ngram: int = 3,
#                       threshold: float = 0.55) -> str:
#     """
#     prev(이전 누적)와 cur(새 가설) 사이의 접미사-접두사 겹침을
#     문자 n-그램 코사인 유사도로 찾고 병합.
#     - overlap 길이를 [min..max]로 바꿔가며 최대 유사도 선택
#     - 임계치 미만이면 그냥 공백 붙임
#     - 포함/중복 케이스 처리
#     """
#     prev_n = _normalize_text(prev)
#     cur_n  = _normalize_text(cur)

#     if not prev_n:
#         return cur.strip()

#     # 완전 포함/중복 방어
#     if prev_n in cur_n:
#         # 새 가설이 더 길면 그걸 사용
#         return cur.strip()
#     if cur_n in prev_n:
#         # 이미 누적에 포함되어 있으면 그대로 반환(증식 방지)
#         return prev

#     # 단어 토큰 기준 최대 가능한 겹침 한도
#     prev_ws = _word_re.findall(prev_n)
#     cur_ws  = _word_re.findall(cur_n)
#     if not prev_ws or not cur_ws:
#         return (prev + " " + cur).strip()

#     max_k = min(max_overlap_words, len(prev_ws), len(cur_ws))
#     best_k = 0
#     best_sim = -1.0

#     for k in range(min_overlap_words, max_k + 1):
#         suffix = _last_k_words(prev_n, k)
#         prefix = _first_k_words(cur_n,  k)
#         sim = _cosine_from_counts(_char_ngrams(suffix, n=ngram),
#                                   _char_ngrams(prefix, n=ngram))
#         if sim > best_sim:
#             best_sim = sim
#             best_k = k

#     if best_sim >= threshold:
#         # 겹치는 앞 k단어는 버리고 나머지만 붙임
#         attach = " ".join(cur_ws[best_k:])
#         if not attach:  # 전부 겹쳤으면 prev 유지
#             return prev
#         return (prev.strip() + " " + attach).strip()
#     else:
#         # 겹침이 약하면 그냥 공백 붙임
#         return (prev.strip() + " " + cur.strip()).strip()

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# ... (device는 이미 위에서 정해둔 걸 재사용)
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
# 길게 흘러들어오는 문장이라면 너무 긴 입력을 잘라주는 게 안전
emb_model.max_seq_length = 128

import re
_word_re = re.compile(r"\S+")

def _normalize(s: str) -> str:
    # 공백/대소문자 정리 (원하면 구두점 제거도 가능)
    return " ".join(s.strip().split()).lower()

def _first_k_words(s: str, k: int) -> str:
    ws = _word_re.findall(s)
    return " ".join(ws[:k]) if k > 0 else ""

def _last_k_words(s: str, k: int) -> str:
    ws = _word_re.findall(s)
    return " ".join(ws[-k:]) if k > 0 else ""

@torch.inference_mode()
def merge_with_embeddings(
    prev: str,
    cur: str,
    *,
    min_overlap_words: int = 1,
    max_overlap_words: int = 8,
    sim_threshold: float = 0.60
) -> str:
    """
    Sentence-Transformers 임베딩으로 prev 접미사 vs cur 접두사 겹침을 찾고 병합.
    - k=min..max 단어 길이로 슬라이딩, 코사인 유사도가 가장 큰 k 채택
    - 유사도 임계치 미만이면 그냥 공백 붙임
    - 포함/중복/빈 문자열 케이스 방어
    """
    prev_n = _normalize(prev)
    cur_n  = _normalize(cur)

    if not cur_n:
        return prev
    if not prev_n:
        return cur

    # 포함관계 처리(증식 방지)
    if prev_n in cur_n:
        # 새 가설이 더 길거나 같다면 cur 채택
        return cur.strip()
    if cur_n in prev_n:
        # 이미 prev가 cur을 포함
        return prev

    prev_ws = _word_re.findall(prev_n)
    cur_ws  = _word_re.findall(cur_n)
    if not prev_ws or not cur_ws:
        return (prev + (" " if prev and cur else "") + cur).strip()

    kmax = min(max_overlap_words, len(prev_ws), len(cur_ws))
    best_k, best_sim = 0, -1.0

    for k in range(min_overlap_words, kmax + 1):
        suffix = _last_k_words(prev_n, k)
        prefix = _first_k_words(cur_n, k)
        if not suffix or not prefix:
            continue

        v1 = emb_model.encode(suffix, convert_to_tensor=True, normalize_embeddings=True)
        v2 = emb_model.encode(prefix, convert_to_tensor=True, normalize_embeddings=True)
        # normalize_embeddings=True를 썼으니 dot = cosine
        sim = torch.dot(v1, v2).item()

        if sim > best_sim:
            best_sim, best_k = sim, k

    if best_sim >= sim_threshold:
        attach = " ".join(cur_ws[best_k:])   # 겹친 앞 k단어는 버리고 나머지만 붙임
        if not attach:
            return prev
        return (prev.strip() + " " + attach).strip()
    else:
        # 겹침이 약하면 그냥 공백 붙임
        return (prev.strip() + " " + cur.strip()).strip()
