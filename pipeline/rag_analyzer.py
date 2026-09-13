"""
3단계: RAG 기반 공격 시그니처 매칭 + LLM 설명 생성.

흐름:
  1. corpus/*.md 문서들을 임베딩해 벡터 인덱스 구축 (최초 1회)
  2. 의심 활동의 정성적 설명(features_to_description 결과)을 쿼리로 임베딩
  3. 가장 유사한 공격 시그니처 문서들을 검색 (의미 기반 유사도)
  4. 검색된 문서들을 근거로 LLM이 "어떤 공격에 가장 가까운가 / 정상인가"를 판정·설명

핵심 설계:
  - 공격 기법마다 별도 탐지모델을 두지 않는다. 문서만 추가하면 새 공격에 대응(확장성).
  - 매칭은 숫자가 아니라 정성적 특성('규칙적인가', '평균 유지되는가')으로 한다.
  - LLM은 경량 모델로 시작, 나중에 교체 가능 (Ollama HTTP 또는 로컬 transformers).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import glob
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


@dataclass(frozen=True)
class SearchResult:
    """검색 결과. 기존 `(name, score, text)` unpack과도 호환된다."""

    name: str
    score: float
    text: str
    meta: dict
    filtered_by: list[str]

    def __iter__(self):
        yield self.name
        yield self.score
        yield self.text

    def __getitem__(self, index: int):
        return (self.name, self.score, self.text)[index]


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"null", "None", ""}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value)
    if value in {"true", "false"}:
        return value == "true"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_front_matter(text: str) -> tuple[dict, str]:
    """YAML-lite front-matter를 파싱하고 본문만 반환한다."""
    if not text.startswith("---\n"):
        return {}, text
    try:
        raw_meta, body = text.split("---\n", 2)[1:]
    except ValueError:
        return {}, text
    meta = {}
    for line in raw_meta.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = _parse_scalar(value)
    return meta, body.lstrip()


class SignatureRetriever:
    """공격 시그니처 문서 코퍼스에 대한 의미 기반 검색기.

    backend:
      "sbert"  : sentence-transformers 임베딩 (권장, 인터넷/모델 필요)
      "tfidf"  : scikit-learn TF-IDF (오프라인 폴백, 모델 다운로드 불필요)
    """

    def __init__(self, corpus_dir: Path = CORPUS_DIR,
                 backend: str = "sbert", model_name: str = "all-MiniLM-L6-v2",
                 include_docs: set[str] | None = None):
        self.backend = backend
        self.docs = []       # (name, text)
        self.meta = {}       # name -> front-matter dict
        self.last_filter_report = {}
        self.include_docs = include_docs
        self.embeddings = None
        self._model = None
        self._vectorizer = None

        if backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        self._load_corpus(corpus_dir)

    def _load_corpus(self, corpus_dir: Path):
        for path in sorted(glob.glob(str(corpus_dir / "*.md"))):
            name = Path(path).stem
            if self.include_docs is not None and name not in self.include_docs:
                continue
            text = Path(path).read_text(encoding="utf-8")
            meta, body = parse_front_matter(text)
            self.meta[name] = meta
            self.docs.append((name, body))
        if not self.docs:
            raise RuntimeError(f"코퍼스 문서를 찾을 수 없습니다: {corpus_dir}")
        texts = [t for _, t in self.docs]

        if self.backend == "sbert":
            self.embeddings = self._model.encode(texts, normalize_embeddings=True)
        elif self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            # 한국어 문서라 문자 n-gram 기반이 단어 토큰화보다 강건함
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            mat = self._vectorizer.fit_transform(texts)
            # 정규화 후 저장
            from sklearn.preprocessing import normalize
            self.embeddings = normalize(mat)
        else:
            raise ValueError(f"알 수 없는 backend: {self.backend}")

        print(f"[RAG:{self.backend}] {len(self.docs)}개 시그니처 문서 인덱싱 완료: "
              f"{[n for n, _ in self.docs]}")

    def _range_filter_failures(self, name: str, features: dict) -> list[str]:
        meta = self.meta.get(name, {})
        failures = []
        if meta.get("multi_gpu_sync") == "required":
            sync_value = features.get("multi_gpu_sync_index", features.get("sync_index"))
            if sync_value is None or float(sync_value) < 0.8:
                failures.append("multi_gpu_sync")
        numeric_keys = (
            "swing_ratio",
            "duty_regularity",
            "ramp_max_w_per_s",
            "high_load_fraction",
        )
        for key in numeric_keys:
            bounds = meta.get(key)
            if bounds is None or key not in features:
                continue
            if not isinstance(bounds, list) or len(bounds) != 2:
                continue
            lo, hi = bounds
            if lo is None or hi is None:
                continue
            value = features.get(key)
            if value is None or not np.isfinite(float(value)):
                continue
            # The synthetic windows are small; use a narrow tolerance so the
            # hard filter remains robust to floating-point and window edge noise.
            if key in {"periodicity_strength", "duty_regularity", "high_load_fraction"}:
                width = max(float(hi) - float(lo), 0.05)
                tol = 0.10 * width
            else:
                width = max(float(hi) - float(lo), abs(float(hi)), abs(float(lo)), 1.0)
                tol = 0.10 * width
            if float(value) < float(lo) - tol or float(value) > float(hi) + tol:
                failures.append(key)
        return failures

    def _candidate_indices(self, features: dict | None) -> list[int]:
        self.last_filter_report = {}
        if features is None:
            return list(range(len(self.docs)))
        indices = []
        for idx, (name, _) in enumerate(self.docs):
            failures = self._range_filter_failures(name, features)
            if failures:
                self.last_filter_report[name] = failures
            else:
                indices.append(idx)
                self.last_filter_report[name] = []
        return indices

    def search(self, query: str, top_k: int = 3, features: dict | None = None):
        """쿼리와 가장 유사한 문서 top_k개를 반환한다.

        features가 주어지면 corpus front-matter의 수치 조건으로 먼저
        후보를 거른 뒤 살아남은 문서만 텍스트 유사도로 정렬한다.
        """
        candidate_indices = self._candidate_indices(features)
        if not candidate_indices:
            return []
        if self.backend == "sbert":
            q_emb = self._model.encode([query], normalize_embeddings=True)[0]
            scores = self.embeddings @ q_emb
        else:  # tfidf
            from sklearn.preprocessing import normalize
            q_vec = normalize(self._vectorizer.transform([query]))
            scores = (self.embeddings @ q_vec.T).toarray().ravel()
        candidate_scores = np.asarray([scores[i] for i in candidate_indices])
        order = np.argsort(-candidate_scores)[:top_k]
        return [
            SearchResult(
                self.docs[candidate_indices[i]][0],
                float(candidate_scores[i]),
                self.docs[candidate_indices[i]][1],
                self.meta.get(self.docs[candidate_indices[i]][0], {}),
                [],
            )
            for i in order
        ]


def analyze_with_llm(description: str, retrieved, context: dict | None = None,
                      backend: str = "ollama", model: str = "gemma3:12b"):
    """검색된 시그니처 문서를 근거로 LLM이 판정·설명 생성.

    backend="ollama": 로컬 Ollama HTTP API 호출 (http://localhost:11434)
    backend="stub":   LLM 없이 규칙 기반 요약 (LLM 준비 전 파이프라인 확인용)
    """
    if not retrieved:
        return {
            "risk": "주의",
            "closest_match": "unknown",
            "reason": "수치 조건을 통과한 시그니처가 없어 알려진 문서로 확정하지 않았습니다.",
            "_backend": f"{backend}:empty-retrieval",
        }
    # 검색 근거 정리
    evidence = "\n\n".join(
        f"[{name}] (유사도 {score:.3f})\n{text[:1200]}"
        for name, score, text in retrieved
    )
    ctx_str = ""
    if context:
        ctx_str = "\n\n관측 맥락:\n" + "\n".join(f"- {k}: {v}" for k, v in context.items())

    prompt = f"""당신은 데이터센터 GPU 전력 이상탐지 분석가입니다.
아래 '관측된 활동'이 알려진 공격 시그니처 중 무엇과 가장 유사한지, 아니면 정상인지
판단하고 근거를 설명하세요. 확정하지 말고 위험도 등급(정상/주의/의심)과 근거를 제시하세요.

## 관측된 활동
{description}{ctx_str}

## 검색된 참고 시그니처 (유사도 순)
{evidence}

## 출력 형식 (JSON)
{{"risk": "정상|주의|의심", "closest_match": "가장 가까운 시그니처 이름 또는 normal",
  "reason": "판단 근거를 2-3문장으로"}}
"""

    if backend == "stub":
        # LLM 없이도 파이프라인이 도는지 확인하기 위한 규칙 기반 대체
        top_name, top_score, _ = retrieved[0]
        top_meta = getattr(retrieved[0], "meta", {})
        risk = "의심" if top_score > 0.35 and top_meta.get("category") != "benign" else "주의"
        if top_meta.get("category") == "benign":
            risk = "정상"
        return {
            "risk": risk,
            "closest_match": top_name,
            "reason": f"(stub) 검색 최상위 문서 '{top_name}'(유사도 {top_score:.3f}) 기준 규칙 판정. "
                      f"실제 배포 시 LLM으로 교체.",
            "_backend": "stub",
        }

    if backend == "ollama":
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            parsed = json.loads(out["response"])
            parsed["_backend"] = f"ollama:{model}"
            return parsed
        except Exception as e:
            return {
                "risk": "분석실패",
                "closest_match": retrieved[0][0],
                "reason": f"Ollama 호출 실패 ({e}). 'ollama serve' 실행 및 모델 pull 확인. "
                          f"우선 backend='stub'로 파이프라인 검증 가능.",
                "_backend": "ollama-error",
            }

    raise ValueError(f"알 수 없는 backend: {backend}")


if __name__ == "__main__":
    # 자체 테스트: features.py의 두 예시 설명을 RAG로 매칭
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import compute_window_features, features_to_description

    # 이 샌드박스는 오프라인이라 tfidf로 검증. 실제 서버에서는 backend="sbert" 권장.
    backend = os.environ.get("RAG_BACKEND", "tfidf")
    retriever = SignatureRetriever(backend=backend)

    t = np.linspace(0, 1, 1000)
    cases = {
        "SWMA류(규칙적 사각파)": 100 + 100 * (np.sign(np.sin(2 * np.pi * 50 * t)) > 0),
        "크립토재킹류(평평한 고부하)": 200 + np.random.randn(1000) * 3,
    }

    for label, signal in cases.items():
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        feats = compute_window_features(signal, sample_hz=1000)
        desc = features_to_description(feats, baseline_mean_w=120)
        print("설명:", desc)
        results = retriever.search(desc, top_k=3)
        print("\nRAG 검색 결과:")
        for name, score, _ in results:
            print(f"  {name}: {score:.3f}")
        verdict = analyze_with_llm(desc, results, backend="stub")
        print("\n판정(stub):", json.dumps(verdict, ensure_ascii=False, indent=2))
