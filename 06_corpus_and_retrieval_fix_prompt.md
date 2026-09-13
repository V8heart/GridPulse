# Corpus 확장 + 검색 구조 수정 (Cursor 프롬프트)

## 배경: 실측으로 확인된 문제

corpus에 신규 문서 6개를 추가하고 tfidf 백엔드로 검색을 테스트한 결과,
**기존 4개 문서가 top-3에서 완전히 밀려났다.**

| 쿼리 | 기존 4개 문서만 | 6개 추가 후 |
|---|---|---|
| SWMA형 (swing 1.85, 주기성 0.95) | swma (0.271) ✅ | llmjacking (0.249) ❌ |
| 크립토재킹형 (swing 1.05, 평탄) | cryptojacking 2위 (0.222) | hidden_ml_training (0.245) ❌ |
| 정상 분산학습형 | — | llmjacking (0.220) ❌ |

원인: `SignatureRetriever`는 문서 전체를 하나의 벡터로 임베딩하고
`TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))`로 문자 n-gram 유사도만 본다.
모든 문서가 동일한 한국어 서술 어휘(평균 전력 / 변동폭 / swing ratio / 주기성 /
전환 / 기울기)를 공유하므로, 문서 수가 늘수록 점수가 평탄해지고(0.21~0.27 구간에
밀집) 오답이 정답을 밀어낸다. 기존 `top1_accuracy: 1.0`은 문서 4개 기준이므로
문서 확장 시 유지된다는 보장이 없다.

## 목표

corpus를 10종으로 확장하되, 텍스트 유사도 단독이 아니라
**수치 조건으로 먼저 거른 뒤 텍스트로 순위를 매기는 2단 검색**으로 바꾼다.

## 작업 1 — corpus 문서에 YAML front-matter 추가

모든 `corpus/*.md` 최상단에 다음 형식의 front-matter를 추가한다.
값은 `pipeline/features.py`의 `compute_window_features()` 출력 키와 정확히 일치시킬 것.

```markdown
---
threat_id: T-SWMA-001
category: attack            # attack | benign
swing_ratio: [1.5, 3.0]     # null 이면 조건 미적용
periodicity_strength: [0.8, 1.0]
duty_regularity: [0.8, 1.0]
mean_power_level: normal    # normal | elevated | any
ramp_level: high            # low | mid | high | any
multi_gpu_sync: optional    # required | optional | not_applicable
min_duration_s: null
---
```

기존 4개(swma, ltma, cryptojacking, normal_workloads)에도 반드시 추가한다.
범위는 `dataset/synthetic/all_v2.csv`의 해당 라벨 윈도우에서 실제 feature
분포를 계산해 5~95 퍼센타일로 정하고, 근거를 커밋 메시지에 남긴다.
임의로 정하지 말 것.

## 작업 2 — `SignatureRetriever`를 2단 검색으로 변경

`pipeline/rag_analyzer.py`:

1. `_load_corpus()`가 front-matter를 파싱해 `self.meta[name]`에 저장하고,
   본문(front-matter 제외)만 임베딩한다. front-matter 텍스트가 n-gram에
   섞이면 또 다른 공통 어휘가 되므로 반드시 제외할 것.
2. `search(query, top_k, features=None)` 시그니처로 확장한다.
   - `features`가 주어지면 각 문서의 수치 조건과 대조해 **하드 필터**를 적용
   - 범위를 벗어나면 후보에서 제외
   - 살아남은 문서만 기존 텍스트 유사도로 정렬
   - `features=None`이면 기존 동작 그대로(하위 호환)
3. 필터 통과 문서가 0개면 `unknown`으로 처리하고 빈 리스트를 반환한다.
   이는 open-set 대응에 부합하므로 예외를 던지지 말 것.
4. 반환값에 `filtered_by: [조건명]`을 포함해 왜 제외됐는지 추적 가능하게 한다.

`pipeline/run_pipeline.py`에서 `retriever.search(desc, features=feats)`로
이미 계산된 feature dict를 넘기도록 호출부를 수정한다.

## 작업 3 — 신규 corpus 문서 6종 추가

첨부한 초안(`corpus_new/`)을 사용하되, **반드시 작업 1의 front-matter를 붙이고
본문의 공통 서술 어휘를 줄여** 문서 간 차이가 드러나게 다듬는다.

| 파일 | 위협 | 근거 |
|---|---|---|
| `hidden_ml_training.md` | 비인가 은닉 ML 학습 | arXiv:2606.19262, MITRE T1496.001 |
| `llmjacking.md` | LLM 서비스 탈취 | MITRE T1496.004 |
| `sponge_attack.md` | 에너지·지연 증폭 공격 | arXiv:2006.03463 (IEEE EuroS&P 2021) |
| `coordinated_multi_gpu.md` | 다중 GPU 협조 변조 | LAA, BlackIoT, Bit2Watt |
| `burst_ramp_attack.md` | 급격 부하 변동 | NERC 대규모 부하 상실 사례 |
| `benign_periodic.md` | 정상이나 공격 유사 (hard negative) | — |

`benign_periodic.md`는 `category: benign`으로 두고, 이 문서가 top-1이면
경보를 발생시키지 않도록 `run_pipeline.py`에서 분기한다.

## 작업 4 — 회귀 검증 (필수)

문서를 늘린 뒤 검색 품질이 떨어지지 않았음을 반드시 수치로 증명한다.

1. `dataset/eval/retrieval_eval.jsonl`에 신규 6종에 대응하는 케이스를 추가
2. `pipeline/eval_retrieval.py`를 **corpus 4개 / 10개 두 조건에서 실행**해
   top-1 정확도를 비교한 표를 `dataset/eval/retrieval_scaling.json`에 저장
3. **10개 조건의 top-1 정확도가 4개 조건보다 낮으면 작업 2의 필터가
   제대로 동작하지 않는 것이므로, 문서를 늘린 채로 마무리하지 말고 보고할 것**
4. tfidf와 sbert 두 백엔드 모두에서 측정한다

## 작업 5 — sbert 절단 확인 (별도, 짧게)

`all-MiniLM-L6-v2`는 최대 입력이 256 토큰이고 한국어는 토큰 팽창이 크다.
현재 corpus 문서(1600~2200자)가 실제로 몇 토큰인지 측정해
`dataset/eval/corpus_token_stats.json`에 기록한다.
256을 초과하면 sbert 모드에서는 문서 앞부분만 반영되므로,
(a) 문서를 짧게 유지하거나 (b) 다국어 임베딩 모델로 교체하는
두 방안을 비교한 메모를 남긴다. 이번에 모델 교체까지 하지는 말 것.

## 표현 규율

- 검색 성능이 떨어지면 숨기지 말고 그대로 보고할 것
- front-matter 수치 범위는 반드시 실제 데이터 분포에서 산출할 것 (임의 지정 금지)
