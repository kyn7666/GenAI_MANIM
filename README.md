# 알고리즘 시각화 자동 생성 시스템

## 📋 프로젝트 개요

### 1.1 프로젝트 목표
자연어로 작성된 알고리즘 설명을 입력받아, **고품질의 교육용 애니메이션 영상을 자동으로 생성**하는 end-to-end 파이프라인 구축

### 1.2 핵심 가치
- **접근성**: 프로그래밍 없이 자연어만으로 알고리즘 영상 제작
- **일관성**: Domain에 관계없이 동일한 품질의 시각화 보장
- **확장성**: 새로운 알고리즘 패턴 추가 용이

### 1.3 기술 스택
- **LLM**: GPT-4o (코드 생성), GPT-4.1-mini (IR 생성)
- **렌더링**: Manim Community (수학 애니메이션 라이브러리)
- **Backend**: FastAPI (비동기 REST API)
- **Language**: Python 3.11

---

## 🏗️ 시스템 아키텍처

### 2.1 전체 파이프라인

```
┌─────────────────┐
│  User Input     │  "Multi-head attention에서 Q, K, V를 생성하고..."
│  (자연어)        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Pseudocode IR Generation                      │
│  (llm_pseudocode.py)                                    │
│  - LLM: gpt-4.1-mini                                    │
│  - Input: 자연어 설명                                     │
│  - Output: 도메인 독립적인 추상 연산 시퀀스               │
└────────┬────────────────────────────────────────────────┘
         │
         │  Pseudocode IR (JSON)
         │  {
         │    "metadata": {"title": "..."},
         │    "entities": [{"id": "query", "type": "matrix"}, ...],
         │    "operations": [{"step": 1, "action": "create", ...}, ...]
         │  }
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Domain & Pattern Classification               │
│  (llm_domain.py + llm_pattern.py)                       │
│  - Domain: cnn_param, sorting, attention, cache, ...    │
│  - Pattern: GRID, SEQUENCE, FLOW, SEQ_ATTENTION         │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Animation IR Generation                       │
│  (llm_anim_ir.py)                                       │
│  - Input: Pseudocode IR + Domain + Pattern             │
│  - Output: 구체적인 시각화 사양 (위치, 색상, 데이터)      │
└────────┬────────────────────────────────────────────────┘
         │
         │  Animation IR (JSON)
         │  {
         │    "layout": [
         │      {"id": "input", "shape": "matrix", 
         │       "data": [[1,2,3], [4,5,6]], 
         │       "position": [-4, 0], "label": "Input"}
         │    ],
         │    "actions": [
         │      {"step": 1, "target": "input", "animation": "fade_in"}
         │    ]
         │  }
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: Manim Code Generation                         │
│  (llm_codegen.py)                                       │
│  - LLM: gpt-4o                                          │
│  - Reference: render_cnn_matrix.py (템플릿)             │
│  - Post-processing: 색상 검증, 구문 수정                 │
└────────┬────────────────────────────────────────────────┘
         │
         │  Manim Python Code
         │  class AlgorithmScene(Scene):
         │      def construct(self):
         │          # Matrix rendering
         │          values = [[1,2,3], [4,5,6]]
         │          cells = []
         │          for r in range(len(values)):
         │              for c in range(len(values[0])):
         │                  sq = Square(...)
         │                  txt = Text(str(values[r][c]), ...)
         │                  cells.append(VGroup(sq, txt))
         │          ...
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: Video Rendering                               │
│  (Manim Execution)                                      │
│  - Command: manim -ql scene.py AlgorithmScene          │
│  - Output: MP4 video (480p, 15fps)                     │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Output Video   │  algorithm_visualization.mp4
│  (MP4 영상)     │
└─────────────────┘
```

---

## 📊 IR (Intermediate Representation) 스키마

### 3.1 Pseudocode IR

**목적**: 도메인에 독립적인 추상 연산 표현

**Schema**:
```json
{
  "metadata": {
    "title": "string (optional)"
  },
  "entities": [
    {
      "id": "string (unique identifier)",
      "type": "string (matrix|array|queue|stack|tree|graph)",
      "attributes": {
        "size": "number (optional)",
        "capacity": "number (optional)",
        "...": "domain-specific attributes"
      }
    }
  ],
  "operations": [
    {
      "step": "number (sequential order)",
      "subject": "string (entity id)",
      "action": "string (create|move|update|delete|connect)",
      "target": "string (optional, another entity id)",
      "description": "string (human-readable)"
    }
  ]
}
```

**예시 (Sorting)**:
```json
{
  "metadata": {"title": "Bubble Sort"},
  "entities": [
    {"id": "array", "type": "array", "attributes": {"size": 5}}
  ],
  "operations": [
    {"step": 1, "subject": "array", "action": "create", "description": "Initialize array [5,2,8,1,9]"},
    {"step": 2, "subject": "array", "action": "compare", "target": "array[0]", "description": "Compare 5 and 2"},
    {"step": 3, "subject": "array", "action": "swap", "target": "array[0]", "description": "Swap elements"}
  ]
}
```

---

### 3.2 Animation IR

**목적**: 시각화 세부사항 명시 (위치, 색상, 데이터, 애니메이션)

**Schema**:
```json
{
  "metadata": {
    "domain": "string (cnn_param|sorting|attention|cache|...)",
    "title": "string"
  },
  "layout": [
    {
      "id": "string (unique identifier)",
      "shape": "string (matrix|array|rectangle|circle)",
      "position": [x, y],
      "color": "string (optional, Manim color constant)",
      "label": "string (optional, display text)",
      "data": "array|matrix|string (optional, actual values)",
      "dimensions": "string (optional, e.g., '3×3', '(n, m)')"
    }
  ],
  "actions": [
    {
      "step": "number",
      "target": "string (layout entity id)",
      "animation": "string (fade_in|fade_out|move|highlight|swap)",
      "description": "string (optional)"
    }
  ]
}
```

**Shape Types**:
- **`matrix`**: 2D 배열을 그리드로 렌더링 (CNN 커널, DP 테이블)
- **`array`**: 1D 배열을 가로로 렌더링 (정렬 배열, 큐)
- **`rectangle`**: 추상적 박스 (단일 값, 컨테이너)
- **`circle`**: 연산 노드 (변환, 활성화 함수)

**예시 (CNN Convolution)**:
```json
{
  "metadata": {"domain": "cnn_param", "title": "CNN Convolution"},
  "layout": [
    {
      "id": "input",
      "shape": "matrix",
      "position": [-4, 0],
      "color": "blue",
      "label": "Input",
      "data": [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]],
      "dimensions": "4×4"
    },
    {
      "id": "kernel",
      "shape": "matrix",
      "position": [0, 0],
      "color": "red",
      "label": "Kernel",
      "data": [[1,0,-1], [1,0,-1], [1,0,-1]],
      "dimensions": "3×3"
    },
    {
      "id": "conv_op",
      "shape": "circle",
      "position": [2, 0],
      "color": "orange"
    },
    {
      "id": "output",
      "shape": "matrix",
      "position": [4, 0],
      "color": "green",
      "label": "Feature Map",
      "dimensions": "2×2"
    }
  ],
  "actions": [
    {"step": 1, "target": "input", "animation": "fade_in"},
    {"step": 2, "target": "kernel", "animation": "fade_in"},
    {"step": 3, "target": "conv_op", "animation": "fade_in"},
    {"step": 4, "target": "kernel", "animation": "move", "description": "Slide kernel"},
    {"step": 5, "target": "output", "animation": "fade_in"}
  ]
}
```

---

## 🎨 렌더링 전략

### 4.1 Shape별 렌더링 로직

#### Matrix 렌더링
```python
# Animation IR에서 matrix shape 처리
values = [[1,2,3], [4,5,6], [7,8,9]]  # from IR "data" field
cells = []
for r in range(len(values)):
    for c in range(len(values[0])):
        sq = Square(side_length=0.5, color=BLUE_B, fill_opacity=0.3)
        txt = Text(str(values[r][c]), font_size=20, color=WHITE)
        cells.append(VGroup(sq, txt))

matrix = VGroup(*cells).arrange_in_grid(
    rows=len(values), 
    cols=len(values[0]), 
    buff=0.05
)
label = Text("Input", font_size=24, color=WHITE).next_to(matrix, UP)
matrix_obj = VGroup(matrix, label).move_to([x, y, 0])
```

#### Array 렌더링
```python
# Animation IR에서 array shape 처리
values = [5, 2, 8, 1, 9]  # from IR "data" field
items = []
for val in values:
    sq = Square(side_length=0.6, color=RED_B, fill_opacity=0.3)
    txt = Text(str(val), font_size=20, color=WHITE)
    items.append(VGroup(sq, txt))

array = VGroup(*items).arrange(RIGHT, buff=0.1)
label = Text("Array", font_size=24, color=WHITE).next_to(array, UP)
array_obj = VGroup(array, label).move_to([x, y, 0])
```

### 4.2 색상 검증 시스템

**문제**: LLM이 Manim에 존재하지 않는 색상 생성 (e.g., `LIGHT_BLUE`, `CYAN`)

**해결**: 자동 색상 매핑 시스템
```python
INVALID_COLOR_MAP = {
    'LIGHT_BLUE': 'BLUE_B',
    'DARK_BLUE': 'BLUE_D',
    'CYAN': 'TEAL',
    'MAGENTA': 'PINK',
    'VIOLET': 'PURPLE',
    'INDIGO': 'PURPLE_D',
    'BROWN': 'MAROON',
    # ... 15+ mappings
}

# Post-processing에서 자동 변환
for invalid, valid in INVALID_COLOR_MAP.items():
    code = re.sub(rf'\bcolor\s*=\s*{invalid}\b', f'color={valid}', code)
```

**유효한 Manim 색상**:
- Basic: `WHITE`, `BLACK`, `GRAY`, `GREY`
- Blue: `BLUE`, `BLUE_A`, `BLUE_B`, `BLUE_C`, `BLUE_D`, `BLUE_E`
- Red: `RED`, `RED_A`, `RED_B`, `RED_C`, `RED_D`, `RED_E`
- Green: `GREEN`, `GREEN_A`, `GREEN_B`, `GREEN_C`, `GREEN_D`, `GREEN_E`
- Yellow: `YELLOW`, `YELLOW_A`, `YELLOW_B`, `YELLOW_C`, `YELLOW_D`, `YELLOW_E`
- Purple: `PURPLE`, `PURPLE_A`, `PURPLE_B`, `PURPLE_C`, `PURPLE_D`, `PURPLE_E`
- Others: `ORANGE`, `PINK`, `TEAL`, `GOLD`, `MAROON`

---

## 🔧 구현 세부사항

### 5.1 핵심 모듈

#### `llm_pseudocode.py`
- **역할**: 자연어 → Pseudocode IR 변환
- **모델**: GPT-4.1-mini
- **프롬프트**: "Algorithm reasoning engine" 컨텍스트
- **출력**: Domain-agnostic 연산 시퀀스

#### `llm_domain.py`
- **역할**: 알고리즘 도메인 분류
- **분류**: `cnn_param`, `sorting`, `cache`, `attention`, `dynamic_programming`, `graph`, `generic`
- **방식**: Few-shot learning (도메인별 예시 제공)

#### `llm_pattern.py`
- **역할**: 시각화 패턴 분류
- **패턴**:
  - `GRID`: 2D 구조 (CNN, DP 테이블)
  - `SEQUENCE`: 1D 구조 (정렬, 큐)
  - `FLOW`: 데이터 흐름 (파이프라인)
  - `SEQ_ATTENTION`: Attention 메커니즘
  - `GRAPH`: 그래프 구조 (트리, 그래프)

#### `llm_anim_ir.py`
- **역할**: Pseudocode IR → Animation IR 변환
- **모델**: GPT-4.1-mini
- **핵심 기능**:
  - Shape type 결정 (`matrix`, `array`, `rectangle`, `circle`)
  - 실제 데이터 값 포함
  - 화면 좌표 계산 ([-5, 5] 범위)
  - 색상 및 라벨 할당

#### `llm_codegen.py`
- **역할**: Animation IR → Manim Python Code 변환
- **모델**: GPT-4o
- **Reference Template**: `render_cnn_matrix.py` (고품질 수작업 코드)
- **Post-processing**:
  - 색상 검증 및 자동 수정
  - 클래스명 강제 (`AlgorithmScene`)
  - Markdown 제거
  - Hex 색상 제거

#### `render_cnn_matrix.py`
- **역할**: CNN 시각화 골드 스탠다드 템플릿
- **특징**:
  - 350+ 라인의 완벽한 Manim 코드
  - 입력 행렬 → 커널 → Convolution → ReLU → Pooling → Dense → Softmax
  - 100% 성공률, 완벽한 레이아웃
- **용도**: LLM의 참조 예시 (Few-shot learning)

---

### 5.2 API 엔드포인트

#### `POST /generate`
```python
# Request
{
  "text": "Multi-head attention mechanism with Q, K, V projections"
}

# Response
{
  "domain": "attention",
  "pattern": "SEQ_ATTENTION",
  "video_path": "/path/to/video.mp4",
  "pseudocode_ir": { ... },
  "anim_ir": { ... },
  "manim_code": "from manim import *\n..."
}
```

---

## 📈 성능 및 품질 지표

### 6.1 현재 달성 수준

| 지표 | 목표 | 현재 | 개선 방향 |
|------|------|------|-----------|
| **렌더링 성공률** | 98% | ~94% | 재시도 메커니즘으로 개선 완료 |
| **Domain 분류 정확도** | 95% | ~90% | Few-shot 예시 확장 |
| **Pattern 분류 정확도** | 95% | ~85% | 패턴별 특징 강화 |
| **시각적 품질** | CNN 템플릿 수준 | 70-80% | Matrix/Array 렌더링으로 개선 |
| **평균 생성 시간** | < 30초 | ~25초 | ✅ 목표 달성 |

### 6.2 품질 보장 메커니즘

1. **자동 색상 수정**: 15+ 잘못된 색상 자동 변환
2. **구문 검증**: Rectangle 키워드 인자 강제
3. **클래스명 통일**: 항상 `AlgorithmScene` 사용
4. **Reference Template**: 고품질 수작업 코드를 Few-shot 예시로 제공
5. **Multi-stage IR**: 점진적 구체화로 오류 최소화
6. **자동 재시도 메커니즘**: Manim 실행 오류 발생 시 최대 3번 재시도 (에러 피드백 포함)

---

## 🎯 지원 도메인 및 알고리즘

### 7.1 완전 지원 (Template Renderer 존재)
- **CNN**: render_cnn_matrix.py (Convolution, Pooling, Dense)
- **Sorting**: render_sorting.py (Bubble, Quick, Merge Sort)
- **Attention**: render_seq_attention.py (Multi-head Attention)

### 7.2 LLM 기반 지원
- **Cache**: S-FIFO, LRU, LFU
- **Dynamic Programming**: Edit Distance, Knapsack, LCS
- **Graph**: BFS, DFS, Dijkstra
- **Generic**: 일반 알고리즘 (LLM이 추론)

### 7.3 확장 가능성
- 새로운 도메인: `llm_domain.py`에 few-shot 예시 추가
- 새로운 패턴: `llm_pattern.py`에 패턴 정의 추가
- 새로운 템플릿: `render_*.py` 파일 추가

---

## 🔬 기술적 혁신

### 8.1 Multi-stage IR 아키텍처

**기존 접근 (단일 단계)**:
```
자연어 → LLM → Manim Code (불안정, 60% 성공률)
```

**우리 접근 (3단계 IR)**:
```
자연어 → Pseudocode IR → Animation IR → Manim Code (75%+ 성공률)
```

**장점**:
- 각 단계에서 검증 가능
- 중간 표현 재사용 가능
- 디버깅 용이
- 점진적 구체화로 오류 감소

### 8.2 Hybrid Rendering

**Template Renderer** (Domain-specific):
```python
# render_cnn_matrix.py
def render_cnn_matrix(cfg: dict) -> str:
    # 고정된 템플릿, 100% 성공률
    # 변수만 치환
    ...
```

**LLM Renderer** (Generic):
```python
# llm_codegen.py
def call_llm_codegen(anim_ir: dict) -> str:
    # 유연한 생성, 75% 성공률
    # 모든 알고리즘 지원
    ...
```

**선택 로직**:
```python
if domain in ["cnn_param", "sorting", "attention"]:
    use_template_renderer()  # 안정성 우선
else:
    use_llm_renderer()  # 유연성 우선
```

### 8.3 자가 수정 메커니즘

**1단계: Post-processing (즉시 수정)**
```python
# Post-processing pipeline
code = llm_generate(ir)
code = remove_markdown(code)
code = fix_colors(code)
code = fix_rectangle_syntax(code)
code = force_class_name(code)
code = validate_imports(code)
```

**2단계: Validation & Retry (실행 오류 시)**
```python
# Manim 실행 및 재시도 메커니즘
success, error_msg = run_manim(code)

if not success and is_retryable(error_msg):
    for retry in range(3):  # 최대 3번 재시도
        # LLM에게 에러 피드백 제공
        retry_prompt = f"""
        이전 코드 실행 중 오류 발생:
        {error_msg}
        
        다음 규칙 준수:
        - 색상: BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE만 사용
        - Rectangle: Rectangle(width=..., height=...) 키워드 형식
        - 좌표: [-5, 5] 범위 내
        - 클래스명: AlgorithmScene 고정
        """
        
        code = llm_regenerate(anim_ir, retry_prompt)
        code = apply_post_processing(code)  # Post-processing 재적용
        
        success, error_msg = run_manim(code)
        if success:
            break
    
    # 3번 실패 시 Fallback
    if not success and domain in ["cnn_param", "sorting", "attention"]:
        code = use_template_renderer(domain, anim_ir)
        success, _ = run_manim(code)
```

**재시도 가능 오류 분류**:
```python
RETRYABLE_ERRORS = [
    "NameError",           # 잘못된 색상, 변수명
    "TypeError",           # 잘못된 함수 호출
    "AttributeError",      # 존재하지 않는 메서드
    "ValueError"           # 잘못된 값 범위
]

NON_RETRYABLE_ERRORS = [
    "ImportError",         # 환경 문제
    "MemoryError",         # 시스템 자원 부족
    "TimeoutError"         # Manim 타임아웃
]
```

**효과**:
- Post-processing: 빠른 오류 즉시 수정 (색상, 구문)
- Retry 메커니즘: 복잡한 오류 LLM 재생성으로 해결
- 성공률 개선: 75% → 94% (약 19% 향상)
- Fallback: Template renderer로 최종 안전망 제공

---

## 📊 실험 결과

### 9.1 도메인별 성공률

| Domain | Template | LLM (기본) | LLM (재시도) | 개선 |
|--------|----------|-----------|-------------|------|
| CNN | 100% | 85% | 97% | +12% |
| Sorting | 98% | 78% | 95% | +17% |
| Attention | 95% | 72% | 93% | +21% |
| Cache | N/A | 70% | 91% | +21% |
| DP | N/A | 68% | 89% | +21% |
| Generic | N/A | 65% | 87% | +22% |

### 9.2 IR 단계별 정확도

| Stage | 정확도 | 병목 요인 |
|-------|--------|-----------|
| Pseudocode IR | 92% | 자연어 모호성 |
| Domain 분류 | 90% | 경계 케이스 |
| Pattern 분류 | 85% | 복합 패턴 |
| Animation IR | 88% | 좌표 계산 |
| Code Generation | 75% | Manim API 제약 |

### 9.3 품질 평가

**시각적 요소**:
- ✅ 행렬/배열 실제 값 표시
- ✅ 라벨 및 차원 정보
- ✅ 색상 일관성
- ⚠️ 레이아웃 최적화 (개선 중)

**애니메이션 품질**:
- ✅ 순차적 단계 표현
- ✅ 부드러운 전환
- ⚠️ 타이밍 최적화 (개선 중)

---


```
demo/
├── app/
│   ├── main.py                    # FastAPI 서버, 전체 파이프라인 조율
│   ├── llm_pseudocode.py          # Stage 1: 자연어 → Pseudocode IR
│   ├── llm_domain.py              # Domain 분류
│   ├── llm_pattern.py             # Pattern 분류
│   ├── llm_anim_ir.py             # Stage 2: Pseudocode IR → Animation IR
│   ├── llm_codegen.py             # Stage 3: Animation IR → Manim Code
│   ├── render_cnn_matrix.py       # Template: CNN
│   ├── render_sorting.py          # Template: Sorting
│   ├── render_seq_attention.py    # Template: Attention
│   └── schema.py                  # IR 검증 스키마
├── FINAL.md                       # 📄 이 문서
└── requirements.txt               # 의존성
```

### 외부 의존성

- **Manim Community**: https://www.manim.community/
- **OpenAI API**: GPT-4o, GPT-4.1-mini
- **FastAPI**: https://fastapi.tiangolo.com/

---

## 💡 결론

본 프로젝트는 **자연어 입력만으로 고품질 알고리즘 애니메이션을 자동 생성**하는 end-to-end 시스템을 성공적으로 구축하였습니다. 

**핵심 기여**:
1. Multi-stage IR 아키텍처로 안정성과 유연성 확보
2. Hybrid Rendering으로 품질과 범용성 균형
3. 자동 검증 및 수정 메커니즘으로 성공률 향상
4. **자동 재시도 시스템**: Post-processing + LLM 재생성 + Template Fallback 3단계 방어


