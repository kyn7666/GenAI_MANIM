# app/patterns.py
from enum import Enum


class PatternType(str, Enum):
    GRID = "grid"
    SEQUENCE = "sequence"
    FLOW = "flow"
    GRAPH = "graph" 
    SEQ_ATTENTION = "seq_attention"


# 도메인 → 패턴 확정 (대표 도메인 처리)
DOMAIN_TO_PATTERN = {
    "cnn_param": PatternType.GRID,
    "sorting": PatternType.SEQUENCE,
    "bubble_sort": PatternType.SEQUENCE,
    "selection_sort": PatternType.SEQUENCE,
    "transformer": PatternType.SEQ_ATTENTION,
    "transformer_attn": PatternType.SEQ_ATTENTION,
    "attention": PatternType.SEQ_ATTENTION,

    "cache": PatternType.FLOW,
    "math": PatternType.FLOW,
    "pipeline": PatternType.FLOW,
    
    # Hash table은 2D grid로 표현 (버킷들을 세로로 배열)
    "hash_table": PatternType.GRID,
    
    # 🆕 Graph 도메인은 GRAPH 패턴 사용
    "graph_traversal": PatternType.GRAPH,
    "shortest_path": PatternType.GRAPH,
    "graph": PatternType.GRAPH,
    
    # Tree도 GRAPH 패턴 사용 (노드 + 엣지 구조)
    "binary_tree": PatternType.GRAPH,
    "tree": PatternType.GRAPH,
    
    # Dynamic programming table은 grid
    "dynamic_programming": PatternType.GRID,
}

VALID_PATTERNS = {
    "grid": PatternType.GRID,
    "sequence": PatternType.SEQUENCE,
    "seq_attention": PatternType.SEQ_ATTENTION,
    "flow": PatternType.FLOW,
    "graph": PatternType.GRAPH,  # 🆕 GRAPH 패턴 추가
}

def resolve_pattern(domain: str, llm_pattern: str) -> PatternType:
    # 1) 도메인 강제 매핑이 있으면 도메인 우선
    if domain in DOMAIN_TO_PATTERN:
        return DOMAIN_TO_PATTERN[domain]

    # 2) LLM 패턴이 정상적으로 감지된 경우
    llm_pattern = llm_pattern.lower()
    if llm_pattern in VALID_PATTERNS:
        return VALID_PATTERNS[llm_pattern]

    # 3) 둘 다 아닌 경우 → fallback: FLOW 패턴
    return PatternType.FLOW
