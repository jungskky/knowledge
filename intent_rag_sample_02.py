"""
biz_talk / small_talk intent 분류 + 조건부 RAG LangGraph 샘플.

- classify_intent: PromptTemplate으로 biz_talk(반도체·업무 지식 검색) vs small_talk(잡담·자유응답) 구분
- biz_talk: TF-IDF 기반 청크 검색 후 컨텍스트 근거 답변 (영문 약어·기술 용어 보존 지시)
- small_talk: 임베딩/검색 없이 LLM만으로 응답

파이프라인: classify_intent → [biz_talk: retrieve → generate_biz] | [small_talk: generate_small]

실행 전: pip install -r requirements.txt
환경 변수: OPENAI_API_KEY (또는 .env)
--mock: API 없이 의도 규칙 분류 + 답변 플레이스홀더
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Intent: 업무(반도체 RAG) vs 잡담
# ---------------------------------------------------------------------------
INTENTS = {
    "biz_talk": (
        "반도체·제조·소자·공정·장비·재료 등 업무/기술 지식이 필요한 질문. "
        "영문 약어만 적은 질문(CMOS, EUV, FinFET, CMP, FEOL 등)도 여기에 해당한다."
    ),
    "small_talk": (
        "인사, 감사, 날씨·취미 등 일상 대화, 또는 검색 없이 가벼운 잡담으로 충분한 경우."
    ),
}

INTENT_LIST_TEXT = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 사용자 메시지를 아래 intent 중 정확히 하나로만 분류한다.\n"
            "규칙:\n"
            "1) 반도체·웨이퍼·리소·에치·증착·수율·공정 노드·장비·소자 구조 등 기술·업무 질의는 biz_talk.\n"
            "2) 영어 대문자 약어 위주 짧은 질문(예: EUV, FinFET, CMP가 뭐야)도 반도체/제조 맥락이면 biz_talk.\n"
            "3) 인사, 감사, 농담, 일상 잡담, 감정 위주 대화는 small_talk.\n"
            "4) 애매하면: 지식 검색이 도움이 되면 biz_talk, 아니면 small_talk.\n"
            "반드시 JSON 한 줄만 출력한다. 다른 설명 금지.\n"
            '형식: {{"intent": "biz_talk" | "small_talk", "confidence": 0.0~1.0, "reason": "한 줄"}}\n'
            f"{INTENT_LIST_TEXT}",
        ),
        ("human", "사용자 메시지:\n{query}"),
    ]
)

BIZ_RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "역할: 반도체 도메인 RAG 어시스턴트.\n"
            "아래 [검색 컨텍스트]에 근거해 답한다. 컨텍스트에 없는 사실은 추측하지 말고 모른다고 한다.\n"
            "영문 약어·기술 용어는 문맥에 맞게 유지하고, 필요하면 한글 설명을 덧붙인다.\n"
            "답은 간결하고 구조화(불릿 가능)한다.",
        ),
        (
            "human",
            "[검색 컨텍스트]\n{context}\n\n"
            "[사용자 질문]\n{query}",
        ),
    ]
)

SMALL_TALK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "역할: 친근한 일상 대화 상대. 검색·문서 없이 자연스럽게 응답한다.\n"
            "반도체 전문 지식을 지어내지 않는다. 필요하면 가볍게만 언급한다.",
        ),
        ("human", "{query}"),
    ]
)


class RAGState02(TypedDict, total=False):
    query: str
    mock: bool
    intent: str
    confidence: float
    reason: str
    context: str
    context_docs: list[str]
    answer: str


@dataclass
class IntentResult02:
    intent: str
    confidence: float
    reason: str

    @classmethod
    def from_json_text(cls, text: str) -> "IntentResult02":
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        intent = data["intent"]
        if intent not in INTENTS:
            raise ValueError(f"알 수 없는 intent: {intent}")
        return cls(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", "")),
        )


def simple_chunk_corpus(text: str, max_chars: int = 450) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars])
    return chunks


class TfidfRetriever:
    """샘플용: 청크 TF-IDF 유사도 검색 (임베딩 API 없이 RAG 데모)."""

    def __init__(self, chunks: list[str], top_k: int = 3):
        self.chunks = chunks
        self.top_k = top_k
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1
        )
        self._matrix = self._vectorizer.fit_transform(chunks)

    def retrieve(self, query: str) -> list[str]:
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix).ravel()
        idx = scores.argsort()[::-1][: self.top_k]
        return [self.chunks[i] for i in idx if scores[i] > 0]


def mock_intent(query: str) -> IntentResult02:
    """API 없이 휴리스틱 분류 (데모용). 영문 약어·반도체 키워드 → biz_talk."""
    q_raw = query.strip()
    q = q_raw.lower()

    small_hints = (
        "안녕",
        "고마워",
        "감사",
        "잘가",
        "날씨",
        "심심",
        "농담",
        "재미",
        "hello",
        "hi ",
        "thanks",
        "how are you",
    )
    if any(h in q for h in small_hints) and len(q_raw) < 80:
        return IntentResult02("small_talk", 0.75, "mock: 인사/일상 키워드")

    biz_hints = (
        "cmos",
        "finfet",
        "euv",
        "duv",
        "etch",
        "depo",
        "cvd",
        "ald",
        "cmp",
        "feol",
        "beol",
        "웨이퍼",
        "리소",
        "반도체",
        "공정",
        "수율",
        "게이트",
        "플라즈마",
    )
    if any(h in q for h in biz_hints):
        return IntentResult02("biz_talk", 0.8, "mock: 반도체/기술 키워드")

    if re.search(r"\b[A-Z]{2,10}\b", q_raw) and len(q_raw) <= 40:
        return IntentResult02("biz_talk", 0.65, "mock: 짧은 영문 약어 위주 → biz 의심")

    return IntentResult02("small_talk", 0.55, "mock: 기본 small_talk")


def build_intent_rag_graph(
    retriever: TfidfRetriever,
    *,
    model: str,
    temperature: float,
):
    llm: ChatOpenAI | None = None

    def get_llm() -> ChatOpenAI:
        nonlocal llm
        if llm is None:
            llm = ChatOpenAI(model=model, temperature=temperature)
        return llm

    def classify_intent_node(state: RAGState02) -> dict:
        if state.get("mock"):
            r = mock_intent(state["query"])
            return {
                "intent": r.intent,
                "confidence": r.confidence,
                "reason": r.reason,
            }
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY가 없습니다. .env를 설정하거나 --mock 을 사용하세요."
            )
        chain = INTENT_CLASSIFIER_PROMPT | get_llm() | StrOutputParser()
        raw = chain.invoke({"query": state["query"]})
        r = IntentResult02.from_json_text(raw)
        return {"intent": r.intent, "confidence": r.confidence, "reason": r.reason}

    def route_after_intent(state: RAGState02) -> str:
        return "retrieve" if state.get("intent") == "biz_talk" else "small_talk"

    def retrieve_node(state: RAGState02) -> dict:
        docs = retriever.retrieve(state["query"])
        ctx = "\n\n---\n\n".join(docs) if docs else "(검색 결과 없음)"
        return {"context_docs": docs, "context": ctx}

    def generate_biz_node(state: RAGState02) -> dict:
        if state.get("mock"):
            n = len(state.get("context_docs") or [])
            return {
                "answer": (
                    f"[biz_talk·mock] 검색 청크 {n}개. "
                    "실제로는 BIZ_RAG_ANSWER_PROMPT로 생성합니다."
                )
            }
        chain = BIZ_RAG_ANSWER_PROMPT | get_llm() | StrOutputParser()
        ans = chain.invoke(
            {"context": state.get("context") or "", "query": state["query"]}
        )
        return {"answer": ans}

    def generate_small_node(state: RAGState02) -> dict:
        if state.get("mock"):
            return {
                "answer": (
                    "[small_talk·mock] 검색 없음. "
                    "실제로는 SMALL_TALK_PROMPT로 생성합니다."
                )
            }
        chain = SMALL_TALK_PROMPT | get_llm() | StrOutputParser()
        ans = chain.invoke({"query": state["query"]})
        return {"answer": ans}

    graph = StateGraph(RAGState02)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_biz", generate_biz_node)
    graph.add_node("generate_small", generate_small_node)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"retrieve": "retrieve", "small_talk": "generate_small"},
    )
    graph.add_edge("retrieve", "generate_biz")
    graph.add_edge("generate_biz", END)
    graph.add_edge("generate_small", END)

    return graph.compile()


def run_pipeline(
    query: str,
    corpus_path: Path,
    mock: bool,
    model: str,
    temperature: float,
) -> None:
    if not mock and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY가 없습니다. .env를 설정하거나 --mock 을 사용하세요."
        )

    text = corpus_path.read_text(encoding="utf-8")
    chunks = simple_chunk_corpus(text)
    retriever = TfidfRetriever(chunks, top_k=3)
    graph = build_intent_rag_graph(retriever, model=model, temperature=temperature)

    final: RAGState02 = graph.invoke({"query": query, "mock": mock})

    intent_payload = {
        "intent": final.get("intent"),
        "confidence": final.get("confidence"),
        "reason": final.get("reason"),
    }
    print("=== Intent 분류 ===")
    print(json.dumps(intent_payload, ensure_ascii=False, indent=2))

    if final.get("intent") == "biz_talk":
        ctx = final.get("context") or ""
        print("\n=== biz_talk: 검색 컨텍스트 (일부) ===")
        print(ctx[:1200] + ("..." if len(ctx) > 1200 else ""))
    else:
        print("\n=== small_talk: 검색 생략 ===")

    print("\n=== 답변 ===")
    print(final.get("answer", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="biz_talk / small_talk 분류 + 조건부 RAG (LangGraph)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="EUV 리소가 DUV보다 어려운 이유를 짧게",
        help="사용자 질의",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "semiconductor_corpus.txt",
        help="biz_talk 검색용 코퍼스",
    )
    parser.add_argument("--mock", action="store_true", help="API 없이 휴리스틱 분류")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI 모델명")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run_pipeline(
        query=args.query,
        corpus_path=args.corpus,
        mock=args.mock,
        model=args.model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
