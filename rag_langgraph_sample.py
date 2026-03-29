"""
RAG + 일상채팅 결합 API 샘플 (LangGraph).

intent: biz_talk (지식/업무 질의 → RAG) | small_talk (일상 대화)
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사용자 발화의 의도를 분류하는 분류기입니다.
반드시 아래 두 값 중 하나만 출력하세요. 다른 설명·따옴표·문장은 넣지 마세요.

- biz_talk: 회사/제품/정책/기술 문서·지식베이스에서 답해야 하는 질문, 업무·전문 정보가 필요한 질문, 사실 조회·비교·절차 문의
- small_talk: 인사, 감사, 농담, 날씨·취미 등 가벼운 잡담, 감정 위로, 의견만 묻는 대화 등 문서 검색 없이도 대화로 응답 가능한 경우

애매하면: 구체적 사실·정책·제품·내부 규정이 필요하면 biz_talk, 그렇지 않으면 small_talk.""",
        ),
        ("human", "사용자 발화:\n{user_message}"),
    ]
)

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 제공된 참고 문서만을 근거로 정확하게 답하는 업무 어시스턴트입니다.
- 참고 문서에 없는 내용은 추측하지 말고 "문서에서 확인할 수 없습니다"라고 말하세요.
- 답변은 한국어로 간결하게 작성하세요.""",
        ),
        (
            "human",
            """참고 문서:
{context}

질문:
{user_message}""",
        ),
    ]
)

SMALL_TALK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 친근하고 자연스러운 일상 대화 상대입니다.
- 짧고 따뜻하게 응답하세요. 불필요하게 길게 쓰지 마세요.
- 업무 문서나 회사 지식을 억지로 끌어오지 마세요.""",
        ),
        ("human", "{user_message}"),
    ]
)


# ---------------------------------------------------------------------------
# State & graph
# ---------------------------------------------------------------------------

Intent = Literal["biz_talk", "small_talk"]


class GraphState(TypedDict):
    user_message: str
    intent: Intent | None
    answer: str
    retrieved_context: str


def _normalize_intent(text: str) -> Intent:
    t = (text or "").strip().lower()
    if "biz_talk" in t:
        return "biz_talk"
    if "small_talk" in t:
        return "small_talk"
    return "small_talk"


def build_sample_vectorstore() -> FAISS:
    """데모용 소규모 지식 베이스."""
    docs = [
        Document(
            page_content=(
                "휴가 정책: 연차는 입사일 기준 매년 15일이 부여됩니다. "
                "반차 신청은 전일 오후 5시까지 HR 시스템에서 합니다."
            ),
            metadata={"source": "hr_policy"},
        ),
        Document(
            page_content=(
                "제품 A는 2024년 2분기에 출시된 SaaS이며, "
                "월 구독료는 팀당 99달러이고 API 호출 한도는 월 10만 건입니다."
            ),
            metadata={"source": "product_a"},
        ),
        Document(
            page_content=(
                "고객 지원 SLA: 유료 고객은 영업일 기준 24시간 내 1차 응답, "
                "긴급(P1) 건은 4시간 내 대응합니다."
            ),
            metadata={"source": "support_sla"},
        ),
    ]
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    return FAISS.from_documents(docs, embeddings)


def create_app(llm: ChatOpenAI | None = None, vectorstore: FAISS | None = None):
    llm = llm or ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    vectorstore = vectorstore or build_sample_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    intent_chain = INTENT_CLASSIFICATION_PROMPT | llm | StrOutputParser()
    rag_chain = RAG_ANSWER_PROMPT | llm | StrOutputParser()
    small_chain = SMALL_TALK_PROMPT | llm | StrOutputParser()

    def classify_intent(state: GraphState) -> GraphState:
        raw = intent_chain.invoke({"user_message": state["user_message"]})
        intent = _normalize_intent(raw)
        return {**state, "intent": intent}

    def route_after_intent(state: GraphState) -> Literal["rag", "small"]:
        return "rag" if state.get("intent") == "biz_talk" else "small"

    def run_rag(state: GraphState) -> GraphState:
        docs = retriever.invoke(state["user_message"])
        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        answer = rag_chain.invoke({"context": context, "user_message": state["user_message"]})
        return {**state, "retrieved_context": context, "answer": answer}

    def run_small_talk(state: GraphState) -> GraphState:
        answer = small_chain.invoke({"user_message": state["user_message"]})
        return {**state, "answer": answer}

    g = StateGraph(GraphState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("rag", run_rag)
    g.add_node("small", run_small_talk)
    g.set_entry_point("classify_intent")
    g.add_conditional_edges("classify_intent", route_after_intent, {"rag": "rag", "small": "small"})
    g.add_edge("rag", END)
    g.add_edge("small", END)

    return g.compile()


def run_once(user_message: str) -> dict:
    """단일 턴 실행: intent, answer, (RAG 시) 검색 컨텍스트 반환."""
    app = create_app()
    result = app.invoke(
        {
            "user_message": user_message,
            "intent": None,
            "answer": "",
            "retrieved_context": "",
        }
    )
    return {
        "intent": result["intent"],
        "answer": result["answer"],
        "retrieved_context": result.get("retrieved_context") or "",
    }


# ---------------------------------------------------------------------------
# HTTP API (FastAPI)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field
    from fastapi import FastAPI, HTTPException

    _compiled_app = None

    def _get_graph():
        global _compiled_app
        if _compiled_app is None:
            _compiled_app = create_app()
        return _compiled_app

    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, description="사용자 메시지")

    class ChatResponse(BaseModel):
        intent: Literal["biz_talk", "small_talk"]
        answer: str
        retrieved_context: str = ""

    api = FastAPI(title="RAG + Small Talk", version="0.1.0")

    @api.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
        graph = _get_graph()
        result = graph.invoke(
            {
                "user_message": req.message,
                "intent": None,
                "answer": "",
                "retrieved_context": "",
            }
        )
        return ChatResponse(
            intent=result["intent"] or "small_talk",
            answer=result["answer"],
            retrieved_context=result.get("retrieved_context") or "",
        )

except ImportError:
    api = None  # type: ignore[misc, assignment]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        if api is None:
            raise SystemExit("FastAPI가 필요합니다: pip install fastapi uvicorn")
        import uvicorn

        uvicorn.run("rag_langgraph_sample:api", host="0.0.0.0", port=8000, reload=False)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY 환경 변수를 설정하세요.")

        samples = [
            "안녕! 오늘 기분이 어때?",
            "제품 A 월 구독료가 얼마야?",
            "고마워, 잘 쉬어!",
            "연차는 몇 일이야?",
        ]
        for q in samples:
            out = run_once(q)
            print("---")
            print("Q:", q)
            print("intent:", out["intent"])
            print("A:", out["answer"])
            if out["retrieved_context"]:
                print("context_snippet:", out["retrieved_context"][:200], "...")
