# rag_self_advanced_sample.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


@dataclass
class SubQueryResult:
    query: str
    needs_retrieval: bool
    answer: str = ""
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    accepted: bool = False


class SimpleRAGSelfAdvanced:
    def __init__(
        self,
        vectorstore: FAISS,
        model: str = "gpt-4o-mini",
        max_iterations: int = 3,
        top_k: int = 4,
    ):
        self.vectorstore = vectorstore
        self.model = model
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.client = OpenAI()

    def _llm(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def decompose_queries(self, user_query: str) -> List[str]:
        prompt = f"""
다음 사용자 질문을 독립적으로 답할 수 있는 서브 질의 목록으로 분해하라.
- 다중 질문이면 각각 분리
- 단일 질문이면 원문 유지
- JSON 배열만 출력

질문:
{user_query}
"""
        out = self._llm([
            {"role": "system", "content": "You decompose user queries into atomic sub-queries."},
            {"role": "user", "content": prompt},
        ])
        try:
            items = json.loads(out)
            if isinstance(items, list) and items:
                return [str(x).strip() for x in items if str(x).strip()]
        except Exception:
            pass
        return [user_query]

    def needs_retrieval(self, sub_query: str) -> bool:
        prompt = f"""
다음 질문이 vector DB 조회가 필요한지 판단하라.
- YES: 사내 문서, 개인 지식, 특정 데이터, 사실 확인이 필요
- NO: 일반 상식, 간단한 설명, 변환, 글쓰기, 코드 구조 설계 정도로 답 가능
- 출력은 YES 또는 NO만

질문: {sub_query}
"""
        out = self._llm([
            {"role": "system", "content": "Classify whether retrieval is needed."},
            {"role": "user", "content": prompt},
        ]).upper()
        return "YES" in out

    def retrieve(self, sub_query: str, k: Optional[int] = None) -> List[Document]:
        k = k or self.top_k
        return self.vectorstore.similarity_search(sub_query, k=k)

    def generate_answer(
        self,
        sub_query: str,
        contexts: List[Document],
        previous_answer: str = "",
    ) -> str:
        ctx_text = "\n\n".join(
            [f"[{i+1}] {d.page_content}" for i, d in enumerate(contexts)]
        )

        prompt = f"""
질문:
{sub_query}

이전 답변(있으면 참고):
{previous_answer}

참고 컨텍스트:
{ctx_text}

요구사항:
- 컨텍스트가 있으면 그 내용을 우선 반영
- 없으면 일반 상식 범위에서만 답변
- 너무 짧으면 안 됨
- 불확실하면 불확실하다고 말함
"""
        return self._llm([
            {"role": "system", "content": "Answer the question faithfully and clearly."},
            {"role": "user", "content": prompt},
        ])

    def validate_answer(
        self,
        sub_query: str,
        answer: str,
        contexts: List[Document],
    ) -> Tuple[bool, str]:
        ctx_text = "\n\n".join([d.page_content for d in contexts[:3]])
        prompt = f"""
다음 답변이 질문과 컨텍스트에 비추어 충분한지 판정하라.
반드시 JSON으로만 출력:
{{"accepted": true/false, "feedback": "짧은 피드백"}}

질문: {sub_query}
답변: {answer}
컨텍스트: {ctx_text}
"""
        out = self._llm([
            {"role": "system", "content": "Validate answer quality and grounding."},
            {"role": "user", "content": prompt},
        ])
        try:
            obj = json.loads(out)
            accepted = bool(obj.get("accepted", False))
            feedback = str(obj.get("feedback", ""))
            return accepted, feedback
        except Exception:
            low = answer.strip()
            if len(low) < 30:
                return False, "Answer too short"
            return True, "Validation fallback accepted"

    def answer_subquery(self, sub_query: str) -> SubQueryResult:
        needs_ret = self.needs_retrieval(sub_query)
        result = SubQueryResult(query=sub_query, needs_retrieval=needs_ret)

        contexts: List[Document] = []
        previous_answer = ""
        k = self.top_k

        for i in range(self.max_iterations):
            result.iterations = i + 1

            if needs_ret:
                contexts = self.retrieve(sub_query, k=k)
            else:
                contexts = []

            answer = self.generate_answer(sub_query, contexts, previous_answer)
            accepted, feedback = self.validate_answer(sub_query, answer, contexts)

            result.answer = answer
            result.contexts = [
                {
                    "content": d.page_content,
                    "metadata": d.metadata,
                }
                for d in contexts
            ]
            result.accepted = accepted

            if accepted:
                return result

            previous_answer = answer
            k = min(k + 1, 8)

        return result

    def synthesize(self, original_query: str, results: List[SubQueryResult]) -> str:
        items = []
        for idx, r in enumerate(results, 1):
            items.append(
                f"[서브질의 {idx}] {r.query}\n"
                f"[답변] {r.answer}\n"
                f"[retrieval={r.needs_retrieval}, accepted={r.accepted}, iterations={r.iterations}]"
            )

        joined = "\n\n".join(items)
        prompt = f"""
원래 질문:
{original_query}

서브 질의별 결과:
{joined}

요구사항:
- 중복 제거
- 서로 연결된 내용은 하나로 합침
- 다중질의면 항목별로 정리
- 자연스러운 최종 답변으로 통합
- 불확실한 내용은 보수적으로 표현
"""
        return self._llm([
            {"role": "system", "content": "Synthesize multi-query answers into one coherent response."},
            {"role": "user", "content": prompt},
        ])

    def answer(self, user_query: str) -> Dict[str, Any]:
        sub_queries = self.decompose_queries(user_query)
        results = [self.answer_subquery(q) for q in sub_queries]
        final_answer = self.synthesize(user_query, results)

        return {
            "query": user_query,
            "sub_queries": sub_queries,
            "results": [
                {
                    "query": r.query,
                    "needs_retrieval": r.needs_retrieval,
                    "answer": r.answer,
                    "accepted": r.accepted,
                    "iterations": r.iterations,
                    "contexts": r.contexts,
                }
                for r in results
            ],
            "final_answer": final_answer,
        }


def build_vectorstore_from_texts(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> FAISS:
    embeddings = OpenAIEmbeddings()
    docs = []
    metadatas = metadatas or [{} for _ in texts]
    for t, m in zip(texts, metadatas):
        docs.append(Document(page_content=t, metadata=m))
    return FAISS.from_documents(docs, embeddings)


if __name__ == "__main__":
    sample_texts = [
        "RAG 시스템은 검색과 생성 단계를 결합한다.",
        "다중 질의는 질문을 여러 서브 질의로 분해해 각각 처리한 뒤 합친다.",
        "반복 정제는 답변 품질이 부족할 때 재검색과 재생성을 수행한다.",
    ]
    vectorstore = build_vectorstore_from_texts(sample_texts)
    rag = SimpleRAGSelfAdvanced(vectorstore=vectorstore)

    query = "RAG 다중질의 처리 방식과 반복 정제 방식 설명해줘"
    result = rag.answer(query)

    print(json.dumps(result, ensure_ascii=False, indent=2))