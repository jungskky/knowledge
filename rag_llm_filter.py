"""
LLM-based text sanitizer for RAG ingestion.

Goal: remove or redact PII (e.g., 담당자 이메일/전화번호) from source text before
storing in a vector DB for embeddings, while preserving business meaning.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

RAG_INGESTION_SANITIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 RAG(Vector DB) 적재 전 문서를 정제하는 텍스트 필터입니다.

목표:
- 주어진 원문에서 개인식별정보(PII) 또는 담당자 연락처 정보를 제거/익명화하여, 벡터DB에 저장되지 않도록 합니다.
- 문서의 업무적 의미/정책/절차/사실 관계는 최대한 유지합니다.

반드시 지킬 규칙:
- 출력은 "정제된 텍스트"만. 어떤 설명/머리말/코드블록/따옴표/JSON도 출력하지 마세요.
- 다음 항목은 원문에 있으면 모두 제거하거나 대체 토큰으로 익명화하세요.
  - 이메일 주소 → [EMAIL_REDACTED]
  - 전화번호(휴대폰/유선/내선/국가번호 포함 가능) → [PHONE_REDACTED]
  - 사람 이름 + 연락처 조합, "담당자/문의/연락/Contact/Owner"로 표시된 개인 정보 줄 → 연락처 부분은 반드시 익명화
  - 주소/주민번호/사번/계정ID 등 식별 가능한 고유 식별자 → [ID_REDACTED] 또는 [ADDRESS_REDACTED]
  - 서명(Signature) 블록(예: 이름/직함/회사/이메일/전화가 반복되는 하단) → PII를 제거한 뒤 필요하면 한 줄로 축약
- 문서 본문에서 PII만 제거하고, 나머지 문장은 가능한 한 그대로 유지하세요.
- 표/목록/줄바꿈 구조를 크게 망가뜨리지 마세요.
""",
        ),
        ("human", "원문:\n{raw_text}\n\n정제된 텍스트:"),
    ]
)


# ---------------------------------------------------------------------------
# Simple regex fallback (safety net)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

# Broad phone patterns (KR + general). Keeps it conservative: redact sequences likely to be phone numbers.
_PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4})(?!\d)",
    flags=re.IGNORECASE,
)

_JUMIN_RE = re.compile(r"\b\d{6}-?\d{7}\b")  # KR resident registration number
_EMPLOYEE_ID_RE = re.compile(r"\b(?:사번|employee\s*id|emp\s*id|id)\s*[:#]?\s*[A-Z0-9-]{4,}\b", re.I)


def _regex_sanitize(text: str) -> str:
    t = text or ""
    t = _EMAIL_RE.sub("[EMAIL_REDACTED]", t)
    t = _PHONE_RE.sub("[PHONE_REDACTED]", t)
    t = _JUMIN_RE.sub("[ID_REDACTED]", t)
    t = _EMPLOYEE_ID_RE.sub("[ID_REDACTED]", t)
    return t


def _collapse_excess_blank_lines(text: str, max_consecutive: int = 2) -> str:
    lines = (text or "").splitlines()
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive:
                out.append("")
        else:
            blank_run = 0
            out.append(line.rstrip())
    return "\n".join(out).strip() + ("\n" if text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SanitizeResult:
    sanitized_text: str
    used_llm: bool


def sanitize_text_for_vector_db(
    raw_text: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_chars: int = 60_000,
) -> SanitizeResult:
    """
    Returns text sanitized for embedding/vector DB storage.

    - Uses LLM if OPENAI_API_KEY exists; otherwise falls back to regex-only.
    - Always applies regex sanitizer at the end as a safety net.
    """

    raw = (raw_text or "").strip()
    if not raw:
        return SanitizeResult(sanitized_text="", used_llm=False)

    clipped = raw if len(raw) <= max_chars else raw[:max_chars]

    used_llm = False
    sanitized = clipped

    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model=model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
        chain = RAG_INGESTION_SANITIZE_PROMPT | llm | StrOutputParser()
        try:
            sanitized = chain.invoke({"raw_text": clipped})
            used_llm = True
        except Exception:
            sanitized = clipped
            used_llm = False

    sanitized = _regex_sanitize(sanitized)
    sanitized = _collapse_excess_blank_lines(sanitized)
    return SanitizeResult(sanitized_text=sanitized, used_llm=used_llm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_all(paths: Iterable[str]) -> str:
    if not paths:
        return sys.stdin.read()
    chunks: list[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            chunks.append(f.read())
    return "\n\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM filter to remove PII before vector DB ingestion.")
    parser.add_argument("paths", nargs="*", help="Input text file paths. If empty, reads stdin.")
    parser.add_argument("--model", default=None, help="Chat model name (default: OPENAI_CHAT_MODEL or gpt-4o-mini).")
    parser.add_argument("--no-llm", action="store_true", help="Force regex-only (ignore OPENAI_API_KEY).")
    parser.add_argument("--max-chars", type=int, default=60_000, help="Max chars to send to the LLM.")
    args = parser.parse_args(argv)

    raw = _read_all(args.paths)
    if args.no_llm:
        os.environ.pop("OPENAI_API_KEY", None)

    result = sanitize_text_for_vector_db(raw, model=args.model, max_chars=args.max_chars)
    sys.stdout.write(result.sanitized_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

