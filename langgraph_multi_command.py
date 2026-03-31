"""
다중 명령 인식·분리·실행 (LangGraph).

한 문장에 여러 요청이 섞인 입력을 개별 명령으로 나눈 뒤, 유형별로 처리합니다.
"""

from __future__ import annotations

import ast
import os
import operator
import re
from datetime import datetime
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------


class SplitCommands(BaseModel):
    """사용자 발화에서 분리된 개별 명령."""

    commands: list[str] = Field(
        description="서로 독립적으로 실행 가능한 짧은 명령 문장들. 순서 유지."
    )


class CommandKind(BaseModel):
    """단일 명령의 처리 유형."""

    kind: Literal["echo", "datetime", "calculate", "chat"] = Field(
        description=(
            "echo: 사용자가 반복해 달라고 한 문장 그대로 출력. "
            "datetime: 현재 날짜/시각 알려달라는 요청. "
            "calculate: 사칙연산 등 수식 계산. "
            "chat: 그 외 일반 대화·질문·요약 등."
        )
    )
    argument: str = Field(
        default="",
        description="echo 시 반복할 문장, calculate 시 수식(숫자와 + - * / 괄호만), 그 외는 빈 문자열 가능.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SPLIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """사용자 발화에 여러 요청이 하나의 문장으로 묶여 있을 수 있습니다.
각 요청을 **서로 독립적으로 실행 가능한 짧은 명령**으로 나누세요.

규칙:
- 한 문장에 명령이 하나뿐이면 길이 1인 목록으로 반환합니다.
- 접속사(그리고, 또, 그다음에 등)나 구두점으로 이어진 요청은 분리합니다.
- 의미가 겹치지 않게, 각 항목은 완전한 요청으로 쓰세요.
- 불필요한 인사·맥락은 첫 명령에만 넣거나 생략합니다.""",
        ),
        ("human", "{user_message}"),
    ]
)

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """다음 단일 명령을 분류하세요.

- echo: "X라고 말해", "Y를 반복해"처럼 특정 문장을 그대로 말해 달라는 경우. argument에 그 문장.
- datetime: 오늘 날짜, 지금 시각, 몇 시 등 시간·날짜 정보 요청.
- calculate: 숫자와 사칙연산(+, -, *, /) 및 괄호로 된 계산 요청. argument에 수식만.
- chat: 위에 해당하지 않는 일반 질문·대화·설명 요청.""",
        ),
        ("human", "명령:\n{command}"),
    ]
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "한국어로 간결하고 정확하게 답하세요.",
        ),
        ("human", "{command}"),
    ]
)


# ---------------------------------------------------------------------------
# Safe calculate (limited arithmetic)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        fn = _ALLOWED_BINOPS[type(node.op)]
        return float(fn(left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_BINOPS:
        v = _eval_ast(node.operand)
        fn = _ALLOWED_BINOPS[type(node.op)]
        return float(fn(v))
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    raise ValueError("허용되지 않은 수식입니다.")


def safe_calculate(expression: str) -> str:
    s = (expression or "").strip().replace(" ", "")
    if not s:
        return "계산할 수식이 없습니다."
    if not re.fullmatch(r"[0-9+\-*/().]+", s):
        return "숫자와 + - * / 괄호만 사용할 수 있습니다."
    try:
        tree = ast.parse(s, mode="eval")
        value = _eval_ast(tree)
        if value == int(value):
            return str(int(value))
        return f"{value:.6g}"
    except Exception as e:  # noqa: BLE001
        return f"계산 오류: {e}"


# ---------------------------------------------------------------------------
# State & graph
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    user_message: str
    commands: list[str]
    results: list[dict[str, Any]]


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)


def create_app(llm: ChatOpenAI | None = None):
    llm = llm or _build_llm()
    splitter = SPLIT_PROMPT | llm.with_structured_output(SplitCommands)
    classifier = CLASSIFY_PROMPT | llm.with_structured_output(CommandKind)
    chat_chain = CHAT_PROMPT | llm

    def parse_commands(state: GraphState) -> GraphState:
        raw = state["user_message"].strip()
        if not raw:
            return {"commands": [], "results": []}
        try:
            out = splitter.invoke({"user_message": raw})
            cmds = [c.strip() for c in out.commands if c and str(c).strip()]
            if not cmds:
                cmds = [raw]
        except Exception:
            cmds = [raw]
        return {"commands": cmds, "results": []}

    def route_after_parse(state: GraphState) -> Literal["execute", "end"]:
        return "end" if not state.get("commands") else "execute"

    def run_all_commands(state: GraphState) -> GraphState:
        cmds = state.get("commands") or []
        rows: list[dict[str, Any]] = []
        for i, cmd in enumerate(cmds, start=1):
            try:
                plan = classifier.invoke({"command": cmd})
            except Exception as e:  # noqa: BLE001
                rows.append(
                    {
                        "index": i,
                        "command": cmd,
                        "kind": "error",
                        "output": f"분류 실패: {e}",
                    }
                )
                continue

            kind = plan.kind
            arg = (plan.argument or "").strip()

            if kind == "echo":
                out = arg if arg else "(반복할 문장 없음)"
            elif kind == "datetime":
                out = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif kind == "calculate":
                out = safe_calculate(arg or cmd)
            else:
                out = chat_chain.invoke({"command": cmd}).content

            rows.append(
                {
                    "index": i,
                    "command": cmd,
                    "kind": kind,
                    "output": out,
                }
            )

        return {"results": rows}

    g = StateGraph(GraphState)
    g.add_node("parse_commands", parse_commands)
    g.add_node("run_all_commands", run_all_commands)
    g.set_entry_point("parse_commands")
    g.add_conditional_edges(
        "parse_commands",
        route_after_parse,
        {"execute": "run_all_commands", "end": END},
    )
    g.add_edge("run_all_commands", END)

    return g.compile()


def run_once(user_message: str) -> dict[str, Any]:
    """단일 턴: 분리된 명령 목록과 각 실행 결과."""
    app = create_app()
    result = app.invoke(
        {
            "user_message": user_message,
            "commands": [],
            "results": [],
        }
    )
    return {
        "commands": result.get("commands") or [],
        "results": result.get("results") or [],
    }


def format_report(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for r in data.get("results") or []:
        lines.append(f"[{r.get('index')}] ({r.get('kind')}) {r.get('command')}")
        lines.append(f"    → {r.get('output')}")
    return "\n".join(lines) if lines else "(실행 결과 없음)"


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 환경 변수를 설정하세요.")

    samples = [
        "안녕하고, 지금 몇 시인지 알려줘, 그리고 (3+5)*2 계산해 줘",
        "오늘 날짜 말해줘",
        '"반가워요"라고 말해 줘',
        "2 곱하기 7은?",
    ]
    for q in samples:
        out = run_once(q)
        print("=" * 60)
        print("입력:", q)
        print("분리된 명령:", out["commands"])
        print(format_report(out))
