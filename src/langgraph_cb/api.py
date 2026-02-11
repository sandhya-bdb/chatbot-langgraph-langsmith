from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from langgraph_cb.graphs.hitl import build_graph


app = FastAPI(title="LangGraph HITL API", version="0.1.0")

graph = build_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    thread_id: str
    response: Optional[str] = None
    approval_prompt: Optional[str] = None
    interrupt_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.invoke({"messages": [{"role": "user", "content": req.message}]}, config=config)
        last = state["messages"][-1]
        content = getattr(last, "content", None) or (
            last.get("content") if isinstance(last, dict) else None
        )
        if isinstance(content, str) and content.startswith("REQUEST_BUY::"):
            return ChatResponse(
                status="approval_required",
                thread_id=thread_id,
                approval_prompt=content,
            )
        return ChatResponse(
            status="completed",
            thread_id=thread_id,
            response=content,
        )
    except GraphInterrupt as exc:
        interrupts = list(exc.args[0]) if exc.args else []
        if interrupts:
            interrupt = interrupts[0]
            return ChatResponse(
                status="approval_required",
                thread_id=thread_id,
                approval_prompt=str(interrupt.value),
                interrupt_id=interrupt.id,
            )
        return ChatResponse(status="approval_required", thread_id=thread_id)


@app.post("/approve", response_model=ChatResponse)
def approve(req: ApprovalRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": req.thread_id}}
    try:
        state = graph.invoke(Command(resume=req.decision), config=config)
        return ChatResponse(
            status="completed",
            thread_id=req.thread_id,
            response=state["messages"][-1].content,
        )
    except GraphInterrupt as exc:
        interrupts = list(exc.args[0]) if exc.args else []
        if interrupts:
            interrupt = interrupts[0]
            return ChatResponse(
                status="approval_required",
                thread_id=req.thread_id,
                approval_prompt=str(interrupt.value),
                interrupt_id=interrupt.id,
            )
        return ChatResponse(status="approval_required", thread_id=req.thread_id)
