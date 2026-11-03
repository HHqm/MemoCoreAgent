"""REST API over a MemoryStore, for services that do not speak MCP.

Each user's memory is separate (`/users/{user}/...`). Requests are handled on a thread pool, so concurrent
searches can share reranker calls when the store uses a BatchingReranker. With a token set, every request
needs `Authorization: Bearer <token>`.
"""

from __future__ import annotations

import hmac
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from memocore.store import Memory, MemoryStore, Utterance


class MessageIn(BaseModel):
    text: str = Field(min_length=1)
    speaker: str = "user"
    at: Optional[datetime] = None  # when it was said; now if omitted
    session: Optional[str] = None  # conversation key; omitted, sessions split on a 30-minute gap
    image: Optional[str] = None  # description of an image shared with the message


class RememberIn(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=1000)


class MemoryOut(BaseModel):
    id: str
    speaker: str
    text: str
    session_time: str
    image: Optional[str] = None
    time_note: Optional[str] = None


class RecallOut(BaseModel):
    context: str  # the memories with ids and dates, ready to put in a prompt
    memories: list[MemoryOut]


def _out(memory: Memory) -> MemoryOut:
    return MemoryOut(**memory.__dict__)


def _ids(ids: str) -> list[str]:
    return [part.strip() for part in ids.split(",") if part.strip()]


def build_app(store: MemoryStore, *, token: str | None = None) -> FastAPI:
    def authorized(authorization: Optional[str] = Header(default=None)) -> None:
        if token and not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")

    app = FastAPI(title="MemoCore", description="Long-term memory stored verbatim, with cited recall.",
                  dependencies=[Depends(authorized)])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/users/{user}/memories", response_model=list[MemoryOut])
    def remember(user: str, body: RememberIn) -> list[MemoryOut]:
        """Store messages verbatim; each gets an id such as D3:12 that answers can cite."""
        utterances = [Utterance(m.text, m.speaker, m.at.replace(tzinfo=None) if m.at else None, m.session, m.image)
                      for m in body.messages]
        return [_out(memory) for memory in store.add_many(user, utterances)]

    @app.get("/users/{user}/recall", response_model=RecallOut)
    def recall(user: str, query: str = Query(min_length=1), budget: int = Query(10, ge=1, le=50)) -> RecallOut:
        """The stored messages most relevant to the query, in conversation order."""
        result = store.search(user, query, budget=budget)
        return RecallOut(context=result.context, memories=[_out(memory) for memory in result.memories])

    @app.get("/users/{user}/memories", response_model=list[MemoryOut])
    def read(user: str, ids: str = Query(description="comma-separated ids, such as D3:12,D5:2")) -> list[MemoryOut]:
        """The exact stored messages behind cited ids."""
        try:
            return [_out(memory) for memory in store.get(user, _ids(ids))]
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/users/{user}/memories")
    def forget(user: str, ids: Optional[str] = None, everything: bool = False) -> dict[str, int]:
        """Delete the given messages, or with everything=true all of the user's memory."""
        if not ids and not everything:
            raise HTTPException(status_code=422, detail="give ids, or everything=true to delete all of this user's memory")
        try:
            return {"deleted": store.forget(user, None if everything else _ids(ids or ""))}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
