from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import Response
from filelock import Timeout
from pydantic import BaseModel, Field

from business_assistant import ask
from orchestrator import route_request
from quote_agent import ReviewDecision
from quote_service import MAX_UPLOAD_BYTES, XLSX_MIME, QuoteService, quote_workbook


app = FastAPI(title="询价报价智能 Agent Demo")
Mode = Literal["mock", "rules", "live"]
InquiryId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


class PendingRepair(BaseModel):
    model: str = ""
    symptom: str = ""
    question: str = ""


class ConversationContext(BaseModel):
    pending: PendingRepair | None = None
    product_query: str = ""
    repair_model: str = ""


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    extraction_mode: Mode = "mock"
    context: ConversationContext = Field(default_factory=ConversationContext)
    use_dense: bool = False
    rerank: bool = False


class SubmitReview(BaseModel):
    interrupt_id: str = Field(min_length=1, max_length=100)
    decisions: list[ReviewDecision] = Field(max_length=1000)


def _quote_response(snapshot) -> dict:
    state = snapshot.values
    pending = snapshot.interrupts[0] if snapshot.interrupts else None
    return {
        "inquiry_id": state["inquiry_id"],
        "source_name": state["source_name"],
        "status": "completed" if "quote" in state else "waiting_review" if pending else "running",
        "interrupt_id": pending.id if pending else None,
        "review_error": state.get("review_error", ""),
        "items": state.get("matched", []),
        "extraction_mode": state.get("llm_mode"),
        "extraction_status": state.get("llm_status"),
        "quote_id": state.get("quote_id"),
        "quote": state.get("quote"),
        "total": state.get("total"),
        "postponed": state.get("postponed", []),
        "audit": state.get("audit", []),
    }


def _handle_quote(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Timeout as exc:
        raise HTTPException(status_code=409, detail="任务正在处理，请稍后重试。") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/assist")
def assist(request: Question) -> dict:
    context = request.context.model_dump(exclude_none=True)
    intent = route_request(request.text, context=context)
    if intent == "quote":
        snapshot = _handle_quote(
            QuoteService().start, source_name="文本询价", text=request.text, mode=request.extraction_mode,
        )
        return {"intent": intent, "quote": _quote_response(snapshot)}
    if intent == "qa":
        answer = ask(request.text, context, use_dense=request.use_dense, rerank=request.rerank)
        return {"intent": intent, "answer": answer, "context": context}
    return {"intent": intent, "message": "请提供询价清单，或提出耗材选型、价格、库存和售后问题。"}


@app.post("/quotes/upload", status_code=201)
def upload_quote(file: Annotated[UploadFile, File()], mode: Annotated[Mode, Query()] = "mock") -> dict:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    snapshot = _handle_quote(
        QuoteService().start, source_name=file.filename or "询价文件", data=data, mode=mode,
    )
    return _quote_response(snapshot)


@app.get("/quotes/{inquiry_id}")
def get_quote(inquiry_id: InquiryId) -> dict:
    return _quote_response(_handle_quote(QuoteService().get, inquiry_id))


@app.post("/quotes/{inquiry_id}/review")
def review_quote(inquiry_id: InquiryId, review: SubmitReview) -> dict:
    snapshot = _handle_quote(
        QuoteService().resume, inquiry_id,
        [decision.model_dump(exclude_unset=True) for decision in review.decisions],
        interrupt_id=review.interrupt_id,
    )
    return _quote_response(snapshot)


@app.get("/quotes/{inquiry_id}/workbook")
def download_quote(inquiry_id: InquiryId) -> Response:
    snapshot = _handle_quote(QuoteService().get, inquiry_id)
    if "quote" not in snapshot.values:
        raise HTTPException(status_code=409, detail="请先完成审核。")
    return Response(
        quote_workbook(snapshot.values), media_type=XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="quote-{inquiry_id}.xlsx"'},
    )
