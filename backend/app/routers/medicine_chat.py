"""Medicine-only DailyMed and local Gemma chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from backend.app.services.medicine_chat import (
    DailyMedUnavailable,
    LocalGemmaUnavailable,
    MedicineChatError,
    answer_medicine_question,
    search_dailymed_labels,
)

router = APIRouter(prefix="/medicine-chat", tags=["medicine chat"])


class MedicineLabelOption(BaseModel):
    setid: str
    title: str
    published_date: str | None = None


class MedicineLabelSearchResponse(BaseModel):
    labels: list[MedicineLabelOption]
    source: str = "U.S. DailyMed"


class MedicineQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    setid: str = Field(min_length=36, max_length=36)
    question: str = Field(min_length=1, max_length=1000)


class MedicineQuestionResponse(BaseModel):
    answer: str
    source_title: str
    source_url: str
    sections: list[str]
    model: str
    retrieved_at: str
    disclaimer: str = (
        "For medicine information only; not a diagnosis or personalized treatment "
        "or dosing recommendation. Confirm directions with the product label and a "
        "doctor or pharmacist."
    )


@router.get(
    "/labels",
    response_model=MedicineLabelSearchResponse,
    summary="Search official DailyMed medicine labels",
)
async def search_labels(
    drug_name: str = Query(min_length=2, max_length=120),
) -> MedicineLabelSearchResponse:
    try:
        labels = await run_in_threadpool(search_dailymed_labels, drug_name.strip())
    except DailyMedUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return MedicineLabelSearchResponse(
        labels=[
            MedicineLabelOption(
                setid=label.setid,
                title=label.title,
                published_date=label.published_date,
            )
            for label in labels
        ]
    )


@router.post(
    "/ask",
    response_model=MedicineQuestionResponse,
    summary="Answer a medicine question from one DailyMed label",
)
async def ask_medicine_question(
    payload: MedicineQuestionRequest,
) -> MedicineQuestionResponse:
    try:
        result = await run_in_threadpool(
            answer_medicine_question,
            setid=payload.setid,
            question=payload.question.strip(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LocalGemmaUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except DailyMedUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except MedicineChatError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return MedicineQuestionResponse(
        answer=result.answer,
        source_title=result.source_title,
        source_url=result.source_url,
        sections=result.sections,
        model=result.model,
        retrieved_at=result.retrieved_at,
    )
