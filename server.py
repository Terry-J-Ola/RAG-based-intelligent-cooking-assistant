from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from rag_service import rag_service


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="RAG 菜谱助手", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")


@app.on_event("startup")
def warm_up() -> None:
    rag_service.initialize()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "RAG 菜谱助手"},
    )


@app.get("/health")
def health() -> dict:
    return rag_service.health()


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        return rag_service.answer(query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成回答失败: {exc}") from exc
