import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import require_env, tavily_post


app = FastAPI(title="Tavily Search API", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    topic: str = Field(default="general", description="Search topic such as general, news, or finance")
    max_results: int = Field(default=5, ge=1, le=20)
    include_answer: str | bool = Field(default="advanced")
    include_raw_content: str | bool = Field(default=False)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    try:
        require_env("TAVILY_API_KEY")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "tavily-search-api",
        "example": "POST /search with {'query': 'latest open-source inference servers'}",
    }


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    try:
        return tavily_post(
            "search",
            {
                "query": payload.query,
                "topic": payload.topic,
                "max_results": payload.max_results,
                "include_answer": payload.include_answer,
                "include_raw_content": payload.include_raw_content,
                "include_favicon": True,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
