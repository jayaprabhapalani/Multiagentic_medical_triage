# main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from models import CreateJobRequest, JobStatusResponse
from state import job_store
from pipeline import create_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] starting up")
    yield
    print("[server] shutting down")


app = FastAPI(
    title="Medical Literature Triage",
    description="Multi-agent async pipeline for clinical research queries",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/research", response_model=JobStatusResponse)
async def create_research_job(request: CreateJobRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    job_id = await create_job(request.query)
    job = await job_store.get_job(job_id)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
    )


@app.get("/research/{job_id}", response_model=JobStatusResponse)
async def get_research_job(job_id: str):
    job = await job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        sub_questions=job.sub_questions,
        results=job.results,
        final_report=job.final_report,
        error=job.error
    )


@app.get("/health")
async def health():
    return {"status": "ok"}