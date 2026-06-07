import pytest
from models import JobStatus, SubQuestion, ResearchResult, ResearchJob, CreateJobRequest, JobStatusResponse
from state import JobStore


# ---------- models ----------

def test_job_status_default():
    job = ResearchJob(job_id="abc", query="test query")
    assert job.status == JobStatus.pending
    assert job.sub_questions == []
    assert job.results == []
    assert job.final_report is None
    assert job.error is None

def test_sub_question_model():
    sq = SubQuestion(id=1, question="What is aspirin?")
    assert sq.id == 1
    assert sq.question == "What is aspirin?"

def test_research_result_score_bounds():
    sq = SubQuestion(id=1, question="test?")
    # valid score
    r = ResearchResult(sub_question=sq, answer="ans", confidence_score=0.8, critique="ok")
    assert r.confidence_score == 0.8

def test_research_result_invalid_score():
    sq = SubQuestion(id=1, question="test?")
    with pytest.raises(Exception):  # pydantic validation error
        ResearchResult(sub_question=sq, answer="ans", confidence_score=1.5, critique="ok")

def test_create_job_request_empty():
    req = CreateJobRequest(query="")
    assert req.query == ""  # model allows it, FastAPI endpoint rejects it


# ---------- state (async) ----------

@pytest.mark.asyncio
async def test_create_and_get_job():
    store = JobStore()
    await store.create_job("job1", "my query")
    job = await store.get_job("job1")
    assert job.job_id == "job1"
    assert job.query == "my query"
    assert job.status == JobStatus.pending

@pytest.mark.asyncio
async def test_update_status():
    store = JobStore()
    await store.create_job("job2", "query")
    await store.update_status("job2", JobStatus.researching)
    job = await store.get_job("job2")
    assert job.status == JobStatus.researching

@pytest.mark.asyncio
async def test_add_result():
    store = JobStore()
    await store.create_job("job3", "query")
    sq = SubQuestion(id=1, question="test?")
    result = ResearchResult(sub_question=sq, answer="ans", confidence_score=0.7, critique="good")
    await store.add_result("job3", result)
    job = await store.get_job("job3")
    assert len(job.results) == 1
    assert job.results[0].confidence_score == 0.7

@pytest.mark.asyncio
async def test_set_final_report():
    store = JobStore()
    await store.create_job("job4", "query")
    await store.set_final_report("job4", "final report text")
    job = await store.get_job("job4")
    assert job.final_report == "final report text"
    assert job.status == JobStatus.complete

@pytest.mark.asyncio
async def test_set_error():
    store = JobStore()
    await store.create_job("job5", "query")
    await store.set_error("job5", "something broke")
    job = await store.get_job("job5")
    assert job.error == "something broke"
    assert job.status == JobStatus.failed

@pytest.mark.asyncio
async def test_get_nonexistent_job():
    store = JobStore()
    job = await store.get_job("doesnotexist")
    assert job is None
