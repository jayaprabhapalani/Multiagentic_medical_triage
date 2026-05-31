# pipeline.py
import asyncio
import uuid
from models import JobStatus, ResearchResult, SubQuestion
from state import job_store
from agents.dispatcher import dispatch
from agents.researcher import research
from agents.critic import critique
from agents.synthesizer import synthesize
from config import settings

# caps total concurrent LLM calls across all workers
llm_semaphore = asyncio.Semaphore(2)


async def worker(
    worker_id: int,
    job_id: str,
    queue: asyncio.Queue,
    query: str
):
    while True:
        try:
            sub_question: SubQuestion = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        print(f"[worker-{worker_id}] picked up: {sub_question.question[:50]}...")

        try:
            # each sub question is assigned as seperate jobs, these are the given to seperate workers
            # first give the subquestion to the research agent 
            async with llm_semaphore:
                answer = await asyncio.wait_for(
                    research(query, sub_question),
                    timeout=settings.researcher_timeout
                )
            # then give the research agent answer to the critique
            async with llm_semaphore:
                score, critic_text = await asyncio.wait_for(
                    critique(query, sub_question, answer),
                    timeout=settings.researcher_timeout
                )

            result = ResearchResult(
                sub_question=sub_question,
                answer=answer,
                confidence_score=score,
                critique=critic_text
            )

            await job_store.add_result(job_id, result)
            print(f"[worker-{worker_id}] completed: {sub_question.id} | score: {score}")

        #handling timeout
        except asyncio.TimeoutError:
            print(f"[worker-{worker_id}] timeout on sub-question {sub_question.id}")
            result = ResearchResult(
                sub_question=sub_question,
                answer="Research timed out for this sub-question.",
                confidence_score=0.0,
                critique="This sub-question could not be completed within the timeout."
            )
            await job_store.add_result(job_id, result)

        #handling exception
        except Exception as e:
            print(f"[worker-{worker_id}] error on sub-question {sub_question.id}: {e}")
            result = ResearchResult(
                sub_question=sub_question,
                answer=f"Research failed: {str(e)}",
                confidence_score=0.0,
                critique="An unexpected error occurred during research."
            )
            await job_store.add_result(job_id, result)

        finally:
            queue.task_done()


async def run_pipeline(job_id: str, query: str):
    try:
        # 1. dispatch
        await job_store.update_status(job_id, JobStatus.dispatching)
        print(f"[pipeline] dispatching job {job_id}")

        sub_questions = await dispatch(query)
        await job_store.set_sub_questions(job_id, sub_questions)
        print(f"[pipeline] got {len(sub_questions)} sub-questions")

        # 2. fill queue
        queue = asyncio.Queue()
        for sq in sub_questions:
            await queue.put(sq)

        # 3. spin up workers + drain queue
        await job_store.update_status(job_id, JobStatus.researching)

        workers = [
            asyncio.create_task(
                worker(i + 1, job_id, queue, query)
            )
            for i in range(settings.max_researchers)
        ]

        await queue.join()

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        print(f"[pipeline] all research complete for job {job_id}")

        # 4. synthesize
        await job_store.update_status(job_id, JobStatus.synthesizing)

        job = await job_store.get_job(job_id)
        report = await synthesize(query, job.results)

        await job_store.set_final_report(job_id, report)
        print(f"[pipeline] job {job_id} complete")

    except Exception as e:
        print(f"[pipeline] fatal error for job {job_id}: {e}")
        await job_store.set_error(job_id, str(e))


async def create_job(query: str) -> str:
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, query)
    asyncio.create_task(run_pipeline(job_id, query))
    return job_id