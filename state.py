import asyncio
from models import ResearchJob, JobStatus,ResearchResult

class JobStore:
    def __init__(self):
        self._jobs:dict[str,ResearchJob]={}
        self._lock=asyncio.Lock()

    async def create_job(self,job_id:str,query:str) -> ResearchJob:
        async with self._lock:
            job=ResearchJob(job_id=job_id,query=query)
            self._jobs[job_id]=job
            return job
        
    async def get_job(self,job_id:str)->ResearchJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_status(self, job_id:str,status:JobStatus):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status=status

    async def set_sub_questions(self,job_id:str,sub_questions):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].sub_questions=sub_questions

    async def add_result(self, job_id:str,result:ResearchResult):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].results.append(result)

    async def set_final_report(self,job_id:str,report:str):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].final_report=report
                self._jobs[job_id].status=JobStatus.complete

    async def set_error(self,job_id:str,error:str):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].error=error
                self._jobs[job_id].status=JobStatus.failed


#singleton -every file import this one instance
job_store=JobStore()
