from pydantic import BaseModel,Field
from enum import Enum
from typing import Optional

"""A job gets created → has a status → sub-questions get generated 
    → each gets researched and scored → everything gets synthesized."""

class JobStatus(str,Enum):
    pending="pending"
    dispatching="dispatching"
    researching="researching"
    synthesizing="synthesizing"
    complete="complete"
    failed="failed"

class SubQuestion(BaseModel):
    id:int
    question:str

class ResearchResult(BaseModel):
    sub_question:SubQuestion
    answer:str
    confidence_score:float=Field(ge=0.0, le=1.0)
    critique:str

class ResearchJob(BaseModel):
    job_id:str
    query:str
    status:JobStatus=JobStatus.pending
    sub_questions:list[SubQuestion]=[]
    results:list[ResearchResult]=[]
    final_report:Optional[str]=None
    error:Optional[str]=None

class CreateJobRequest(BaseModel):
    query:str

class JobStatusResponse(BaseModel):
    job_id:str
    status:JobStatus
    sub_questions:list[SubQuestion]=[]
    results:list[ResearchResult]=[]
    final_report:Optional[str]=None
    error:Optional[str]=None