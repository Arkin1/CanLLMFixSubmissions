from __future__ import annotations
from pydantic import BaseModel
from typing import Optional

class Test(BaseModel):
    input: str
    output: str

    def __str__(self):
        return f'#Input#: {self.input}\n #Output#: {self.output}\n'


class Problem(BaseModel):
    problem_id: str
    description: str
    input_format: str
    output_format: str
    examples: list[Test]
    note: Optional[str]
    tests: list[Test]
    time_limit:float
    memory_limit:float
    input_mode:str
    generated_checker:Optional[str]


class Submission(BaseModel):
    submission_id: str
    source_code: str
    programming_language: str
    verdict: str
    ds_verdict: bool
    anchor: Submission = None

class ModelInfo(BaseModel):
    vendor: str
    model_name: str

class Loss(BaseModel):
    test_loss: float
    num_add_lines_loss: float
    num_deleted_lines_loss: float
    num_total_lines_loss: float
    
class GeneratedLLMResult(BaseModel):
    generated_result_id: str
    method_name: str
    llm_result: str
    source_code: str
    model_info: ModelInfo
    loss: Optional[Loss] = None

class ResultAnalysis(BaseModel):
    problem_id: str
    submission: Submission
    generated_results: list[GeneratedLLMResult]





