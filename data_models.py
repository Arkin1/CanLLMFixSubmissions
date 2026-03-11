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
    problem_id: str
    source_code: str
    programming_language: str
    verdict: str
    ds_verdict: bool
    anchor: Optional[Submission] = None

class ModelInfo(BaseModel):
    vendor: str
    model_name: str

class RewardContext(BaseModel):
    num_lines_buggy_solution: int
    num_lines_correct_solution: int
    num_lines_generated_solution: int
    num_added_lines_baseline: int
    num_deleted_lines_baseline: int
    num_added_lines_generated: int
    num_deleted_lines_generated: int
    num_common_lines_baseline: int
    num_common_lines_generated: int
    test_pass:float
    similarity_baseline:float
    similarity_generated:float

class Reward(BaseModel):
    context: Optional[RewardContext] = None
    test_reward: float
    similarity_reward: float
    total_reward: float
    
class GeneratedLLMResult(BaseModel):
    generated_result_id: str
    method_name: str
    llm_result: str
    source_code: str
    model_info: ModelInfo
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reward: Optional[Reward] = None

class ResultAnalysis(BaseModel):
    problem_id: str
    submission: Submission
    generated_results: list[GeneratedLLMResult]





