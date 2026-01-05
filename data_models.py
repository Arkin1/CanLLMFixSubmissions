from dataclasses import dataclass

@dataclass
class Test:
    input: str
    output: str

@dataclass
class Problem:
    problem_id: str
    description: str 
    input_format: str
    output_format: str
    examples: str
    note: str
    tests: list[Test]

@dataclass
class Submission:
    submission_id: str
    source_code: str
    verdict: str
    ds_verdict: str

@dataclass
class ModelInfo:
    vendor: str
    model_name: str

@dataclass
class GeneratedLLMResult:
    generated_result_id: str
    method_name: str
    llm_result: str
    source_code: str
    ds_verdict: str
    model_info: ModelInfo

@dataclass
class ResultAnalysis:
    problem: Problem
    submission: Submission
    generated_results: list[GeneratedLLMResult]





