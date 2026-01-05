from genai.endpoints import get_llm_api
from genai.utils import read_prompt, CodeParser
from data_models import Problem, Submission, GeneratedLLMResult, ModelInfo
from utils import RetryExecution
import logging

logger = logging.getLogger()

class LLMMethod():
    def __init__(self, model_name:str):
        self.model_name = model_name

    def predict(self, problem: Problem, submission: Submission) -> GeneratedLLMResult:
        pass

class NaiveFixBugLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str):
        super().__init__(model_name)
        self.llm = get_llm_api(vendor, model_name)
        self.prompt_name = 'prompt_naive_fix_bug'

    def predict(self, problem: Problem, submission: Submission) -> GeneratedLLMResult:
        prompt = read_prompt(self.prompt_name,
                             problem_description = problem.description,
                             problem_input_format = problem.input_format,
                             problem_output_format = problem.output_format,
                             problem_example = str(problem.examples),
                             note = problem.note,
                             submission_verdict = submission.verdict,
                             submission_code = submission.source_code)
        def f():
            return self.llm.generate(prompt)
        
        with RetryExecution("NaiveFixBugLLMMethod", 3, logger) as retry:
            result = retry(f)

        return GeneratedLLMResult(submission.submission_id + '_NaiveFixBugLLMMethod',
                                  "_NaiveFixBugLLMMethod",
                                  result,
                                  CodeParser.extract_code(result),
                                  None,
                                  ModelInfo(self.llm.vendor, self.llm.model_name))
        

class GenerateFromScratchLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str):
        super().__init__(model_name)
        self.llm = get_llm_api(vendor, model_name)
        self.prompt_name = 'prompt_generate_solution'

    def predict(self, problem: Problem, submission: Submission) -> GeneratedLLMResult:
        prompt = read_prompt(self.prompt_name,
                             problem_description = problem.description,
                             problem_input_format = problem.input_format,
                             problem_output_format = problem.output_format,
                             problem_example = str(problem.examples),
                             note = problem.note,
                             submission_verdict = submission.verdict)
        def f():
            return self.llm.generate(prompt)
        
        with RetryExecution("GenerateFromScratchLLMMethod", 3, logger) as retry:
            result = retry(f)

        return GeneratedLLMResult(submission.submission_id + '_GenerateFromScratchLLMMethod',
                                  'GenerateFromScratchLLMMethod',
                                  result,
                                  CodeParser.extract_code(result),
                                  None,
                                  ModelInfo(self.llm.vendor, self.llm.model_name))
    

_available_methods = {"NaiveFixBugLLMMethod": NaiveFixBugLLMMethod,
                      "GenerateFromScratchLLMMethod": GenerateFromScratchLLMMethod}

def create_method(name:str, **kwargs):
    return _available_methods[name](**kwargs)
