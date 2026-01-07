from genai.endpoints import get_llm_api
from genai.utils import read_prompt, CodeParser
from data_models import Problem, Submission, GeneratedLLMResult, ModelInfo, Loss
from utils import compile_and_test, count_differences, ProblemsManager
from utils import RetryExecution
import logging

logger = logging.getLogger()

class LLMMethod():
    def __init__(self, model_name:str):
        self.model_name = model_name

    def predict(self, problem: Problem, submission: Submission) -> GeneratedLLMResult:
        pass
    async def compute_loss(self, problem_manager: ProblemsManager, problem:Problem, submission: Submission, result: GeneratedLLMResult):
        test_loss = await problem_manager.evaluate_submission(problem.problem_id, result.source_code, "GNU C++", 'soft')
        num_add_baseline, num_del_baseline = count_differences(submission.source_code, submission.anchor.source_code)
        num_add_generated, num_del_generated = count_differences(result.source_code, submission.anchor.source_code)

        add_loss = num_add_generated - num_add_baseline
        del_loss = num_del_generated - num_del_baseline

        lines_loss = add_loss + del_loss
        
        return Loss(test_loss = test_loss, 
                    num_add_lines_loss=add_loss, 
                    num_deleted_lines_loss=del_loss, 
                    num_total_lines_loss=lines_loss)

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
                             problem_example = "\n".join([str(e) for e in problem.examples]),
                             note = problem.note,
                             submission_verdict = submission.verdict,
                             submission_code = submission.source_code)
        def f():
            return self.llm.generate(prompt)
        
        with RetryExecution("NaiveFixBugLLMMethod", 3, logger) as retry:
            result = retry(f)

        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_NaiveFixBugLLMMethod',
                                  method_name = "_NaiveFixBugLLMMethod",
                                  llm_result = result,
                                  source_code = CodeParser.extract_code(result),
                                  model_info = ModelInfo(vendor = self.llm.vendor, model_name = self.llm.model_name),
                                  loss=None)
 

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

        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_GenerateFromScratchLLMMethod',
                                  method_name = "GenerateFromScratchLLMMethod",
                                  llm_result = result,
                                  source_code = CodeParser.extract_code(result),
                                  model_info = ModelInfo(vendor = self.llm.vendor, model_name = self.llm.model_name),
                                  loss=None)

_available_methods = {"NaiveFixBugLLMMethod": NaiveFixBugLLMMethod,
                      "GenerateFromScratchLLMMethod": GenerateFromScratchLLMMethod}

def create_method(name:str, **kwargs):
    return _available_methods[name](**kwargs)
