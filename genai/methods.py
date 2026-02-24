from genai.endpoints import get_llm_api
from genai.utils import read_prompt, CodeParser
from data_models import Problem, Submission, GeneratedLLMResult, ModelInfo, Loss
from utils import compile_and_test, count_differences, ProblemsManager
from utils import RetryExecution
import logging
import dspy
from dspy_modules import BugFixerSignature
import asyncio
from utils import preprocess_line

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger()

class LLMMethod():
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager = None):
        self.model_name = model_name
        self.vendor = vendor
        self.problems_manager = problems_manager
    
    def fit(self, train_submissions: list[Submission], val_submissions: list[Submission]):
        raise NotImplementedError()

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        pass
    
    async def compute_loss(self, problem_id:str, buggy_code:str, correct_code:str, generated_code:str):
        test_loss = await self.problems_manager.evaluate_submission(problem_id, generated_code, "GNU C++", 'soft')
        num_add_baseline, num_del_baseline = count_differences(buggy_code, correct_code)
        num_add_generated, num_del_generated = count_differences(generated_code, correct_code)

        add_loss = abs(num_add_generated - num_add_baseline)
        del_loss = abs(num_del_generated - num_del_baseline)

        lines_loss = add_loss + del_loss

        mx_value = len(correct_code.splitlines()) * 2
        
        if (abs(test_loss - 1) < 1e-4):
            total_loss = lines_loss
        else:
            total_loss = mx_value
        
        total_normalized_loss = (mx_value - total_loss) / mx_value

        return Loss(test_loss = test_loss, 
                    num_add_lines_loss=add_loss, 
                    num_deleted_lines_loss=del_loss, 
                    num_total_lines_loss=lines_loss,
                    total_loss = total_loss,
                    total_normalized_loss = total_normalized_loss)


class NaiveFixBugLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager = None, **model_kwargs):
        super().__init__(vendor, model_name, problems_manager)
        self.llm = get_llm_api(vendor, model_name, **model_kwargs)
        self.prompt_name = 'prompt_naive_fix_bug'

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        problem = self.problems_manager.get_info_problem(submission.problem_id)
        prompt = read_prompt(self.prompt_name,
                             problem_description = problem.description,
                             problem_input_format = problem.input_format,
                             problem_output_format = problem.output_format,
                             problem_example = "\n".join([str(e) for e in problem.examples]),
                             note = problem.note,
                             submission_verdict = submission.verdict,
                             submission_code = submission.source_code)
        def f():
            return self.llm(prompt)[0]
        
        with RetryExecution("NaiveFixBugLLMMethod", 3, logger) as retry:
            result = retry(f)

        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_NaiveFixBugLLMMethod',
                                  method_name = "NaiveFixBugLLMMethod",
                                  llm_result = result,
                                  source_code = CodeParser.extract_code(result),
                                  model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                  loss=None)
 

class GenerateFromScratchLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager = None, **model_kwargs):
        super().__init__(vendor, model_name, problems_manager)
        self.llm = get_llm_api(vendor, model_name, **model_kwargs)
        self.prompt_name = 'prompt_generate_solution'

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        problem = self.problems_manager.get_info_problem(submission.problem_id)
        prompt = read_prompt(self.prompt_name,
                             problem_description = problem.description,
                             problem_input_format = problem.input_format,
                             problem_output_format = problem.output_format,
                             problem_example = str(problem.examples),
                             note = problem.note,
                             submission_verdict = submission.verdict)
        def f():
            return self.llm(prompt)[0]
        
        with RetryExecution("GenerateFromScratchLLMMethod", 3, logger) as retry:
            result = retry(f)

        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_GenerateFromScratchLLMMethod',
                                  method_name = "GenerateFromScratchLLMMethod",
                                  llm_result = result,
                                  source_code = CodeParser.extract_code(result),
                                  model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                  loss=None)
                                  

class DSPyOptimizedLLMMethod(LLMMethod):
        def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager = None, **model_kwargs):
            super().__init__(vendor, model_name, problems_manager)
            self.llm = get_llm_api(vendor, model_name, **model_kwargs)
            dspy.configure(lm=self.llm)
            self.cot = dspy.ChainOfThought(BugFixerSignature)
            self.cot.load("optimized_qa.json")
            self.problems_manger = problems_manager

        def fit(self, train_submissions: list[Submission], val_submissions: list[Submission]):
            def metric(example, pred, trace=None):
                try:
                    # Check if we are already in an environment with a running loop
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                loss_co = self.compute_loss(problem_id = example.problem_id, 
                            buggy_code = example.buggy_code, 
                            correct_code = example.fixed_code, 
                            generated_code = CodeParser.extract_code(pred.fixed_code))
                
                if running_loop:
                    loss = running_loop.run_until_complete(loss_co)
                else:
                    loss = asyncio.run(loss_co)
                return loss.total_normalized_loss
            
            train_dataset = []
            val_dataset = []
            for submission in train_submissions:
                problem = self.problems_manager.get_info_problem(submission.problem_id)
                train_dataset.append(dspy.Example(
                            problem_id = problem.problem_id,
                            problem_description=problem.description,
                            input_format=problem.input_format,
                            output_format=problem.output_format,
                            examples=str(problem.examples),
                            note=problem.note or "None",
                            submission_verdict=submission.verdict,
                            buggy_code=submission.source_code,
                            fixed_code=submission.anchor.source_code).with_inputs(
                            "problem_description", "input_format", "output_format", 
                            "examples", "note", "submission_verdict", "buggy_code"
                        ))
                
            for submission in val_submissions:
                problem = self.problems_manager.get_info_problem(submission.problem_id)
                val_dataset.append(dspy.Example(
                            problem_id = problem.problem_id,
                            problem_description=problem.description,
                            input_format=problem.input_format,
                            output_format=problem.output_format,
                            examples=str(problem.examples),
                            note=problem.note or "None",
                            submission_verdict=submission.verdict,
                            buggy_code=submission.source_code,
                            fixed_code=submission.anchor.source_code).with_inputs(
                            "problem_description", "input_format", "output_format", 
                            "examples", "note", "submission_verdict", "buggy_code"
                        ))

            
            guesser = dspy.MIPROv2(metric = metric, auto = 'light')
            optimized_program = guesser.compile(self.cot, trainset=train_dataset, valset=val_dataset)
            optimized_program.save("optimized_qa.json")
            optimized_program = optimized_program


        
        def predict(self, submission: Submission) -> GeneratedLLMResult:
            problem = self.problems_manager.get_info_problem(submission.problem_id)
            def f():
                prediction = self.cot(problem_id = problem.problem_id, 
                     problem_description = problem.description,
                     input_format = problem.input_format,
                     output_format = problem.output_format,
                     examples = str(problem.examples),
                     note = problem.note,
                     submission_verdict = submission.verdict,
                     buggy_code = submission.source_code)
                
                return prediction.fixed_code
            
            with RetryExecution("DSPyOptimizedLLMMethod", 3, logger) as retry:
                result = retry(f)

            return GeneratedLLMResult(generated_result_id = submission.submission_id + 'DSPyOptimizedLLMMethod',
                                    method_name = "DSPyOptimizedLLMMethod",
                                    llm_result = result,
                                    source_code = CodeParser.extract_code(result),
                                    model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                    loss=None)


_available_methods = {"NaiveFixBugLLMMethod": NaiveFixBugLLMMethod,
                      "GenerateFromScratchLLMMethod": GenerateFromScratchLLMMethod,
                      "DSPyOptimizedLLMMethod": DSPyOptimizedLLMMethod}

def create_method(name:str, problems_manager: ProblemsManager = None, **kwargs):
    return _available_methods[name](problems_manager = problems_manager, **kwargs)

