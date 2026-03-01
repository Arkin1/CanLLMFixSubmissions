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
from typing import Optional

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

        add_loss = num_add_generated - num_add_baseline
        del_loss = num_del_generated - num_del_baseline

        lines_loss = add_loss + del_loss

        mx_value = (len(correct_code.splitlines()) * 2 - num_add_baseline - num_del_baseline)
        
        if (abs(test_loss - 1) < 1e-4):
            total_loss = lines_loss
        else:
            total_loss = mx_value
        
        if total_loss < 0 : # Generated solution is equal or closer to the correct solution than the buggy solution.
            total_normalized_loss = 1.0
        else: 
            total_normalized_loss = (mx_value - total_loss) / mx_value

            if total_normalized_loss < 0: #There are more lines added or deleted than the number of lines in the correct code.
                total_normalized_loss = 0

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
        dspy.configure(lm=self.llm)
        self.cot = dspy.ChainOfThought("prompt->fixed_code")

        self.prompt_name = 'prompt_naive_fix_bug'

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        problem = self.problems_manager.get_info_problem(submission.problem_id)
        prompt = read_prompt(self.prompt_name,
                             problem_description = problem.description,
                             problem_input_format = problem.input_format,
                             problem_output_format = problem.output_format,
                             problem_example = "\n".join([str(e) for e in problem.examples]),
                             note = problem.note or "None",
                             submission_verdict = submission.verdict,
                             submission_code = submission.source_code)
        def f():
            return self.cot(prompt = prompt)
        
        with dspy.settings.context(llm = self.llm, track_usage=True):
            with RetryExecution("NaiveFixBugLLMMethod", 3, logger) as retry:
                result = retry(f)

        lm_usage = list(result.get_lm_usage().values())[0]
        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_NaiveFixBugLLMMethod',
                                  method_name = "NaiveFixBugLLMMethod",
                                  llm_result = result.fixed_code,
                                  source_code = CodeParser.extract_code(result.fixed_code),
                                  model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                  loss=None,
                                  prompt_tokens = lm_usage['prompt_tokens'],
                                  completion_tokens=lm_usage['completion_tokens'],
                                  total_tokens = lm_usage['total_tokens'])
 

class GenerateFromScratchLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager = None, **model_kwargs):
        super().__init__(vendor, model_name, problems_manager)
        self.llm = get_llm_api(vendor, model_name, **model_kwargs)
        dspy.configure(lm=self.llm)
        self.cot = dspy.ChainOfThought("prompt->generated_code")
        self.prompt_name = 'prompt_generate_solution'

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        problem = self.problems_manager.get_info_problem(submission.problem_id)
        prompt = read_prompt(self.prompt_name,
                             problem_description = problem.description,
                             problem_input_format = problem.input_format,
                             problem_output_format = problem.output_format,
                             problem_example = str(problem.examples),
                             note = problem.note or "None",
                             submission_verdict = submission.verdict)
        def f():
            return self.cot(prompt = prompt)
        
        
        with dspy.settings.context(llm = self.llm, track_usage=True):
            with RetryExecution("GenerateFromScratchLLMMethod", 3, logger) as retry:
                result = retry(f)
                
        lm_usage = list(result.get_lm_usage().values())[0]
        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_GenerateFromScratchLLMMethod',
                                  method_name = "GenerateFromScratchLLMMethod",
                                  llm_result = result.generated_code,
                                  source_code = CodeParser.extract_code(result.generated_code),
                                  model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                  loss=None,
                                  prompt_tokens = lm_usage['prompt_tokens'],
                                  completion_tokens=lm_usage['completion_tokens'],
                                  total_tokens = lm_usage['total_tokens'])
                                  

class DSPyOptimizedLLMMethod(LLMMethod):
        def __init__(self, 
                     vendor:str, 
                     model_name:str, 
                     problems_manager:ProblemsManager = None, 
                     model_path: str = None,
                     model_output_path: str = None,
                     **model_kwargs):
            super().__init__(vendor, model_name, problems_manager)
            self.llm = get_llm_api(vendor, model_name, **model_kwargs)
            dspy.configure(lm=self.llm)
            self.cot = dspy.ChainOfThought(BugFixerSignature)
            self.model_path = model_path
            self.model_output_path = model_output_path
            if self.model_path:
                self.cot.load(self.model_path)
            self.problems_manger = problems_manager

        def fit(self, train_submissions: list[Submission], val_submissions: list[Submission]):
            if not self.model_output_path:
                raise ValueError("Model Output Path is not set")
            
            def metric(gold: dspy.Example,
                       pred: dspy.Prediction,
                       trace = None,
                       pred_name: Optional[str] = None,
                       pred_trace = None):
                try:
                    # Check if we are already in an environment with a running loop
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                loss_co = self.compute_loss(problem_id = gold.problem_id, 
                            buggy_code = gold.buggy_code, 
                            correct_code = gold.fixed_code, 
                            generated_code = CodeParser.extract_code(pred.fixed_code))
                
                if running_loop:
                    loss = running_loop.run_until_complete(loss_co)
                else:
                    loss = asyncio.run(loss_co)

                feedback_text = ""
                if loss.test_loss < 1:
                    feedback_text = f"The proposed code fails on {(1 - loss.test_loss)*100}% of tests. The fix is not solving the bug."
                else:
                    if loss.total_normalized_loss < 1:
                        feedback_text = f"All the tests pass, but you are not sticking to the original code. You should fix the bug by modifying as few code as possible."
                    else:
                        feedback_text = f"You generated a good bug fix!"
                
                return dspy.Prediction(score = loss.total_normalized_loss, feedback = feedback_text)
                                         
            
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

            
            guesser = dspy.GEPA(metric = metric, 
                                auto = 'light',
                                reflection_lm = self.llm,
                                track_stats = True)
            optimized_program = guesser.compile(self.cot, trainset=train_dataset, valset=val_dataset)
            optimized_program.save(self.model_output_path)

        
        def predict(self, submission: Submission) -> GeneratedLLMResult:
            problem = self.problems_manager.get_info_problem(submission.problem_id)
            def f():
                prediction = self.cot(
                     problem_id = problem.problem_id, 
                     problem_description = problem.description,
                     input_format = problem.input_format,
                     output_format = problem.output_format,
                     examples = str(problem.examples),
                     note = problem.note or "None",
                     submission_verdict = submission.verdict,
                     buggy_code = submission.source_code).with_inputs(
                            "problem_description", "input_format", "output_format", 
                            "examples", "note", "submission_verdict", "buggy_code")
                
                return prediction
            
            with dspy.settings.context(llm = self.llm, track_usage=True):
                with RetryExecution("DSPyOptimizedLLMMethod", 3, logger) as retry:
                    result = retry(f)

            lm_usage = list(result.get_lm_usage().values())[0]

            return GeneratedLLMResult(generated_result_id = submission.submission_id + 'DSPyOptimizedLLMMethod',
                                    method_name = "DSPyOptimizedLLMMethod",
                                    llm_result = result.fixed_code,
                                    source_code = CodeParser.extract_code(result.fixed_code),
                                    model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                    loss=None,
                                    prompt_tokens = lm_usage['prompt_tokens'],
                                    completion_tokens=lm_usage['completion_tokens'],
                                    total_tokens = lm_usage['total_tokens'])


_available_methods = {"NaiveFixBugLLMMethod": NaiveFixBugLLMMethod,
                      "GenerateFromScratchLLMMethod": GenerateFromScratchLLMMethod,
                      "DSPyOptimizedLLMMethod": DSPyOptimizedLLMMethod}

def create_method(name:str, problems_manager: ProblemsManager = None, **kwargs):
    return _available_methods[name](problems_manager = problems_manager, **kwargs)

