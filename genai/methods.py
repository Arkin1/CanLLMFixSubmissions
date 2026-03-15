from genai.endpoints import get_llm_api
from genai.utils import CodeParser
from data_models import Problem, Submission, GeneratedLLMResult, ModelInfo, Reward,RewardContext
from utils import compile_and_test, count_differences, ProblemsManager
from utils import RetryExecution
import logging
import dspy
from dspy_modules import BugFixerSignature, GeneratedSolutionFromScratchSignature
import asyncio
from utils import clean_source_code
from typing import Optional

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger()

class LLMMethod():
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager):
        self.model_name = model_name
        self.vendor = vendor
        self.problems_manager = problems_manager
    
    def fit(self, train_submissions: list[Submission], val_submissions: list[Submission]):
        raise NotImplementedError()

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        pass
    
    async def compute_reward(self, problem_id:str, buggy_code:str, correct_code:str, generated_code:str):
        generated_code = clean_source_code(generated_code)
        test_reward = await self.problems_manager.evaluate_submission(problem_id, generated_code, "GNU C++", 'soft')

        num_lines_buggy_code = len(buggy_code.splitlines())
        num_lines_correct_code = len(correct_code.splitlines())
        num_lines_generated_code = len(generated_code.splitlines())

        num_add_baseline, num_del_baseline = count_differences(buggy_code, correct_code)
        num_common_baseline = num_lines_buggy_code - num_del_baseline

        num_add_generated, num_del_generated = count_differences(buggy_code, generated_code)
        num_common_generated = num_lines_buggy_code - num_del_generated

        jaccard_similarity_baseline = num_common_baseline / (num_common_baseline + num_add_baseline + num_del_baseline)
        jaccard_similarity_generated = num_common_generated / (num_common_generated + num_add_generated + num_del_generated)

        if jaccard_similarity_baseline == 0:
            raise Exception("Baseline similarity is 0, cannot compute reward")
        
        similarity_reward = min(1, jaccard_similarity_generated /  jaccard_similarity_baseline)

        context = RewardContext(num_lines_buggy_solution = num_lines_buggy_code,
                                num_lines_correct_solution = num_lines_correct_code,
                                num_lines_generated_solution = num_lines_generated_code,
                                num_added_lines_baseline = num_add_baseline,
                                num_deleted_lines_baseline = num_del_baseline,
                                num_added_lines_generated = num_add_generated,
                                num_deleted_lines_generated = num_del_generated,
                                num_common_lines_baseline = num_common_baseline,
                                num_common_lines_generated = num_common_generated,
                                test_pass = test_reward,
                                similarity_baseline = jaccard_similarity_baseline,
                                similarity_generated = jaccard_similarity_generated)
        
        return Reward(context = context,
                      test_reward = float((test_reward  > 0.99)), 
                      similarity_reward = similarity_reward,
                      total_reward = (test_reward  > 0.99) * similarity_reward)

class NaiveFixBugLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager, **model_kwargs):
        super().__init__(vendor, model_name, problems_manager)
        self.llm = get_llm_api(vendor, model_name, **model_kwargs)
        dspy.configure(lm=self.llm)
        self.model = dspy.Predict(BugFixerSignature)

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        problem = self.problems_manager.get_info_problem(submission.problem_id)
        def f():
            prediction = self.model(
                     problem_description = problem.description,
                     input_format = problem.input_format,
                     output_format = problem.output_format,
                     examples = str(problem.examples),
                     note = problem.note or "None",
                     submission_verdict = submission.verdict,
                     buggy_code = submission.source_code)
                
            return prediction
        
        with dspy.settings.context(lm = self.llm, track_usage=True):
            with RetryExecution("NaiveFixBugLLMMethod", 3, logger) as retry:
                result = retry(f)

        lm_usage = list(result.get_lm_usage().values())[0]
        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_NaiveFixBugLLMMethod',
                                  method_name = "NaiveFixBugLLMMethod",
                                  llm_result = result.fixed_code,
                                  source_code = CodeParser.extract_code(result.fixed_code),
                                  model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                  reward=None,
                                  prompt_tokens = lm_usage['prompt_tokens'],
                                  completion_tokens=lm_usage['completion_tokens'],
                                  total_tokens = lm_usage['total_tokens'])
 

class GenerateFromScratchLLMMethod(LLMMethod):
    def __init__(self, vendor:str, model_name:str, problems_manager:ProblemsManager, **model_kwargs):
        super().__init__(vendor, model_name, problems_manager)
        self.llm = get_llm_api(vendor, model_name, **model_kwargs)
        dspy.configure(lm=self.llm)
        self.model = dspy.Predict(GeneratedSolutionFromScratchSignature)

    def predict(self, submission: Submission) -> GeneratedLLMResult:
        problem = self.problems_manager.get_info_problem(submission.problem_id)

        def f():
            prediction = self.model(
                     problem_description = problem.description,
                     input_format = problem.input_format,
                     output_format = problem.output_format,
                     examples = str(problem.examples),
                     note = problem.note or "None",
                     submission_verdict = submission.verdict)
            return prediction
        
        
        with dspy.settings.context(lm = self.llm, track_usage=True):
            with RetryExecution("GenerateFromScratchLLMMethod", 3, logger) as retry:
                result = retry(f)
                
        lm_usage = list(result.get_lm_usage().values())[0]
        return GeneratedLLMResult(generated_result_id = submission.submission_id + '_GenerateFromScratchLLMMethod',
                                  method_name = "GenerateFromScratchLLMMethod",
                                  llm_result = result.generated_code,
                                  source_code = CodeParser.extract_code(result.generated_code),
                                  model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                  reward=None,
                                  prompt_tokens = lm_usage['prompt_tokens'],
                                  completion_tokens=lm_usage['completion_tokens'],
                                  total_tokens = lm_usage['total_tokens'])
                                  

class DSPyOptimizedLLMMethod(LLMMethod):
        def __init__(self, 
                     vendor:str, 
                     model_name:str, 
                     problems_manager:ProblemsManager, 
                     reflection_model_name:str = None, 
                     model_path: str = None,
                     seed:int = 0,
                     model_output_path: str = None,
                     **model_kwargs):
            super().__init__(vendor, model_name, problems_manager)
            self.llm = get_llm_api(vendor, model_name, **model_kwargs)
            if reflection_model_name:
                self.reflection_model = get_llm_api(vendor, reflection_model_name, **model_kwargs)
            dspy.configure(lm=self.llm)
            self.model = dspy.Predict(BugFixerSignature)
            self.model_path = model_path
            self.model_output_path = model_output_path
            if self.model_path:
                self.model.load(self.model_path)
            
            self.seed = seed
            self.problems_manager = problems_manager

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

                reward_co = self.compute_reward(problem_id = gold.problem_id, 
                            buggy_code = gold.buggy_code, 
                            correct_code = gold.fixed_code, 
                            generated_code = CodeParser.extract_code(pred.fixed_code))
                
                if running_loop:
                    reward = running_loop.run_until_complete(reward_co)
                else:
                    reward = asyncio.run(reward_co)

                feedback_text = ""
                if reward.test_reward < 1:
                    feedback_text = f"Your score is 0. The proposed code fails on {(1 - reward.context.test_pass)*100}% of tests. The fix is not solving the bug."
                else:
                    if reward.total_reward < 1:
                        feedback_text = f"Your score is {reward.total_reward}. All the tests pass, but you are not sticking to the original code. You should fix the bug by modifying as few code lines as possible."
                    else:
                        feedback_text = f"You generated a good bug fix!"
                
                return dspy.Prediction(score = reward.total_reward, feedback = feedback_text)
                                         
            
            train_dataset = []
            val_dataset = []
            for submission in train_submissions:
                problem = self.problems_manager.get_info_problem(submission.problem_id)
                train_dataset.append(dspy.Example(
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
                                reflection_lm = self.reflection_model,
                                track_stats = True,
                                seed = self.seed)
            optimized_program = guesser.compile(self.model, trainset=train_dataset, valset=val_dataset)
            optimized_program.save(self.model_output_path)

        
        def predict(self, submission: Submission) -> GeneratedLLMResult:
            problem = self.problems_manager.get_info_problem(submission.problem_id)
            def f():
                prediction = self.model(
                     problem_description = problem.description,
                     input_format = problem.input_format,
                     output_format = problem.output_format,
                     examples = str(problem.examples),
                     note = problem.note or "None",
                     submission_verdict = submission.verdict,
                     buggy_code = submission.source_code)
                
                return prediction
            
            with dspy.settings.context(llm = self.llm, track_usage=True):
                with RetryExecution("DSPyOptimizedLLMMethod", 3, logger) as retry:
                    result = retry(f)

            lm_usage = list(result.get_lm_usage().values())[0]

            return GeneratedLLMResult(generated_result_id = submission.submission_id + '_DSPyOptimizedLLMMethod',
                                    method_name = "DSPyOptimizedLLMMethod",
                                    llm_result = result.fixed_code,
                                    source_code = CodeParser.extract_code(result.fixed_code),
                                    model_info = ModelInfo(vendor = self.vendor, model_name = self.model_name),
                                    reward=None,
                                    prompt_tokens = lm_usage['prompt_tokens'],
                                    completion_tokens=lm_usage['completion_tokens'],
                                    total_tokens = lm_usage['total_tokens'])


_available_methods = {"NaiveFixBugLLMMethod": NaiveFixBugLLMMethod,
                      "GenerateFromScratchLLMMethod": GenerateFromScratchLLMMethod,
                      "DSPyOptimizedLLMMethod": DSPyOptimizedLLMMethod}

def create_method(name:str, problems_manager: ProblemsManager = None, **kwargs):
    return _available_methods[name](problems_manager = problems_manager, **kwargs)

