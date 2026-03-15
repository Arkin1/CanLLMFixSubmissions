import dspy
from pydantic import Field
from typing import Optional

class GeneratedSolutionFromScratchSignature(dspy.Signature):
    """
    Generate a solution for the given competitive programming problem.
    """
    problem_id: int = dspy.InputField(description = "The id of the problem. It is used for reference and debugging purposes, but you should not rely on it to generate the solution.")
    problem_description:str = dspy.InputField(description = "Problem description.")
    input_format:str = dspy.InputField(description = "How the input should be formatted")
    output_format: str = dspy.InputField(description = "How the output should be formatted")
    examples: str = dspy.InputField(description = "Examples of input and output. Input is marked by #Input# and output by #Output# ") 
    note: Optional[str] = dspy.InputField(description = "An explanation for an input and an output or a remark.")
    submission_verdict: str = dspy.InputField(description = "The verdict of the buggy submission (Wrong Answer, Time Limit Exceeded etc.)")

    generated_code: str = dspy.OutputField(description = "The generated code. Please encapsulate it into ```cpp```")

class BugFixerSignature(dspy.Signature):
    """
    Fix the bug in the buggy code for the given competitive programming problem by adding, deleting or modifying as few lines as possible.
    In other words, you must adhere to the given buggy code and change it as little as possible to make it work. You can add, delete or modify lines of code, but you cannot rewrite the whole solution. 
    The more lines you change, the more points you lose. 
    You should try to find the bug and fix it, not to rewrite the whole solution.
    """
    problem_id: int = dspy.InputField(description = "The id of the problem. It is used for reference and debugging purposes, but you should not rely on it to generate the solution.")
    problem_description:str = dspy.InputField(description = "Problem description.")
    input_format:str = dspy.InputField(description = "How the input should be formatted")
    output_format: str = dspy.InputField(description = "How the output should be formatted")
    examples: str = dspy.InputField(description = "Examples of input and output. Input is marked by #Input# and output by #Output# ") 
    note: Optional[str] = dspy.InputField(description = "An explanation for an input and an output or a remark.")
    submission_verdict: str = dspy.InputField(description = "The verdict of the buggy submission (Wrong Answer, Time Limit Exceeded etc.)")
    buggy_code: str = dspy.InputField(description = "The buggy code")

    fixed_code: str = dspy.OutputField(description = "The entire fixed code. Please encapsulate it into ```cpp```")