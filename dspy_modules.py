import dspy
from pydantic import Field
from typing import Optional

class GeneratedSolutionFromScratchSignature(dspy.Signature):
    """
    You are an expert competitive programmer and you must help your peers in fixing bugs. You will receive a competitive programming problem with a description, example inputs / outputs. Please generate me the correct solution for the code.
    I REALLY EMPHASIS THAT I WANT YOU TO GIVE ME A SINGLE PIECE OF CODE THAT CONTAINS THE ENTIRE SOLUTION. DON'T SPLIT IT. DON'T USE COMMENTS. Please encapsulate the code with ```cpp ```
    """
    problem_description:str = dspy.InputField(description = "Problem description.")
    input_format:str = dspy.InputField(description = "How the input should be formatted")
    output_format: str = dspy.InputField(description = "How the output should be formatted")
    examples: str = dspy.InputField(description = "Examples of input and output. Input is marked by #Input# and output by #Output# ") 
    note: Optional[str] = dspy.InputField(description = "An explanation for an input and an output or a remark.")
    submission_verdict: str = dspy.InputField(description = "The verdict of the buggy submission (Wrong Answer, Time Limit Exceeded etc.)")

    generated_code: str = dspy.OutputField(description = "The generated code. Please encapsulate it into ```cpp```")

class BugFixerSignature(dspy.Signature):
    """
    You are an expert competitive programmer and you must help your peers in fixing bugs. You will receive a competitive programming problem with a description, example inputs / outputs and a submission with a verdict. Your job is to identify the problem in the submission and fix the bug. 
    Note that you must stick to the original submission as much as possible, in other words, you should identify the minimum lines to add or to delete to fix the bug. You should provide me the entire fixed code.
    I REALLY EMPHASIS THAT I WANT YOU TO GIVE ME A SINGLE PIECE OF CODE THAT CONTAINS THE ENTIRE SOLUTION. DON'T SPLIT IT. DON'T USE COMMENTS. Please encapsulate the code with ```cpp ```
    """
    problem_description:str = dspy.InputField(description = "Problem description.")
    input_format:str = dspy.InputField(description = "How the input should be formatted")
    output_format: str = dspy.InputField(description = "How the output should be formatted")
    examples: str = dspy.InputField(description = "Examples of input and output. Input is marked by #Input# and output by #Output# ") 
    note: Optional[str] = dspy.InputField(description = "An explanation for an input and an output or a remark.")
    submission_verdict: str = dspy.InputField(description = "The verdict of the buggy submission (Wrong Answer, Time Limit Exceeded etc.)")
    buggy_code: str = dspy.InputField(description = "The buggy code")

    fixed_code: str = dspy.OutputField(description = "The fixed code. Please encapsulate it into ```cpp```")