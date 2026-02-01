import dspy
from pydantic import Field
from typing import Optional

class BugFixerSignature(dspy.Signature):
    """
    Expert competitive programmer fixing bugs. 
    Stick to original code, provide a single block, no comments. 
    Don't regenerate the entire solution from scratch, because you are not fixing a bug in this way
    """
    problem_id:str = dspy.InputField(problem_id = "The id of the problem")
    problem_description:str = dspy.InputField(description = "Problem description.")
    input_format:str = dspy.InputField(description = "How the input should be formatted")
    output_format: str = dspy.InputField(description = "How the output should be formatted")
    examples: str = dspy.InputField(description = "Examples of input and output. Input is marked by #Input# and output by #Output# ") 
    note: Optional[str] = dspy.InputField(description = "An explanation for an input and an output or a remark.")
    submission_verdict: str = dspy.InputField(description = "The verdict of the buggy submission (Wrong Answer, TLE etc.)")
    buggy_code: str = dspy.InputField(description = "The buggy code")

    fixed_code: str = dspy.OutputField(description = "The fixed code. Please encapsulate it into ```cpp```")