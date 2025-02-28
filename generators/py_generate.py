from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, \
                            generic_generate_internal_tests, \
                            generic_generate_self_reflection, \
                            generate_self_reflection_diverse, \
                            generate_self_reflection_diverse_oneshot

from typing import Optional, List, Union
import ast
import re
from .parse import parse_code_block, add_code_block

PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only python code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation (restate the function signature)."
PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(1, 2) == 4 # output: -1

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a + b
```
'''

PY_REFLEXION_FEW_SHOT = '''Example 1:
[previous impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert fullJustify([], 10) == [] # output: ['          ']
assert fullJustify([], 0) == [] # output: ['']

[reflection on previous impl]:
The implementation failed the test cases where the input list of words is empty. The issue arises because the code does not handle the case where there are no words to process. As a result, it still appends a line with spaces to the result list, even when there are no words. To fix this issue, we should add a condition at the beginning of the function to check if the input list is empty, and return an empty list if it is. This will ensure that the function returns the correct output for empty input lists.

[improved impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```
END EXAMPLES

'''
PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Python programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Python programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."
PY_SELF_REFLECTION_FEW_SHOT = """Example 1:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []  
Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]
[self-reflection]:
The implementation failed the where no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.

Example 2:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []
Tests failing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3] # output: list index out of range
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5] # output: list index out of range
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: list index out of range
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3] # output: list index out of range
[self-reflection]:
The implementation failed 4 out of the 7 test cases due to an IndexError. The issue stems from the while loop while current_sum + nums[right] <= target:, which directly accesses nums[right] without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to while right < len(nums) and current_sum + nums[right] <= target:. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError.
END OF EXAMPLES
"""

PY_TEST_GENERATION_FEW_SHOT = """Examples:
func signature:
def add3Numbers(x, y, z):
    \"\"\" Add three numbers together.
    This function takes three numbers as input and returns the sum of the three numbers.
    \"\"\"
unit tests:
assert add3Numbers(1, 2, 3) == 6
assert add3Numbers(-1, 2, 3) == 4
assert add3Numbers(1, -2, 3) == 2
assert add3Numbers(1, 2, -3) == 0
assert add3Numbers(-3, -2, -1) == -6
assert add3Numbers(0, 0, 0) == 0
"""

PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring.

{PY_TEST_GENERATION_FEW_SHOT}"""

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring."""


# ----------------------------------
# Added these prompts
# ----------------------------------

PY_TEST_EXTRACTION_CHAT_INSTRUCTION = \
"""You are an AI coding assistant that can extract unit tests from a given function signature and docstring. 
The docstring includes example use cases starting after the string "Examples:". 
Additionally, you can synthesize extra valid unit tests based on the information from the function signature and docstring."""

PY_EXTRACT_TEST_CASES_INSTRUCTION  = \
"""
Given a Python function signature along with a docstring that describes the 
function's purpose and provides example use cases starting after the string 
"Examples:", generate unit tests using assert statements based on the provided examples. 
Additionally, synthesize extra valid unit tests using the information from 
the function signature and the docstring.

**Input:**

```python
from typing import List

def minPath(grid: List[List[int]], k: int) -> List[int]:
    Given a grid with N rows and N columns (N >= 2) and a positive integer k, 
    each cell of the grid contains a value. Every integer in the range [1, N * N]
    inclusive appears exactly once on the cells of the grid.

    You have to find the minimum path of length k in the grid. You can start
    from any cell, and in each step you can move to any of the neighbor cells,
    in other words, you can go to cells which share an edge with you current
    cell.
    Please note that a path of length k means visiting exactly k cells (not
    necessarily distinct).
    You CANNOT go off the grid.
    A path A (of length k) is considered less than a path B (of length k) if
    after making the ordered lists of the values on the cells that A and B go
    through (let's call them lst_A and lst_B), lst_A is lexicographically less
    than lst_B, in other words, there exist an integer index i (1 <= i <= k)
    such that lst_A[i] < lst_B[i] and for any j (1 <= j < i) we have
    lst_A[j] = lst_B[j].
    It is guaranteed that the answer is unique.
    Return an ordered list of the values on the cells that the minimum path go through.

    Examples:    
    >>> minPath([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
    [1, 2, 1]

    >>> minPath([[5, 9, 3], [4, 1, 6], [7, 8, 2]], 1)
    [1]
```

**Output:**

```python
unit tests:
assert minPath([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3) == [1, 2, 1]
assert minPath([[5, 9, 3], [4, 1, 6], [7, 8, 2]], 1) == [1]
# Additional synthesized tests
assert minPath([[10, 11, 12], [13, 14, 15], [16, 17, 18]], 2) == [1, 10]
assert minPath([[21, 22, 23], [24, 25, 26], [27, 28, 29]], 4) == [1, 10, 1, 10]
assert minPath([[30, 31, 32], [33, 34, 35], [36, 37, 38]], 1) == [1]
```

**Instructions for the LLM:**

1. Extract the examples provided after the string "Examples:" in the docstring.
2. Convert each example into an assert statement for unit testing.
3. Synthesize additional valid unit tests based on the function signature and docstring.
4. Ensure the assert statements follow this format:
    - `assert function_name(arguments) == expected_output`
5. Use the constraints and descriptions in the docstring to create meaningful additional test cases.
6. The output should be a block of assert statements separated by newline characters.
7. Do not include any additional text or comments in the output.
"""

PY_DIVERSE_REFLECTION_CHAT_INSTRUCTION = \
"""You are a Python programming assistant. 
You will be given a function implementation, unit tests, and previously generated reflections. 
Write a terse and insightful explanation of why the implementation is incorrect based on the tests. 
Ensure your reflections are accurate and leverage previous reflections to avoid repetition. 
Aim for diversity in your explanations while prioritizing the correctness of your hints. 
Only provide the few sentence description in your answer, not the implementation.
"""

PY_DIVERSE_REFLECTION_ONESHOT_CHAT_INSTRUCTION = \
"""You are a Python programming assistant. You will be given a function implementation, unit tests, and previously generated reflections.
Write 3 unique and diverse reflections to fix the problem. Each reflection should follow this structure:

Problem: {a terse description of the identified problem}.
Fix: {proposed fix or hint to fix the identified problem}.

Ensure your reflections are accurate and leverage previous reflections to avoid repetition.
Aim for diversity in your explanations while prioritizing the correctness of your hints.
Add "\n\n" at the end of each proposed reflection.
"""

PY_DIVERSE_REFLECTION_FEW_SHOT = \
"""
Example 1:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []
Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]
[self-reflection]:
**Problem**: The implementation failed the test where no subarray fulfills the condition. The issue is due to the use of `>=` instead of `>` in the condition to update the result.  
**Fix**: Change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length by replacing `>=` with `>` in the condition.  

**Problem**: The current implementation does not handle the case where the sum of any subarray exceeds the target, leading to incorrect results.  
**Fix**: Add a condition to check if the sum of the subarray exceeds the target before updating the result.  

**Problem**: The algorithm might not correctly identify the longest subarray when multiple subarrays have the same length but different sums.  
**Fix**: Introduce an additional check to ensure that the subarray with the sum closest to the target is selected when lengths are equal.  

**Problem**: The code does not consider the possibility of an empty array or a target that is too small to be achieved by any subarray.  
**Fix**: Implement a preliminary check to return an empty array if the target is smaller than the smallest element in `nums` or if `nums` is empty.  

---

Example 2:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []
Tests failing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3] # output: list index out of range
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5] # output: list index out of range
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: list index out of range
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3] # output: list index out of range
[self-reflection]:
**Problem**: The implementation fails due to an IndexError caused by accessing `nums[right]` without bounds checking.  
**Fix**: Add a bounds check for the `right` variable in the `while current_sum + nums[right] <= target:` loop by modifying the condition to `while right < len(nums) and current_sum + nums[right] <= target:`.  

**Problem**: The initial while loop does not account for the scenario where the right index exceeds the length of the array.  
**Fix**: Introduce a boundary condition to ensure `right` remains within the array limits during the initial accumulation phase.  

**Problem**: The implementation incorrectly attempts to extend the subarray beyond the array's bounds, leading to a runtime error.  
**Fix**: Implement a check within the loop to ensure `right` does not exceed the length of `nums` before accessing `nums[right]`.  

**Problem**: The code does not correctly handle cases where the target sum is less than any single element in the array, resulting in a failure to return an empty array.  
**Fix**: Add a preliminary check to return an empty array if no subarray can possibly meet the target.  

END OF EXAMPLES
"""

PY_DoT_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. ALWAYS WRITE your full implementation (restate the function signature)."
# """ 
# You are an AI Python assistant. 
# You will be given your past function implementation, a series of unit tests, and a reflection identifying the problem and suggesting a potential fix. 
# Using the provided reflection, WRITE the full implementation of the function (restate the function signature). 
# Ensure the new implementation addresses the identified problem and incorporates the suggested fix.
# THE OUTPUT SHOULD ALWAYS CONTAIN GENERATED PYTHON CODE ALONG WITH TEST CASES AND THEIR OUTPUT BUT NO TEXT.
# """


class PyGenerator(Generator):

    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "python"),
            self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT
        )


    def self_reflection_diverse(self, func: str, feedback: str, model: ModelBase, diverse_reflections: list) -> str:
        return generate_self_reflection_diverse(
                func=func,
                feedback=feedback,
                model=model,
                self_reflection_chat_instruction=PY_DIVERSE_REFLECTION_CHAT_INSTRUCTION,
                add_code_block=lambda x: add_code_block(x, "python"),
                self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT,
                previous_reflections=diverse_reflections
        )

        
    def self_reflection_diverse_oneshot(self, func: str, feedback: str, model: ModelBase, diverse_reflections: list) -> str:
        return generate_self_reflection_diverse_oneshot(
                func=func,
                feedback=feedback,
                model=model,
                self_reflection_chat_instruction=PY_DIVERSE_REFLECTION_ONESHOT_CHAT_INSTRUCTION,
                add_code_block=lambda x: add_code_block(x, "python"),
                self_reflection_few_shot=PY_DIVERSE_REFLECTION_FEW_SHOT,
                previous_reflections=diverse_reflections
        )


    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.,
        ref_chat_instruction=None,
    ) -> Union[str, List[str]]:
        
        if ref_chat_instruction == 'dot':
            ref_chat_instruction = PY_DoT_CHAT_INSTRUCTION
        else:
            reflexion_chat_instruction = PY_REFLEXION_CHAT_INSTRUCTION
        
        return generic_generate_func_impl(
            func_sig=func_sig,
            model=model,
            strategy=strategy,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            reflexion_chat_instruction=ref_chat_instruction,
            reflexion_few_shot=PY_REFLEXION_FEW_SHOT_ADD,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "python"),
            add_code_block=lambda x: add_code_block(x, "python"),
        )
        

    def internal_tests(self, func_sig: str, model: ModelBase, max_num_tests: int = 5) -> List[str]:
        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if "assert" in test]
        """
        Generates tests for a function.
        """
        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            
            test_generation_few_shot=PY_TEST_GENERATION_FEW_SHOT,
            test_generation_chat_instruction=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            
            # extract tests cases from doc strings and synthesize additional
            # test_generation_few_shot=PY_EXTRACT_TEST_CASES_INSTRUCTION,
            # test_generation_chat_instruction=PY_TEST_EXTRACTION_CHAT_INSTRUCTION,
            
            test_generation_completion_instruction=PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
            parse_tests=parse_tests,
            is_syntax_valid=py_is_syntax_valid,
        )


DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
