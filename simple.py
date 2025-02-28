from pprint import pprint
from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List

from gpt_usage import gpt_usage

SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."

def run_simple(
        dataset: List[dict],
        model_name: str,
        language: str,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False
    ) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)
    
    
    failed_probs = []
    num_items = len(dataset)
    num_success = 0
    for i, item in enumerate_resume(dataset, log_path):
        try:
            cur_pass = 0
            is_solved = False
            cur_func_impl = ""
            while cur_pass < pass_at_k:
                cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
                
                assert isinstance(cur_func_impl, str)
                is_passing = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout = 20 if is_leetcode else 10)
                if is_passing:
                    is_solved = True
                    num_success += 1
                    break
                cur_pass += 1
            item["solution"] = cur_func_impl
            
            llm_cost = gpt_usage(backend=model_name)
            print(llm_cost)
            
            item["is_solved"] = is_solved
            item['cost'] = llm_cost['cost']
            item['completion_tokens'] = llm_cost['completion_tokens']
            item['prompt_tokens'] = llm_cost['prompt_tokens']
            write_jsonl(log_path, [item], append=True)
            
        except:
            failed_probs.append(item['task_id'])
            continue

        print("Failed problems:")
        pprint(failed_probs)
        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
