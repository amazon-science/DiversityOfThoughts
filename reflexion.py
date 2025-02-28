from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List
from gpt_usage import gpt_usage

def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    visible_tests: any = None,
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    
    print("Running Reflexion")
    
    for i, item in enumerate_resume(dataset, log_path):
        try:
            cur_pass = 0
            is_solved = False
            reflections = []
            implementations = []
            test_feedback = []
            # cur_func_impl = ""
            cur_func_impl = None
            while cur_pass < pass_at_k and not is_solved:
                if is_leetcode:
                    tests_i = item['visible_tests']
                else:
                    if visible_tests:
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[item['entry_point']]['given_tests']

                    elif 'mbpp' in log_path.lower():
                        print("using visible test cases")
                        tests_i = item['visible_tests']

                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(item["prompt"], model, 1)
                        
                        # Use original test cases
                        # print("using original test cases")
                        # tests_i = [case.lstrip().replace('candidate', item['entry_point']) for case in item['test'].split('\n')[1:-1] if 'assert' in case]

                # first attempt
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(feedback)

                # if solved, exit early
                if is_passing:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    is_solved = is_passing
                    num_success += int(is_passing)
                    break

                # use self-reflection to iteratively improve
                cur_iter = 1
                cur_feedback = feedback
                while cur_iter < max_iters:
                    # get self-reflection
                    reflection = gen.self_reflection(
                        cur_func_impl, cur_feedback, model)
                    reflections += [reflection]

                    # apply self-reflection in the next attempt
                    if isinstance(model, tuple):
                        model = model[0]
                        
                    new_func_impl = None
                    while new_func_impl is None:
                        new_func_impl = gen.func_impl(
                            func_sig=item["prompt"],
                            model=model,
                            strategy="reflexion",
                            prev_func_impl=cur_func_impl,
                            feedback=cur_feedback,
                            self_reflection=reflection,
                        )
                    cur_func_impl = new_func_impl
                    
                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)
                    
                    # check if all internal unit tests pass
                    is_passing, cur_feedback, _ = exe.execute(
                        cur_func_impl, tests_i)
                    test_feedback.append(cur_feedback)

                    # if solved, check if it passes the real tests, exit early
                    if is_passing or cur_iter == max_iters - 1:
                        is_passing = exe.evaluate(
                            item["entry_point"], cur_func_impl, item["test"], timeout=10)
                        if is_passing:
                            item["solution"] = cur_func_impl
                            is_solved = True
                            num_success += 1
                        break

                    cur_iter += 1
                cur_pass += 1
                
        except:
            continue
        
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item['cost'] = llm_cost['cost']
        item['completion_tokens'] = llm_cost['completion_tokens']
        item['prompt_tokens'] = llm_cost['prompt_tokens']
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')

    print(colored(gpt_usage(backend=model_name), 'blue'))