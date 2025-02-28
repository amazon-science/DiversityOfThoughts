import random
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List

from gpt_usage import gpt_usage

def run_dot(
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
    
    primary_key = 'entry_point' #"task_id" if "task_id" in dataset[0].keys() else "name" #'entry_point' for HumanEval

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        
        try:
        
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            all_levels_reflections_scores = []
            all_levels_implementations = []
            cur_func_impl = None
            while cur_pass < pass_at_k and not is_solved:
                if is_leetcode:
                    tests_i = item['visible_tests']
                else:

                    if visible_tests:
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[item[primary_key]]['given_tests'] #'task_id'

                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(item["prompt"], model, 1)
                        
                # first attempt
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(item["prompt"], model, "simple", temperature=1.)
                
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(feedback)
                
                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=20)
                    is_solved = is_passing
                    num_success += int(is_passing)
                    print(is_solved, num_success)
                    break

                # conditional sampling on prior reflections to promote diversity
                cur_iter = 1
                cur_feedback = feedback
                while cur_iter < max_iters:
                    
                    # Iterative sampling
                    # get 3 self-reflection-diverse
                    # for i in range(3):
                    #     print(f"{i}th disc reflection")
                    #     reflection = gen.self_reflection_diverse(
                    #         cur_func_impl, cur_feedback, model, diverse_reflections)
                    #     diverse_reflections += [reflection]
                    # div_reflections = diverse_reflections.copy()
                    
                    # one-shot sampling
                    # get multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse_oneshot(
                        cur_func_impl, cur_feedback, model, diverse_reflections).split("\n\n")
                    
                    # filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    diverse_reflections += div_reflections

                    
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    
                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    
                    
                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 3):
                        
                        #re-init executor
                        del exe
                        exe = executor_factory(language, is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()
                        
                        # apply self-reflection in the next attempt
                        new_func_impl = None
                        while new_func_impl is None:
                            new_func_impl = gen.func_impl(
                                func_sig=item["prompt"],
                                model=model,
                                strategy="reflexion",
                                prev_func_impl=cur_func_impl_copy,
                                feedback=cur_feedback,
                                self_reflection=reflection,
                                temperature=1.0,
                                ref_chat_instruction='dot'
                            )
                        cur_func_impl = new_func_impl

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:
                            
                            print("regenerating func impl.")
                            continue

                        # Will be used later to sample a probable solution
                        temp_implementations.append(cur_func_impl)
                    
                        # check if all internal unit tests pass
                        is_passing, cur_feedback, _ = exe.execute(
                            cur_func_impl, tests_i)
                        test_feedback.append(cur_feedback)
                        div_reflections_feedbacks.append(cur_feedback)
                        
                        # measures total number of failed unit tests
                        reflections_scores.append((len(tests_i) - cur_feedback.split("Tests failed:")[1].count('assert')) + 1e-8)

                        # increment ref-id counter
                        ref_id += 1
                        pbar.update(1)

                        # if solved, check if it passes the real tests, exit early
                        if is_passing or cur_iter == max_iters - 1: 
                            is_passing = exe.evaluate(
                                item["entry_point"], cur_func_impl, item["test"], timeout=10)
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            break
                    
                    pbar.close()
                    
                    #log reflection scores and given level implementations
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)
                    
                    if is_solved:
                        break
                    
                    #sample likely implementation
                    print(reflections_scores)
                    sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                    cur_func_impl = temp_implementations[sampled_impl_idx]
                    
                    # set cur_feedback to the corresponding sampled div-reflection
                    cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    cur_iter += 1
                cur_pass += 1
            
        except Exception as e:
            print(colored(f"Error: {e}", 'red'))
            print('-----------------')

            
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        item["is_solved"] = is_solved
        item["diverse_reflections"] = diverse_reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item['all_levels_reflections_scores'] = all_levels_reflections_scores
        item['all_levels_implementations'] = all_levels_implementations
        item['cost'] = llm_cost['cost']
        item['completion_tokens'] = llm_cost['completion_tokens']
        item['prompt_tokens'] = llm_cost['prompt_tokens']
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
        
    print(colored(gpt_usage(backend=model_name), 'blue'))
