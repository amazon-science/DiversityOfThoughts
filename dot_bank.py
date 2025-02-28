import random
import pickle as pkl
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored

from utils import enumerate_resume_dotbank, \
                  make_printv, \
                  write_jsonl, \
                  resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List

from gpt_usage import gpt_usage

#memory bank imports
from scipy.spatial import distance
from memory_utils import get_cohere_embedding, \
                         get_openai_embedding, \
                         get_top_k_closest, \
                         get_random_k_indices
import generators.py_generate as py_generate
import generators.generator_utils as gen_utils
from generators.parse import parse_code_block, add_code_block
import os

def run_dot_bank(
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
    
    print("Running DoT-Bank")
    
    #init memory bank related file paths
    root_path = '/'.join(log_path.split('/')[:-1])
    mem_bank_file_path = root_path + '/mem_bank.pkl'
    failed_probs_path = root_path + '/failed_probs.pkl'
    
    # check if memory-bank already exists
    if os.path.exists(mem_bank_file_path):
        with open(mem_bank_file_path, 'rb') as f:
            memory_bank = pkl.load(f)
    else:
        # initialize memory bank
        memory_bank = {
            "positive_trajectories": [],
            "negative_trajectories": [],
        }
        
    if os.path.exists(failed_probs_path):
        with open(failed_probs_path, 'rb') as f:
            failed_problems = pkl.load(f)
    else:
        # store all problems that failed visible/synthetic tests in the first pass
        failed_problems = []
    
    primary_key = "task_id" if "task_id" in dataset[0].keys() else "name" #'entry_point'
    
    # First Pass
    for i, item in enumerate_resume_dotbank(dataset, log_path):
        
        cur_pass = 0
        is_solved = False
        diverse_reflections = []
        implementations = []
        test_feedback = []
        all_levels_reflections_scores = []
        all_levels_implementations = []
        cur_func_impl = None
        
        cur_prob_passed = False
        
        try:
        
            while cur_pass < pass_at_k and not is_solved:
                if is_leetcode:
                    tests_i = item['visible_tests']
                else:

                    if visible_tests:
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[item['entry_point']]['given_tests']
                        
                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(item["prompt"], model, 1)
                        

                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(item["prompt"], model, "simple", temperature=1.)
                
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(feedback)
                
                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:
                    
                    # populate memory bank if first attempt passed all visible tests
                    trajectory = {
                                    "task_id": item[primary_key],
                                    "prompt": item["prompt"],
                                    "gen_solution": cur_func_impl,
                                    "prompt_embedding": get_openai_embedding([item["prompt"]]),
                                }
                
                    # update memory bank
                    cur_prob_passed = True
                    memory_bank['positive_trajectories'].append(trajectory)
                    
                    # evaluate on hidden test cases
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
                    
                    # iterative sampling
                    # # get self-reflection-diverse
                    # reflection = gen.self_reflection_diverse(
                    #     cur_func_impl, cur_feedback, model, diverse_reflections)
                    # diverse_reflections += [reflection]
                    
                    # one-shot sampling
                    # get multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse_oneshot(
                        cur_func_impl, cur_feedback, model, diverse_reflections).split("\n\n")
                    
                    # filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    
                    # revisit later
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
                            print("skipping func impl.")
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
                            # setting based on visible/synthetic tests
                            cur_prob_passed = True

                            is_passing = exe.evaluate(
                                item["entry_point"], cur_func_impl, item["test"], timeout=10)
                            
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += int(is_passing)
                                
                            break
                    
                    pbar.close()
                    
                    #log reflection scores and given level implementations
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)

                    #sample likely implementation
                    print(reflections_scores)
                    sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                    cur_func_impl = temp_implementations[sampled_impl_idx]
                    
                    # set cur_feedback to the corresponding sampled div-reflection
                    cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    # populate memory bank
                    visible_tests_status, _, _ = exe.execute(cur_func_impl, tests_i)

                    if cur_iter == max_iters - 1 or visible_tests_status:
                        trajectory = {
                                        "task_id": item[primary_key],
                                        "prompt": item["prompt"],
                                        "gen_solution": cur_func_impl,
                                        "reflection": cur_feedback,
                                        "test_feedback": test_feedback[sampled_impl_idx],
                                        "prev_implementation": cur_func_impl_copy,
                                        "prompt_embedding": get_openai_embedding([item["prompt"]]),
                                        "refection_embedding": get_openai_embedding([cur_feedback]),
                                    }                           
                        if visible_tests_status:
                            cur_prob_passed = True
                            memory_bank['positive_trajectories'].append(trajectory)
                        else:
                            memory_bank['negative_trajectories'].append(trajectory)
                    
                    if is_solved:                        
                        break
                    
                    cur_iter += 1
                cur_pass += 1
                
        except:
            continue
        

            
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)
        
        if cur_prob_passed:
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
            
        else:
            failed_problems.append(item)
            
        #write mem-bank to file
        with open(mem_bank_file_path, 'wb') as f:
            pkl.dump(memory_bank, f)
            
        #update failed_probs.pkl
        with open(failed_probs_path, 'wb') as f:
            pkl.dump(failed_problems, f)


    print("finished first pass")
    
    memory_bank = pkl.load(open(mem_bank_file_path, 'rb'))
    failed_problems = pkl.load(open(failed_probs_path, 'rb'))

    
    # reset num_items and num_success for 2nd pass
    num_items = len(failed_problems)
    num_success = 0
    
    # # Second pass
    for i, item in enumerate(failed_problems):
        
        try:
        
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            all_levels_reflections_scores = []
            all_levels_implementations = []
            cur_func_impl = ""
            
            visible_tests_status = False
            
            while cur_pass < pass_at_k and not is_solved:
                if is_leetcode:
                    tests_i = item['visible_tests']
                else:

                    if visible_tests:
                        # Use visible test cases
                        print("using visible test cases")                
                        # tests_i = visible_tests[item[primary_key]]['given_tests']
                        tests_i = visible_tests[item['entry_point']]['given_tests']

                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(item["prompt"], model, 1)
                        

                # inject similar problems trajectory into context
                curr_emb = get_openai_embedding([item['prompt']]) 
                        
                top_k_indices, cosine_similarities = get_top_k_closest(memory_bank['positive_trajectories'], curr_emb[:, None], k=1)

                closest_match = [memory_bank['positive_trajectories'][i] for i in top_k_indices]

                PY_SIMPLE_CHAT_INSTRUCTION = (
                    "You are an AI that only responds with python code, NOT ENGLISH.\n"
                    "You will be given a function signature and its docstring by the user.\n"
                    "Write your full implementation (restate the function signature).\n"
                    f"Here are {len(closest_match)} problems and their solutions.\n\n"
                    + ''.join(
                        f"[Problem {i+1}]\n"
                        "```python\n"
                        f"{example['prompt']}\n"
                        "```\n"
                        f"[Solution {i+1}]\n"
                        "```python\n"
                        f"{example['gen_solution']}\n"
                        "```\n\n"
                        for i, example in enumerate(closest_match)
                    )
                )

                # first attempt
                cur_func_impl = gen_utils.generic_generate_func_impl(
                                            func_sig=item["prompt"],
                                            model=model,
                                            strategy='simple',
                                            num_comps=1,
                                            temperature=1.,
                                            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
                                            simple_completion_instruction=py_generate.PY_SIMPLE_COMPLETION_INSTRUCTION,
                                            code_block_instruction=py_generate.USE_PYTHON_CODEBLOCK_INSTRUCTION,
                                            parse_code_block=lambda x: parse_code_block(x, "python"),
                                            add_code_block=lambda x: add_code_block(x, "python"),
                                            prev_func_impl=None,
                                            feedback=None,
                                            self_reflection=None,
                                            reflexion_chat_instruction=None,
                                            reflexion_few_shot=None,
                                            reflexion_completion_instruction=None
                                        )
                
                
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(feedback)
                
                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:
                    visible_tests_status = True # passed visible/Synthetic test cases
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
                    
                    # iterative sampling
                    # # get self-reflection-diverse
                    # reflection = gen.self_reflection_diverse(
                    #     cur_func_impl, cur_feedback, model, diverse_reflections)
                    # diverse_reflections += [reflection]
                    
                    # one-shot sampling
                    # get multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse_oneshot(
                        cur_func_impl, cur_feedback, model, diverse_reflections).split("\n\n")
                    
                    # filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    
                    # revisit later
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
                        
                        # inject similar problems trajectory into context based on similary in reflection
                        curr_emb = get_openai_embedding([reflection])
                        filtered_trajectories = [traj for traj in memory_bank['positive_trajectories'] if "refection_embedding" in traj.keys()]
                        
                        if len(filtered_trajectories):
                            top_k_indices, cosine_similarities = get_top_k_closest( filtered_trajectories, 
                                                                                    curr_emb[:, None], 
                                                                                    k=1, 
                                                                                    similarity_axis = "refection_embedding")
                        else:
                            top_k_indices, cosine_similarities = get_top_k_closest( filtered_trajectories, 
                                                                                    curr_emb[:, None], 
                                                                                    k=1, 
                                                                                    similarity_axis = "prompt_embedding")
                            
                        closest_match = filtered_trajectories[top_k_indices[0]]
                        
                        # apply self-reflection in the next attempt
                        PY_FEW_SHOT = f'''Example 1:
    [previous impl]:
    ```python
    {closest_match['prev_implementation']}
    ```

    [unit test results from previous impl]:
    {closest_match["test_feedback"][0]}

    [reflection on previous impl]:
    {closest_match['reflection']}

    [improved impl]:
    ```python
    {closest_match['gen_solution']}
    ```
    '''
                        cur_func_impl = gen_utils.generic_generate_func_impl(
                                                    func_sig=item["prompt"],
                                                    model=model,
                                                    strategy='reflexion',
                                                    num_comps=1,
                                                    temperature=1.,
                                                    simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
                                                    simple_completion_instruction=py_generate.PY_SIMPLE_COMPLETION_INSTRUCTION,
                                                    code_block_instruction=py_generate.USE_PYTHON_CODEBLOCK_INSTRUCTION,
                                                    parse_code_block=lambda x: parse_code_block(x, "python"),
                                                    add_code_block=lambda x: add_code_block(x, "python"),
                                                    prev_func_impl=cur_func_impl_copy,
                                                    feedback=cur_feedback,
                                                    self_reflection=reflection,
                                                    reflexion_chat_instruction=py_generate.PY_REFLEXION_CHAT_INSTRUCTION,
                                                    reflexion_few_shot=PY_FEW_SHOT,
                                                    reflexion_completion_instruction=py_generate.PY_REFLEXION_COMPLETION_INSTRUCTION
                                                    )

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:

                            print("skipping func impl.")
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

                    #sample likely implementation
                    print(reflections_scores)
                    sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                    cur_func_impl = temp_implementations[sampled_impl_idx]
                    
                    # set cur_feedback to the corresponding sampled div-reflection
                    cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    # populate memory bank
                    visible_tests_status, _, _ = exe.execute(cur_func_impl, tests_i)
                    if cur_iter == max_iters - 1:
                        trajectory = {
                                        "task_id": item[primary_key], 
                                        "prompt": item["prompt"],
                                        "gen_solution": cur_func_impl,
                                        "reflection": cur_feedback,
                                        "prompt_embedding": get_openai_embedding([item["prompt"]]),
                                        "refection_embedding": get_openai_embedding([cur_feedback]),
                                    }                           
                        if visible_tests_status:
                            memory_bank['positive_trajectories'].append(trajectory)
                        else:
                            memory_bank['negative_trajectories'].append(trajectory)
                    
                    if is_solved:
                        break
                    
                    cur_iter += 1
                cur_pass += 1
                
        except:
            continue
            
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
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}')

        
    print(colored(gpt_usage(backend=model_name), 'blue'))
