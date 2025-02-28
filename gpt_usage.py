completion_tokens = prompt_tokens = cache_creation_input_tokens= cache_read_input_tokens = 0

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens, cache_creation_input_tokens, cache_read_input_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.015 + prompt_tokens / 1000 *  0.005
    elif backend == "gpt-4o-mini":
        cost = completion_tokens / 1000 * 0.0006 + prompt_tokens / 1000 *  0.00015
    elif backend == "o1":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.015
    elif backend == "o1-mini":
        cost = completion_tokens / 1000 * 0.012 + prompt_tokens / 1000 *  0.003
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "claude_3_sonnet" or backend == "claude_35_sonnet":
        cost = completion_tokens / 1000 * 0.015 + prompt_tokens / 1000 * 0.003
    elif backend == "llama3_1_8b":
        cost = completion_tokens / 1000 * 0.00022 + prompt_tokens / 1000 * 0.00022
    elif backend == "llama3_1_70b":
        cost = completion_tokens / 1000 * 0.00099 + prompt_tokens / 1000 * 0.00099
    elif backend == "llama3_1_405b":
        cost = completion_tokens / 1000 * 0.016 + prompt_tokens / 1000 * 0.00532 
        
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost, "cache_creation_input_tokens" : cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens}