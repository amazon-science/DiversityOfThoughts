# set --model to one of the supported models from {gpt-4o, gpt-4o-mini, o1, gpt-3.5-turbo, claude_35_sonnet, llama3_1_405b, llama3_1_70b, llama3_1_8b}
# Llama and Sonnet models require Bedrock API setup
# GPT models can be used by setting API key using the command: export OPENAI_API_KEY="{paste api key here}"

# DoT GPT-4o-mini on HumanEval Full
python main.py \
    --run_name "DoT_gpt-4o-mini_pass1_humanEvalFull_2" \
    --root_dir {SAVE FOLDER PATH} \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "gpt-4o-mini" \
    --pass_at_k "1" \
    --max_iters "3" \
    --verbose


# DoT-bank GPT-4o-mini on HumanEval Full
python main.py \
    --run_name "DoTBank_gpt-4o-mini_pass1_humanEvalFull" \
    --root_dir {SAVE FOLDER PATH} \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "gpt-4o-mini" \
    --pass_at_k "1" \
    --max_iters "3" \
    --verbose

# DoT GPT-4o-mini on LeetCodeHardGym
python main.py \
    --run_name "DoT_gpt-4o_mini_LC_Hard40" \
    --root_dir {SAVE FOLDER PATH} \
    --dataset_path benchmarks/LC_hard_uncontaminated.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "gpt-4o-mini" \
    --pass_at_k "1" \
    --max_iters "5" \
    --is_leetcode \
    --verbose

# DoT-bank GPT-4o-mini on LeetCodeHardGym
python main.py \
    --run_name "DoTBank_gpt-4o_mini_LC_Hard40" \
    --root_dir {SAVE FOLDER PATH} \
    --dataset_path benchmarks/LC_hard_uncontaminated.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "gpt-4o-mini" \
    --pass_at_k "1" \
    --max_iters "5" \
    --is_leetcode \
    --verbose